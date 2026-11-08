#pragma once

// [EXPERIMENTAL] A bounded cache of routed experts (Kolibri only).
//
// The stacked expert tensors of every layer stay in the GGUF: they are mapped, but never touched.
// Each layer instead has slot tensors of n_slots experts in device memory. Before the expert
// matmuls of a ubatch, a CPU op maps the router's ids to slots, reading every expert that is not in
// a slot with pread (F_NOCACHE on macOS unless buffered, so the bytes do not stay in the page cache); the matmuls
// then read the slots. The router chooses every expert: the expert weights keep its own ids, and a
// prediction only ever decides what to read ahead.
//
// With waves, a ubatch whose experts do not fit the slots is not capped: its expert matmuls run once
// per wave of (slots - n_expert_used)/2 of its experts. The graph, fixed at build, holds as few waves
// as it takes with up to slots - n_expert_used experts each, and a large union grows its waves to
// fit. Before each wave a CPU op loads the wave and queues the next one behind it; the pairs of other
// waves read some slot of the wave and are multiplied by 0, so the sum over the waves adds every
// pair's own result once.

#include "llama.h"
#include "llama-expert-sidefile.h"
#include "llama-expert-slots.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// [EXPERIMENTAL] the decode timing that --expert-cache-stats reports, with or without the cache:
// the context adds every ubatch of one token
struct llama_decode_diag {
    uint64_t tokens        = 0; // decode ubatches, one token each
    double   wall_seconds  = 0; // their wall time: graph build or reuse, compute and the output copy, synchronized
    uint64_t splits        = 0; // the scheduler's graph splits, summed over them
    uint64_t metal_commits = 0; // Metal command buffers committed during them (process-wide, so with
                                // concurrent contexts also those of the others)
    bool     has_metal     = false; // whether a Metal backend counts its commits (else metal_commits is null)

    // contexts of one model decode concurrently: they update the fields above only through these
    void set_metal(bool has);
    void add(double seconds, uint64_t n_splits, uint64_t n_commits);

    // the fields above as JSON members, without braces
    std::string json_fields() const;

    // the stats of a model without the cache: the decode timing and the process counters
    // since start(); write() puts them into path, when the model is freed
    std::string path;
    void start();
    std::string stats_json() const;
    void write() const;

private:
    mutable std::mutex mtx;

    uint64_t disk_read0 = 0;
    uint64_t pageins0   = 0;
    uint64_t faults0    = 0;
};

struct llama_expert_cache {
    // the gate, up and down tensors of one layer, with their data offsets in the file
    struct tensors {
        ggml_tensor * t[3]    = {};
        size_t        offs[3] = {};
    };

    // userdata of the remap op of one layer
    struct layer_ud {
        llama_expert_cache * cache;
        int il;
    };

    // userdata of the wave ops of one layer and wave
    struct wave_ud {
        llama_expert_cache * cache;
        int il;
        int w;
    };

    llama_expert_cache(const llama_expert_cache_params & p, int n_layer, int n_expert, int n_expert_used);
    ~llama_expert_cache();

    // init takes the stacked expert tensors and their offsets in the file the loader opened as fd,
    // allocates each layer's slots for that layer's device, opens the file for its own reads and
    // starts the I/O threads; it throws on failure. The slots are host memory, which pread writes:
    // a CPU buffer, or host memory the device uses in place (buffer_from_host_ptr, e.g. Metal).
    void init(const std::vector<tensors> & stacked, const std::vector<ggml_backend_dev_t> & devs, int fd);

    // whether t is one of the stacked expert tensors this cache reads in place of the model's
    bool holds(const ggml_tensor * t) const;

    // the largest ubatch whose experts always fit: the fewest slots of a layer over n_expert_used;
    // unbounded with waves
    uint32_t max_ubatch() const;

    // the same without waves: the GPU remap's slot ids are sized for it, and a larger ubatch runs
    // on the CPU remap (see gpu_graph)
    uint32_t gate_ubatch() const;

    // whether waves are on: asked for, and every layer has at least 3*n_expert_used slots
    bool waves_on() const;

    // the waves layer il's expert matmuls take for a ubatch of n_tokens, 1 when its experts always fit
    int n_waves(int il, int n_tokens) const;

    // the graph nodes the waves add for a ubatch of n_tokens
    uint32_t wave_nodes(int n_tokens) const;

    // one context per model
    bool attach();
    void detach();
    bool in_use() const { return attached; }

    // the slot ids of the router's selected experts [n_expert_used, n_tokens], from a CPU op
    ggml_tensor * build_remap(ggml_context * ctx, ggml_tensor * selected_experts, int il);

    // [EXPERIMENTAL] the GPU remap (--expert-cache-gpu-remap): no CPU op; the layer's matmuls read
    // their ids from the slot ids, which the Metal backend's gate fills from the layer's table
    bool gpu_remap() const { return gpu_remap_on; }
    void bind_gate(ggml_tensor * mul_mat_id, int il) const;

    // whether the graph of a ubatch of n_tokens runs behind the gates: with the GPU remap, every
    // ubatch whose experts always fit the slots. A larger one, which only waves allow, takes the CPU
    // remap and its waves. Without waves every graph is gated, as the context caps its ubatch anyway
    // (only a reserved graph can be larger). The graph's n_tokens decides, so a reused graph keeps
    // its path.
    bool gpu_graph(uint32_t n_tokens) const { return gpu_remap_on && (!waves_on() || n_tokens <= gate_ubatch()); }

    // after a graph on the CPU remap in a GPU remap context: the waves moved experts the tables do
    // not know of, so every layer's table is copied again, once the loads are done
    void refresh_tables();

    // the Metal backend calls these, with the GPU idle; see ggml_metal_graph_compute_gated
    void gate_remap(int il, const int32_t * ids, int n_ids, int n_tokens, bool stopped);
    void gate_run(int n_tokens, bool stopped);

    // after a graph with experts: every layer has been through its gate, else one left the GPU;
    // has_outputs false: the last layer has no rows, hence no gate
    void gate_check_graph(bool has_outputs);

    // after a graph that did not complete: the next one starts at layer 0 again
    void gate_abandon_graph();

    // wave w's slot ids [n_expert_used, n_tokens] for the contiguous router ids, from a CPU op that
    // runs after after (the previous wave's sum, or null): it plans the waves at w 0, loads wave w
    // and waits for it
    ggml_tensor * build_wave_ids(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * after, int il, int w);

    // 1 for the pairs of wave w, else 0 [1, n_expert_used, n_tokens], from a CPU op after wave_ids
    ggml_tensor * build_wave_mask(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * wave_ids, int il, int w);

    ggml_tensor * gate(int il) const { return slot_t[il].t[0]; }
    ggml_tensor * up  (int il) const { return slot_t[il].t[1]; }
    ggml_tensor * down(int il) const { return slot_t[il].t[2]; }

    // the ops' work
    void remap(int il, const ggml_tensor * ids, ggml_tensor * dst);
    void remap_ids(int il, const int32_t * ids, int n_ids, int n_tokens, int32_t * out);
    void wave_ids(int il, int w, const ggml_tensor * ids, ggml_tensor * dst);
    void wave_mask(int il, int w, const ggml_tensor * ids, ggml_tensor * dst);

    // the decode timing to add to the stats; the model owns it
    void set_decode_diag(const llama_decode_diag * d) { diag = d; }

    std::string stats_json() const;

    // the wait mode as --expert-cache-wait spells it
    std::string wait_name() const;

private:
    bool load(int il, int e, int s, uint64_t & preads);
    void begin(int il, int n_tokens);
    void observe(int il);
    [[noreturn]] void fail(int il, int n_tokens);

    const int n_layer, n_expert, n_expert_used;
    std::vector<int> n_slots; // per layer
    int    n_threads;
    int    prefetch_k;
    int    prefetch_half_life;
    double prefetch_mix;
    std::string stats_path;
    bool buffered; // misses read through the page cache
    std::string sidefile_path;
    bool gpu_remap_on;
    llama_expert_wait wait; // how a remap waits for its loads
    bool waves;    // asked for: see waves_on()

    std::vector<tensors>  stacked;
    std::vector<tensors>  slot_t; // the slot tensors; offs unused
    std::vector<size_t>   expert_bytes; // per layer
    std::vector<layer_ud> ud;
    std::vector<std::vector<wave_ud>> wud; // per layer, per wave

    // the GPU remap: per layer, the table {n_expert}, each expert's slot or -1, and the slot ids
    // {n_expert_used, 2*gate_ubatch}: the slot ids, then the gate's log of the router's ids
    std::vector<ggml_tensor *> gate_table;
    std::vector<ggml_tensor *> gate_slots;
    std::vector<int32_t>       gate_scratch;
    int      gate_seen = 0;           // layers through their gate in this graph
    int      gate_il   = -1;          // the last of them
    uint64_t gate_runs[2]  = {0, 0};  // command buffers per phase
    uint64_t gate_stops[2] = {0, 0};  // those a gate stopped

    // the slots of the layers of one device
    struct group {
        ggml_backend_dev_t    dev  = nullptr;
        ggml_context *        ctx  = nullptr;
        ggml_backend_buffer_t buf  = nullptr;
        void *                host = nullptr; // the mapping the device's buffer wraps, if any
        size_t                size = 0;
    };
    std::vector<group> groups;
    size_t slot_bytes() const;

    std::string path;
    int fd = -1;

    // with --expert-cache-sidefile the misses come from here, one pread per expert; fd only checks the GGUF's identity
    std::unique_ptr<llama_expert_sidefile> sidefile;
    std::vector<std::array<llama_expert_stacked, 3>> rows; // the GGUF path's ranges, per layer

    std::unique_ptr<llama_expert_slots>  slots;
    std::unique_ptr<llama_expert_markov> markov;
    std::vector<int> pred;

    std::atomic<bool> attached{false};
    int64_t tok_first = 0; // the first token of the ubatch being computed
    int64_t tok_next  = 0;
    bool    decode    = false;

    // the remap op's time per phase, prefill and decode; minus the wait on loads, it is the
    // bookkeeping's own cost (remap_seconds in the stats)
    double op_seconds[2] = {0, 0};
    const llama_decode_diag * diag = nullptr;

    // process counters at init, for the deltas in the stats
    uint64_t disk_read0 = 0;
    uint64_t pageins0   = 0;
    uint64_t faults0    = 0;
};
