#pragma once

// [EXPERIMENTAL] A bounded cache of routed experts (Kolibri only).
//
// The stacked expert tensors of every layer stay in the GGUF: they are mapped, but never touched.
// Each layer instead has slot tensors of n_slots experts in device memory. Before the expert
// matmuls of a ubatch, a CPU op maps the router's ids to slots, reading every expert that is not in
// a slot with pread (F_NOCACHE on macOS unless buffered, so the bytes do not stay in the page cache); the matmuls
// then read the slots. The router chooses every expert: the expert weights keep its own ids, and a
// prediction only ever decides what to read ahead.

#include "llama.h"
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

    llama_expert_cache(const llama_expert_cache_params & p, int n_layer, int n_expert, int n_expert_used);
    ~llama_expert_cache();

    // init takes the stacked expert tensors and their offsets in the file the loader opened as fd,
    // allocates each layer's slots for that layer's device, opens the file for its own reads and
    // starts the I/O threads; it throws on failure. The slots are host memory, which pread writes:
    // a CPU buffer, or host memory the device uses in place (buffer_from_host_ptr, e.g. Metal).
    void init(const std::vector<tensors> & stacked, const std::vector<ggml_backend_dev_t> & devs, int fd);

    // the largest ubatch whose experts always fit: the fewest slots of a layer over n_expert_used
    uint32_t max_ubatch() const;

    // one context per model
    bool attach();
    void detach();
    bool in_use() const { return attached; }

    // the slot ids of the router's selected experts [n_expert_used, n_tokens], from a CPU op
    ggml_tensor * build_remap(ggml_context * ctx, ggml_tensor * selected_experts, int il);

    ggml_tensor * gate(int il) const { return slot_t[il].t[0]; }
    ggml_tensor * up  (int il) const { return slot_t[il].t[1]; }
    ggml_tensor * down(int il) const { return slot_t[il].t[2]; }

    // the remap op's work
    void remap(int il, const ggml_tensor * ids, ggml_tensor * dst);

    // the decode timing to add to the stats; the model owns it
    void set_decode_diag(const llama_decode_diag * d) { diag = d; }

    std::string stats_json() const;

private:
    bool load(int il, int e, int s);

    const int n_layer, n_expert, n_expert_used;
    std::vector<int> n_slots; // per layer
    int    n_threads;
    int    prefetch_k;
    int    prefetch_half_life;
    double prefetch_mix;
    std::string stats_path;
    bool buffered; // misses read through the page cache

    std::vector<tensors>  stacked;
    std::vector<tensors>  slot_t; // the slot tensors; offs unused
    std::vector<size_t>   expert_bytes; // per layer
    std::vector<layer_ud> ud;

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
