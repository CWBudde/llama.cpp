#pragma once

// [EXPERIMENTAL] A bounded cache of routed experts (Kolibri only).
//
// The stacked expert tensors of every layer stay in the GGUF: they are mapped, but never touched.
// Each layer instead has slot tensors of n_slots experts in device memory. Before the expert
// matmuls of a ubatch, a CPU op maps the router's ids to slots, reading every expert that is not in
// a slot with pread (F_NOCACHE on macOS, so the bytes do not stay in the page cache); the matmuls
// then read the slots. The router chooses every expert: the expert weights keep its own ids, and a
// prediction only ever decides what to read ahead.

#include "llama.h"
#include "llama-expert-slots.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

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
    // allocates the slots in buft, opens the file for its own reads and starts the I/O threads;
    // it throws on failure
    void init(const std::vector<tensors> & stacked, int fd, ggml_backend_buffer_type_t buft);

    // the largest ubatch whose experts always fit: the fewest slots of a layer over n_expert_used
    uint32_t max_ubatch() const;

    // one context per model
    bool attach();
    void detach();

    // the slot ids of the router's selected experts [n_expert_used, n_tokens], from a CPU op
    ggml_tensor * build_remap(ggml_context * ctx, ggml_tensor * selected_experts, int il);

    ggml_tensor * gate(int il) const { return slot_t[il].t[0]; }
    ggml_tensor * up  (int il) const { return slot_t[il].t[1]; }
    ggml_tensor * down(int il) const { return slot_t[il].t[2]; }

    // the remap op's work
    void remap(int il, const ggml_tensor * ids, ggml_tensor * dst);

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

    std::vector<tensors>  stacked;
    std::vector<tensors>  slot_t; // the slot tensors; offs unused
    std::vector<size_t>   expert_bytes; // per layer
    std::vector<layer_ud> ud;

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::string path;
    int fd = -1;

    std::unique_ptr<llama_expert_slots>  slots;
    std::unique_ptr<llama_expert_markov> markov;
    std::vector<int> pred;

    bool    attached  = false;
    int64_t tok_first = 0; // the first token of the ubatch being computed
    int64_t tok_next  = 0;
    bool    decode    = false;

    // process counters at init, for the deltas in the stats
    uint64_t disk_read0 = 0;
    uint64_t pageins0   = 0;
    uint64_t faults0    = 0;
};
