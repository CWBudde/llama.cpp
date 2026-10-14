#pragma once

// Bookkeeping of a bounded expert cache, independent of ggml: which expert of a layer sits in which
// slot, which loads are queued or running, and what they cost. The loads themselves go through a
// callback, so the logic can be tested without a model.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// counters of one phase (prefill or decode)
struct llama_expert_cache_counts {
    uint64_t steps          = 0; // remaps: one per layer and ubatch
    uint64_t needed         = 0; // distinct experts the remaps needed
    uint64_t hits           = 0; // needed and ready
    uint64_t hits_loading   = 0; // needed while a prefetch of it was queued or running: awaited, not repeated
    uint64_t misses         = 0; // needed and not in the cache: loaded on demand
    uint64_t evictions      = 0; // a ready expert replaced
    uint64_t demand_bytes   = 0; // read for misses
    uint64_t prefetch_bytes = 0; // read for prefetches
    uint64_t prefetch_issued    = 0;
    uint64_t prefetch_used      = 0; // a prefetched expert needed before it was evicted
    uint64_t prefetch_wasted    = 0; // a prefetched expert evicted before it was needed
    uint64_t prefetch_cancelled = 0; // dropped from the queue before it started
    uint64_t prefetch_skipped   = 0; // not issued: resident or loading already, or its layer in flight
    uint64_t load_failures      = 0;
    double   wait_seconds = 0; // remaps blocked on loads: the cost on the critical path
    double   io_seconds   = 0; // the loads' own durations, summed over the I/O threads
    double   io_prefetch_seconds = 0; // the part of io_seconds that prefetches took: overlapped with
                                      // compute, unless a remap awaited the prefetch (hits_loading)
};

struct llama_expert_slots {
    // load(layer, expert, slot) reads one expert into one slot and returns false on failure
    using load_fn = std::function<bool(int layer, int expert, int slot)>;

    llama_expert_slots(std::vector<int> n_slots, int n_expert, std::vector<size_t> expert_bytes, int n_threads, load_fn load);
    ~llama_expert_slots();

    llama_expert_slots(const llama_expert_slots &) = delete;
    llama_expert_slots & operator=(const llama_expert_slots &) = delete;

    // remap maps the router's ids of one ubatch of layer l (n ids) to slots, loading every expert
    // that is not ready and waiting for all of them; out gets the slot of each id. The layer is in
    // flight from then until the next remap: no prefetch replaces its slots. It returns false, with
    // nothing changed, when the ubatch needs more distinct experts than the layer has slots or an
    // id is out of range, and false after a demand load failed.
    bool remap(int l, const int32_t * ids, int n, bool decode, int32_t * out);

    // prefetch queues loads of the experts of layer l that are neither ready nor loading, best
    // first, into its least recently used slots. Nothing is queued for the layer in flight.
    void prefetch(int l, const int * experts, int n, bool decode);

    // distinct experts of the last remap, in the order the ids met them
    const std::vector<int> & last_needed() const { return needed; }

    int n_slots(int l) const { return (int) layers[l].slots.size(); }
    int min_slots() const;

    // the slot an expert of layer l sits in, ready or loading, or -1
    int slot_of(int l, int e) const;

    // waits until no load is queued or running
    void drain();

    // whether a demand load failed, and which: a remap then returns false
    bool failure(int & l, int & e) const;

    llama_expert_cache_counts counts(bool decode) const;

private:
    enum slot_state { EMPTY, LOADING, READY };

    struct slot {
        int        expert   = -1;
        slot_state state    = EMPTY;
        uint64_t   last_use = 0;
        bool       prefetched   = false; // loaded by a prefetch and not needed since
        bool       queued       = false; // its load waits in a queue
        bool       demand       = false; // its load is a demand load
        bool       phase_decode = false; // the phase of the step that issued its load
    };

    struct layer {
        std::vector<slot> slots;
        std::vector<int>  where; // per expert: its slot, or -1
    };

    struct job {
        int l, slot;
    };

    int pick_victim(int l) const;
    void assign(int l, int s, int e, bool demand, bool decode);
    void worker();
    llama_expert_cache_counts & counts_of(bool decode) { return decode ? c_decode : c_prefill; }

    const int n_expert;
    const std::vector<size_t> expert_bytes; // per layer
    load_fn load;

    std::vector<layer> layers;
    int      in_flight = -1;
    uint64_t tick      = 0;
    std::vector<int>  needed;
    std::vector<char> keep; // per expert: needed by the remap running now

    llama_expert_cache_counts c_prefill;
    llama_expert_cache_counts c_decode;

    mutable std::mutex      mtx;
    std::condition_variable cv_work; // a job was queued, or stop
    std::condition_variable cv_done; // a load finished
    std::deque<job> q_demand;
    std::deque<job> q_prefetch;
    int  running = 0;
    bool stop    = false;
    bool demand_failed = false;
    int  failed_layer  = -1;
    int  failed_expert = -1;
    std::vector<std::thread> threads;
};

// A port of the decayed-count Markov predictor of the repository's internal/locality (predict.go),
// same-token horizon only: before layer l of a decode step it ranks the experts of layer l+1 from
// layer l-1's experts and layer l+1's decayed use counts.
struct llama_expert_markov {
    llama_expert_markov(int n_layer, int n_expert, int half_life, double mix);

    // observe: the step (token first, layer l), now run, used the distinct experts ids
    void observe(int l, int64_t first, bool decode, const int * ids, int n);

    // predict ranks, before the step (token first, layer l), the experts of layer l+1; it returns
    // that layer and its k best experts in out, best first, the lower expert first of equal
    // scores, or -1 and nothing when it predicts nothing
    int predict(int l, int64_t first, bool decode, int k, std::vector<int> & out);

    size_t bytes() const;

private:
    const int    n_layer, n_expert;
    const int    half_life;
    const double mix;

    std::vector<double>  freq;  // per layer and expert: its uses, decayed to the token in seen
    std::vector<double>  sum;   // per layer
    std::vector<int64_t> seen;  // per layer
    std::vector<float>   cross; // per layer l, from expert of l-2, to expert of l
    std::vector<float>   x_row;
    std::vector<std::vector<int>> last; // per layer: its latest decoded set
    std::vector<int64_t> last_tok;      // per layer: that set's token, -1 for none
    std::vector<double>  score;
};

// the indexes of the k best scores, best first, the lower index first of equal ones (predict.go's topK)
void llama_expert_top_k(const std::vector<double> & score, int k, std::vector<int> & out);
