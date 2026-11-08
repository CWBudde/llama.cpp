#pragma once

// Bookkeeping of a bounded expert cache, independent of ggml: which expert of a layer sits in which
// slot, which loads are queued or running, and what they cost. The loads themselves go through a
// callback, so the logic can be tested without a model.

#include <chrono>
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
    uint64_t loads  = 0; // experts read into a slot
    uint64_t preads = 0; // the system calls those reads took, failed loads included
    double   wait_seconds = 0; // remaps blocked on loads: the cost on the critical path
    double   io_seconds   = 0; // the loads' own durations, summed over the I/O threads
    double   io_prefetch_seconds = 0; // the part of io_seconds that prefetches took: overlapped with
                                      // compute, unless a remap awaited the prefetch (hits_loading)
    uint64_t waves       = 0; // waves staged with experts of their own (prefill ubatches above the slots)
    uint64_t waves_built = 0; // the passes the graph ran: waves, plus those a plan left empty (full GEMMs, all parked)
    uint64_t wave_parked = 0; // pairs of other waves parked on a slot of the staged wave, masked to 0
};

// how a remap or a wave stage waits for the loads it needs (--expert-cache-wait). Every mode checks
// the same predicate under the lock and ends on the same events; they differ in what the thread does
// between two checks.
struct llama_expert_wait {
    enum mode_t {
        SLEEP,  // on the condition variable the I/O threads notify
        SPIN,   // busy: release the lock, spin about 10 us with a pause hint, take it again
        HYBRID, // SPIN for at most spin_us per remap or wave stage, then SLEEP
        QOS,    // SLEEP, the waiting thread raised once to QOS_CLASS_USER_INTERACTIVE (Apple; elsewhere SLEEP)
    };
    mode_t mode    = SLEEP;
    int    spin_us = 0; // HYBRID only
};

// the experts per wave for a layer of n_slots slots and k experts per token: a wave, the next
// one's preloads and up to k parking slots must fit at once, so 0 (no waves) below 3k slots
int llama_expert_wave_cap(int n_slots, int k);

// the waves a graph holds for at most n_touch experts: every wave runs its GEMMs whether a plan
// fills it or not, so as few as it takes with waves of up to n_slots - k experts. A plan that
// keeps to the cap of llama_expert_wave_cap fits in them, unless its union is large; then its
// waves grow, and fewer of the next wave's experts load ahead.
int llama_expert_wave_count(int n_touch, int n_slots, int k);

struct llama_expert_slots {
    // load(layer, expert, slot, preads) reads one expert into one slot, adds its read calls to
    // preads and returns false on failure
    using load_fn = std::function<bool(int layer, int expert, int slot, uint64_t & preads)>;

    llama_expert_slots(std::vector<int> n_slots, int n_expert, std::vector<size_t> expert_bytes, int n_threads, load_fn load,
                       llama_expert_wait wait = {});
    ~llama_expert_slots();

    llama_expert_slots(const llama_expert_slots &) = delete;
    llama_expert_slots & operator=(const llama_expert_slots &) = delete;

    // remap maps the router's ids of one ubatch of layer l (n ids) to slots, loading every expert
    // that is not ready and waiting for all of them; out gets the slot of each id. The layer is in
    // flight from then until the next remap: no prefetch replaces its slots. It returns false, with
    // nothing changed, when the ubatch needs more distinct experts than the layer has slots or an
    // id is out of range, and false after a demand load failed.
    bool remap(int l, const int32_t * ids, int n, bool decode, int32_t * out);

    // Waves run a ubatch that needs more distinct experts than a layer has slots in passes over
    // its union of experts. plan_waves cuts the union of layer l's n ids (rows of k per token) into
    // waves of at most cap experts, the least used first, so the most used are what the slots
    // hold afterwards; first use breaks ties. With n_built > 0 a large union gets larger waves, so
    // that it takes at most n_built. It returns the number of waves, or -1 for an id out of range,
    // a cap below k, or waves that would leave the layer fewer than k other slots.
    int plan_waves(int l, const int32_t * ids, int n, int k, int cap, int n_built = 0);

    // stage_wave loads wave w of layer l's plan, queues the next wave's loads behind it and waits
    // for wave w. out gets the slot of each id of wave w; every other id gets a slot of a ready
    // expert, distinct within its row of k, for the masked GEMM. The slots out names and the
    // experts of wave w stay put until the next stage of the layer, so the GEMMs may read them
    // until then. A wave beyond the plan (the graph holds what the largest union takes) parks every id.
    // It returns false after a demand load failed.
    bool stage_wave(int l, int w, const int32_t * ids, int n, int k, int32_t * out);

    // wave_mask sets mask[i] to 1 if ids[i] is in wave w of layer l's plan, else to 0
    void wave_mask(int l, int w, const int32_t * ids, int n, float * mask) const;

    // the wave of expert e in layer l's plan, or -1
    int wave_of(int l, int e) const;

    // prefetch queues loads of the experts of layer l that are neither ready nor loading, best
    // first, into its least recently used slots. Nothing is queued for the layer in flight.
    void prefetch(int l, const int * experts, int n, bool decode);

    // distinct experts of the last remap or wave plan, in the order the ids met them
    const std::vector<int> & last_needed() const { return needed; }

    int n_slots(int l) const { return (int) layers[l].slots.size(); }
    int min_slots() const;

    // the slot an expert of layer l sits in, ready or loading, or -1
    int slot_of(int l, int e) const;

    // [EXPERIMENTAL] layer l's table for the GPU remap, n_expert entries: an expert's slot when it
    // holds its bytes, else -1. A loading expert is -1, so a gate that reads the table stops there.
    void copy_table(int l, int32_t * table) const;

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
        std::vector<int>  wave;      // per expert: its wave in the plan, or -1
        std::vector<char> preloaded; // per expert: loaded ahead by the previous stage, counted there
        std::vector<std::vector<int>> plan; // the experts of each wave
    };

    struct job {
        int l, slot;
    };

    // the spin budget of the waits of one remap or wave stage (HYBRID)
    struct wait_budget {
        bool started = false;
        std::chrono::steady_clock::time_point spin_end;
    };

    // one wait for a load to end: the caller holds the lock and checks its predicate again after
    // it; the lock is held again on return
    void wait_load(std::unique_lock<std::mutex> & lock, wait_budget & budget);

    int pick_victim(int l, int w_now = -1) const;
    void drop_prefetches(int l);
    int  take_victim(int l, std::unique_lock<std::mutex> & lock, wait_budget & budget, llama_expert_cache_counts & c, bool block,
                     int w_now = -1);
    void assign(int l, int s, int e, bool demand, bool decode);
    void worker();
    llama_expert_cache_counts & counts_of(bool decode) { return decode ? c_decode : c_prefill; }

    const int n_expert;
    const std::vector<size_t> expert_bytes; // per layer
    load_fn load;
    const llama_expert_wait wait;

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
