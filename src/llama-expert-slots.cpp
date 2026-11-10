#include "llama-expert-slots.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif

llama_expert_slots::llama_expert_slots(std::vector<int> n_slots, int n_expert, std::vector<size_t> expert_bytes, int n_threads, load_fn load,
                                       llama_expert_wait wait)
    : n_expert(n_expert), expert_bytes(std::move(expert_bytes)), load(std::move(load)), wait(wait), keep(n_expert, 0) {
    assert(n_slots.size() == this->expert_bytes.size());
    layers.resize(n_slots.size());
    for (size_t l = 0; l < n_slots.size(); ++l) {
        layers[l].slots.resize(std::max(n_slots[l], 0));
        layers[l].where.assign(n_expert, -1);
        layers[l].wave.assign(n_expert, -1);
        layers[l].preloaded.assign(n_expert, 0);
    }
    for (int i = 0; i < std::max(n_threads, 1); ++i) {
        threads.emplace_back([this] { worker(); });
    }
}

llama_expert_slots::~llama_expert_slots() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
        q_demand.clear();
        q_prefetch.clear();
    }
    cv_work.notify_all();
    for (auto & t : threads) {
        t.join();
    }
}

int llama_expert_slots::min_slots() const {
    int m = layers.empty() ? 0 : (int) layers[0].slots.size();
    for (const auto & L : layers) {
        m = std::min(m, (int) L.slots.size());
    }
    return m;
}

int llama_expert_slots::slot_of(int l, int e) const {
    std::lock_guard<std::mutex> lock(mtx);
    return layers[l].where[e];
}

void llama_expert_slots::fill_table(int l, int32_t * table) const {
    const auto & L = layers[l];
    for (int e = 0; e < n_expert; ++e) {
        const int s = L.where[e];
        table[e] = s >= 0 && L.slots[s].state == READY ? s : -1;
    }
}

void llama_expert_slots::copy_table(int l, int32_t * table) const {
    std::lock_guard<std::mutex> lock(mtx);
    fill_table(l, table);
}

// off the critical path of a gate, like drain(): it sleeps in every mode
void llama_expert_slots::refresh_tables(int32_t * const * tables) {
    std::unique_lock<std::mutex> lock(mtx);
    cv_done.wait(lock, [this] { return q_demand.empty() && q_prefetch.empty() && running == 0; });
    for (int l = 0; l < (int) layers.size(); ++l) {
        fill_table(l, tables[l]);
    }
}

llama_expert_cache_counts llama_expert_slots::counts(bool decode) const {
    std::lock_guard<std::mutex> lock(mtx);
    return decode ? c_decode : c_prefill;
}

void llama_expert_slots::add_skipped(bool decode, int n) {
    std::lock_guard<std::mutex> lock(mtx);
    counts_of(decode).skipped += n;
}

bool llama_expert_slots::failure(int & l, int & e) const {
    std::lock_guard<std::mutex> lock(mtx);
    l = failed_layer;
    e = failed_expert;
    return demand_failed;
}

// a pause hint for a busy wait: the core stays awake and at its clock, without hammering the pipeline
static inline void cpu_relax() {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
    __yield();
#endif
}

// the lock released for d of busy waiting, then taken again
static void spin_unlocked(std::unique_lock<std::mutex> & lock, std::chrono::steady_clock::duration d) {
    lock.unlock();
    const auto until = std::chrono::steady_clock::now() + d;
    while (std::chrono::steady_clock::now() < until) {
        cpu_relax();
    }
    lock.lock();
}

// the calling thread raised to the interactive QoS class, once per thread (Apple only)
static void raise_qos_once() {
#if defined(__APPLE__)
    static thread_local bool raised = false;
    if (!raised) {
        raised = true;
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
#endif
}

void llama_expert_slots::wait_load(std::unique_lock<std::mutex> & lock, wait_budget & budget) {
    constexpr auto spin_step = std::chrono::microseconds(10);
    switch (wait.mode) {
        case llama_expert_wait::SPIN:
            spin_unlocked(lock, spin_step);
            return;
        case llama_expert_wait::HYBRID: {
            const auto now = std::chrono::steady_clock::now();
            if (!budget.started) {
                budget.started  = true;
                budget.spin_end = now + std::chrono::microseconds(wait.spin_us);
            }
            if (now < budget.spin_end) {
                spin_unlocked(lock, std::min<std::chrono::steady_clock::duration>(spin_step, budget.spin_end - now));
                return;
            }
            // the caller checked its predicate under this lock, which the wait releases atomically: no
            // notify is lost between the check and the sleep
            cv_done.wait(lock);
            return;
        }
        case llama_expert_wait::QOS:
            raise_qos_once();
            cv_done.wait(lock);
            return;
        case llama_expert_wait::SLEEP:
        default:
            cv_done.wait(lock);
            return;
    }
}

// not a remap's wait: the teardown's, off the critical path, sleeps in every mode
void llama_expert_slots::drain() {
    std::unique_lock<std::mutex> lock(mtx);
    cv_done.wait(lock, [this] { return q_demand.empty() && q_prefetch.empty() && running == 0; });
}

// an empty slot, else the least recently used ready slot whose expert the running remap does not
// need; never a slot that is loading. While wave w_now of a plan stages, experts of later waves go
// last: the ubatch needs them again.
int llama_expert_slots::pick_victim(int l, int w_now) const {
    const auto & L = layers[l];
    int  best       = -1;
    bool best_later = false;
    for (int s = 0; s < (int) L.slots.size(); ++s) {
        const slot & x = L.slots[s];
        if (x.state == EMPTY) {
            return s;
        }
        if (x.state != READY || keep[x.expert]) {
            continue;
        }
        const bool later = w_now >= 0 && L.wave[x.expert] > w_now;
        if (best < 0 || later < best_later || (later == best_later && x.last_use < L.slots[best].last_use)) {
            best       = s;
            best_later = later;
        }
    }
    return best;
}

// the caller holds the lock: queued prefetches of layer l whose expert nothing pins are dropped,
// and their slots are free again
void llama_expert_slots::drop_prefetches(int l) {
    auto & L = layers[l];
    for (auto it = q_prefetch.begin(); it != q_prefetch.end();) {
        slot & x = layers[it->l].slots[it->slot];
        if (it->l == l && !keep[x.expert]) {
            counts_of(x.phase_decode).prefetch_cancelled++;
            L.where[x.expert] = -1;
            x = slot{};
            it = q_prefetch.erase(it);
        } else {
            ++it;
        }
    }
}

// the caller holds the lock: a victim slot of layer l (pick_victim), its ready expert evicted, or
// -1 if there is none and block is false; with block, it waits for a load to finish until there is
// one, through wait_load on the caller's budget
int llama_expert_slots::take_victim(int l, std::unique_lock<std::mutex> & lock, wait_budget & budget, llama_expert_cache_counts & c, bool block,
                                    int w_now) {
    auto & L = layers[l];
    int s;
    while ((s = pick_victim(l, w_now)) < 0) {
        if (!block) {
            return -1;
        }
        wait_load(lock, budget);
    }
    slot & x = L.slots[s];
    if (x.state == READY) {
        c.evictions++;
        if (x.prefetched) {
            counts_of(x.phase_decode).prefetch_wasted++;
        }
        L.where[x.expert] = -1;
    }
    return s;
}

// the caller holds the lock; slot s of layer l is empty or ready and no longer mapped
void llama_expert_slots::assign(int l, int s, int e, bool demand, bool decode) {
    slot & x = layers[l].slots[s];
    x.expert       = e;
    x.state        = LOADING;
    x.last_use     = tick;
    x.prefetched   = !demand;
    x.queued       = true;
    x.demand       = demand;
    x.phase_decode = decode;
    layers[l].where[e] = s;
    (demand ? q_demand : q_prefetch).push_back({l, s});
}

bool llama_expert_slots::remap(int l, const int32_t * ids, int n, bool decode, int32_t * out) {
    std::unique_lock<std::mutex> lock(mtx);
    auto & L = layers[l];
    wait_budget budget; // one spin budget for all of this remap's waits

    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= n_expert) {
            return false;
        }
    }
    needed.clear();
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (!keep[e]) {
            keep[e] = 1;
            needed.push_back(e);
        }
    }
    auto release = [&] {
        for (int e : needed) {
            keep[e] = 0;
        }
    };
    if (needed.size() > L.slots.size()) {
        release();
        return false;
    }

    in_flight = l;
    tick++;
    auto & c = counts_of(decode);
    c.steps++;
    c.needed += needed.size();

    // queued prefetches of this layer that it does not need are dropped: their slots are free again
    drop_prefetches(l);

    // hits; a queued prefetch of a needed expert moves to the demand queue
    for (int e : needed) {
        const int s = L.where[e];
        if (s < 0) {
            continue;
        }
        slot & x = L.slots[s];
        if (x.state == READY) {
            c.hits++;
        } else {
            c.hits_loading++;
            if (x.queued && !x.demand) {
                for (auto it = q_prefetch.begin(); it != q_prefetch.end(); ++it) {
                    if (it->l == l && it->slot == s) {
                        q_prefetch.erase(it);
                        q_demand.push_front({l, s});
                        break;
                    }
                }
            }
        }
        if (x.prefetched) {
            counts_of(x.phase_decode).prefetch_used++;
            x.prefetched = false;
        }
        x.last_use = tick;
    }

    // misses go to the least recently used slots the ubatch does not need; when every such slot
    // is loading, the remap waits for one of those loads
    auto load_missing = [&] {
        for (int e : needed) {
            if (L.where[e] >= 0) {
                continue;
            }
            const int s = take_victim(l, lock, budget, c, true);
            c.misses++;
            assign(l, s, e, true, decode);
        }
        cv_work.notify_all();
    };
    load_missing();

    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    for (;;) {
        if (demand_failed) {
            ok = false;
            break;
        }
        bool pending = false;
        bool lost    = false; // a prefetch of a needed expert failed: load it on demand
        for (int e : needed) {
            const int s = L.where[e];
            if (s < 0) {
                lost = true;
            } else if (L.slots[s].state != READY) {
                pending = true;
            }
        }
        if (lost) {
            load_missing();
            continue;
        }
        if (!pending) {
            break;
        }
        wait_load(lock, budget);
    }
    c.wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (ok) {
        for (int i = 0; i < n; ++i) {
            out[i] = L.where[ids[i]];
        }
    }
    release();
    return ok;
}

int llama_expert_wave_cap(int n_slots, int k) {
    return k > 0 && n_slots >= 3*k ? (n_slots - k)/2 : 0;
}

int llama_expert_wave_count(int n_touch, int n_slots, int k) {
    return n_slots > k ? (n_touch + (n_slots - k) - 1)/(n_slots - k) : 0;
}

int llama_expert_slots::plan_waves(int l, const int32_t * ids, int n, int k, int cap, int n_built) {
    std::lock_guard<std::mutex> lock(mtx);
    auto & L = layers[l];
    if (k <= 0 || cap < k) {
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= n_expert) {
            return -1;
        }
    }

    // the union in first-use order
    std::vector<int> uses(n_expert, 0);
    std::vector<int> uniq;
    for (int i = 0; i < n; ++i) {
        if (uses[ids[i]]++ == 0) {
            uniq.push_back(ids[i]);
        }
    }
    // waves of cap experts, larger only if a union would otherwise take more waves than the graph holds
    const int size = n_built > 0 ? std::max(cap, ((int) uniq.size() + n_built - 1)/n_built) : cap;
    if (size + k > (int) L.slots.size()) {
        return -1;
    }

    for (const auto & wave : L.plan) {
        for (int e : wave) {
            L.wave[e]      = -1;
            L.preloaded[e] = 0;
        }
    }
    L.plan.clear();

    needed = uniq; // last_needed(): the union in first-use order
    // stably sorted by use count, least used first
    std::stable_sort(uniq.begin(), uniq.end(), [&](int a, int b) { return uses[a] < uses[b]; });
    for (size_t i = 0; i < uniq.size(); ++i) {
        if (i % size == 0) {
            L.plan.emplace_back();
        }
        L.plan.back().push_back(uniq[i]);
        L.wave[uniq[i]] = (int) L.plan.size() - 1;
    }

    in_flight = l;
    drop_prefetches(l);
    auto & c = c_prefill;
    c.steps++;
    c.needed += uniq.size();
    return (int) L.plan.size();
}

bool llama_expert_slots::stage_wave(int l, int w, const int32_t * ids, int n, int k, int32_t * out) {
    std::unique_lock<std::mutex> lock(mtx);
    auto & L = layers[l];
    auto & c = c_prefill;
    wait_budget budget; // one spin budget for all of this stage's waits
    static const std::vector<int> none;
    const auto & cur = w < (int) L.plan.size() ? L.plan[w] : none;
    const auto & nxt = w + 1 < (int) L.plan.size() ? L.plan[w + 1] : none;
    in_flight = l;
    tick++;

    std::vector<int> pinned;
    auto pin = [&](int e) {
        if (!keep[e]) {
            keep[e] = 1;
            pinned.push_back(e);
        }
    };
    auto release = [&] {
        for (int e : pinned) {
            keep[e] = 0;
        }
    };
    for (int e : cur) {
        pin(e);
    }
    // a wave of fewer than k experts parks on ready slots of others as well, the most recent first
    std::vector<int> pool;
    if ((int) cur.size() < k) {
        std::vector<int> ready;
        for (int s = 0; s < (int) L.slots.size(); ++s) {
            if (L.slots[s].state == READY && !keep[L.slots[s].expert]) {
                ready.push_back(s);
            }
        }
        std::stable_sort(ready.begin(), ready.end(), [&](int a, int b) { return L.slots[a].last_use > L.slots[b].last_use; });
        if ((int) ready.size() < k - (int) cur.size()) {
            release();
            return false;
        }
        for (int i = 0; i < k - (int) cur.size(); ++i) {
            pin(L.slots[ready[i]].expert);
            pool.push_back(ready[i]);
        }
    }
    // what of the next wave is resident or loading already stays, as far as this wave still fits
    // beside it (a wave can be larger than the cap)
    int room = (int) L.slots.size() - (int) pinned.size();
    for (int e : nxt) {
        if (room > 0 && L.where[e] >= 0 && !keep[e]) {
            pin(e);
            room--;
        }
    }

    // this wave: what the previous stage loaded ahead was counted there
    auto load_missing = [&] {
        for (int e : cur) {
            if (L.where[e] >= 0) {
                continue;
            }
            const int s = take_victim(l, lock, budget, c, true, w);
            c.misses++;
            assign(l, s, e, true, false);
        }
    };
    for (int e : cur) {
        const int s = L.where[e];
        if (L.preloaded[e]) {
            L.preloaded[e] = 0;
        } else if (s >= 0) {
            (L.slots[s].state == READY ? c.hits : c.hits_loading)++;
        }
        if (s >= 0) {
            slot & x = L.slots[s];
            if (x.prefetched) {
                counts_of(x.phase_decode).prefetch_used++;
                x.prefetched = false;
            }
            x.last_use = tick;
        }
    }
    load_missing();
    // the next wave loads behind this one, into slots this stage does not pin and not over its own
    // experts; what finds no slot loads at its own stage
    std::vector<int> held;
    for (int e : nxt) {
        if (L.where[e] >= 0 && !keep[e]) {
            keep[e] = 1;
            held.push_back(e);
        }
    }
    for (int e : nxt) {
        if (L.where[e] >= 0) {
            continue;
        }
        const int s = take_victim(l, lock, budget, c, false, w);
        if (s < 0) {
            break;
        }
        c.misses++;
        assign(l, s, e, true, false);
        L.preloaded[e] = 1;
    }
    for (int e : held) {
        keep[e] = 0;
    }
    cv_work.notify_all();

    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    for (;;) {
        if (demand_failed) {
            ok = false;
            break;
        }
        bool pending = false;
        bool lost    = false;
        for (int e : cur) {
            const int s = L.where[e];
            if (s < 0) {
                lost = true;
            } else if (L.slots[s].state != READY) {
                pending = true;
            }
        }
        if (lost) {
            load_missing();
            cv_work.notify_all();
            continue;
        }
        if (!pending) {
            break;
        }
        wait_load(lock, budget);
    }
    c.wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        release();
        return false;
    }
    c.waves_built++;
    if (!cur.empty()) {
        c.waves++;
    }

    // the wave's own pairs first, then each other pair of a row on a pool slot the row does not use
    for (int e : cur) {
        pool.push_back(L.where[e]);
    }
    std::vector<int32_t> used;
    for (int r = 0; r < n; r += k) {
        const int m = std::min(k, n - r);
        used.clear();
        for (int i = r; i < r + m; ++i) {
            out[i] = -1;
            if (L.wave[ids[i]] == w) {
                out[i] = L.where[ids[i]];
                used.push_back(out[i]);
            }
        }
        size_t p = 0;
        for (int i = r; i < r + m; ++i) {
            if (out[i] >= 0) {
                continue;
            }
            while (std::find(used.begin(), used.end(), pool[p]) != used.end()) {
                p++;
            }
            out[i] = pool[p];
            used.push_back(pool[p]);
            c.wave_parked++;
        }
    }
    release();
    return true;
}

void llama_expert_slots::wave_mask(int l, int w, const int32_t * ids, int n, float * mask) const {
    std::lock_guard<std::mutex> lock(mtx);
    const auto & L = layers[l];
    for (int i = 0; i < n; ++i) {
        mask[i] = ids[i] >= 0 && ids[i] < n_expert && L.wave[ids[i]] == w ? 1.0f : 0.0f;
    }
}

int llama_expert_slots::wave_of(int l, int e) const {
    std::lock_guard<std::mutex> lock(mtx);
    return e >= 0 && e < n_expert ? layers[l].wave[e] : -1;
}

void llama_expert_slots::prefetch(int l, const int * experts, int n, bool decode) {
    std::lock_guard<std::mutex> lock(mtx);
    auto & L = layers[l];
    auto & c = counts_of(decode);
    if (l == in_flight) {
        c.prefetch_skipped += n;
        return;
    }
    for (int i = 0; i < n; ++i) {
        const int e = experts[i];
        if (L.where[e] >= 0) {
            c.prefetch_skipped++;
            continue;
        }
        const int s = pick_victim(l);
        if (s < 0) {
            c.prefetch_skipped++;
            continue;
        }
        slot & x = L.slots[s];
        if (x.state == READY) {
            c.evictions++;
            if (x.prefetched) {
                counts_of(x.phase_decode).prefetch_wasted++;
            }
            L.where[x.expert] = -1;
        }
        c.prefetch_issued++;
        assign(l, s, e, false, decode);
    }
    cv_work.notify_all();
}

void llama_expert_slots::worker() {
    std::unique_lock<std::mutex> lock(mtx);
    for (;;) {
        cv_work.wait(lock, [this] { return stop || !q_demand.empty() || !q_prefetch.empty(); });
        if (stop) {
            return;
        }
        auto & q = q_demand.empty() ? q_prefetch : q_demand;
        const job j = q.front();
        q.pop_front();

        slot & x = layers[j.l].slots[j.slot];
        x.queued = false;
        const int  e      = x.expert;
        const bool demand = x.demand;
        const bool decode = x.phase_decode;
        running++;

        // the slot is loading and in no queue: no remap, prefetch or cancellation touches it now
        lock.unlock();
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t preads = 0;
        const bool ok = load(j.l, e, j.slot, preads);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        lock.lock();

        running--;
        auto & c = counts_of(decode);
        c.io_seconds += dt;
        c.preads     += preads;
        if (!demand) {
            c.io_prefetch_seconds += dt;
        }
        if (ok) {
            x.state = READY;
            c.loads++;
            (demand ? c.demand_bytes : c.prefetch_bytes) += expert_bytes[j.l];
        } else {
            c.load_failures++;
            layers[j.l].where[e] = -1;
            x = slot{};
            if (demand) {
                demand_failed = true;
                failed_layer  = j.l;
                failed_expert = e;
            }
        }
        cv_done.notify_all();
    }
}

void llama_expert_top_k(const std::vector<double> & score, int k, std::vector<int> & out) {
    out.clear();
    for (int b = 0; b < (int) score.size(); ++b) {
        const double v = score[b];
        if ((int) out.size() < k) {
            out.push_back(b);
        } else if (!(v > score[out[k-1]])) {
            continue;
        }
        int i = (int) out.size() - 1;
        while (i > 0 && score[out[i-1]] < v) {
            out[i] = out[i-1];
            i--;
        }
        out[i] = b;
    }
}

llama_expert_markov::llama_expert_markov(int n_layer, int n_expert, int half_life, double mix)
    : n_layer(n_layer), n_expert(n_expert), half_life(std::max(half_life, 1)), mix(mix),
      freq((size_t) n_layer*n_expert), sum(n_layer), seen(n_layer, 0),
      cross((size_t) n_layer*n_expert*n_expert), x_row((size_t) n_layer*n_expert),
      last(n_layer), last_tok(n_layer, -1), score(n_expert) {}

size_t llama_expert_markov::bytes() const {
    return cross.size()*sizeof(float) + x_row.size()*sizeof(float) + freq.size()*sizeof(double) + sum.size()*sizeof(double);
}

void llama_expert_markov::observe(int l, int64_t first, bool decode, const int * ids, int n) {
    const size_t N = n_expert;
    if (const int64_t d = first - seen[l]; d > 0) {
        const double f = std::pow(0.5, double(d)/double(half_life));
        for (size_t e = 0; e < N; ++e) {
            freq[l*N + e] *= f;
        }
        sum[l] *= f;
        seen[l] = first;
    }
    for (int i = 0; i < n; ++i) {
        freq[l*N + ids[i]]++;
    }
    sum[l] += double(n);
    if (!decode) {
        last_tok[l] = -1;
        return;
    }
    if (l >= 2 && last_tok[l-2] == first) {
        for (int a : last[l-2]) {
            float * r = &cross[(l*N + a)*N];
            for (int i = 0; i < n; ++i) {
                r[ids[i]]++;
            }
            x_row[l*N + a] += float(n);
        }
    }
    last[l].assign(ids, ids + n);
    last_tok[l] = first;
}

int llama_expert_markov::predict(int l, int64_t first, bool decode, int k, std::vector<int> & out) {
    out.clear();
    if (!decode || k <= 0) {
        return -1;
    }
    const int target = l + 1;
    const int from   = l - 1;
    if (l < 1 || target >= n_layer || last_tok[from] != first) {
        return -1;
    }
    const size_t N = n_expert;
    std::fill(score.begin(), score.end(), 0.0);
    double den = 0;
    for (int a : last[from]) {
        den += double(x_row[target*N + a]);
    }
    if (den > 0 && mix > 0) {
        for (int a : last[from]) {
            const float * r = &cross[(target*N + a)*N];
            for (size_t b = 0; b < N; ++b) {
                score[b] += double(r[b]);
            }
        }
        for (size_t b = 0; b < N; ++b) {
            score[b] *= mix / den;
        }
    }
    if (const double s = sum[target]; s > 0 && mix < 1) {
        for (size_t b = 0; b < N; ++b) {
            score[b] += (1 - mix) * freq[target*N + b] / s;
        }
    }
    llama_expert_top_k(score, std::min(k, n_expert), out);
    return target;
}
