#include "llama-expert-slots.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>

llama_expert_slots::llama_expert_slots(std::vector<int> n_slots, int n_expert, std::vector<size_t> expert_bytes, int n_threads, load_fn load)
    : n_expert(n_expert), expert_bytes(std::move(expert_bytes)), load(std::move(load)), keep(n_expert, 0) {
    assert(n_slots.size() == this->expert_bytes.size());
    layers.resize(n_slots.size());
    for (size_t l = 0; l < n_slots.size(); ++l) {
        layers[l].slots.resize(std::max(n_slots[l], 0));
        layers[l].where.assign(n_expert, -1);
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

llama_expert_cache_counts llama_expert_slots::counts(bool decode) const {
    std::lock_guard<std::mutex> lock(mtx);
    return decode ? c_decode : c_prefill;
}

bool llama_expert_slots::failure(int & l, int & e) const {
    std::lock_guard<std::mutex> lock(mtx);
    l = failed_layer;
    e = failed_expert;
    return demand_failed;
}

void llama_expert_slots::drain() {
    std::unique_lock<std::mutex> lock(mtx);
    cv_done.wait(lock, [this] { return q_demand.empty() && q_prefetch.empty() && running == 0; });
}

// an empty slot, else the least recently used ready slot whose expert the running remap does not
// need; never a slot that is loading
int llama_expert_slots::pick_victim(int l) const {
    const auto & L = layers[l];
    int best = -1;
    for (int s = 0; s < (int) L.slots.size(); ++s) {
        const slot & x = L.slots[s];
        if (x.state == EMPTY) {
            return s;
        }
        if (x.state == READY && !keep[x.expert] && (best < 0 || x.last_use < L.slots[best].last_use)) {
            best = s;
        }
    }
    return best;
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
            int s;
            while ((s = pick_victim(l)) < 0) {
                cv_done.wait(lock);
            }
            slot & x = L.slots[s];
            if (x.state == READY) {
                c.evictions++;
                if (x.prefetched) {
                    counts_of(x.phase_decode).prefetch_wasted++;
                }
                L.where[x.expert] = -1;
            }
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
        cv_done.wait(lock);
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
        const bool ok = load(j.l, e, j.slot);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        lock.lock();

        running--;
        auto & c = counts_of(decode);
        c.io_seconds += dt;
        if (!demand) {
            c.io_prefetch_seconds += dt;
        }
        if (ok) {
            x.state = READY;
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
