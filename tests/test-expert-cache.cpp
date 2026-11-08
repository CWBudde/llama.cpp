// Tests of the expert cache's bookkeeping (src/llama-expert-slots.*) with a fake loader, of its
// Markov predictor against the predictions of the Go original, and of its two read paths
// (src/llama-expert-sidefile.*) on a synthetic GGUF and side-file:
//
//   test-expert-cache [markov-fixture.json]
//   test-expert-cache --sidefile GGUF SIDEFILE [EXPERTS_PER_LAYER]
//
// The fixture is testdata/locality/markov-fixture.json of the kolibri-llama-cpp repository; without
// it the predictor check is skipped.
//
// --sidefile checks a real pair instead: a side-file written by kolibri-sidefile, opened as the
// model's load opens it, with the ranges taken from the GGUF's own header as the model's loader
// takes them. Every expert (or EXPERTS_PER_LAYER spread over each layer) reads the same bytes on
// both paths, at 3 and at 1 pread.

#include "../src/llama-expert-cache.h"
#include "../src/llama-expert-sidefile.h"
#include "../src/llama-expert-slots.h"
#include "arg.h"
#include "common.h"
#include "speculative.h"

#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#define HAVE_SIDEFILE_TESTS 1
#define HAVE_THREAD_CPUTIME 1
#endif
#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif
#include <future>

static int n_failed = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #cond); n_failed++; } } while (0)

// fake device memory: the expert each slot holds; a load may wait for a gate and may fail
struct fake {
    int n_layer, n_slots;
    std::vector<std::atomic<int>>  mem;
    std::vector<std::atomic<bool>> busy;
    std::atomic<bool> overlap{false}; // a load started on a slot another load was writing
    std::atomic<int>  n_loads{0};
    std::mutex mtx;
    std::set<std::pair<int, int>> gated;   // (layer, expert): wait for open
    std::set<std::pair<int, int>> failing; // (layer, expert): fail once
    std::atomic<bool> open{false};
    std::atomic<int>  delay_us{0};

    fake(int n_layer, int n_slots) : n_layer(n_layer), n_slots(n_slots), mem(n_layer*n_slots), busy(n_layer*n_slots) {
        for (auto & m : mem) {
            m = -1;
        }
    }

    bool load(int l, int e, int s) {
        if (busy[l*n_slots + s].exchange(true)) {
            overlap = true;
        }
        bool wait = false, fail = false;
        {
            std::lock_guard<std::mutex> lock(mtx);
            wait = gated.count({l, e}) > 0;
            fail = failing.erase({l, e}) > 0;
        }
        while (wait && !open) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        }
        if (!fail) {
            mem[l*n_slots + s] = e;
        }
        n_loads++;
        busy[l*n_slots + s] = false;
        return !fail;
    }

    int at(int l, int s) const { return mem[l*n_slots + s]; }

    // waits until a load of layer l runs
    void wait_busy(int l) const {
        for (;;) {
            for (int s = 0; s < n_slots; ++s) {
                if (busy[l*n_slots + s]) {
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
};

static std::unique_ptr<llama_expert_slots> make(fake & f, int n_expert, int n_threads, size_t bytes = 100, llama_expert_wait wait = {},
                                                llama_expert_skip skip = {}) {
    return std::make_unique<llama_expert_slots>(std::vector<int>(f.n_layer, f.n_slots), n_expert,
        std::vector<size_t>(f.n_layer, bytes), n_threads, [&f](int l, int e, int s, uint64_t & preads) { preads++; return f.load(l, e, s); }, wait, skip);
}

// every id maps to a slot that holds it
static bool mapped(const fake & f, int l, const std::vector<int32_t> & ids, const std::vector<int32_t> & out) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (out[i] < 0 || out[i] >= f.n_slots || f.at(l, out[i]) != ids[i]) {
            return false;
        }
    }
    return true;
}

static bool remap(llama_expert_slots & c, int l, std::vector<int32_t> ids, std::vector<int32_t> & out, bool decode = true) {
    out.assign(ids.size(), -1);
    return c.remap(l, ids.data(), (int) ids.size(), decode, out.data());
}

// remap with the router weights of the ids, n_used ids per token; skipped gets the skip flags
static bool remap_w(llama_expert_slots & c, int l, std::vector<int32_t> ids, std::vector<float> w, int n_used,
                    std::vector<int32_t> & out, std::vector<uint8_t> & skipped, bool decode = true) {
    out.assign(ids.size(), -1);
    skipped.assign(ids.size(), 0xff);
    return c.remap(l, ids.data(), (int) ids.size(), decode, out.data(), w.data(), n_used, skipped.data());
}

// opens the fake's gate after a pause, so that a remap waits for a gated load
struct opener {
    std::thread t;
    explicit opener(fake & f) : t([&f] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); f.open = true; }) {}
    ~opener() { t.join(); }
};

static void test_lru() {
    fake f(1, 3);
    auto c = make(f, 10, 2);
    std::vector<int32_t> out;
    CHECK(remap(*c, 0, {0, 1, 2}, out) && mapped(f, 0, {0, 1, 2}, out));
    CHECK(remap(*c, 0, {0}, out) && mapped(f, 0, {0}, out));
    // 1 and 2 were used once, before 0: the least recently used is 1, the lower slot of the two
    CHECK(remap(*c, 0, {3}, out) && mapped(f, 0, {3}, out));
    CHECK(c->slot_of(0, 1) < 0 && c->slot_of(0, 2) >= 0 && c->slot_of(0, 0) >= 0);
    const auto n = c->counts(true);
    CHECK(n.steps == 3 && n.needed == 5 && n.misses == 4 && n.hits == 1 && n.evictions == 1);
    CHECK(n.demand_bytes == 400 && n.prefetch_bytes == 0 && n.io_prefetch_seconds == 0);
    CHECK(c->counts(false).steps == 0);
    // duplicate ids in a ubatch need one slot
    CHECK(remap(*c, 0, {3, 0, 3, 0}, out) && mapped(f, 0, {3, 0, 3, 0}, out) && out[0] == out[2]);
}

static void test_pinned() {
    // the least recently used expert is needed by the ubatch itself: the other one goes
    fake f(1, 2);
    auto c = make(f, 10, 1);
    std::vector<int32_t> out;
    CHECK(remap(*c, 0, {5}, out));
    CHECK(remap(*c, 0, {6}, out));
    CHECK(remap(*c, 0, {5, 7}, out) && mapped(f, 0, {5, 7}, out));
    CHECK(c->slot_of(0, 6) < 0);
}

static void test_pinned_loading() {
    // two slots: a needed expert, and a prefetch still loading; the miss must wait for the
    // prefetch's slot rather than take the needed expert's, the only other one
    fake f(2, 2);
    f.gated.insert({1, 3});
    auto c = make(f, 10, 2);
    std::vector<int32_t> out;
    CHECK(remap(*c, 1, {5}, out));
    CHECK(remap(*c, 0, {0}, out)); // layer 1 is no longer in flight
    const int p = 3;
    c->prefetch(1, &p, 1, true);
    f.wait_busy(1);
    {
        opener o(f);
        CHECK(remap(*c, 1, {5, 7}, out) && mapped(f, 1, {5, 7}, out));
    }
    CHECK(c->slot_of(1, 3) < 0);
    // one miss per remap and no second load of the needed expert: only the prefetch is evicted
    const auto n = c->counts(true);
    CHECK(n.misses == 3 && n.hits == 1 && n.evictions == 1 && n.prefetch_wasted == 1);
}

static void test_capacity() {
    fake f(1, 2);
    auto c = make(f, 10, 1);
    std::vector<int32_t> out;
    CHECK(!remap(*c, 0, {1, 2, 3}, out));
    CHECK(c->counts(true).steps == 0 && f.n_loads == 0);
    CHECK(remap(*c, 0, {1, 2, 1}, out) && mapped(f, 0, {1, 2, 1}, out));
}

static void test_prefetch_awaited() {
    // a needed expert whose prefetch is still loading is awaited, not loaded twice
    fake f(3, 4);
    f.gated.insert({2, 7});
    auto c = make(f, 10, 2);
    std::vector<int32_t> out;
    CHECK(remap(*c, 0, {1}, out));
    const int p = 7;
    c->prefetch(2, &p, 1, true);
    {
        opener o(f);
        CHECK(remap(*c, 2, {7}, out) && mapped(f, 2, {7}, out));
    }
    const auto n = c->counts(true);
    CHECK(n.hits_loading == 1 && n.misses == 1 && n.prefetch_issued == 1 && n.prefetch_used == 1);
    CHECK(f.n_loads == 2 && n.wait_seconds > 0.005);
    c->drain();
    const auto m = c->counts(true);
    CHECK(m.prefetch_bytes == 100 && m.io_prefetch_seconds > 0.005 && m.io_prefetch_seconds <= m.io_seconds);
}

static void test_cancel() {
    // one I/O thread, held by a gated prefetch of layer 3; of the two prefetches queued behind it,
    // the needed one moves to the demand queue and the other is dropped
    fake f(4, 4);
    f.gated.insert({3, 0});
    auto c = make(f, 10, 1);
    std::vector<int32_t> out;
    CHECK(remap(*c, 1, {9}, out));
    const int p[] = {0, 1, 2};
    c->prefetch(3, p, 3, true);
    f.wait_busy(3);
    {
        opener o(f);
        CHECK(remap(*c, 3, {1}, out) && mapped(f, 3, {1}, out));
    }
    c->drain();
    const auto n = c->counts(true);
    CHECK(n.prefetch_issued == 3 && n.prefetch_cancelled == 1 && n.hits_loading == 1 && n.prefetch_used == 1);
    CHECK(c->slot_of(3, 2) < 0 && c->slot_of(3, 0) >= 0);
    CHECK(n.prefetch_bytes == 200); // 0 ran, 1 ran from the demand queue, 2 never ran
}

static void test_in_flight() {
    // no prefetch into the layer in flight; a loading slot is never a victim
    fake f(2, 2);
    f.gated.insert({1, 4});
    auto c = make(f, 10, 2);
    std::vector<int32_t> out;
    CHECK(remap(*c, 0, {1}, out));
    const int p0 = 2;
    c->prefetch(0, &p0, 1, true);
    CHECK(c->slot_of(0, 2) < 0 && c->counts(true).prefetch_skipped == 1);

    // layer 1: one slot loads a gated prefetch; a ubatch of two other experts must wait for it
    const int p1 = 4;
    c->prefetch(1, &p1, 1, true);
    f.wait_busy(1);
    {
        opener o(f);
        CHECK(remap(*c, 1, {5, 6}, out) && mapped(f, 1, {5, 6}, out));
    }
    CHECK(!f.overlap);
    const auto n = c->counts(true);
    CHECK(n.prefetch_wasted == 1 && n.evictions == 1);
}

static void test_failures() {
    fake f(1, 3);
    auto c = make(f, 10, 1);
    std::vector<int32_t> out;
    // a failed prefetch falls back to a demand load
    f.failing.insert({0, 3});
    const int p = 3;
    c->prefetch(0, &p, 1, false);
    c->drain();
    CHECK(c->slot_of(0, 3) < 0);
    CHECK(remap(*c, 0, {3}, out, false) && mapped(f, 0, {3}, out));
    const auto n = c->counts(false);
    CHECK(n.load_failures == 1 && n.misses == 1);
    // a failed demand load fails the remap and is reported
    f.failing.insert({0, 4});
    CHECK(!remap(*c, 0, {4}, out, false));
    int l = -1, e = -1;
    CHECK(c->failure(l, e) && l == 0 && e == 4);
}

static void test_stress() {
    // random ubatches and prefetches over 4 layers, with slow loads: every remap maps every id to
    // a slot holding it, and the slots of the layer in flight stay intact until the next remap
    fake f(4, 8);
    f.delay_us = 50;
    auto c = make(f, 32, 4);
    std::mt19937 rng(7);
    std::vector<int32_t> out;
    bool ok = true;
    int n_remaps = 0;
    for (int step = 0; step < 400 && ok; ++step) {
        const int l = step % 4;
        const int n = 1 + rng() % 8;
        std::vector<int32_t> ids;
        for (int i = 0; i < n; ++i) {
            ids.push_back(rng() % 12 + (rng() % 4 == 0 ? 20 : 0));
        }
        ok = remap(*c, l, ids, out, n == 1) && mapped(f, l, ids, out);
        n_remaps++;
        for (int t = 1; t <= 2; ++t) {
            std::vector<int> pf;
            for (int i = 0; i < 4; ++i) {
                pf.push_back(rng() % 32);
            }
            c->prefetch((l + t) % 4, pf.data(), (int) pf.size(), true);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(rng() % 200));
        ok = ok && mapped(f, l, ids, out);
    }
    c->drain();
    CHECK(ok && n_remaps == 400);
    CHECK(!f.overlap);
    const auto d = c->counts(true), p = c->counts(false);
    CHECK(d.hits + d.hits_loading + d.misses == d.needed && p.hits + p.hits_loading + p.misses == p.needed);
    CHECK(d.demand_bytes + p.demand_bytes == 100*(d.misses + p.misses));
    CHECK(d.prefetch_issued > 0 && d.prefetch_used > 0);
}

// a layer's misses are issued concurrently over the I/O threads: all four threads are inside a load
// of one remap at once, or no load gets past the barrier (and a timeout fails the check)
#define CHECK_IN(what, cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: FAIL %s: %s\n", __FILE__, __LINE__, what, #cond); n_failed++; } } while (0)

// what a waiting thread saw
struct waited {
    bool   ok   = false;
    double wall = 0;  // seconds in the call
    double cpu  = -1; // the thread's CPU seconds in the call, -1 where unknown
    int    qos_before = -1; // the thread's QoS class before and after the call (Apple), else -1
    int    qos_after  = -1;
};

static double thread_cpu_seconds() {
#ifdef HAVE_THREAD_CPUTIME
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#else
    return -1;
#endif
}

// runs fn on a new thread, as the decode thread that waits for loads. A wait that does not end
// within 5 s fails the test and ends the process at once: the cache that holds the waiter cannot
// be destroyed, so the test must not hang on it.
static waited on_thread(const char * what, const std::function<bool()> & fn) {
    auto done = std::make_shared<std::promise<waited>>();
    auto res  = done->get_future();
    std::thread t([done, fn] {
        waited w;
#ifdef __APPLE__
        w.qos_before = (int) qos_class_self();
#endif
        const double c0 = thread_cpu_seconds();
        const auto   t0 = std::chrono::steady_clock::now();
        w.ok   = fn();
        w.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        w.cpu  = c0 < 0 ? -1 : thread_cpu_seconds() - c0;
#ifdef __APPLE__
        w.qos_after = (int) qos_class_self();
#endif
        done->set_value(w);
    });
    if (res.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        fprintf(stderr, "%s:%d: FAIL %s: the wait did not end within 5 s\n", __FILE__, __LINE__, what);
        fflush(stderr);
        std::_Exit(1);
    }
    t.join();
    return res.get();
}

// every wait mode: a remap that waits for slow demand loads, one that waits for a loading slot to
// become a victim, and one whose demand load fails all end, with the expert each slot holds; spin
// burns the waiting core and the others do not; qos raises the waiting thread and the others leave it
static void test_wait_modes() {
    struct mode {
        const char *      name;
        llama_expert_wait w;
    };
    const mode modes[] = {
        { "sleep",     { llama_expert_wait::SLEEP,  0 } },
        { "spin",      { llama_expert_wait::SPIN,   0 } },
        { "hybrid:50", { llama_expert_wait::HYBRID, 50 } },
        { "qos",       { llama_expert_wait::QOS,    0 } },
    };
    for (const auto & m : modes) {
        const bool spins = m.w.mode == llama_expert_wait::SPIN;
        const bool qos   = m.w.mode == llama_expert_wait::QOS;
        char what[128];

        // slow demand loads: the remap waits in its main loop
        {
            fake f(1, 4);
            f.delay_us = 2000;
            auto c = make(f, 10, 2, 100, m.w);
            std::vector<int32_t> out;
            snprintf(what, sizeof(what), "wait %s, slow demand loads", m.name);
            const waited r = on_thread(what, [&] { return remap(*c, 0, {1, 2, 3}, out); });
            CHECK_IN(what, r.ok && mapped(f, 0, {1, 2, 3}, out));
            const auto n = c->counts(true);
            CHECK_IN(what, n.misses == 3 && n.loads == 3 && n.wait_seconds > 0.001 && n.wait_seconds <= r.wall);
#ifdef __APPLE__
            CHECK_IN(what, qos ? r.qos_after == (int) QOS_CLASS_USER_INTERACTIVE : r.qos_after == r.qos_before);
#endif
        }

        // two slots, a needed expert and a gated prefetch: the miss waits for the prefetch's slot,
        // then for its own load (test_pinned_loading)
        {
            fake f(2, 2);
            f.gated.insert({1, 3});
            auto c = make(f, 10, 2, 100, m.w);
            std::vector<int32_t> out;
            CHECK(remap(*c, 1, {5}, out));
            CHECK(remap(*c, 0, {0}, out));
            const int p = 3;
            c->prefetch(1, &p, 1, true);
            f.wait_busy(1);
            snprintf(what, sizeof(what), "wait %s, loading victim", m.name);
            waited r;
            {
                opener o(f);
                r = on_thread(what, [&] { return remap(*c, 1, {5, 7}, out); });
            }
            CHECK_IN(what, r.ok && mapped(f, 1, {5, 7}, out) && !f.overlap);
            CHECK_IN(what, c->slot_of(1, 3) < 0 && c->counts(true).prefetch_wasted == 1);
            CHECK_IN(what, r.wall > 0.005);
            if (r.cpu >= 0) {
                // the gate opens after 20 ms: a spin keeps the core busy, a sleep (or a 50 us spin) does not
                CHECK_IN(what, spins ? r.cpu > 0.3 * r.wall : r.cpu < 0.3 * r.wall);
            }
        }

        // a failed demand load ends the wait
        {
            fake f(1, 2);
            f.delay_us = 2000;
            f.failing.insert({0, 4});
            auto c = make(f, 10, 1, 100, m.w);
            std::vector<int32_t> out;
            snprintf(what, sizeof(what), "wait %s, failed load", m.name);
            const waited r = on_thread(what, [&] { return remap(*c, 0, {4}, out); });
            int l = -1, e = -1;
            CHECK_IN(what, !r.ok && c->failure(l, e) && l == 0 && e == 4);
        }

        // random ubatches and prefetches with slow loads, as test_stress, shorter
        {
            fake f(4, 8);
            f.delay_us = 50;
            auto c = make(f, 32, 4, 100, m.w);
            std::mt19937 rng(11);
            snprintf(what, sizeof(what), "wait %s, stress", m.name);
            const waited r = on_thread(what, [&] {
                std::vector<int32_t> out;
                bool ok = true;
                for (int step = 0; step < 200 && ok; ++step) {
                    const int l = step % 4;
                    std::vector<int32_t> ids(1 + rng() % 6);
                    for (auto & id : ids) {
                        id = (int32_t) (rng() % 32);
                    }
                    ok = remap(*c, l, ids, out) && mapped(f, l, ids, out);
                    const int pf[] = { (int) (rng() % 32), (int) (rng() % 32) };
                    c->prefetch((l + 2) % 4, pf, 2, true);
                }
                return ok;
            });
            CHECK_IN(what, r.ok && !f.overlap);
        }
    }
}

// --expert-cache-wait: the four modes reach the model's params; anything else is refused
static void test_wait_arg() {
    struct good {
        const char *            arg;
        llama_expert_cache_wait mode;
        int32_t                 us;
    };
    const good goods[] = {
        { "sleep",     LLAMA_EXPERT_CACHE_WAIT_SLEEP,  0 },
        { "spin",      LLAMA_EXPERT_CACHE_WAIT_SPIN,   0 },
        { "hybrid:50", LLAMA_EXPERT_CACHE_WAIT_HYBRID, 50 },
        { "qos",       LLAMA_EXPERT_CACHE_WAIT_QOS,    0 },
    };
    // the parser wants a model, which it does not open
    auto parse = [](common_params & params, std::vector<std::string> args) {
        args.insert(args.begin(), { "test-expert-cache", "-m", "none.gguf" });
        std::vector<char *> argv;
        for (auto & a : args) {
            argv.push_back(a.data());
        }
        return common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_COMMON);
    };
    for (const auto & g : goods) {
        common_params params;
        CHECK_IN(g.arg, parse(params, { "--expert-cache", "16", "--expert-cache-wait", g.arg }));
        CHECK_IN(g.arg, params.expert_cache_wait == g.mode && params.expert_cache_wait_us == g.us);
        const llama_model_params mp = common_model_params_to_llama(params);
        CHECK_IN(g.arg, mp.expert_cache && mp.expert_cache->wait_mode == g.mode && mp.expert_cache->wait_spin_us == g.us);
    }
    for (const char * bad : { "busy", "", "SPIN", "spin:5", "hybrid", "hybrid:", "hybrid:0", "hybrid:-5", "hybrid:5x", "qos:1" }) {
        common_params params;
        CHECK_IN(bad, !parse(params, { "--expert-cache-wait", bad }));
        CHECK_IN(bad, params.expert_cache_wait == LLAMA_EXPERT_CACHE_WAIT_SLEEP);
    }
    // the default is the sleep
    common_params params;
    CHECK(params.expert_cache_wait == LLAMA_EXPERT_CACHE_WAIT_SLEEP && params.expert_cache_wait_us == 0);
    const llama_expert_cache_params ec = llama_expert_cache_default_params();
    CHECK(ec.wait_mode == LLAMA_EXPERT_CACHE_WAIT_SLEEP && ec.wait_spin_us == 0);
    CHECK(llama_expert_wait{}.mode == llama_expert_wait::SLEEP);
}

// the counters a skip may not change when it is off
static bool same_counts(const llama_expert_cache_counts & a, const llama_expert_cache_counts & b) {
    return a.steps == b.steps && a.needed == b.needed && a.hits == b.hits && a.hits_loading == b.hits_loading &&
           a.misses == b.misses && a.evictions == b.evictions && a.loads == b.loads && a.demand_bytes == b.demand_bytes &&
           a.skipped == b.skipped && a.skipped_ids == b.skipped_ids;
}

static bool none_skipped(const std::vector<uint8_t> & sk) {
    return std::all_of(sk.begin(), sk.end(), [](uint8_t x) { return x == 0; });
}

static void test_skip_off() {
    // (a) a threshold of 0, or no weights, maps and counts as the cache without the rule
    const std::vector<std::vector<int32_t>> steps = {
        {0, 1, 2, 3, 4, 5}, {0, 1, 2, 6, 7, 8}, {9, 1, 2, 3, 10, 11}, {12, 13, 14, 15, 10, 11},
    };
    const std::vector<float> w = {0.9f, 0.8f, 0.7f, 0.05f, 0.04f, 0.03f};
    fake f0(1, 8), f1(1, 8), f2(1, 8);
    auto c0 = make(f0, 16, 1);
    auto c1 = make(f1, 16, 1, 100, {}, {0.0f, 1}); // threshold 0, weights given
    auto c2 = make(f2, 16, 1, 100, {}, {0.1f, 1}); // a threshold, no weights
    for (const auto & ids : steps) {
        std::vector<int32_t> o0, o1, o2;
        std::vector<uint8_t> sk;
        CHECK(remap(*c0, 0, ids, o0) && mapped(f0, 0, ids, o0));
        CHECK(remap_w(*c1, 0, ids, w, 6, o1, sk) && o1 == o0 && none_skipped(sk));
        CHECK(remap(*c2, 0, ids, o2) && o2 == o0);
        CHECK(c1->last_needed() == c0->last_needed());
    }
    CHECK(same_counts(c0->counts(true), c1->counts(true)) && same_counts(c0->counts(true), c2->counts(true)));
    CHECK(c0->counts(true).skipped == 0 && f0.n_loads == f1.n_loads && f0.n_loads == f2.n_loads);
}

static void test_skip_rule() {
    fake f(1, 8);
    auto c = make(f, 16, 1, 100, {}, {0.1f, 1});
    std::vector<int32_t> out;
    std::vector<uint8_t> sk;
    CHECK(remap_w(*c, 0, {0, 1, 2, 3, 4, 5}, {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f}, 6, out, sk) && mapped(f, 0, {0, 1, 2, 3, 4, 5}, out));
    CHECK(f.n_loads == 6 && none_skipped(sk));

    // (b, f) expert 9, a true miss, is 2nd in id order but 5th by weight and below the threshold: no
    // load, the slot of the rank-1 expert (0). Expert 4, 6th and below it too, is resident: mapped.
    CHECK(remap_w(*c, 0, {0, 9, 1, 2, 3, 4}, {0.9f, 0.05f, 0.8f, 0.7f, 0.6f, 0.01f}, 6, out, sk));
    CHECK(f.n_loads == 6 && c->slot_of(0, 9) < 0);
    CHECK(out[1] == out[0] && f.at(0, out[1]) == 0);
    CHECK(sk == std::vector<uint8_t>({0, 1, 0, 0, 0, 0}));
    CHECK(mapped(f, 0, {0, 1, 2, 3, 4}, {out[0], out[2], out[3], out[4], out[5]}));
    // the predictor still sees the router's whole selection
    CHECK(c->last_needed() == std::vector<int>({0, 9, 1, 2, 3, 4}));
    auto n = c->counts(true);
    CHECK(n.skipped == 1 && n.skipped_ids == 1 && n.misses == 6 && n.hits == 5 && n.needed == 11);

    // (f) the other way round: expert 10 is 5th in id order but first by weight, every weight below
    // the threshold: loaded. Experts 2, 3 and 4 (ranks 4 to 6) are resident.
    CHECK(remap_w(*c, 0, {0, 1, 2, 3, 10, 4}, {0.09f, 0.08f, 0.07f, 0.06f, 0.095f, 0.01f}, 6, out, sk) && mapped(f, 0, {0, 1, 2, 3, 10, 4}, out));
    CHECK(f.n_loads == 7 && none_skipped(sk));

    // (c) misses at ranks 1 to 3 are loaded, however low their weights
    CHECK(remap_w(*c, 0, {11, 12, 13, 0, 1, 2}, {0.09f, 0.08f, 0.07f, 0.06f, 0.05f, 0.04f}, 6, out, sk) && mapped(f, 0, {11, 12, 13, 0, 1, 2}, out));
    CHECK(f.n_loads == 10 && none_skipped(sk));

    // (d) a miss at rank 5 with a weight at the threshold is loaded; 12 at rank 6 is resident
    CHECK(remap_w(*c, 0, {0, 1, 2, 11, 14, 12}, {0.9f, 0.8f, 0.7f, 0.6f, 0.1f, 0.05f}, 6, out, sk) && mapped(f, 0, {0, 1, 2, 11, 14, 12}, out));
    CHECK(f.n_loads == 11 && none_skipped(sk));

    // the counters add up: every distinct expert of a step is a hit, a miss or skipped
    n = c->counts(true);
    CHECK(n.hits + n.hits_loading + n.misses == n.needed && n.needed + n.skipped == 5*6);
    CHECK(n.skipped == 1 && n.loads == 11);
}

static void test_skip_loading() {
    // (e) an expert whose prefetch is still loading is awaited at rank 5 below the threshold, not skipped
    fake f(2, 8);
    f.gated.insert({1, 14});
    auto c = make(f, 16, 2, 100, {}, {0.1f, 1});
    std::vector<int32_t> out;
    std::vector<uint8_t> sk;
    CHECK(remap_w(*c, 1, {0, 1, 2, 3, 4, 5}, {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f}, 6, out, sk));
    CHECK(remap(*c, 0, {0}, out)); // layer 1 is not in flight now
    const int p = 14;
    c->prefetch(1, &p, 1, true);
    {
        opener o(f);
        CHECK(remap_w(*c, 1, {0, 1, 2, 3, 14, 4}, {0.9f, 0.8f, 0.7f, 0.6f, 0.05f, 0.01f}, 6, out, sk) && mapped(f, 1, {0, 1, 2, 3, 14, 4}, out));
    }
    CHECK(none_skipped(sk));
    const auto n = c->counts(true);
    CHECK(n.hits_loading == 1 && n.prefetch_used == 1 && n.skipped == 0 && n.skipped_ids == 0);
}

static void test_skip_ubatch() {
    // (g) a prefill ubatch of two tokens
    fake f(1, 8);
    auto c = make(f, 16, 1, 100, {}, {0.1f, 1});
    std::vector<int32_t> out;
    std::vector<uint8_t> sk;
    CHECK(remap_w(*c, 0, {0, 1, 2, 3, 4, 5}, {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f}, 6, out, sk));
    // expert 15 is 5th and below the threshold for token 0, but first for token 1: loaded for both
    const std::vector<int32_t> a = {0, 1, 2, 3, 15, 4,  15, 0, 1, 2, 3, 4};
    CHECK(remap_w(*c, 0, a, {0.9f, 0.8f, 0.7f, 0.6f, 0.05f, 0.01f,  0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f}, 6, out, sk, false));
    CHECK(mapped(f, 0, a, out) && none_skipped(sk) && f.n_loads == 7);
    CHECK(c->counts(false).skipped == 0);
    // expert 9 is 5th and below the threshold for both: skipped for both, each mapped to its own rank-1 expert
    CHECK(remap_w(*c, 0, {0, 1, 2, 3, 9, 4,  1, 0, 2, 3, 9, 4},
                  {0.9f, 0.8f, 0.7f, 0.6f, 0.05f, 0.01f,  0.9f, 0.8f, 0.7f, 0.6f, 0.05f, 0.01f}, 6, out, sk, false));
    CHECK(f.n_loads == 7 && c->slot_of(0, 9) < 0);
    CHECK(sk == std::vector<uint8_t>({0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 1, 0}));
    CHECK(out[4] == out[0] && f.at(0, out[4]) == 0 && out[10] == out[6] && f.at(0, out[10]) == 1);
    const auto n = c->counts(false);
    CHECK(n.skipped == 1 && n.skipped_ids == 2 && n.needed == 6 + 5);
}

static void test_skip_map_rank() {
    // (h) skip_map_rank 2 maps a skipped id to the slot of its token's rank-2 expert
    for (int rank : {1, 2}) {
        fake f(1, 8);
        auto c = make(f, 16, 1, 100, {}, {0.1f, rank});
        std::vector<int32_t> out;
        std::vector<uint8_t> sk;
        CHECK(remap_w(*c, 0, {0, 1, 2, 3, 4, 5}, {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f}, 6, out, sk));
        // rank 1 is expert 1 (position 2), rank 2 expert 0 (position 0)
        CHECK(remap_w(*c, 0, {0, 9, 1, 2, 3, 4}, {0.8f, 0.05f, 0.9f, 0.7f, 0.6f, 0.01f}, 6, out, sk));
        CHECK(sk[1] == 1 && f.n_loads == 6);
        CHECK(out[1] == (rank == 1 ? out[2] : out[0]) && f.at(0, out[1]) == (rank == 1 ? 1 : 0));
    }
}

static bool parse_args(common_params & params, std::vector<std::string> args) {
    // the parser wants a model, which it does not open
    args.insert(args.begin(), { "test-expert-cache", "-m", "none.gguf" });
    std::vector<char *> argv;
    for (auto & a : args) {
        argv.push_back(a.data());
    }
    return common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_COMMON);
}

// what the cache's constructor throws, or "" when it accepts p
static std::string cache_refusal(const llama_expert_cache_params & p) {
    try {
        llama_expert_cache c(p, 4, 16, 6);
        return "";
    } catch (const std::exception & e) {
        return e.what();
    }
}

static void test_skip_arg() {
    for (const auto & [arg, v] : std::vector<std::pair<const char *, float>>{ { "0.05", 0.05f }, { "0", 0.0f }, { "1", 1.0f } }) {
        common_params params;
        CHECK_IN(arg, parse_args(params, { "--expert-cache", "16", "--expert-cache-skip", arg }));
        CHECK_IN(arg, params.expert_cache_skip == v);
        const llama_model_params mp = common_model_params_to_llama(params);
        CHECK_IN(arg, mp.expert_cache && mp.expert_cache->skip_weight == v && mp.expert_cache->skip_map_rank == 1);
    }
    for (const char * bad : { "-0.1", "-1", "nan", "inf", "-inf", "abc", "" }) {
        common_params params;
        CHECK_IN(bad, !parse_args(params, { "--expert-cache-skip", bad }));
        CHECK_IN(bad, params.expert_cache_skip == 0.0f);
    }
    // the default is off
    {
        common_params params;
        CHECK(params.expert_cache_skip == 0.0f);
        const llama_expert_cache_params ec = llama_expert_cache_default_params();
        CHECK(ec.skip_weight == 0.0f && ec.skip_map_rank == 1);
        CHECK(llama_expert_skip{}.weight == 0.0f && llama_expert_skip{}.map_rank == 1);
    }
    // the skip alone still hands the model its params, so that the load refuses it without a cache
    {
        common_params params;
        CHECK(parse_args(params, { "--expert-cache-skip", "0.05" }));
        const llama_model_params mp = common_model_params_to_llama(params);
        CHECK(mp.expert_cache && mp.expert_cache->skip_weight == 0.05f && mp.expert_cache->n_slots == 0);
    }
#if defined(__APPLE__) || defined(__linux__)
    // the environment variable
    {
        setenv("LLAMA_ARG_EXPERT_CACHE_SKIP", "0.25", 1);
        common_params params;
        const bool ok = parse_args(params, { "--expert-cache", "16" });
        unsetenv("LLAMA_ARG_EXPERT_CACHE_SKIP");
        CHECK(ok && params.expert_cache_skip == 0.25f);
    }
#endif
    // the constructor's refusals
    llama_expert_cache_params p = llama_expert_cache_default_params();
    p.n_slots = 8;
    CHECK(cache_refusal(p).empty());
    p.skip_weight = 0.05f;
    CHECK(cache_refusal(p).empty());
    p.gpu_remap = true;
    CHECK(cache_refusal(p).find("--expert-cache-skip") != std::string::npos);
    p.gpu_remap = false;
    for (int rank : {0, 3, -1}) {
        p.skip_map_rank = rank;
        CHECK_IN("map rank", cache_refusal(p).find("map rank") != std::string::npos);
    }
    p.skip_map_rank = 2;
    CHECK(cache_refusal(p).empty());
    for (float w : { -0.5f, NAN, INFINITY }) {
        p.skip_weight = w;
        CHECK_IN("weight", !cache_refusal(p).empty());
    }
}

static void test_concurrent_misses() {
    std::atomic<int> inside{0};
    std::atomic<int> most{0};
    llama_expert_slots c(std::vector<int>(1, 8), 16, std::vector<size_t>(1, 100), 4, [&](int, int, int, uint64_t & preads) {
        preads++;
        const int now = ++inside;
        for (int prev = most; now > prev && !most.compare_exchange_weak(prev, now);) {
        }
        const auto t0 = std::chrono::steady_clock::now();
        while (most < 4 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        inside--;
        return true;
    });
    std::vector<int32_t> out;
    CHECK(remap(c, 0, {0, 1, 2, 3, 4, 5}, out));
    CHECK(most == 4);
    const auto d = c.counts(true);
    CHECK(d.misses == 6 && d.loads == 6 && d.preads == 6 && d.load_failures == 0);
}

// a load's read calls are counted in its phase, failed loads too, and only finished loads count as loads
static void test_preads_counted() {
    fake f(1, 4);
    llama_expert_slots c(std::vector<int>(1, 4), 8, std::vector<size_t>(1, 100), 1, [&f](int l, int e, int s, uint64_t & preads) {
        preads += 3;
        return f.load(l, e, s);
    });
    std::vector<int32_t> out;
    CHECK(remap(c, 0, {0, 1}, out, false));
    CHECK(remap(c, 0, {1, 2}, out, true));
    auto p = c.counts(false), d = c.counts(true);
    CHECK(p.loads == 2 && p.preads == 6);
    CHECK(d.loads == 1 && d.preads == 3 && d.hits == 1);
    f.failing.insert({0, 3});
    CHECK(!remap(c, 0, {3}, out, true));
    d = c.counts(true);
    CHECK(d.loads == 1 && d.preads == 6 && d.load_failures == 1);
}

static std::string hex32(const uint8_t * p) {
    std::string s;
    char b[3];
    for (int i = 0; i < 32; ++i) {
        snprintf(b, sizeof(b), "%02x", p[i]);
        s += b;
    }
    return s;
}

static void test_sha256() {
    const auto sum = [](const std::string & m) {
        uint8_t out[32];
        llama_expert_sha256(m.data(), m.size(), out);
        return hex32(out);
    };
    CHECK(sum("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sum("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sum("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(sum(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

#ifdef HAVE_SIDEFILE_TESTS
// A synthetic GGUF and its side-file: a header of random bytes, then three layers of stacked rows
// in the file's own order (down first) with a different row size for down. The table is written here
// as kolibri-sidefile writes it; tools/gguf/check_expert_cache.py reads what the tool itself writes.
struct sidefile_fixture {
    static constexpr int      n_layer  = 3;
    static constexpr int      n_expert = 8;
    static constexpr uint32_t align    = 16384;
    static constexpr size_t   head     = 1000;
    static constexpr size_t   row_gate = 12345;
    static constexpr size_t   row_down = 33000;

    std::string dir, gguf_path, side_path;
    std::vector<uint8_t> gguf;
    std::vector<std::array<llama_expert_stacked, 3>> stacked;
    std::vector<uint8_t> table; // the side-file's table, unpadded

    sidefile_fixture() {
        const char * tmp = getenv("TMPDIR");
        std::string tmpl = std::string(tmp ? tmp : "/tmp") + "/test-expert-cache-XXXXXX";
        CHECK(mkdtemp(tmpl.data()) != nullptr);
        dir = tmpl;
        gguf_path = dir + "/m.gguf";
        side_path = dir + "/m.gguf.experts";

        std::mt19937 rng(11);
        gguf.resize(head);
        stacked.resize(n_layer);
        const auto append_rows = [&](llama_expert_stacked & t, size_t row, uint32_t type) {
            t = { (uint64_t) gguf.size(), (uint64_t) row, type };
            for (size_t i = 0; i < row*n_expert; ++i) {
                gguf.push_back(0);
            }
        };
        for (auto & lt : stacked) {
            append_rows(lt[2], row_down, 23);
            append_rows(lt[0], row_gate, 18);
            append_rows(lt[1], row_gate, 18);
        }
        for (auto & b : gguf) {
            b = (uint8_t) (rng() | 1);
        }
        write_gguf();
        table = build_table();
        write_side();
    }

    ~sidefile_fixture() {
        unlink(gguf_path.c_str());
        unlink(side_path.c_str());
        rmdir(dir.c_str());
    }

    static void put_u32(std::vector<uint8_t> & b, uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            b.push_back((uint8_t) (v >> (8*i)));
        }
    }
    static void put_u64(std::vector<uint8_t> & b, uint64_t v) {
        put_u32(b, (uint32_t) v);
        put_u32(b, (uint32_t) (v >> 32));
    }
    static void set_u64(std::vector<uint8_t> & b, size_t at, uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            b[at + i] = (uint8_t) (v >> (8*i));
        }
    }
    static void set_u32(std::vector<uint8_t> & b, size_t at, uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            b[at + i] = (uint8_t) (v >> (8*i));
        }
    }

    static constexpr size_t table_bytes_of() { return 128 + (size_t) n_layer*3*24 + (size_t) n_layer*n_expert*8; }
    static constexpr uint32_t version_next() { return llama_expert_sidefile::version + 1; }

    size_t extent() const { return (2*row_gate + row_down + align - 1)/align*align; }
    size_t table_bytes() const { return table_bytes_of(); }
    size_t data_start() const { return (table_bytes() + align - 1)/align*align; }
    size_t file_size() const { return data_start() + (size_t) n_layer*n_expert*extent(); }
    size_t offset_at(int il, int e) const { return 128 + (size_t) n_layer*3*24 + ((size_t) il*n_expert + e)*8; }

    static void seal(std::vector<uint8_t> & t) {
        memset(&t[88], 0, 32);
        uint8_t sum[32];
        llama_expert_sha256(t.data(), t.size(), sum);
        memcpy(&t[88], sum, 32);
    }

    std::vector<uint8_t> build_table() const {
        std::vector<uint8_t> t;
        for (const char c : std::string("KOLIEXPT")) {
            t.push_back((uint8_t) c);
        }
        put_u32(t, llama_expert_sidefile::version);
        put_u32(t, align);
        put_u32(t, n_layer);
        put_u32(t, n_expert);
        put_u64(t, table_bytes());
        put_u64(t, file_size());
        put_u64(t, gguf.size());
        put_u64(t, head);
        uint8_t sum[32];
        llama_expert_sha256(gguf.data(), head, sum);
        t.insert(t.end(), sum, sum + 32);
        t.resize(128);
        for (const auto & lt : stacked) {
            for (const auto & r : lt) {
                put_u64(t, r.offset);
                put_u64(t, r.row_bytes);
                put_u32(t, r.type);
                put_u32(t, 0);
            }
        }
        for (int il = 0; il < n_layer; ++il) {
            for (int e = 0; e < n_expert; ++e) {
                put_u64(t, data_start() + ((size_t) il*n_expert + e)*extent());
            }
        }
        seal(t);
        return t;
    }

    void write_gguf() const {
        std::ofstream(gguf_path, std::ios::binary).write((const char *) gguf.data(), (std::streamsize) gguf.size());
    }

    // the side-file with the table t; data as the format says, or its first size bytes
    void write_side(const std::vector<uint8_t> & t, size_t size) const {
        std::vector<uint8_t> f(file_size());
        memcpy(f.data(), t.data(), std::min(t.size(), f.size()));
        for (int il = 0; il < n_layer; ++il) {
            for (int e = 0; e < n_expert; ++e) {
                size_t at = data_start() + ((size_t) il*n_expert + e)*extent();
                for (int k = 0; k < 3; ++k) {
                    const auto & r = stacked[il][k];
                    memcpy(&f[at], &gguf[r.offset + e*r.row_bytes], r.row_bytes);
                    at += r.row_bytes;
                }
            }
        }
        f.resize(size);
        std::ofstream(side_path, std::ios::binary).write((const char *) f.data(), (std::streamsize) f.size());
    }
    void write_side() const { write_side(table, file_size()); }
};

// a model load's check: the error says what failed, and nothing else opens
static std::string refusal(const sidefile_fixture & x, int fd_gguf) {
    try {
        llama_expert_sidefile side(x.side_path, x.stacked, sidefile_fixture::n_expert, fd_gguf, false);
    } catch (const std::exception & e) {
        return e.what();
    }
    return "";
}

// each of the three tensors receives the GGUF path's bytes on both paths; one read against three
static void test_sidefile_slots() {
    sidefile_fixture x;
    const int fd = open(x.gguf_path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    for (const bool buffered : {false, true}) {
        llama_expert_sidefile side(x.side_path, x.stacked, sidefile_fixture::n_expert, fd, buffered);
        CHECK(side.alignment() == sidefile_fixture::align && side.max_extent() == x.extent());
        std::vector<uint8_t> a[3], b[3];
        for (int il = 0; il < sidefile_fixture::n_layer; ++il) {
            for (int e = 0; e < sidefile_fixture::n_expert; ++e) {
                char * pa[3], * pb[3];
                for (int k = 0; k < 3; ++k) {
                    const auto n = x.stacked[il][k].row_bytes;
                    a[k].assign(n + 2, 0xEE); // guards: a row must not run over its slot
                    b[k].assign(n + 2, 0xEE);
                    pa[k] = (char *) a[k].data() + 1;
                    pb[k] = (char *) b[k].data() + 1;
                }
                uint64_t reads_gguf = 0, reads_side = 0;
                CHECK(llama_expert_read_rows(fd, x.gguf_path, x.stacked[il], e, pa, reads_gguf));
                CHECK(side.read(il, e, pb, reads_side));
                CHECK(reads_gguf == 3 && reads_side == 1);
                for (int k = 0; k < 3; ++k) {
                    const auto & r = x.stacked[il][k];
                    CHECK(a[k] == b[k] && a[k].front() == 0xEE && a[k].back() == 0xEE);
                    CHECK(memcmp(pa[k], &x.gguf[r.offset + e*r.row_bytes], r.row_bytes) == 0);
                }
            }
        }
        // several threads read concurrently into their own staging buffers
        std::atomic<int> bad{0};
        std::vector<std::thread> threads;
        for (int th = 0; th < 4; ++th) {
            threads.emplace_back([&, th] {
                std::vector<uint8_t> g(sidefile_fixture::row_gate), u(sidefile_fixture::row_gate), d(sidefile_fixture::row_down);
                char * dst[3] = { (char *) g.data(), (char *) u.data(), (char *) d.data() };
                for (int i = 0; i < 200; ++i) {
                    const int il = (th + i) % sidefile_fixture::n_layer;
                    const int e  = (th*3 + i) % sidefile_fixture::n_expert;
                    uint64_t reads = 0;
                    const auto & r = x.stacked[il];
                    if (!side.read(il, e, dst, reads) || reads != 1 ||
                        memcmp(dst[2], &x.gguf[r[2].offset + e*r[2].row_bytes], r[2].row_bytes) != 0) {
                        bad++;
                    }
                }
            });
        }
        for (auto & t : threads) {
            t.join();
        }
        CHECK(bad == 0);
    }
    close(fd);
}

// a read past the end of the file fails
static void test_sidefile_read_failure() {
    sidefile_fixture x;
    const int fd = open(x.gguf_path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    {
        llama_expert_sidefile side(x.side_path, x.stacked, sidefile_fixture::n_expert, fd, false);
        CHECK(truncate(x.side_path.c_str(), (off_t) x.data_start()) == 0);
        std::vector<uint8_t> g(sidefile_fixture::row_gate), u(sidefile_fixture::row_gate), d(sidefile_fixture::row_down);
        char * dst[3] = { (char *) g.data(), (char *) u.data(), (char *) d.data() };
        uint64_t reads = 0;
        CHECK(!side.read(0, 0, dst, reads));
        CHECK(reads == 1);
    }
    close(fd);
}

// every wrong table is refused, by name of what is wrong
// in pieces or at once, across the piece size
static void test_sha256_file() {
    const char * tmp = getenv("TMPDIR");
    std::string path = std::string(tmp ? tmp : "/tmp") + "/test-expert-sha256-XXXXXX";
    const int fd = mkstemp(path.data());
    CHECK(fd >= 0);
    const size_t piece = 1024*1024;
    std::vector<uint8_t> data(3*piece + 17);
    std::mt19937 rng(7);
    for (auto & b : data) {
        b = (uint8_t) rng();
    }
    CHECK(write(fd, data.data(), data.size()) == (ssize_t) data.size());
    for (const size_t n : { (size_t) 0, (size_t) 1, (size_t) 55, (size_t) 56, (size_t) 63, (size_t) 64, (size_t) 65,
                            piece - 1, piece, piece + 1, 2*piece, data.size() }) {
        uint8_t want[32], got[32];
        llama_expert_sha256(data.data(), n, want);
        CHECK(llama_expert_sha256_file(fd, n, got));
        CHECK(hex32(want) == hex32(got));
    }
    uint8_t got[32];
    CHECK(!llama_expert_sha256_file(fd, data.size() + 1, got));
    close(fd);
    unlink(path.c_str());
}

static void test_sidefile_refusals() {
    sidefile_fixture x;
    const int fd = open(x.gguf_path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    CHECK(refusal(x, fd).empty());

    const auto with_table = [&](const std::function<void(std::vector<uint8_t> &)> & edit, bool seal = true) {
        auto t = x.table;
        edit(t);
        if (seal) {
            sidefile_fixture::seal(t);
        }
        x.write_side(t, x.file_size());
    };
    const auto expect = [&](const char * what, const char * needle) {
        const std::string msg = refusal(x, fd);
        if (msg.find(needle) == std::string::npos) {
            fprintf(stderr, "%s:%d: FAIL %s: refusal \"%s\", want \"%s\"\n", __FILE__, __LINE__, what, msg.c_str(), needle);
            n_failed++;
        }
    };
    using F = sidefile_fixture;

    with_table([](std::vector<uint8_t> & t) { t[200] ^= 1; }, false);
    expect("a flipped byte in the tensor table, checksum left", "table checksum");
    with_table([&](std::vector<uint8_t> & t) { t[F::table_bytes_of() - 1] ^= 1; }, false);
    expect("a flipped byte in the offsets, checksum left", "table checksum");
    with_table([](std::vector<uint8_t> & t) { t[90] ^= 1; }, false);
    expect("a flipped byte in the checksum", "table checksum");
    with_table([](std::vector<uint8_t> & t) { t[0] ^= 1; });
    expect("bad magic", "bad magic");
    with_table([](std::vector<uint8_t> & t) { F::set_u32(t, 8, F::version_next()); });
    expect("a version from the future", "version");
    with_table([](std::vector<uint8_t> & t) { F::set_u32(t, 16, F::n_layer - 1); });
    expect("fewer layers", "layers, the model has");
    with_table([](std::vector<uint8_t> & t) { F::set_u32(t, 20, F::n_expert + 1); });
    expect("more experts", "experts per layer, the model has");
    with_table([](std::vector<uint8_t> & t) { F::set_u64(t, 24, F::table_bytes_of() + 8); });
    expect("a table of another length", "table of");
    with_table([&](std::vector<uint8_t> & t) { F::set_u64(t, 32, x.file_size() + F::align); });
    expect("a larger file size in the table", "file size");
    with_table([](std::vector<uint8_t> & t) { F::set_u32(t, 12, 0); });
    expect("an alignment of zero, which the extents divide by", "alignment");
    with_table([](std::vector<uint8_t> & t) { F::set_u32(t, 12, 512); });
    expect("an alignment below the device's", "alignment");
    with_table([](std::vector<uint8_t> & t) { F::set_u32(t, 12, 20000); });
    expect("an alignment that is not a power of two", "alignment");
    with_table([](std::vector<uint8_t> & t) { F::set_u64(t, 128 + 8, F::row_gate + 1); });
    expect("another row size", "row bytes");
    with_table([](std::vector<uint8_t> & t) { F::set_u32(t, 128 + 2*24 + 16, 18); });
    expect("another type for down", "ggml type");
    with_table([](std::vector<uint8_t> & t) { F::set_u64(t, 128, F::head + 8); });
    expect("another GGUF offset", "GGUF offset");
    with_table([&](std::vector<uint8_t> & t) { F::set_u64(t, x.offset_at(0, 1), x.data_start() + x.extent() + 1); });
    expect("a misaligned extent", "not aligned");
    with_table([&](std::vector<uint8_t> & t) { F::set_u64(t, x.offset_at(0, 0), 0); });
    expect("an extent over the table", "out of range");
    with_table([&](std::vector<uint8_t> & t) { F::set_u64(t, x.offset_at(2, 7), x.file_size()); });
    expect("an extent past the end", "out of range");
    with_table([&](std::vector<uint8_t> & t) { F::set_u64(t, x.offset_at(1, 3), ~(uint64_t) 0 - F::align + 1); });
    expect("an extent whose end overflows", "out of range");
    with_table([&](std::vector<uint8_t> & t) { F::set_u64(t, x.offset_at(1, 3), x.data_start() + 2*x.extent()); });
    expect("two experts in one extent", "overlaps");
    with_table([](std::vector<uint8_t> & t) { F::set_u64(t, 40, F::head + 1); });
    expect("another GGUF's size", "another GGUF");
    with_table([](std::vector<uint8_t> & t) { t[56] ^= 1; });
    expect("another GGUF's header", "another GGUF");
    // the table is checksummed, so this is what a crafted one says; nothing the size of the claim is allocated
    with_table([&](std::vector<uint8_t> & t) { F::set_u64(t, 48, x.gguf.size()); });
    expect("a GGUF header as long as the GGUF", "reaches into the experts");
    with_table([](std::vector<uint8_t> & t) { F::set_u64(t, 48, F::head + 1); });
    expect("a GGUF header one byte into the experts", "reaches into the experts");
    with_table([](std::vector<uint8_t> & t) { F::set_u64(t, 48, F::head - 1); });
    expect("a shorter GGUF header", "another GGUF");

    // a truncated or extended file, with a good table
    x.write_side(x.table, x.file_size() - F::align);
    expect("a truncated file", "file size");
    x.write_side(x.table, x.table_bytes() - 1);
    expect("a truncated table", "shorter than its table");
    x.write_side(x.table, 100);
    expect("a truncated header", "too short");
    x.write_side();
    CHECK(truncate(x.side_path.c_str(), (off_t) (x.file_size() + 4096)) == 0);
    expect("an extended file", "file size");
    x.write_side();
    CHECK(refusal(x, fd).empty());

    // the GGUF's identity, read from the file itself
    close(fd);
    auto header_changed = x.gguf;
    header_changed[10] ^= 1;
    std::ofstream(x.gguf_path, std::ios::binary).write((const char *) header_changed.data(), (std::streamsize) header_changed.size());
    int fd2 = open(x.gguf_path.c_str(), O_RDONLY);
    CHECK(refusal(x, fd2).find("another GGUF") != std::string::npos);
    close(fd2);

    // no file at all: no fallback
    unlink(x.side_path.c_str());
    {
        x.write_gguf();
        fd2 = open(x.gguf_path.c_str(), O_RDONLY);
        const std::string msg = refusal(x, fd2);
        CHECK(msg.find("cannot open") != std::string::npos && msg.find("no fallback") != std::string::npos);
        close(fd2);
    }
}

// --sidefile: a real pair, as described at the top
static int test_sidefile_pair(const char * gguf_path, const char * side_path, int per_layer) {
    ggml_context * ctx = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx };
    gguf_context * g = gguf_init_from_file(gguf_path, ip);
    if (!g) {
        fprintf(stderr, "cannot read %s\n", gguf_path);
        return 1;
    }
    const int64_t arch_key = gguf_find_key(g, "general.architecture");
    const std::string arch = gguf_get_val_str(g, arch_key);
    const int n_layer = (int) gguf_get_val_u32(g, gguf_find_key(g, (arch + ".block_count").c_str()));
    int n_expert = 0;
    std::vector<std::array<llama_expert_stacked, 3>> stacked(n_layer);
    static const char * kinds[3] = { "gate", "up", "down" };
    for (int il = 0; il < n_layer; ++il) {
        for (int k = 0; k < 3; ++k) {
            const std::string name = "blk." + std::to_string(il) + ".ffn_" + kinds[k] + "_exps.weight";
            const int64_t id = gguf_find_tensor(g, name.c_str());
            const ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
            if (id < 0 || !t) {
                fprintf(stderr, "no tensor %s\n", name.c_str());
                return 1;
            }
            n_expert = (int) t->ne[2];
            // as llama_model_loader's weight and the cache's slot tensor compute them
            stacked[il][k] = { (uint64_t) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id)),
                               (uint64_t) ggml_row_size(t->type, t->ne[0]) * (uint64_t) t->ne[1], (uint32_t) t->type };
        }
    }
    const int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cannot open %s\n", gguf_path);
        return 1;
    }
    int failed = 0;
    uint64_t experts = 0, reads_gguf = 0, reads_side = 0, bytes = 0;
    try {
        llama_expert_sidefile side(side_path, stacked, n_expert, fd, false);
        const int step = per_layer > 0 ? std::max(1, n_expert / per_layer) : 1;
        for (int il = 0; il < n_layer; ++il) {
            std::vector<uint8_t> a[3], b[3];
            char * pa[3], * pb[3];
            for (int k = 0; k < 3; ++k) {
                a[k].resize(stacked[il][k].row_bytes);
                b[k].resize(stacked[il][k].row_bytes);
                pa[k] = (char *) a[k].data();
                pb[k] = (char *) b[k].data();
            }
            // the first and the last expert of each layer, always
            for (int e = 0; e < n_expert; e = (e + step < n_expert || e == n_expert - 1) ? e + step : n_expert - 1) {
                const uint64_t r0 = reads_gguf, r1 = reads_side;
                if (!llama_expert_read_rows(fd, gguf_path, stacked[il], e, pa, reads_gguf) || !side.read(il, e, pb, reads_side)) {
                    fprintf(stderr, "layer %d expert %d: read failed\n", il, e);
                    failed++;
                    continue;
                }
                experts++;
                for (int k = 0; k < 3; ++k) {
                    bytes += a[k].size();
                    if (a[k] != b[k]) {
                        fprintf(stderr, "layer %d expert %d %s: bytes differ\n", il, e, kinds[k]);
                        failed++;
                    }
                }
                if (reads_gguf - r0 != 3 || reads_side - r1 != 1) {
                    fprintf(stderr, "layer %d expert %d: %llu and %llu preads\n", il, e,
                        (unsigned long long) (reads_gguf - r0), (unsigned long long) (reads_side - r1));
                    failed++;
                }
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "refused: %s\n", e.what());
        failed++;
    }
    close(fd);
    gguf_free(g);
    ggml_free(ctx);
    printf("sidefile pair: %llu experts of %d layers x %d, %llu bytes, %llu preads from the GGUF, %llu from the side-file, %d differences\n",
        (unsigned long long) experts, n_layer, n_expert, (unsigned long long) bytes, (unsigned long long) reads_gguf, (unsigned long long) reads_side, failed);
    return failed > 0 ? 1 : 0;
}
#endif
// the GPU remap's table: an expert's slot when the slot holds it, else -1
static void test_gate_table() {
    fake f(2, 4);
    std::vector<int32_t> table(16), out;
    auto c = make(f, 16, 2);
    auto slot_of_table = [&](int e) { c->copy_table(0, table.data()); return table[e]; };

    CHECK(slot_of_table(3) == -1); // nothing yet
    CHECK(remap(*c, 0, {1, 2, 3, 4}, out));
    for (int e : {1, 2, 3, 4}) {
        const int s = slot_of_table(e);
        CHECK(s >= 0 && f.at(0, s) == e);
    }

    // the eviction takes the expert out of the table
    CHECK(remap(*c, 0, {2, 3, 4, 5}, out));
    CHECK(slot_of_table(1) == -1);
    CHECK(slot_of_table(5) >= 0 && f.at(0, slot_of_table(5)) == 5);

    // a loading expert is not in the table, though it has a slot: a gate must stop there
    f.gated.insert({0, 6});
    CHECK(remap(*c, 1, {0}, out)); // layer 0 is no longer in flight
    int pf = 6;
    c->prefetch(0, &pf, 1, true);
    f.wait_busy(0);
    CHECK(c->slot_of(0, 6) >= 0);
    CHECK(slot_of_table(6) == -1);
    f.open = true;
    c->drain();
    CHECK(slot_of_table(6) >= 0 && f.at(0, slot_of_table(6)) == 6);

    // a failed demand load leaves nothing in the table
    f.failing.insert({0, 9});
    CHECK(!remap(*c, 0, {9}, out));
    CHECK(slot_of_table(9) == -1);
}

// the GPU remap defers a hit layer's bookkeeping until the command buffer has completed: it must do
// the same as a remap per layer, and every load must find the GPU idle
static void test_gate_deferred() {
    const int n_layer = 8, n_slots = 12, n_expert = 64, k = 3, n_tokens = 400;

    fake a(n_layer, n_slots), b(n_layer, n_slots);
    auto ca = make(a, n_expert, 2);
    auto cb = make(b, n_expert, 2);

    std::mt19937 rng(7);
    std::vector<std::vector<int32_t>> tables(n_layer, std::vector<int32_t>(n_expert, -1));

    uint64_t n_stops = 0, n_resident_read = 0;
    std::vector<int32_t> out;

    for (int tok = 0; tok < n_tokens; ++tok) {
        std::vector<std::vector<int32_t>> ids(n_layer);
        for (int l = 0; l < n_layer; ++l) {
            // skewed, so that layers hit and miss
            while ((int) ids[l].size() < k) {
                const int e = std::min<int>(n_expert - 1, (int) std::abs(std::normal_distribution<double>(0, 10)(rng)));
                if (std::find(ids[l].begin(), ids[l].end(), e) == ids[l].end()) {
                    ids[l].push_back(e);
                }
            }
        }

        // a remap per layer, as the CPU op does
        for (int l = 0; l < n_layer; ++l) {
            CHECK(remap(*ca, l, ids[l], out));
        }

        // the gates: while the "GPU runs", only the table decides, and nothing may load
        int reported = 0;
        for (int pos = 0; pos < n_layer;) {
            const int n0 = b.n_loads;
            int stop = -1;
            for (int l = pos; l < n_layer; ++l) {
                for (int e : ids[l]) {
                    if (tables[l][e] < 0) {
                        stop = l;
                    }
                }
                if (stop >= 0) {
                    break;
                }
                // what this layer's matmuls read is the expert itself, as long as no load runs
                for (int e : ids[l]) {
                    CHECK(b.at(l, tables[l][e]) == e);
                    n_resident_read++;
                }
            }
            CHECK(b.n_loads == n0);

            // the buffer has completed: the bookkeeping of the layers that passed, in order
            const int pass = stop >= 0 ? stop : n_layer;
            for (int l = reported; l < pass; ++l) {
                CHECK(remap(*cb, l, ids[l], out));
            }
            reported = std::max(reported, pass);
            CHECK(b.n_loads == n0); // hits read nothing

            if (stop >= 0) {
                n_stops++;
                CHECK(remap(*cb, stop, ids[stop], out));
                cb->copy_table(stop, tables[stop].data());
                reported = stop + 1;
                pos = stop; // resumes at this gate, which now passes
            } else {
                pos = n_layer;
            }
        }
    }
    CHECK(n_stops > 0 && n_resident_read > 0);

    const auto x = ca->counts(true), y = cb->counts(true);
    CHECK(x.steps == y.steps && x.needed == y.needed && x.hits == y.hits && x.misses == y.misses);
    CHECK(x.evictions == y.evictions && x.demand_bytes == y.demand_bytes && x.hits_loading == 0 && y.hits_loading == 0);
    CHECK(y.misses > 0 && y.hits > 0);
    CHECK(a.n_loads == b.n_loads && !b.overlap);
}

static void test_top_k() {
    std::vector<int> out;
    llama_expert_top_k({0.5, 2.0, 1.0, 2.0, 0.0}, 3, out);
    CHECK((out == std::vector<int>{1, 3, 2}));
    llama_expert_top_k({0.0, 0.0, 0.0}, 2, out);
    CHECK((out == std::vector<int>{0, 1}));
}

static void test_markov_fixture(const char * path) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "cannot read %s\n", path);
        n_failed++;
        return;
    }
    const auto j = nlohmann::json::parse(in);
    const int n_layer = j["layers"], n_expert = j["n_expert"], k = j["k"];
    const auto & steps = j["steps"];
    size_t n_pred = 0;
    for (size_t pi = 0; pi < j["params"].size(); ++pi) {
        const auto & p = j["params"][pi];
        llama_expert_markov m(n_layer, n_expert, p["half_life_tokens"].get<int>(), p["mix"].get<double>());
        const auto & preds = j["predictions"][pi];
        std::vector<int> out;
        size_t bad = 0;
        for (size_t i = 0; i < steps.size(); ++i) {
            const auto & s = steps[i];
            const int     l      = s["layer"];
            const int64_t first  = s["first"];
            const bool    decode = s["decode"];
            const auto    ex     = s["experts"].get<std::vector<int>>();
            const int target = m.predict(l, first, decode, k, out);
            std::vector<int> got = {target};
            got.insert(got.end(), out.begin(), out.end());
            const auto want = preds[i].get<std::vector<int>>();
            if (got != want && bad++ == 0) {
                fprintf(stderr, "params %zu, step %zu (token %lld, layer %d): predicted", pi, i, (long long) first, l);
                for (int x : got) { fprintf(stderr, " %d", x); }
                fprintf(stderr, ", Go");
                for (int x : want) { fprintf(stderr, " %d", x); }
                fprintf(stderr, "\n");
            }
            n_pred += target >= 0;
            m.observe(l, first, decode, ex.data(), (int) ex.size());
        }
        CHECK(bad == 0);
    }
    printf("markov fixture: %zu steps, %zu predictions over %zu parameter sets, as the Go predictor\n", steps.size(), n_pred, j["params"].size());
}

// a draft model gets the target's params except its expert cache: Kolibri only, one context per model
static void test_draft_params() {
    common_params target;
    target.expert_cache        = 128;
    target.expert_cache_layers = { 128, 64 };
    target.expert_cache_stats  = "stats.json";
    target.expert_cache_sidefile = "experts.side";
    target.expert_cache_gpu_remap = true;
    target.expert_cache_skip   = 0.05f;
    const common_params draft = common_base_params_to_speculative(target);
    CHECK(draft.expert_cache == 0);
    CHECK(draft.expert_cache_layers.empty());
    CHECK(draft.expert_cache_stats.empty());
    CHECK(draft.expert_cache_sidefile.empty());
    // the remap without slots is refused at load, so the draft would not load
    CHECK(!draft.expert_cache_gpu_remap);
    CHECK(draft.expert_cache_skip == 0.0f);
    CHECK(target.expert_cache == 128);
    CHECK(target.expert_cache_gpu_remap);
}

int main(int argc, char ** argv) {
#ifdef HAVE_SIDEFILE_TESTS
    if (argc >= 4 && strcmp(argv[1], "--sidefile") == 0) {
        return test_sidefile_pair(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 0);
    }
#endif
    test_lru();
    test_pinned();
    test_pinned_loading();
    test_capacity();
    test_prefetch_awaited();
    test_cancel();
    test_in_flight();
    test_failures();
    test_stress();
    test_concurrent_misses();
    test_wait_modes();
    test_wait_arg();
    test_skip_off();
    test_skip_rule();
    test_skip_loading();
    test_skip_ubatch();
    test_skip_map_rank();
    test_skip_arg();
    test_preads_counted();
    test_gate_table();
    test_gate_deferred();
    test_top_k();
    test_sha256();
#ifdef HAVE_SIDEFILE_TESTS
    test_sidefile_slots();
    test_sidefile_read_failure();
    test_sha256_file();
    test_sidefile_refusals();
#endif
    test_draft_params();
    if (argc > 1) {
        test_markov_fixture(argv[1]);
    } else {
        printf("markov fixture: skipped (no path given)\n");
    }
    if (n_failed > 0) {
        fprintf(stderr, "%d checks failed\n", n_failed);
        return 1;
    }
    printf("all expert cache tests passed\n");
    return 0;
}
