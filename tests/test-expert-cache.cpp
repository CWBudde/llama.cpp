// Tests of the expert cache's bookkeeping (src/llama-expert-slots.*) with a fake loader, and of its
// Markov predictor against the predictions of the Go original:
//
//   test-expert-cache [markov-fixture.json]
//
// The fixture is testdata/locality/markov-fixture.json of the kolibri-llama-cpp repository; without
// it the predictor check is skipped.

#include "../src/llama-expert-slots.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <utility>
#include <vector>

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

static std::unique_ptr<llama_expert_slots> make(fake & f, int n_expert, int n_threads, size_t bytes = 100) {
    return std::make_unique<llama_expert_slots>(std::vector<int>(f.n_layer, f.n_slots), n_expert,
        std::vector<size_t>(f.n_layer, bytes), n_threads, [&f](int l, int e, int s) { return f.load(l, e, s); });
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

int main(int argc, char ** argv) {
    test_lru();
    test_pinned();
    test_pinned_loading();
    test_capacity();
    test_prefetch_awaited();
    test_cancel();
    test_in_flight();
    test_failures();
    test_stress();
    test_top_k();
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
