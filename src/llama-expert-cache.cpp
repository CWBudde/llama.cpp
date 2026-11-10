#include "llama-expert-cache.h"

#include "llama-impl.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <fcntl.h>
#if TARGET_OS_OSX
#include <libproc.h> // macOS only: iOS, tvOS and visionOS do not ship it
#define LLAMA_EXPERT_CACHE_PROC_RUSAGE
#endif
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

// the counters this platform measures; the others are reported as null
struct process_counts {
    uint64_t disk_read = 0;
    uint64_t pageins   = 0;
    uint64_t faults    = 0;
    uint64_t max_phys_footprint = 0;
#if defined(LLAMA_EXPERT_CACHE_PROC_RUSAGE)
    static constexpr bool has_io     = true; // disk_read, pageins, max_phys_footprint
#else
    static constexpr bool has_io     = false;
#endif
#if defined(__APPLE__) || defined(__linux__)
    static constexpr bool has_faults = true;
#else
    static constexpr bool has_faults = false;
#endif
};

std::string json_u64(bool measured, uint64_t v) {
    return measured ? std::to_string(v) : std::string("null");
}

// s as a JSON string, quotes included
std::string json_string(const std::string & s) {
    std::string out = "\"";
    for (const unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    out += format("\\u%04x", c);
                } else {
                    out += (char) c;
                }
        }
    }
    return out + "\"";
}

process_counts process_now() {
    process_counts c;
#if defined(LLAMA_EXPERT_CACHE_PROC_RUSAGE)
    rusage_info_v4 ri = {};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *) &ri) == 0) {
        c.disk_read          = ri.ri_diskio_bytesread;
        c.pageins            = ri.ri_pageins;
        c.max_phys_footprint = ri.ri_lifetime_max_phys_footprint;
    }
#endif
#if defined(__APPLE__) || defined(__linux__)
    rusage ru = {};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        c.faults = ru.ru_majflt;
    }
#endif
    return c;
}

void remap_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * ud = (llama_expert_cache::layer_ud *) userdata;
    ud->cache->remap(ud->il, dst->src[0], dst);
}

void wave_ids_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * ud = (llama_expert_cache::wave_ud *) userdata;
    ud->cache->wave_ids(ud->il, ud->w, dst->src[0], dst);
}

void wave_mask_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * ud = (llama_expert_cache::wave_ud *) userdata;
    ud->cache->wave_mask(ud->il, ud->w, dst->src[0], dst);
}

// extra: further members, each with a leading ", "
std::string counts_json(const llama_expert_cache_counts & c, const std::string & extra) {
    return format(
        "{\"steps\": %llu, \"needed\": %llu, \"hits\": %llu, \"hits_loading\": %llu, \"misses\": %llu, "
        "\"skipped\": %llu, \"evictions\": %llu, \"demand_bytes\": %llu, \"prefetch_bytes\": %llu, \"prefetch_issued\": %llu, "
        "\"prefetch_used\": %llu, \"prefetch_wasted\": %llu, \"prefetch_cancelled\": %llu, "
        "\"prefetch_skipped\": %llu, \"load_failures\": %llu, \"wait_seconds\": %.6f, \"io_seconds\": %.6f, "
        "\"io_prefetch_seconds\": %.6f, \"loads\": %llu, \"preads\": %llu, \"waves\": %llu, \"waves_built\": %llu, \"wave_parked\": %llu%s}",
        (unsigned long long) c.steps, (unsigned long long) c.needed, (unsigned long long) c.hits,
        (unsigned long long) c.hits_loading, (unsigned long long) c.misses, (unsigned long long) c.skipped, (unsigned long long) c.evictions,
        (unsigned long long) c.demand_bytes, (unsigned long long) c.prefetch_bytes,
        (unsigned long long) c.prefetch_issued, (unsigned long long) c.prefetch_used,
        (unsigned long long) c.prefetch_wasted, (unsigned long long) c.prefetch_cancelled,
        (unsigned long long) c.prefetch_skipped, (unsigned long long) c.load_failures, c.wait_seconds, c.io_seconds,
        c.io_prefetch_seconds, (unsigned long long) c.loads, (unsigned long long) c.preads, (unsigned long long) c.waves,
        (unsigned long long) c.waves_built, (unsigned long long) c.wave_parked, extra.c_str());
}

std::string process_json(const process_counts & pc, uint64_t disk_read0, uint64_t pageins0, uint64_t faults0) {
    return format("{\"disk_bytes_read\": %s, \"pageins\": %s, \"major_faults\": %s, \"max_phys_footprint\": %s}",
        json_u64(pc.has_io, pc.disk_read - disk_read0).c_str(), json_u64(pc.has_io, pc.pageins - pageins0).c_str(),
        json_u64(pc.has_faults, pc.faults - faults0).c_str(), json_u64(pc.has_io, pc.max_phys_footprint).c_str());
}

} // namespace

void llama_decode_diag::set_metal(bool has) {
    std::lock_guard<std::mutex> lock(mtx);
    has_metal = has;
}

void llama_decode_diag::add(double seconds, uint64_t n_splits, uint64_t n_commits) {
    std::lock_guard<std::mutex> lock(mtx);
    tokens++;
    wall_seconds  += seconds;
    splits        += n_splits;
    metal_commits += n_commits;
}

std::string llama_decode_diag::json_fields() const {
    std::lock_guard<std::mutex> lock(mtx);
    return format("\"tokens\": %llu, \"wall_seconds\": %.6f, \"splits\": %llu, \"metal_commits\": %s",
        (unsigned long long) tokens, wall_seconds, (unsigned long long) splits, json_u64(has_metal, metal_commits).c_str());
}

void llama_decode_diag::start() {
    const auto pc = process_now();
    disk_read0 = pc.disk_read;
    pageins0   = pc.pageins;
    faults0    = pc.faults;
}

void llama_decode_diag::write() const {
    const std::string stats = stats_json();
    LLAMA_LOG_INFO("%s: %s\n", __func__, stats.c_str());
    std::ofstream out(path);
    out << stats << "\n";
    if (!out) {
        LLAMA_LOG_ERROR("%s: cannot write %s\n", __func__, path.c_str());
    }
}

std::string llama_decode_diag::stats_json() const {
    return format("{\"expert_cache\": null, \"decode\": {%s}, \"process\": %s}",
        json_fields().c_str(), process_json(process_now(), disk_read0, pageins0, faults0).c_str());
}

llama_expert_cache::llama_expert_cache(const llama_expert_cache_params & p, int n_layer, int n_expert, int n_expert_used)
    : n_layer(n_layer), n_expert(n_expert), n_expert_used(n_expert_used),
      n_threads(std::max(p.n_threads, 1)), prefetch_k(std::max(p.prefetch_k, 0)),
      prefetch_half_life(std::max(p.prefetch_half_life, 1)), prefetch_mix(p.prefetch_mix),
      stats_path(p.stats_path ? p.stats_path : ""), buffered(p.buffered), sidefile_path(p.sidefile_path ? p.sidefile_path : ""), gpu_remap_on(p.gpu_remap), waves(p.waves) {
    switch (p.wait_mode) {
        case LLAMA_EXPERT_CACHE_WAIT_SLEEP:  wait.mode = llama_expert_wait::SLEEP;  break;
        case LLAMA_EXPERT_CACHE_WAIT_SPIN:   wait.mode = llama_expert_wait::SPIN;   break;
        case LLAMA_EXPERT_CACHE_WAIT_HYBRID: wait.mode = llama_expert_wait::HYBRID; break;
        case LLAMA_EXPERT_CACHE_WAIT_QOS:    wait.mode = llama_expert_wait::QOS;    break;
        default:
            throw std::runtime_error(format("expert cache: unknown wait mode %d", (int) p.wait_mode));
    }
    if (wait.mode == llama_expert_wait::HYBRID) {
        if (p.wait_spin_us < 1) {
            throw std::runtime_error(format("expert cache: hybrid wait with a spin of %d us, want > 0", (int) p.wait_spin_us));
        }
        wait.spin_us = p.wait_spin_us;
    }
    if (!std::isfinite(p.skip_weight) || p.skip_weight < 0) {
        throw std::runtime_error(format("expert cache: skip weight %g, want a finite weight >= 0", (double) p.skip_weight));
    }
    if (p.skip_map_rank != 1 && p.skip_map_rank != 2) {
        throw std::runtime_error(format("expert cache: skip map rank %d, want 1 or 2", (int) p.skip_map_rank));
    }
    skip_weight   = p.skip_weight;
    skip_map_rank = p.skip_map_rank;
    if (skip_on() && !gpu_remap_on) {
        // the rule lives in the Metal gate, which ranks the weights and zeroes them in place
        throw std::runtime_error("expert cache: --expert-cache-skip needs --expert-cache-gpu-remap");
    }
    if (gpu_remap_on && prefetch_k > 0) {
        // a prefetch writes a slot while the GPU runs; the gate relies on an idle GPU for every write
        throw std::runtime_error("expert cache: --expert-cache-gpu-remap does not work with --expert-prefetch");
    }
    if (p.n_slots_layer && p.n_layer_slots > 0) {
        if (p.n_layer_slots != n_layer) {
            throw std::runtime_error(format("expert cache: %d per-layer slot counts for %d layers", p.n_layer_slots, n_layer));
        }
        n_slots.assign(p.n_slots_layer, p.n_slots_layer + n_layer);
    } else {
        n_slots.assign(n_layer, p.n_slots);
    }
    for (int & s : n_slots) {
        if (s < n_expert_used) {
            throw std::runtime_error(format("expert cache: %d slots per layer, at least %d needed (the experts of one token)", s, n_expert_used));
        }
        s = std::min(s, n_expert);
    }
    if (prefetch_mix < 0 || prefetch_mix > 1) {
        throw std::runtime_error(format("expert cache: prefetch mix %g, want [0, 1]", prefetch_mix));
    }
    if (waves && !waves_on()) {
        LLAMA_LOG_WARN("%s: waves need %d slots per layer, the fewest are %d: the ubatch stays capped\n",
            __func__, 3*n_expert_used, *std::min_element(n_slots.begin(), n_slots.end()));
    }
    wud.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        for (int w = 0; w < n_waves(il, n_expert, n_expert_used); ++w) {
            wud[il].push_back({this, il, w});
        }
    }
}

llama_expert_cache::~llama_expert_cache() {
    // the I/O threads go first: they write into the slots and read the file
    if (slots) {
        slots->drain();
    }
    const std::string stats = slots ? stats_json() : "";
    slots.reset();
    if (!stats.empty()) {
        LLAMA_LOG_INFO("%s: %s\n", __func__, stats.c_str());
        if (!stats_path.empty()) {
            std::ofstream out(stats_path);
            out << stats << "\n";
            if (!out) {
                LLAMA_LOG_ERROR("%s: cannot write %s\n", __func__, stats_path.c_str());
            }
        }
    }
#if defined(__APPLE__) || defined(__linux__)
    if (fd >= 0) {
        close(fd);
    }
#endif
    for (auto & gr : groups) {
        if (gr.buf) {
            ggml_backend_buffer_free(gr.buf);
        }
#if defined(__APPLE__) || defined(__linux__)
        if (gr.host) {
            munmap(gr.host, gr.size);
        }
#endif
        if (gr.ctx) {
            ggml_free(gr.ctx);
        }
    }
}

size_t llama_expert_cache::slot_bytes() const {
    size_t n = 0;
    for (const auto & gr : groups) {
        n += gr.buf ? ggml_backend_buffer_get_size(gr.buf) : 0;
    }
    return n;
}

bool llama_expert_cache::holds(const ggml_tensor * t) const {
    for (const auto & layer : stacked) {
        for (const ggml_tensor * s : layer.t) {
            if (s == t) {
                return true;
            }
        }
    }
    return false;
}

void llama_expert_cache::init(const std::vector<tensors> & stacked_in, const std::vector<ggml_backend_dev_t> & devs, int fd_loader) {
    GGML_ASSERT((int) stacked_in.size() == n_layer && (int) devs.size() == n_layer);
    stacked = stacked_in;

    if (gpu_remap_on) {
        for (int il = 0; il < n_layer; ++il) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(devs[il]);
            if (!reg || strcmp(ggml_backend_reg_name(reg), "MTL") != 0) {
                throw std::runtime_error(format("expert cache: --expert-cache-gpu-remap needs every expert layer on Metal, and layer %d is on %s",
                    il, ggml_backend_dev_name(devs[il])));
            }
        }
    }

    // one group of slot tensors per device: the stacked tensors' first two dimensions, n_slots experts
    std::vector<int> group_of(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        int g = 0;
        while (g < (int) groups.size() && groups[g].dev != devs[il]) {
            g++;
        }
        if (g == (int) groups.size()) {
            groups.push_back({});
            groups[g].dev = devs[il];
        }
        group_of[il] = g;
    }
    for (auto & gr : groups) {
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead()*5*n_layer,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        gr.ctx = ggml_init(ip);
        if (!gr.ctx) {
            throw std::runtime_error("expert cache: ggml_init failed");
        }
    }
    static const char * names[3] = { "ffn_gate_exps_slots", "ffn_up_exps_slots", "ffn_down_exps_slots" };
    slot_t.resize(n_layer);
    expert_bytes.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        size_t bytes = 0;
        for (int k = 0; k < 3; ++k) {
            const ggml_tensor * s = stacked[il].t[k];
            if (!s || s->ne[2] != n_expert || !ggml_is_contiguous(s)) {
                throw std::runtime_error(format("expert cache: layer %d has no contiguous stacked tensor %s of %d experts", il, names[k], n_expert));
            }
            ggml_tensor * t = ggml_new_tensor_3d(groups[group_of[il]].ctx, s->type, s->ne[0], s->ne[1], n_slots[il]);
            ggml_format_name(t, "blk.%d.%s", il, names[k]);
            GGML_ASSERT(t->nb[2] == s->nb[2]);
            slot_t[il].t[k] = t;
            bytes += s->nb[2];
        }
        expert_bytes[il] = bytes;
    }

    if (gpu_remap_on) {
        gate_table.resize(n_layer);
        gate_slots.resize(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * tb = ggml_new_tensor_1d(groups[group_of[il]].ctx, GGML_TYPE_I32, n_expert);
            ggml_format_name(tb, "blk.%d.ffn_moe_table", il);
            ggml_tensor * sl = ggml_new_tensor_2d(groups[group_of[il]].ctx, GGML_TYPE_I32, n_expert_used, 2*(int64_t) gate_ubatch());
            ggml_format_name(sl, "blk.%d.ffn_moe_slot_ids", il);
            // what the Metal backend needs to know of it
            sl->op_params[0] = il;
            sl->op_params[1] = (int32_t) gate_ubatch();
            // [lossy] the skip's threshold, as bits, and the rank its ids map to
            memcpy(&sl->op_params[2], &skip_weight, sizeof(float));
            sl->op_params[3] = skip_map_rank;
            gate_table[il] = tb;
            gate_slots[il] = sl;
        }
    }

    for (auto & gr : groups) {
        const char * dev_name = ggml_backend_dev_name(gr.dev);
        if (ggml_backend_dev_type(gr.dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            // the device's own buffer type is plain host memory, unlike the repacking extra types
            gr.buf = ggml_backend_alloc_ctx_tensors_from_buft(gr.ctx, ggml_backend_dev_buffer_type(gr.dev));
        } else {
#if defined(__APPLE__) || defined(__linux__)
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(gr.dev, &props);
            if (!props.caps.buffer_from_host_ptr) {
                throw std::runtime_error(format("expert cache: %s cannot use host memory for the slots (no buffer_from_host_ptr); "
                    "the cache needs the expert layers on the CPU or on a device that can, such as Metal", dev_name));
            }
            // host memory the device uses in place: the I/O threads read into it, nothing is copied;
            // an anonymous mapping, so slots never written cost no memory
            const size_t align = (size_t) std::max(sysconf(_SC_PAGESIZE), 4096L); // also covers the buffer's own alignment
            size_t max_tensor = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(gr.ctx); t; t = ggml_get_next_tensor(gr.ctx, t)) {
                gr.size += GGML_PAD(ggml_nbytes(t), align);
                max_tensor = std::max(max_tensor, ggml_nbytes(t));
            }
            gr.host = mmap(nullptr, gr.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
            if (gr.host == MAP_FAILED) {
                gr.host = nullptr;
                throw std::runtime_error(format("expert cache: cannot map %zu bytes for the slots: %s", gr.size, strerror(errno)));
            }
            gr.buf = ggml_backend_dev_buffer_from_host_ptr(gr.dev, gr.host, gr.size, max_tensor);
            if (gr.buf) {
                ggml_tallocr ta = ggml_tallocr_new(gr.buf);
                for (ggml_tensor * t = ggml_get_first_tensor(gr.ctx); t; t = ggml_get_next_tensor(gr.ctx, t)) {
                    if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error(format("expert cache: cannot place %s in the slot buffer of %s", ggml_get_name(t), dev_name));
                    }
                }
            }
#else
            throw std::runtime_error(format("expert cache: the slots of %s need host memory the device can use, not supported on this platform", dev_name));
#endif
        }
        if (!gr.buf) {
            throw std::runtime_error(format("expert cache: cannot allocate the slots for %s", dev_name));
        }
        // the slots hold weights: the scheduler runs the expert matmuls where they are
        ggml_backend_buffer_set_usage(gr.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

#if defined(__APPLE__)
    char p[MAXPATHLEN];
    if (fcntl(fd_loader, F_GETPATH, p) == -1) {
        throw std::runtime_error(format("expert cache: F_GETPATH: %s", strerror(errno)));
    }
    path = p;
    fd = open(p, O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(format("expert cache: open %s: %s", p, strerror(errno)));
    }
    // reads go to the device, not through the page cache, unless asked to be buffered
    if (!buffered) {
        fcntl(fd, F_NOCACHE, 1);
    }
#elif defined(__linux__)
    buffered = true; // no page-cache bypass here
    path = format("/proc/self/fd/%d", fd_loader);
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(format("expert cache: open %s: %s", path.c_str(), strerror(errno)));
    }
#else
    GGML_UNUSED(fd_loader);
    throw std::runtime_error("expert cache: not supported on this platform");
#endif
#if defined(__APPLE__) || defined(__linux__)
    // the cache has its own open file, since F_NOCACHE would reach the loader's through a dup. A path
    // can name another file by now, and the offsets would read wrong weights without an error
    struct stat st_loader = {};
    struct stat st_own = {};
    if (fstat(fd_loader, &st_loader) != 0 || fstat(fd, &st_own) != 0) {
        throw std::runtime_error(format("expert cache: fstat of %s: %s", path.c_str(), strerror(errno)));
    }
    if (st_loader.st_dev != st_own.st_dev || st_loader.st_ino != st_own.st_ino) {
        throw std::runtime_error(format("expert cache: %s was replaced while the model loaded", path.c_str()));
    }
#endif

    rows.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        for (int k = 0; k < 3; ++k) {
            const ggml_tensor * t = slot_t[il].t[k];
            rows[il][k] = { (uint64_t) stacked[il].offs[k], (uint64_t) t->nb[2], (uint32_t) t->type };
        }
    }
    if (!sidefile_path.empty()) {
        sidefile = std::make_unique<llama_expert_sidefile>(sidefile_path, rows, n_expert, fd, buffered);
    }

    ud.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        ud[il] = { this, il };
    }
    for (ggml_tensor * tb : gate_table) {
        std::fill_n((int32_t *) tb->data, n_expert, -1); // no expert has a slot yet
    }
    slots = std::make_unique<llama_expert_slots>(n_slots, n_expert, expert_bytes, n_threads,
        [this](int il, int e, int s, uint64_t & preads) { return load(il, e, s, preads); }, wait);
    if (prefetch_k > 0) {
        markov = std::make_unique<llama_expert_markov>(n_layer, n_expert, prefetch_half_life, prefetch_mix);
    }

    const auto pc = process_now();
    disk_read0 = pc.disk_read;
    pageins0   = pc.pageins;
    faults0    = pc.faults;

    const std::string pf = markov
        ? format("markov same-token k %d, half-life %d, mix %g (%.1f MiB)", prefetch_k, prefetch_half_life, prefetch_mix, markov->bytes()/1024.0/1024.0)
        : std::string("off");
    std::string where;
    for (const auto & gr : groups) {
        where += format("%s%.2f MiB for %s", where.empty() ? "" : ", ", ggml_backend_buffer_get_size(gr.buf)/1024.0/1024.0, ggml_backend_dev_name(gr.dev));
    }
    const std::string sk = skip_on() ? format(", skip of one-token misses at rank >= 4 below %g (lossy)", (double) skip_weight) : std::string();
    LLAMA_LOG_INFO("%s: %s, %d to %d slots per layer of %d experts, %s, %d I/O threads, prefetch %s, wait %s, reading %s%s%s%s\n",
        __func__, gpu_remap_on ? "GPU remap" : "CPU remap", *std::min_element(n_slots.begin(), n_slots.end()), *std::max_element(n_slots.begin(), n_slots.end()),
        n_expert, where.c_str(), n_threads, pf.c_str(), wait_name().c_str(), sidefile ? sidefile_path.c_str() : path.c_str(), buffered ? " (buffered)" : "",
        waves_on() ? ", waves" : "", sk.c_str());
}

uint32_t llama_expert_cache::max_ubatch() const {
    if (waves_on()) {
        return UINT32_MAX;
    }
    return gate_ubatch();
}

uint32_t llama_expert_cache::gate_ubatch() const {
    return std::max(1, *std::min_element(n_slots.begin(), n_slots.end()) / n_expert_used);
}

bool llama_expert_cache::waves_on() const {
    return waves && llama_expert_wave_cap(*std::min_element(n_slots.begin(), n_slots.end()), n_expert_used) > 0;
}

int llama_expert_cache::n_waves(int il, int n_tokens, int k) const {
    // a k up to n_expert_used takes at most the waves allocated for it (wud); a larger k (warmup
    // routes all experts) leaves no room for waves
    const int n_touch = (int) std::min<int64_t>(n_expert, (int64_t) n_tokens*k);
    if (!waves_on() || k > n_expert_used || n_touch <= n_slots[il]) {
        return 1;
    }
    return llama_expert_wave_count(n_touch, n_slots[il], k);
}

uint32_t llama_expert_cache::wave_nodes(int n_tokens) const {
    uint32_t n = 0;
    for (int il = 0; il < n_layer; ++il) {
        const int nw = n_waves(il, n_tokens, n_expert_used);
        if (nw > 1) {
            n += 32u*nw; // per wave: the two ops, a dependency view, the expert matmuls and activation, mask and sum
        }
    }
    return n;
}

bool llama_expert_cache::attach() {
    return !attached.exchange(true);
}

void llama_expert_cache::detach() {
    attached = false;
}

ggml_tensor * llama_expert_cache::build_remap(ggml_context * ctx0, ggml_tensor * selected_experts, int il) {
    ggml_tensor * ids = ggml_cont(ctx0, selected_experts); // [n_expert_used, n_tokens]
    ggml_format_name(ids, "ffn_moe_topk_cont-%d", il);
    ggml_tensor * args[1] = { ids };
    ggml_tensor * out = ggml_custom_4d(ctx0, GGML_TYPE_I32, ids->ne[0], ids->ne[1], 1, 1, args, 1, remap_op, 1, &ud[il]);
    ggml_format_name(out, "ffn_moe_slots-%d", il);
    return out;
}

void llama_expert_cache::remap(int il, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    remap_ids(il, (const int32_t *) ids->data, (int) ggml_nelements(ids), (int) ids->ne[1], (int32_t *) dst->data);
}

void llama_expert_cache::bind_gate(ggml_tensor * mul_mat_id, int il, ggml_tensor * weights) const {
    if (mul_mat_id->op != GGML_OP_MUL_MAT_ID) {
        throw std::runtime_error("expert cache: --expert-cache-gpu-remap needs plain expert matmuls (no LoRA adapter)");
    }
    mul_mat_id->src[3] = gate_slots[il];
    mul_mat_id->src[4] = gate_table[il];
    // [lossy] the skip: the gate ranks the ids' weights and zeroes those of the ids it skips, so the
    // weights are computed before the matmul, and nothing may read them before it
    if (weights) {
        GGML_ASSERT(skip_on() && weights->type == GGML_TYPE_F32);
        mul_mat_id->src[5] = weights;
    }
}

void llama_expert_cache::gate_remap(int il, const int32_t * ids, int n_ids, int n_tokens, bool stopped) {
    // A graph compute finds its own gates, so a layer cut in two gets two, and a second remap of the
    // same layer would count and pin it twice. Hence no cut inside a layer, before that remap.
    if (il <= gate_il) {
        GGML_ABORT("expert cache: --expert-cache-gpu-remap: layer %d reached its Metal gate again in the same graph, after layer %d: "
                   "a cb_eval that stops between a layer's matmuls cuts it in two, and this mode cannot run that", il, gate_il);
    }
    gate_il = il;
    gate_seen++;
    gate_scratch.resize(n_ids);
    // [lossy] the gate logs an id it skipped as -1 - id: it is counted, and neither pinned nor loaded
    int n_kept = n_ids;
    if (skip_on() && n_tokens == 1) {
        gate_kept.clear();
        for (int i = 0; i < n_ids; ++i) {
            if (ids[i] >= 0) {
                gate_kept.push_back(ids[i]);
            }
        }
        n_kept = (int) gate_kept.size();
        // ranks 1 to 3 are never skipped
        GGML_ASSERT(n_kept >= std::min(n_ids, 3));
        ids = gate_kept.data();
    }
    remap_ids(il, ids, n_kept, n_tokens, gate_scratch.data());
    if (n_kept < n_ids) {
        slots->add_skipped(decode, n_ids - n_kept);
    }
    // only a stopped layer has new residents; the gate that resumes reads them from the table
    if (stopped) {
        slots->copy_table(il, (int32_t *) gate_table[il]->data);
    }
}

void llama_expert_cache::refresh_tables() {
    GGML_ASSERT(gpu_remap_on);
    std::vector<int32_t *> tables(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        tables[il] = (int32_t *) gate_table[il]->data;
    }
    slots->refresh_tables(tables.data());
}

void llama_expert_cache::gate_run(int n_tokens, bool stopped) {
    const int d = n_tokens == 1;
    gate_runs[d]++;
    gate_stops[d] += stopped ? 1 : 0;
}

void llama_expert_cache::gate_check_graph(bool has_outputs) {
    const int want = has_outputs ? n_layer : n_layer - 1;
    if (gate_seen != want) {
        GGML_ABORT("expert cache: --expert-cache-gpu-remap: %d of %d expert layers went through the Metal gate; the others left the GPU", gate_seen, want);
    }
    gate_abandon_graph();
}

void llama_expert_cache::gate_abandon_graph() {
    gate_seen = 0;
    gate_il   = -1;
}

ggml_tensor * llama_expert_cache::build_wave_ids(ggml_context * ctx0, ggml_tensor * ids, ggml_tensor * after, int il, int w) {
    GGML_ASSERT(w < (int) wud[il].size());
    // the op reads one element of the previous wave's sum: it runs once those matmuls are done, and
    // only then may the slots they read be reloaded
    ggml_tensor * args[2] = { ids, after ? ggml_view_1d(ctx0, after, 1, 0) : nullptr };
    ggml_tensor * out = ggml_custom_4d(ctx0, GGML_TYPE_I32, ids->ne[0], ids->ne[1], 1, 1, args, after ? 2 : 1, wave_ids_op, 1, &wud[il][w]);
    ggml_format_name(out, "ffn_moe_wave_slots-%d-%d", il, w);
    return out;
}

ggml_tensor * llama_expert_cache::build_wave_mask(ggml_context * ctx0, ggml_tensor * ids, ggml_tensor * wave_ids, int il, int w) {
    ggml_tensor * args[2] = { ids, wave_ids };
    ggml_tensor * out = ggml_custom_4d(ctx0, GGML_TYPE_F32, 1, ids->ne[0], ids->ne[1], 1, args, 2, wave_mask_op, 1, &wud[il][w]);
    ggml_format_name(out, "ffn_moe_wave_mask-%d-%d", il, w);
    return out;
}

// a graph runs its layers in order; layer 0 starts the next ubatch (the last layer may see fewer
// tokens, only the outputs)
void llama_expert_cache::begin(int il, int n_tokens) {
    if (il == 0) {
        tok_first = tok_next;
        tok_next += n_tokens;
        decode    = n_tokens == 1;
    }
}

void llama_expert_cache::observe(int il) {
    if (markov) {
        const auto & needed = slots->last_needed();
        markov->observe(il, tok_first, decode, needed.data(), (int) needed.size());
        const int target = markov->predict(il + 1, tok_first, decode, prefetch_k, pred);
        if (target >= 0) {
            slots->prefetch(target, pred.data(), (int) pred.size(), decode);
        }
    }
}

void llama_expert_cache::fail(int il, int n_tokens) {
    int l = -1;
    int e = -1;
    if (slots->failure(l, e)) {
        GGML_ABORT("expert cache: reading expert %d of layer %d from %s failed", e, l, sidefile ? sidefile_path.c_str() : path.c_str());
    }
    GGML_ABORT("expert cache: layer %d needs more distinct experts in one ubatch of %d tokens than its %d slots",
        il, n_tokens, n_slots[il]);
}

void llama_expert_cache::remap_ids(int il, const int32_t * ids, int n_ids, int n_tokens, int32_t * out) {
    const auto t0 = std::chrono::steady_clock::now();
    begin(il, n_tokens);
    if (!slots->remap(il, ids, n_ids, decode, out)) {
        fail(il, n_tokens);
    }
    observe(il);
    op_seconds[decode] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void llama_expert_cache::wave_ids(int il, int w, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    const int k = (int) ids->ne[0]; // the graph's top-k, which n_waves planned the graph with
    const auto t0 = std::chrono::steady_clock::now();
    const int n_tokens = (int) ids->ne[1];
    const int n = (int) ggml_nelements(ids);
    const auto * id = (const int32_t *) ids->data;
    if (w == 0) {
        begin(il, n_tokens);
        const int n_built   = n_waves(il, n_tokens, k);
        GGML_ASSERT(n_built > 1);
        const int n_planned = slots->plan_waves(il, id, n, k, llama_expert_wave_cap(n_slots[il], k), n_built);
        GGML_ASSERT(n_planned >= 0 && n_planned <= (int) wud[il].size());
        observe(il);
    }
    if (!slots->stage_wave(il, w, id, n, k, (int32_t *) dst->data)) {
        fail(il, n_tokens);
    }
    op_seconds[decode] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void llama_expert_cache::wave_mask(int il, int w, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) == ggml_nelements(ids));
    slots->wave_mask(il, w, (const int32_t *) ids->data, (int) ggml_nelements(ids), (float *) dst->data);
}

bool llama_expert_cache::load(int il, int e, int s, uint64_t & preads) {
    char * dst[3];
    for (int k = 0; k < 3; ++k) {
        const ggml_tensor * t = slot_t[il].t[k];
        dst[k] = (char *) t->data + (size_t) s*t->nb[2];
    }
    if (sidefile) {
        return sidefile->read(il, e, dst, preads);
    }
    return llama_expert_read_rows(fd, path, rows[il], e, dst, preads);
}

std::string llama_expert_cache::wait_name() const {
    switch (wait.mode) {
        case llama_expert_wait::SPIN:   return "spin";
        case llama_expert_wait::HYBRID: return format("hybrid:%d", wait.spin_us);
        case llama_expert_wait::QOS:    return "qos";
        case llama_expert_wait::SLEEP:  break;
    }
    return "sleep";
}

std::string llama_expert_cache::stats_json() const {
    const auto pc = process_now();
    const int s_min = *std::min_element(n_slots.begin(), n_slots.end());
    const int s_max = *std::max_element(n_slots.begin(), n_slots.end());
    const auto c_pre = slots->counts(false);
    const auto c_dec = slots->counts(true);
    // the remap op's own time: all of it but the wait on loads
    const auto remap = [](double op, double wait) { return format(", \"remap_seconds\": %.6f", std::max(op - wait, 0.0)); };
    // the GPU remap's command buffers, and those a gate stopped
    const auto gate = [this](int d) {
        return gpu_remap_on ? format(", \"gpu_runs\": %llu, \"gpu_stops\": %llu", (unsigned long long) gate_runs[d], (unsigned long long) gate_stops[d]) : std::string();
    };
    const std::string x_pre = remap(op_seconds[0], c_pre.wait_seconds) + gate(0);
    const std::string x_dec = remap(op_seconds[1], c_dec.wait_seconds) + gate(1) + (diag ? ", " + diag->json_fields() : std::string());
    return format(
        "{\"expert_cache\": {\"slots_min\": %d, \"slots_max\": %d, \"slot_bytes\": %zu, \"expert_bytes\": %zu, "
        "\"io_threads\": %d, \"prefetch_k\": %d, \"prefetch_half_life\": %d, \"prefetch_mix\": %g, \"buffered\": %s, \"gpu_remap\": %s, \"wait\": \"%s\", \"waves\": %s, \"skip_weight\": %g, \"file\": %s, \"sidefile\": %s}, "
        "\"prefill\": %s, \"decode\": %s, \"process\": %s}",
        s_min, s_max, slot_bytes(), expert_bytes.empty() ? (size_t) 0 : expert_bytes[0],
        n_threads, markov ? prefetch_k : 0, prefetch_half_life, prefetch_mix, buffered ? "true" : "false", gpu_remap_on ? "true" : "false", wait_name().c_str(),
        waves_on() ? "true" : "false", (double) skip_weight, json_string(path).c_str(), sidefile ? json_string(sidefile_path).c_str() : "null",
        counts_json(c_pre, x_pre).c_str(), counts_json(c_dec, x_dec).c_str(),
        process_json(pc, disk_read0, pageins0, faults0).c_str());
}
