#include "llama-expert-cache.h"

#include "llama-impl.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>

#if defined(__APPLE__)
#include <fcntl.h>
#include <libproc.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/resource.h>
#include <unistd.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

// the counters this platform measures; the others are reported as null
struct process_counts {
    uint64_t disk_read = 0;
    uint64_t pageins   = 0;
    uint64_t faults    = 0;
    uint64_t max_phys_footprint = 0;
#if defined(__APPLE__)
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
#if defined(__APPLE__)
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

// extra: further members, each with a leading ", "
std::string counts_json(const llama_expert_cache_counts & c, const std::string & extra) {
    return format(
        "{\"steps\": %llu, \"needed\": %llu, \"hits\": %llu, \"hits_loading\": %llu, \"misses\": %llu, "
        "\"evictions\": %llu, \"demand_bytes\": %llu, \"prefetch_bytes\": %llu, \"prefetch_issued\": %llu, "
        "\"prefetch_used\": %llu, \"prefetch_wasted\": %llu, \"prefetch_cancelled\": %llu, "
        "\"prefetch_skipped\": %llu, \"load_failures\": %llu, \"wait_seconds\": %.6f, \"io_seconds\": %.6f, "
        "\"io_prefetch_seconds\": %.6f%s}",
        (unsigned long long) c.steps, (unsigned long long) c.needed, (unsigned long long) c.hits,
        (unsigned long long) c.hits_loading, (unsigned long long) c.misses, (unsigned long long) c.evictions,
        (unsigned long long) c.demand_bytes, (unsigned long long) c.prefetch_bytes,
        (unsigned long long) c.prefetch_issued, (unsigned long long) c.prefetch_used,
        (unsigned long long) c.prefetch_wasted, (unsigned long long) c.prefetch_cancelled,
        (unsigned long long) c.prefetch_skipped, (unsigned long long) c.load_failures, c.wait_seconds, c.io_seconds,
        c.io_prefetch_seconds, extra.c_str());
}

std::string process_json(const process_counts & pc, uint64_t disk_read0, uint64_t pageins0, uint64_t faults0) {
    return format("{\"disk_bytes_read\": %s, \"pageins\": %s, \"major_faults\": %s, \"max_phys_footprint\": %s}",
        json_u64(pc.has_io, pc.disk_read - disk_read0).c_str(), json_u64(pc.has_io, pc.pageins - pageins0).c_str(),
        json_u64(pc.has_faults, pc.faults - faults0).c_str(), json_u64(pc.has_io, pc.max_phys_footprint).c_str());
}

} // namespace

std::string llama_decode_diag::json_fields() const {
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
      stats_path(p.stats_path ? p.stats_path : ""), buffered(p.buffered) {
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

void llama_expert_cache::init(const std::vector<tensors> & stacked_in, const std::vector<ggml_backend_dev_t> & devs, int fd_loader) {
    GGML_ASSERT((int) stacked_in.size() == n_layer && (int) devs.size() == n_layer);
    stacked = stacked_in;

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
            /*.mem_size   =*/ ggml_tensor_overhead()*3*n_layer,
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

    ud.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        ud[il] = { this, il };
    }
    slots = std::make_unique<llama_expert_slots>(n_slots, n_expert, expert_bytes, n_threads,
        [this](int il, int e, int s) { return load(il, e, s); });
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
    LLAMA_LOG_INFO("%s: %d to %d slots per layer of %d experts, %s, %d I/O threads, prefetch %s, reading %s%s\n",
        __func__, *std::min_element(n_slots.begin(), n_slots.end()), *std::max_element(n_slots.begin(), n_slots.end()),
        n_expert, where.c_str(), n_threads, pf.c_str(), path.c_str(), buffered ? " (buffered)" : "");
}

uint32_t llama_expert_cache::max_ubatch() const {
    return std::max(1, *std::min_element(n_slots.begin(), n_slots.end()) / n_expert_used);
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
    const auto t0 = std::chrono::steady_clock::now();
    const int n_tokens = (int) ids->ne[1];

    // a graph runs its layers in order; layer 0 starts the next ubatch (the last layer may see
    // fewer tokens, only the outputs)
    if (il == 0) {
        tok_first = tok_next;
        tok_next += n_tokens;
        decode    = n_tokens == 1;
    }

    if (!slots->remap(il, (const int32_t *) ids->data, (int) ggml_nelements(ids), decode, (int32_t *) dst->data)) {
        int l = -1;
        int e = -1;
        if (slots->failure(l, e)) {
            GGML_ABORT("expert cache: reading expert %d of layer %d from %s failed", e, l, path.c_str());
        }
        GGML_ABORT("expert cache: layer %d needs more distinct experts in one ubatch of %d tokens than its %d slots",
            il, n_tokens, n_slots[il]);
    }

    if (markov) {
        const auto & needed = slots->last_needed();
        markov->observe(il, tok_first, decode, needed.data(), (int) needed.size());
        const int target = markov->predict(il + 1, tok_first, decode, prefetch_k, pred);
        if (target >= 0) {
            slots->prefetch(target, pred.data(), (int) pred.size(), decode);
        }
    }
    op_seconds[decode] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool llama_expert_cache::load(int il, int e, int s) {
#if defined(__APPLE__) || defined(__linux__)
    for (int k = 0; k < 3; ++k) {
        const ggml_tensor * t = slot_t[il].t[k];
        const size_t n   = t->nb[2];
        char *       dst = (char *) t->data + (size_t) s*n;
        const off_t  off = (off_t) (stacked[il].offs[k] + (size_t) e*n);
        size_t done = 0;
        while (done < n) {
            const ssize_t r = pread(fd, dst + done, n - done, off + (off_t) done);
            if (r <= 0) {
                if (r < 0 && errno == EINTR) {
                    continue;
                }
                LLAMA_LOG_ERROR("%s: pread of %zu bytes at %lld in %s: %s\n", __func__, n - done,
                    (long long) (off + (off_t) done), path.c_str(), r < 0 ? strerror(errno) : "end of file");
                return false;
            }
            done += (size_t) r;
        }
    }
    return true;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(e);
    GGML_UNUSED(s);
    return false;
#endif
}

std::string llama_expert_cache::stats_json() const {
    const auto pc = process_now();
    const int s_min = *std::min_element(n_slots.begin(), n_slots.end());
    const int s_max = *std::max_element(n_slots.begin(), n_slots.end());
    const auto c_pre = slots->counts(false);
    const auto c_dec = slots->counts(true);
    // the remap op's own time: all of it but the wait on loads
    const auto remap = [](double op, double wait) { return format(", \"remap_seconds\": %.6f", std::max(op - wait, 0.0)); };
    const std::string x_pre = remap(op_seconds[0], c_pre.wait_seconds);
    const std::string x_dec = remap(op_seconds[1], c_dec.wait_seconds) + (diag ? ", " + diag->json_fields() : std::string());
    return format(
        "{\"expert_cache\": {\"slots_min\": %d, \"slots_max\": %d, \"slot_bytes\": %zu, \"expert_bytes\": %zu, "
        "\"io_threads\": %d, \"prefetch_k\": %d, \"prefetch_half_life\": %d, \"prefetch_mix\": %g, \"buffered\": %s, \"file\": %s}, "
        "\"prefill\": %s, \"decode\": %s, \"process\": %s}",
        s_min, s_max, slot_bytes(), expert_bytes.empty() ? (size_t) 0 : expert_bytes[0],
        n_threads, markov ? prefetch_k : 0, prefetch_half_life, prefetch_mix, buffered ? "true" : "false", json_string(path).c_str(),
        counts_json(c_pre, x_pre).c_str(), counts_json(c_dec, x_dec).c_str(),
        process_json(pc, disk_read0, pageins0, faults0).c_str());
}
