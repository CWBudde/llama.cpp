#include "llama-expert-sidefile.h"

#include "llama-impl.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#if defined(__APPLE__) || defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr char     sidefile_magic[8]  = { 'K', 'O', 'L', 'I', 'E', 'X', 'P', 'T' };
constexpr size_t   sidefile_header    = 128;
constexpr size_t   sidefile_tensor    = 24;
constexpr size_t   sidefile_checksum  = 88;
constexpr uint32_t sidefile_min_align = 4096;
const char * const expert_kinds[3]    = { "gate", "up", "down" };

uint32_t load_u32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

uint64_t load_u64(const uint8_t * p) {
    return (uint64_t) load_u32(p) | ((uint64_t) load_u32(p + 4) << 32);
}

std::string hex(const uint8_t * p, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        s += format("%02x", p[i]);
    }
    return s;
}

uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void sha256_block(uint32_t h[8], const uint8_t * p) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t) p[4*i] << 24) | ((uint32_t) p[4*i + 1] << 16) | ((uint32_t) p[4*i + 2] << 8) | (uint32_t) p[4*i + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
        const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + mj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

#if defined(__APPLE__) || defined(__linux__)
// reads all n bytes at off; a short read goes on, each pread is counted
bool pread_all(int fd, const std::string & path, char * dst, size_t n, off_t off, uint64_t & preads) {
    size_t done = 0;
    while (done < n) {
        preads++;
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
    return true;
}

// a load's read buffer, freed when the I/O thread ends. Page-aligned: as far as XNU's cluster layer goes
// (not measured here), a direct F_NOCACHE read needs the destination aligned as well as the file offset,
// and a slot's rows mostly are not, as their sizes are no multiples of a page or a device block.
struct staging {
    void * data = nullptr;
    size_t size = 0;
    ~staging() { free(data); }

    bool reserve(size_t n) {
        if (n <= size) {
            return true;
        }
        void * p = nullptr;
        if (posix_memalign(&p, (size_t) sysconf(_SC_PAGESIZE), n) != 0) {
            return false;
        }
        free(data);
        data = p;
        size = n;
        return true;
    }
};
#endif

} // namespace

void llama_expert_sha256(const void * data, size_t n, uint8_t out[32]) {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    const uint8_t * p = (const uint8_t *) data;
    size_t left = n;
    for (; left >= 64; left -= 64, p += 64) {
        sha256_block(h, p);
    }
    uint8_t tail[128] = {};
    if (left > 0) {
        memcpy(tail, p, left);
    }
    tail[left] = 0x80;
    const size_t blocks = left + 9 <= 64 ? 1 : 2;
    const uint64_t bits = (uint64_t) n * 8;
    for (int i = 0; i < 8; ++i) {
        tail[blocks*64 - 1 - i] = (uint8_t) (bits >> (8*i));
    }
    for (size_t b = 0; b < blocks; ++b) {
        sha256_block(h, tail + b*64);
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[4*i + j] = (uint8_t) (h[i] >> (24 - 8*j));
        }
    }
}

bool llama_expert_read_rows(int fd, const std::string & path, const std::array<llama_expert_stacked, 3> & stacked, int e,
        char * const dst[3], uint64_t & preads) {
#if defined(__APPLE__) || defined(__linux__)
    for (int k = 0; k < 3; ++k) {
        const size_t n   = (size_t) stacked[k].row_bytes;
        const off_t  off = (off_t) (stacked[k].offset + (uint64_t) e*n);
        if (!pread_all(fd, path, dst[k], n, off, preads)) {
            return false;
        }
    }
    return true;
#else
    GGML_UNUSED(fd);
    GGML_UNUSED(path);
    GGML_UNUSED(stacked);
    GGML_UNUSED(e);
    GGML_UNUSED(dst);
    GGML_UNUSED(preads);
    return false;
#endif
}

llama_expert_sidefile::llama_expert_sidefile(const std::string & path_in, const std::vector<std::array<llama_expert_stacked, 3>> & stacked,
        int n_expert, int fd_gguf, bool buffered)
    : path(path_in), n_expert(n_expert) {
    const auto fail = [&](const std::string & why) {
        return std::runtime_error(format("expert cache: side-file %s: %s", path.c_str(), why.c_str()));
    };
#if defined(__APPLE__) || defined(__linux__)
    // the table is read through the page cache; F_NOCACHE comes on after it, for the experts
    const auto read_at = [&](int f, void * dst, size_t n, off_t off) {
        size_t done = 0;
        while (done < n) {
            const ssize_t r = pread(f, (char *) dst + done, n - done, off + (off_t) done);
            if (r < 0 && errno == EINTR) {
                continue;
            }
            if (r <= 0) {
                return false;
            }
            done += (size_t) r;
        }
        return true;
    };

    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw fail(format("cannot open: %s (there is no fallback to the GGUF)", strerror(errno)));
    }
    try {
        struct stat st = {};
        if (fstat(fd, &st) != 0) {
            throw fail(format("fstat: %s", strerror(errno)));
        }
        const uint64_t actual = (uint64_t) st.st_size;
        if (actual < sidefile_header) {
            throw fail(format("%llu bytes are too short for a header of %zu (truncated?)", (unsigned long long) actual, sidefile_header));
        }

        std::vector<uint8_t> table(sidefile_header);
        if (!read_at(fd, table.data(), table.size(), 0)) {
            throw fail("cannot read the header");
        }
        if (memcmp(table.data(), sidefile_magic, sizeof(sidefile_magic)) != 0) {
            throw fail("bad magic: not a side-file");
        }
        const uint32_t ver = load_u32(&table[8]);
        if (ver != version) {
            throw fail(format("version %u, this build reads version %u", ver, version));
        }
        align = load_u32(&table[12]);
        const uint32_t n_layer_t  = load_u32(&table[16]);
        const uint32_t n_expert_t = load_u32(&table[20]);
        const uint64_t table_len  = load_u64(&table[24]);
        const uint64_t size_table    = load_u64(&table[32]);
        const uint64_t gguf_size  = load_u64(&table[40]);
        const uint64_t gguf_head  = load_u64(&table[48]);

        // the counts first: they bound what the rest reads
        if (n_layer_t != stacked.size()) {
            throw fail(format("%u layers, the model has %zu", n_layer_t, stacked.size()));
        }
        if (n_expert_t != (uint32_t) n_expert) {
            throw fail(format("%u experts per layer, the model has %d", n_expert_t, n_expert));
        }
        const uint64_t n_tensor = (uint64_t) n_layer_t*3;
        const uint64_t n_extent = (uint64_t) n_layer_t*n_expert_t;
        const uint64_t table_want = sidefile_header + n_tensor*sidefile_tensor + n_extent*sizeof(uint64_t);
        if (table_len != table_want) {
            throw fail(format("table of %llu bytes, %u layers of %u experts take %llu", (unsigned long long) table_len,
                n_layer_t, n_expert_t, (unsigned long long) table_want));
        }
        if (actual < table_len) {
            throw fail(format("file size %llu, shorter than its table of %llu bytes (truncated)", (unsigned long long) actual, (unsigned long long) table_len));
        }
        table.resize(table_len);
        if (!read_at(fd, table.data() + sidefile_header, table_len - sidefile_header, sidefile_header)) {
            throw fail("cannot read the table");
        }

        uint8_t want[32];
        memcpy(want, &table[sidefile_checksum], sizeof(want));
        memset(&table[sidefile_checksum], 0, sizeof(want));
        uint8_t got[32];
        llama_expert_sha256(table.data(), table.size(), got);
        memcpy(&table[sidefile_checksum], want, sizeof(want));
        if (memcmp(got, want, sizeof(want)) != 0) {
            throw fail(format("table checksum mismatch: sha256 %s, the table says %s (corrupt: rebuild it with kolibri-sidefile write)",
                hex(got, 32).c_str(), hex(want, 32).c_str()));
        }

        if (size_table != actual) {
            throw fail(format("file size %llu, the table says %llu (truncated or extended)", (unsigned long long) actual, (unsigned long long) size_table));
        }
        if (align < sidefile_min_align || (align & (align - 1)) != 0 || actual % align != 0) {
            throw fail(format("alignment %u: want a power of two of at least %u that divides the file's %llu bytes",
                align, sidefile_min_align, (unsigned long long) actual));
        }

        // the rows must be the very ranges the GGUF path reads
        row_bytes.resize(n_layer_t);
        extent_bytes.resize(n_layer_t);
        const uint8_t * e_tensor = &table[sidefile_header];
        for (uint32_t il = 0; il < n_layer_t; ++il) {
            uint64_t rows = 0;
            for (int k = 0; k < 3; ++k, e_tensor += sidefile_tensor) {
                const auto & s = stacked[il][k];
                const uint64_t off = load_u64(e_tensor);
                const uint64_t row = load_u64(e_tensor + 8);
                const uint32_t typ = load_u32(e_tensor + 16);
                if (row != s.row_bytes) {
                    throw fail(format("layer %u %s: row bytes %llu, the model has %llu", il, expert_kinds[k], (unsigned long long) row, (unsigned long long) s.row_bytes));
                }
                if (typ != s.type) {
                    throw fail(format("layer %u %s: ggml type %u, the model has %u", il, expert_kinds[k], typ, s.type));
                }
                if (off != s.offset) {
                    throw fail(format("layer %u %s: GGUF offset %llu, the model has %llu", il, expert_kinds[k], (unsigned long long) off, (unsigned long long) s.offset));
                }
                row_bytes[il][k] = row;
                rows += row;
            }
            extent_bytes[il] = (size_t) ((rows + align - 1)/align*align);
            largest = std::max(largest, extent_bytes[il]);
        }

        // the extents: aligned, behind the table, inside the file, overlapping nothing
        const uint64_t data_start = (table_len + align - 1)/align*align;
        offsets.resize(n_extent);
        std::vector<std::pair<uint64_t, uint64_t>> spans(n_extent); // offset, index
        const uint8_t * e_offset = &table[sidefile_header + n_tensor*sidefile_tensor];
        for (uint64_t i = 0; i < n_extent; ++i, e_offset += sizeof(uint64_t)) {
            const uint64_t off = load_u64(e_offset);
            const uint32_t il  = (uint32_t) (i/n_expert_t);
            const uint32_t e   = (uint32_t) (i%n_expert_t);
            if (off % align != 0) {
                throw fail(format("layer %u expert %u: extent at %llu, not aligned to %u", il, e, (unsigned long long) off, align));
            }
            if (off < data_start || off > actual || actual - off < extent_bytes[il]) {
                throw fail(format("layer %u expert %u: extent [%llu, +%zu) out of range [%llu, %llu)", il, e, (unsigned long long) off,
                    extent_bytes[il], (unsigned long long) data_start, (unsigned long long) actual));
            }
            offsets[i] = off;
            spans[i]   = { off, i };
        }
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size(); ++i) {
            const uint64_t prev = spans[i-1].second;
            if (spans[i-1].first + extent_bytes[prev/n_expert_t] > spans[i].first) {
                const uint64_t cur = spans[i].second;
                throw fail(format("layer %llu expert %llu overlaps layer %llu expert %llu at %llu",
                    (unsigned long long) (prev/n_expert_t), (unsigned long long) (prev%n_expert_t),
                    (unsigned long long) (cur/n_expert_t), (unsigned long long) (cur%n_expert_t), (unsigned long long) spans[i].first));
            }
        }

        // the GGUF's identity: its size and the hash of its header
        struct stat sg = {};
        if (fstat(fd_gguf, &sg) != 0) {
            throw fail(format("fstat of the GGUF: %s", strerror(errno)));
        }
        if ((uint64_t) sg.st_size != gguf_size) {
            throw fail(format("written from another GGUF: its size was %llu, this one has %llu", (unsigned long long) gguf_size, (unsigned long long) sg.st_size));
        }
        if (gguf_head > gguf_size) {
            throw fail(format("GGUF header of %llu bytes in a GGUF of %llu", (unsigned long long) gguf_head, (unsigned long long) gguf_size));
        }
        std::vector<uint8_t> head(gguf_head);
        if (!read_at(fd_gguf, head.data(), head.size(), 0)) {
            throw fail("cannot read the GGUF's header");
        }
        llama_expert_sha256(head.data(), head.size(), got);
        if (memcmp(got, &table[56], sizeof(got)) != 0) {
            throw fail(format("written from another GGUF: its header's sha256 was %s, this one's is %s", hex(&table[56], 32).c_str(), hex(got, 32).c_str()));
        }

        // reads go to the device, not through the page cache, unless asked to be buffered
#if defined(__APPLE__)
        if (!buffered) {
            fcntl(fd, F_NOCACHE, 1);
        }
#else
        GGML_UNUSED(buffered);
#endif
    } catch (...) {
        close(fd);
        fd = -1;
        throw;
    }
#else
    GGML_UNUSED(stacked);
    GGML_UNUSED(fd_gguf);
    GGML_UNUSED(buffered);
    throw fail("not supported on this platform");
#endif
}

llama_expert_sidefile::~llama_expert_sidefile() {
#if defined(__APPLE__) || defined(__linux__)
    if (fd >= 0) {
        close(fd);
    }
#endif
}

bool llama_expert_sidefile::read(int il, int e, char * const dst[3], uint64_t & preads) const {
#if defined(__APPLE__) || defined(__linux__)
    static thread_local staging stage;
    const size_t n = extent_bytes[il];
    if (!stage.reserve(n)) {
        LLAMA_LOG_ERROR("%s: cannot allocate %zu bytes to read into\n", __func__, n);
        return false;
    }
    const uint64_t off = offsets[(uint64_t) il*n_expert + e];
    if (!pread_all(fd, path, (char *) stage.data, n, (off_t) off, preads)) {
        return false;
    }
    const char * src = (const char *) stage.data;
    for (int k = 0; k < 3; ++k) {
        memcpy(dst[k], src, (size_t) row_bytes[il][k]);
        src += row_bytes[il][k];
    }
    return true;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(e);
    GGML_UNUSED(dst);
    GGML_UNUSED(preads);
    return false;
#endif
}
