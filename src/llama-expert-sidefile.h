#pragma once

// [EXPERIMENTAL] The expert cache's reads, independent of ggml: from the GGUF with one pread per
// row, or from a side-file with one pread per expert (Kolibri only).
//
// The side-file, written by the kolibri-llama-cpp repository's cmd/kolibri-sidefile, holds every
// expert's gate, up and down rows, contiguous, in one extent aligned for F_NOCACHE. Opening it
// checks its table against the model and the GGUF, and refuses any mismatch. It does not hash the
// data, about 31 GB at IQ3: kolibri-sidefile verify checks every byte against the GGUF, and only
// it does.
//
// The layout, little endian:
//
//    0  char[8]   magic "KOLIEXPT"
//    8  u32       version
//   12  u32       alignment of the extents
//   16  u32       layers
//   20  u32       experts per layer
//   24  u64       table bytes: this header, the tensor entries and the extent offsets
//   32  u64       file size
//   40  u64       GGUF size
//   48  u64       GGUF header bytes: everything before its data section
//   56  byte[32]  sha256 of those GGUF header bytes
//   88  byte[32]  sha256 of the table, with these 32 bytes zero
//   120 byte[8]   zero
//   128 layers*3 tensor entries, gate, up, down of each layer: u64 absolute offset of expert 0 in
//       the GGUF, u64 row bytes (one expert), u32 ggml type, u32 zero
//   then layers*experts extent offsets, u64, absolute in the side-file
//
// An extent holds an expert's rows and zeros to the alignment; the reads take all of it.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// sha256 of n bytes: the side-file's table and the GGUF's header
void llama_expert_sha256(const void * data, size_t n, uint8_t out[32]);

// one stacked tensor, as the model has it
struct llama_expert_stacked {
    uint64_t offset    = 0; // absolute offset of expert 0 in the GGUF
    uint64_t row_bytes = 0; // one expert's rows
    uint32_t type      = 0; // ggml type
};

// reads expert e's three rows from the GGUF, one pread per row, into dst; false after logging a failure.
// preads counts the system calls, also those of a failed load.
bool llama_expert_read_rows(int fd, const std::string & path, const std::array<llama_expert_stacked, 3> & stacked, int e,
        char * const dst[3], uint64_t & preads);

struct llama_expert_sidefile {
    static constexpr uint32_t version = 1;

    // opens the side-file and checks its table; throws, saying what failed, on any mismatch. Without
    // buffered, reads bypass the page cache (F_NOCACHE on macOS). fd_gguf is the model's file.
    llama_expert_sidefile(const std::string & path, const std::vector<std::array<llama_expert_stacked, 3>> & stacked,
            int n_expert, int fd_gguf, bool buffered);
    ~llama_expert_sidefile();

    llama_expert_sidefile(const llama_expert_sidefile &) = delete;
    llama_expert_sidefile & operator=(const llama_expert_sidefile &) = delete;

    // reads expert e of layer il with one pread of its extent (more only after a short read) into a
    // page-aligned staging buffer of the calling thread, and copies the three rows to dst, exactly
    // the bytes llama_expert_read_rows gives. Safe to call from several threads.
    bool read(int il, int e, char * const dst[3], uint64_t & preads) const;

    const std::string & file() const { return path; }

    // the extents' common alignment, and the largest extent: one I/O thread's staging buffer
    uint32_t alignment() const { return align; }
    size_t   max_extent() const { return largest; }

private:
    std::string path;
    int      fd    = -1;
    int      n_expert;
    uint32_t align = 0;
    size_t   largest = 0;
    std::vector<std::array<uint64_t, 3>> row_bytes; // per layer
    std::vector<size_t>   extent_bytes; // per layer, the padding included
    std::vector<uint64_t> offsets;      // per layer and expert
};
