// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

// diagnostics: the Metal command buffers this process has committed so far (all devices, keep-alive
// work excluded); also reachable through the registry as "ggml_backend_metal_get_n_commits"
GGML_BACKEND_API uint64_t ggml_backend_metal_get_n_commits(void);

// [EXPERIMENTAL] the expert cache's GPU remap: a MUL_MAT_ID whose src[3] and src[4] are set reads
// its ids from src[3], which a gate kernel fills from the table src[4] before the layer's first
// matmul. The gate stops the command buffer when an id has no slot; the backend then calls remap
// with the GPU idle, and resumes at that gate. See ggml_metal_graph_compute_gated.
//
// remap: called in layer order, once per gate of a graph: for a gate that passed, after its run
//        has completed (stopped false), and for a gate that stopped (stopped true), which must
//        load the ids' experts and update the table before it returns
// run:   called after every command buffer of a graph of n_tokens tokens; stopped says whether a
//        gate stopped it
struct ggml_backend_metal_expert_gate {
    void (*remap)(void * user, int il, const int32_t * ids, int n_ids, int n_tokens, bool stopped);
    void (*run)  (void * user, int n_tokens, bool stopped);
    void * user;
};

// also reachable through the registry as "ggml_backend_metal_set_expert_gate"
GGML_BACKEND_API void ggml_backend_metal_set_expert_gate(ggml_backend_t backend, const struct ggml_backend_metal_expert_gate * gate);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

#ifdef __cplusplus
}
#endif
