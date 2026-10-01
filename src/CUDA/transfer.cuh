#ifndef CUDA_TRANSFER_CUH
#define CUDA_TRANSFER_CUH

#include <cstddef>     // size_t

#ifdef __cplusplus
extern "C" {
#endif

// GPU-side mirror of the context window and target that live in Portion.
// All pointers are device memory — do NOT dereference on the CPU.
// Uses void* to hide __nv_bfloat16 from the CPU compiler.
struct GpuPortion {
  void *d_context_window[2] = {nullptr, nullptr}; // device ptr: X  (128 tokens * 4 dim)
  void *d_target[2] = {nullptr, nullptr}; // device ptr: Y  (1  token  * 4 dim)

  size_t context_ptr = sizeof(d_context_window) / sizeof(d_context_window[0]);
  size_t target_ptr = sizeof(d_target) / sizeof(d_target[0]);

  cudaStream_t d_context_stream[2] = {nullptr, nullptr};
  cudaStream_t d_target_stream[2] = {nullptr, nullptr};
  
  size_t context_elements         = 0;       // total bfloat16 elements in X
  size_t target_elements          = 0;       // total bfloat16 elements in Y
};

// GPU-side mirror of the AI weights and biases.
struct GpuWeights {
  void* d_weights = nullptr; // device ptr to __nv_bfloat16
  void* d_bias    = nullptr; // device ptr to __nv_bfloat16
  size_t weight_elements = 0;
  size_t bias_elements   = 0;
};

// GPU-side activations buffer
struct GpuActivations {
  void* d_output = nullptr;  // device ptr to __nv_bfloat16
  size_t output_elements = 0;
};

#ifdef BUILDING_CUDA_DLL
  #define CUDA_API __declspec(dllexport)
#else
  #define CUDA_API __declspec(dllimport)
#endif

// Upload context window (X) and target (Y) from CPU bfloat16 vectors to GPU.
// Allocates device memory on first call; reallocates only if size changes.
// Returns false and prints to stderr if any CUDA call fails.
CUDA_API bool upload_to_gpu(const void *h_context, size_t context_elements,
                            const void *h_target,  size_t target_elements,
                            GpuPortion *gpu, int buffer_index);

CUDA_API bool download_from_gpu(void *h_context, size_t context_elements,
                                void *h_target,  size_t target_elements,
                                const GpuPortion *gpu, int buffer_index);

// Upload weights and biases from CPU to GPU
CUDA_API bool upload_ai_weights(const float* h_weights, size_t weight_elements,
                                const float* h_bias,    size_t bias_elements,
                                GpuWeights* gpu);

// Execute the forward pass: y = ReLU(W*x + b)
// x is gpu_portion->d_context_window
// W and b are in gpu_weights
// y is stored in gpu_activations->d_output
CUDA_API bool forward_pass(const GpuPortion* gpu_portion,
                           const GpuWeights* gpu_weights,
                           GpuActivations* gpu_activations,
                           size_t input_dim, size_t output_dim, int buffer_index);

// Free all device allocations inside GpuPortion, GpuWeights, and GpuActivations.
CUDA_API void free_gpu(GpuPortion *gpu);
CUDA_API void free_gpu_weights(GpuWeights *gpu);
CUDA_API void free_gpu_activations(GpuActivations *gpu);

// ── Pinned (page-locked) host memory wrappers ────────────────────────────────
// These exist so that g++ (MinGW) never needs to link against cudart.lib.
// ALL cudaHostAlloc / cudaFreeHost calls live inside the MSVC-compiled DLL.
//
// pinned_alloc: allocates 'bytes' of cudaHostAllocWriteCombined memory.
//   Returns a valid host pointer on success, nullptr on failure.
CUDA_API void* pinned_alloc(size_t bytes);

// pinned_free: releases a buffer previously returned by pinned_alloc.
//   Safe to call with nullptr.
CUDA_API void  pinned_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // CUDA_TRANSFER_CUH
