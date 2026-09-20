#ifndef CUDA_TRANSFER_CUH
#define CUDA_TRANSFER_CUH

#include <cuda_bf16.h> // __nv_bfloat16
#include <cstddef>     // size_t

// GPU-side mirror of the context window and target that live in Portion.
// All pointers are device memory — do NOT dereference on the CPU.
struct GpuPortion {
  __nv_bfloat16 *d_context_window = nullptr; // device ptr: X  (128 tokens * 4 dim)
  __nv_bfloat16 *d_target         = nullptr; // device ptr: Y  (1  token  * 4 dim)
  size_t context_elements         = 0;       // total bfloat16 elements in X
  size_t target_elements          = 0;       // total bfloat16 elements in Y
};

// Upload context window (X) and target (Y) from CPU bfloat16 vectors to GPU.
// Allocates device memory on first call; reallocates only if size changes.
// Returns false and prints to stderr if any CUDA call fails.
bool upload_to_gpu(const __nv_bfloat16 *h_context, size_t context_elements,
                   const __nv_bfloat16 *h_target,  size_t target_elements,
                   GpuPortion &gpu);

// Free all device allocations inside GpuPortion.
void free_gpu(GpuPortion &gpu);

#endif // CUDA_TRANSFER_CUH
