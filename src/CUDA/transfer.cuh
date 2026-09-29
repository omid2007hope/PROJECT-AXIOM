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
  void *d_context_window = nullptr; // device ptr: X  (128 tokens * 4 dim)
  void *d_target         = nullptr; // device ptr: Y  (1  token  * 4 dim)
  size_t context_elements         = 0;       // total bfloat16 elements in X
  size_t target_elements          = 0;       // total bfloat16 elements in Y
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
                            GpuPortion *gpu);

// Download context window (X) and target (Y) from GPU to CPU bfloat16 vectors for verification.
CUDA_API bool download_from_gpu(void *h_context, size_t context_elements,
                                void *h_target,  size_t target_elements,
                                const GpuPortion *gpu);

// Free all device allocations inside GpuPortion.
CUDA_API void free_gpu(GpuPortion *gpu);

#ifdef __cplusplus
}
#endif

#endif // CUDA_TRANSFER_CUH
