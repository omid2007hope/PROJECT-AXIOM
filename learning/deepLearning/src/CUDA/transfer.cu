#include "transfer.cuh"
#include <cuda_bf16.h> // __nv_bfloat16
#include <cstdio>  // fprintf, stderr
#include <cstdlib> // nullptr

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper: check a CUDA return code and print a human-readable error.
// ─────────────────────────────────────────────────────────────────────────────
static bool cuda_check(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d — %s\n", file, line,
            cudaGetErrorString(err));
    return false;
  }
  return true;
}
#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

// ─────────────────────────────────────────────────────────────────────────────
// upload_to_gpu
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __declspec(dllexport) bool upload_to_gpu(const void *h_context, size_t context_elements,
                   const void *h_target,  size_t target_elements,
                   GpuPortion *gpu) {

  if (!gpu || !h_context || !h_target || context_elements == 0 || target_elements == 0) {
    fprintf(stderr, "Invalid memory parameters in upload_to_gpu\n");
    return false;};

  // ── Context Window (X) ────────────────────────────────────────────────────
  if (gpu->context_elements != context_elements) {
    if (gpu->d_context_window) {
      if (!CUDA_CHECK(cudaFree(gpu->d_context_window))) return false;
      gpu->d_context_window = nullptr;
    }
    if (!CUDA_CHECK(cudaMalloc(&gpu->d_context_window,
                               context_elements * sizeof(__nv_bfloat16))))
      return false;
    gpu->context_elements = context_elements;
  }

  if (!CUDA_CHECK(cudaMemcpy(gpu->d_context_window, h_context,
                             context_elements * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice)))
    return false;

  // ── Target (Y) ────────────────────────────────────────────────────────────
  if (gpu->target_elements != target_elements) {
    if (gpu->d_target) {
      if (!CUDA_CHECK(cudaFree(gpu->d_target))) return false;
      gpu->d_target = nullptr;
    }
    if (!CUDA_CHECK(cudaMalloc(&gpu->d_target,
                               target_elements * sizeof(__nv_bfloat16))))
      return false;
    gpu->target_elements = target_elements;
  }

  if (!CUDA_CHECK(cudaMemcpy(gpu->d_target, h_target,
                             target_elements * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice)))
    return false;

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// download_from_gpu
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __declspec(dllexport) bool download_from_gpu(void *h_context, size_t context_elements,
                       void *h_target,  size_t target_elements,
                       const GpuPortion *gpu) {
  if (!gpu) return false;

  if (h_context && gpu->d_context_window && context_elements == gpu->context_elements) {
    if (!CUDA_CHECK(cudaMemcpy(h_context, gpu->d_context_window,
                               context_elements * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost)))
      return false;
  }

  if (h_target && gpu->d_target && target_elements == gpu->target_elements) {
    if (!CUDA_CHECK(cudaMemcpy(h_target, gpu->d_target,
                               target_elements * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost)))
      return false;
  }

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// free_gpu
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __declspec(dllexport) void free_gpu(GpuPortion *gpu) {
  if (!gpu) return;
  if (gpu->d_context_window) {
    cudaFree(gpu->d_context_window);
    gpu->d_context_window = nullptr;
    gpu->context_elements = 0;
  }
  if (gpu->d_target) {
    cudaFree(gpu->d_target);
    gpu->d_target = nullptr;
    gpu->target_elements = 0;
  }
}
