#include "transfer.cuh"
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
//
// Copies the CPU bfloat16 context window (X) and target (Y) into device RAM.
//
// Memory management:
//   - First call: cudaMalloc allocates both device buffers.
//   - Subsequent calls where size hasn't changed: skips realloc, goes straight
//     to cudaMemcpy (zero extra syscall cost).
//   - Size change: frees old allocation then reallocates to new size.
//
// The CPU vectors use std::bfloat16_t (C++23 <stdfloat>).
// CUDA uses __nv_bfloat16. They have the same 16-bit IEEE 754 BF16 layout,
// so a raw pointer cast is safe — no conversion kernel needed.
// ─────────────────────────────────────────────────────────────────────────────
bool upload_to_gpu(const __nv_bfloat16 *h_context, size_t context_elements,
                   const __nv_bfloat16 *h_target,  size_t target_elements,
                   GpuPortion &gpu) {

  // ── Context Window (X) ────────────────────────────────────────────────────
  if (gpu.context_elements != context_elements) {
    if (gpu.d_context_window) {
      if (!CUDA_CHECK(cudaFree(gpu.d_context_window))) return false;
      gpu.d_context_window = nullptr;
    }
    if (!CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gpu.d_context_window),
                               context_elements * sizeof(__nv_bfloat16))))
      return false;
    gpu.context_elements = context_elements;
  }

  if (!CUDA_CHECK(cudaMemcpy(gpu.d_context_window, h_context,
                             context_elements * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice)))
    return false;

  // ── Target (Y) ────────────────────────────────────────────────────────────
  if (gpu.target_elements != target_elements) {
    if (gpu.d_target) {
      if (!CUDA_CHECK(cudaFree(gpu.d_target))) return false;
      gpu.d_target = nullptr;
    }
    if (!CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gpu.d_target),
                               target_elements * sizeof(__nv_bfloat16))))
      return false;
    gpu.target_elements = target_elements;
  }

  if (!CUDA_CHECK(cudaMemcpy(gpu.d_target, h_target,
                             target_elements * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice)))
    return false;

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// free_gpu
//
// Releases both device allocations and resets the struct to a clean state.
// Safe to call multiple times (guards against double-free).
// ─────────────────────────────────────────────────────────────────────────────
void free_gpu(GpuPortion &gpu) {
  if (gpu.d_context_window) {
    cudaFree(gpu.d_context_window);
    gpu.d_context_window = nullptr;
    gpu.context_elements = 0;
  }
  if (gpu.d_target) {
    cudaFree(gpu.d_target);
    gpu.d_target = nullptr;
    gpu.target_elements = 0;
  }
}
