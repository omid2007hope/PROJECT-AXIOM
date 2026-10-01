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

    for (size_t each_context_ptr = 0; each_context_ptr <= gpu->context_ptr ; ++each_context_ptr) {
    
    // ── Context Window (X) ────────────────────────────────────────────────────
    if (gpu->context_elements != context_elements) {

        if (gpu->d_context_window[each_context_ptr]) {
          
          // Free the memory if an error accrued.
          if (!CUDA_CHECK(cudaFree(gpu->d_context_window[each_context_ptr]))) return false;
          gpu->d_context_window[each_context_ptr] = nullptr;
          
        }
        
        if (!CUDA_CHECK(cudaMalloc(&gpu->d_context_window[each_context_ptr],
                                   context_elements * sizeof(__nv_bfloat16)))) return false;
        gpu->context_elements = context_elements;
      
  } // end

  if (!CUDA_CHECK(cudaMemcpy(gpu->d_context_window[each_context_ptr], h_context,
                             context_elements * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice)))
    return false;
  
  };

  for (size_t each_target_ptr = 0; each_target_ptr <= gpu->target_ptr;
       ++each_target_ptr) {
    
        // ── Target (Y) ────────────────────────────────────────────────────────────
        if (gpu->target_elements != target_elements) {

          if (gpu->d_target[each_target_ptr]) {

            if (!CUDA_CHECK(cudaFree(gpu->d_target[each_target_ptr]))) return false;
            gpu->d_target[each_target_ptr] = nullptr;
          }
          
          if (!CUDA_CHECK(cudaMalloc(&gpu->d_target[each_target_ptr],
                                     target_elements * sizeof(__nv_bfloat16))))  return false;
          gpu->target_elements = target_elements;

        }
      
        if (!CUDA_CHECK(cudaMemcpy(gpu->d_target[each_target_ptr], h_target,
                                   target_elements * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice))) return false;

        return true;
      }
  };


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

// ─────────────────────────────────────────────────────────────────────────────
// pinned_alloc / pinned_free
// Thin wrappers so the g++ (MinGW) side never links against cudart.lib.
// cudaHostAllocWriteCombined bypasses CPU L1/L2 cache and enables true
// async DMA across the PCIe bus.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __declspec(dllexport) void* pinned_alloc(size_t bytes) {
  void* ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocWriteCombined);
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA] pinned_alloc failed (%zu bytes): %s\n",
            bytes, cudaGetErrorString(err));
    return nullptr;
  }
  return ptr;
}

extern "C" __declspec(dllexport) void pinned_free(void* ptr) {
  if (ptr) cudaFreeHost(ptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// upload_ai_weights
// Converts float32 host weights/biases to __nv_bfloat16 and uploads to VRAM.
// Allocates device buffers on first call; reallocates only if size changes.
// ─────────────────────────────────────────────────────────────────────────────

// Kernel: convert float32 → bfloat16 in-place on device
__global__ static void float_to_bf16_kernel(const float* __restrict__ src,
                                             __nv_bfloat16* __restrict__ dst,
                                             size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// GpuWeights is declared in transfer.cuh

extern "C" __declspec(dllexport) bool upload_ai_weights(
    const float* h_weights, size_t weight_elements,
    const float* h_bias,    size_t bias_elements,
    GpuWeights* gpu) {

  if (!gpu || !h_weights || !h_bias || weight_elements == 0 || bias_elements == 0) {
    fprintf(stderr, "Invalid parameters in upload_ai_weights\n");
    return false;
  }

  // ── Weights ───────────────────────────────────────────────────────────────
  if (gpu->weight_elements != weight_elements) {
    if (gpu->d_weights) { if (!CUDA_CHECK(cudaFree(gpu->d_weights))) return false; gpu->d_weights = nullptr; }
    if (!CUDA_CHECK(cudaMalloc(&gpu->d_weights, weight_elements * sizeof(__nv_bfloat16)))) return false;
    gpu->weight_elements = weight_elements;
  }

  // Temporary float32 staging buffer on device
  float* d_tmp_w = nullptr;
  if (!CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp_w), weight_elements * sizeof(float)))) return false;
  if (!CUDA_CHECK(cudaMemcpy(d_tmp_w, h_weights, weight_elements * sizeof(float), cudaMemcpyHostToDevice))) {
    cudaFree(d_tmp_w); return false;
  }
  int threads = 256;
  int blocks  = static_cast<int>((weight_elements + threads - 1) / threads);
  float_to_bf16_kernel<<<blocks, threads>>>(d_tmp_w, static_cast<__nv_bfloat16*>(gpu->d_weights), weight_elements);
  if (!CUDA_CHECK(cudaGetLastError())) { cudaFree(d_tmp_w); return false; }
  cudaFree(d_tmp_w);

  // ── Bias ──────────────────────────────────────────────────────────────────
  if (gpu->bias_elements != bias_elements) {
    if (gpu->d_bias) { if (!CUDA_CHECK(cudaFree(gpu->d_bias))) return false; gpu->d_bias = nullptr; }
    if (!CUDA_CHECK(cudaMalloc(&gpu->d_bias, bias_elements * sizeof(__nv_bfloat16)))) return false;
    gpu->bias_elements = bias_elements;
  }

  float* d_tmp_b = nullptr;
  if (!CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp_b), bias_elements * sizeof(float)))) return false;
  if (!CUDA_CHECK(cudaMemcpy(d_tmp_b, h_bias, bias_elements * sizeof(float), cudaMemcpyHostToDevice))) {
    cudaFree(d_tmp_b); return false;
  }
  blocks = static_cast<int>((bias_elements + threads - 1) / threads);
  float_to_bf16_kernel<<<blocks, threads>>>(d_tmp_b, static_cast<__nv_bfloat16*>(gpu->d_bias), bias_elements);
  if (!CUDA_CHECK(cudaGetLastError())) { cudaFree(d_tmp_b); return false; }
  cudaFree(d_tmp_b);

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// free_gpu_weights & free_gpu_activations
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __declspec(dllexport) void free_gpu_weights(GpuWeights *gpu) {
  if (!gpu) return;
  if (gpu->d_weights) { cudaFree(gpu->d_weights); gpu->d_weights = nullptr; }
  if (gpu->d_bias)    { cudaFree(gpu->d_bias);    gpu->d_bias    = nullptr; }
  gpu->weight_elements = 0;
  gpu->bias_elements   = 0;
}

extern "C" __declspec(dllexport) void free_gpu_activations(GpuActivations *gpu) {
  if (!gpu) return;
  if (gpu->d_output) { cudaFree(gpu->d_output); gpu->d_output = nullptr; }
  gpu->output_elements = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward_pass: y = ReLU(W*x + b)
// ─────────────────────────────────────────────────────────────────────────────
// A simple matrix-vector multiplication kernel in __nv_bfloat16.
__global__ static void forward_pass_kernel(
    const __nv_bfloat16* __restrict__ W, // [out_dim, in_dim]
    const __nv_bfloat16* __restrict__ x, // [in_dim]
    const __nv_bfloat16* __restrict__ b, // [out_dim]
    __nv_bfloat16* __restrict__ y,       // [out_dim]
    size_t in_dim, size_t out_dim) {
  
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < out_dim) {
    float sum = __bfloat162float(b[row]);
    for (size_t col = 0; col < in_dim; ++col) {
      sum += __bfloat162float(W[row * in_dim + col]) * __bfloat162float(x[col]);
    }
    // ReLU activation
    if (sum < 0.0f) sum = 0.0f;
    y[row] = __float2bfloat16(sum);
  }
}

extern "C" __declspec(dllexport) bool forward_pass(const GpuPortion* gpu_portion,
                                                   const GpuWeights* gpu_weights,
                                                   GpuActivations* gpu_activations,
                                                   size_t input_dim, size_t output_dim) {
  if (!gpu_portion || !gpu_weights || !gpu_activations) return false;
  if (!gpu_portion->d_context_window || !gpu_weights->d_weights || !gpu_weights->d_bias) return false;

  // Allocate output buffer if needed
  if (gpu_activations->output_elements != output_dim) {
    if (gpu_activations->d_output) {
      if (!CUDA_CHECK(cudaFree(gpu_activations->d_output))) return false;
      gpu_activations->d_output = nullptr;
    }
    if (!CUDA_CHECK(cudaMalloc(&gpu_activations->d_output, output_dim * sizeof(__nv_bfloat16)))) return false;
    gpu_activations->output_elements = output_dim;
  }

  int threads = 256;
  int blocks = static_cast<int>((output_dim + threads - 1) / threads);
  
  forward_pass_kernel<<<blocks, threads>>>(
      static_cast<const __nv_bfloat16*>(gpu_weights->d_weights),
      static_cast<const __nv_bfloat16*>(gpu_portion->d_context_window),
      static_cast<const __nv_bfloat16*>(gpu_weights->d_bias),
      static_cast<__nv_bfloat16*>(gpu_activations->d_output),
      input_dim, output_dim
  );
  
  if (!CUDA_CHECK(cudaGetLastError())) return false;
  if (!CUDA_CHECK(cudaDeviceSynchronize())) return false; // Synchronize for safety in this early phase
  
  return true;
}
