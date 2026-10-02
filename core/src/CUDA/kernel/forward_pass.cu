#include "forward_pass.cuh"
#include "../util/cuda_utils.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ static void forward_pass_kernel(
    const __nv_bfloat16* __restrict__ W, 
    const __nv_bfloat16* __restrict__ x, 
    const __nv_bfloat16* __restrict__ b, 
    __nv_bfloat16* __restrict__ y,       
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

extern "C" CUDA_API bool forward_pass(const GpuPortion* gpu_portion,
                                      const GpuWeights* gpu_weights,
                                      GpuActivations* gpu_activations,
                                      size_t input_dim, size_t output_dim, int buffer_index) {
  if (!gpu_portion || !gpu_weights || !gpu_activations) return false;
  if (!gpu_portion->d_context_window[buffer_index] || !gpu_weights->d_weights || !gpu_weights->d_bias) return false;

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
  
  forward_pass_kernel<<<blocks, threads, 0, (cudaStream_t)gpu_portion->stream[buffer_index]>>>(
      static_cast<const __nv_bfloat16*>(gpu_weights->d_weights),
      static_cast<const __nv_bfloat16*>(gpu_portion->d_context_window[buffer_index]),
      static_cast<const __nv_bfloat16*>(gpu_weights->d_bias),
      static_cast<__nv_bfloat16*>(gpu_activations->d_output),
      input_dim, output_dim
  );
  
  if (!CUDA_CHECK(cudaGetLastError())) return false;
  
  return true;
}
