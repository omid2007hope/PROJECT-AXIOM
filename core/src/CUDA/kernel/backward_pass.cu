#include "backward_pass.cuh"
#include "../util/cuda_utils.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ static void backward_pass_kernel(
    __nv_bfloat16* __restrict__ W, 
    const __nv_bfloat16* __restrict__ x, 
    __nv_bfloat16* __restrict__ b, 
    const __nv_bfloat16* __restrict__ y, 
    const __nv_bfloat16* __restrict__ target, 
    size_t in_dim, size_t out_dim, float learning_rate) {

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < out_dim) {
    float y_val = __bfloat162float(y[row]);
    float t_val = __bfloat162float(target[row]);
    float dy = y_val - t_val;
    
    // Gradient of ReLU
    float dz = (y_val > 0.0f) ? dy : 0.0f;
    
    float b_val = __bfloat162float(b[row]);
    b[row] = __float2bfloat16(b_val - learning_rate * dz);
    
    for (size_t col = 0; col < in_dim; ++col) {
      float x_val = __bfloat162float(x[col]);
      float dw = dz * x_val;
      
      float w_val = __bfloat162float(W[row * in_dim + col]);
      W[row * in_dim + col] = __float2bfloat16(w_val - learning_rate * dw);
    }
  }
}

extern "C" CUDA_API bool backward_pass(const GpuPortion *gpu_portion,
                                       GpuWeights *gpu_weights,
                                       const GpuActivations *gpu_activations,
                                       size_t input_dim, size_t output_dim,
                                       float learning_rate, int buffer_index) {

  // Safe Guard
  if (!gpu_portion || !gpu_weights || !gpu_activations || 
      !gpu_portion->d_context_window[buffer_index] || !gpu_portion->d_target[buffer_index] || 
      !gpu_weights->d_weights || !gpu_weights->d_bias || !gpu_activations->d_output) {
    return false;
  }

  Grid grid(output_dim);

  backward_pass_kernel<<<grid.block, grid.thread, grid.index, (cudaStream_t)gpu_portion->stream[buffer_index]>>>(
      static_cast<__nv_bfloat16*>(gpu_weights->d_weights),
      static_cast<const __nv_bfloat16*>(gpu_portion->d_context_window[buffer_index]),
      static_cast<__nv_bfloat16*>(gpu_weights->d_bias),
      static_cast<const __nv_bfloat16*>(gpu_activations->d_output),
      static_cast<const __nv_bfloat16*>(gpu_portion->d_target[buffer_index]),
      input_dim, output_dim, learning_rate
  );
  
  if (!CUDA_CHECK(cudaGetLastError())) return false;
  
  return true;
}
