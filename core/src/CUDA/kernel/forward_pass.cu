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

  /*
  blockDim.x is the number of apartments in each building (e.g., 10 apartments
  per building).

  blockIdx.x is the building number (e.g., Building #3).


  threadIdx.x is the specific apartment number inside that building (e.g.,
  Apartment #5).

  Your Global ID is your absolute position on the street: (Building 3 × 10
  apartments) + Apartment 5 = The 35th apartment overall.
*/

  /*
  blockDim.x (Block Size):
  This is a fixed constant: it is always 256 for every thread. It never changes.

  blockIdx.x (Block Number):
  Because we only need 4 outputs, we only launched 1 block. So for all 256
threads, the block number is always 0. It never changes.

  threadIdx.x (The Thread's ID):
  This is the only number that changes! It is each worker's badge number: 0, 1,
2, 3, 4, ... 255.

*/


size_t row = blockIdx.x * blockDim.x + threadIdx.x;

if (row < out_dim) {
  float sum = __bfloat162float(b[row]);

  for (size_t col = 0; col < in_dim; ++col) {
      sum += __bfloat162float(W[row * in_dim + col]) * __bfloat162float(x[col]);
    }

    // ReLU activation
    if (sum < 0.0f)
      sum = 0.0f;
    
    y[row] = __float2bfloat16(sum);
  }
}

extern "C" CUDA_API bool forward_pass(const GpuPortion *gpu_portion,
                                      const GpuWeights *gpu_weights,
                                      GpuActivations *gpu_activations,
                                      size_t input_dim, size_t output_dim,
                                      int buffer_index) {

  // Safe Guard
  if (!gpu_portion || !gpu_weights || !gpu_activations || !gpu_portion->d_context_window[buffer_index] || !gpu_weights->d_weights || !gpu_weights->d_bias)
    return false;
  
  if (gpu_activations->output_elements != output_dim) {
    if (gpu_activations->d_output) {
      if (!CUDA_CHECK(cudaFree(gpu_activations->d_output))) return false;
      gpu_activations->d_output = nullptr;
    }

    if (!CUDA_CHECK(cudaMalloc(&gpu_activations->d_output, output_dim * sizeof(__nv_bfloat16)))) return false;
    gpu_activations->output_elements = output_dim;
  }

  forward_pass_kernel<<<Grid->block, Grid->thread, Grid->index, (cudaStream_t)gpu_portion->stream[buffer_index]>>>(
      static_cast<const __nv_bfloat16*>(gpu_weights->d_weights),
      static_cast<const __nv_bfloat16*>(gpu_portion->d_context_window[buffer_index]),
      static_cast<const __nv_bfloat16*>(gpu_weights->d_bias),
      static_cast<__nv_bfloat16*>(gpu_activations->d_output),
      input_dim, output_dim
  );
  
  if (!CUDA_CHECK(cudaGetLastError())) return false;
  
  return true;
}
