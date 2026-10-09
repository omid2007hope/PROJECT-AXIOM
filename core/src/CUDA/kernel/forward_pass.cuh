#ifndef FORWARD_PASS_CUH
#define FORWARD_PASS_CUH

#include "../util/gpu_types.cuh"

#ifdef __cplusplus
extern "C" {
#endif

CUDA_API bool forward_pass(const GpuPortion* gpu_portion,
			   const GpuWeights* gpu_weights,
			   GpuActivations* gpu_activations,
			   size_t input_dim,
			   size_t output_dim,
			   int buffer_index);

#ifdef __cplusplus
}
#endif

#endif // FORWARD_PASS_CUH
