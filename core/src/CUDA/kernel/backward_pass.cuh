#ifndef BACKWARD_PASS_CUH
#define BACKWARD_PASS_CUH

#include "../util/gpu_types.cuh"

#ifdef __cplusplus
extern "C" {
#endif

CUDA_API bool backward_pass(const GpuPortion* gpu_portion,
			    GpuWeights* gpu_weights,
			    const GpuActivations* gpu_activations,
			    size_t input_dim,
			    size_t output_dim,
			    float learning_rate,
			    int buffer_index);

CUDA_API bool backward_pass_with_embeddings(const GpuPortion* gpu_portion,
					    GpuEmbeddings* gpu_embeddings,
					    GpuWeights* gpu_weights,
					    const GpuActivations* gpu_activations,
					    size_t input_dim,
					    size_t output_dim,
					    float learning_rate,
					    int buffer_index,
					    bool use_adamw);

#ifdef __cplusplus
}
#endif

#endif // BACKWARD_PASS_CUH
