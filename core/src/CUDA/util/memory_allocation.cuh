#ifndef MEMORY_ALLOCATION_CUH
#define MEMORY_ALLOCATION_CUH

#include "gpu_types.cuh"

#ifdef __cplusplus
extern "C" {
#endif

CUDA_API void free_gpu(GpuPortion* gpu);
CUDA_API void free_gpu_weights(GpuWeights* gpu);
CUDA_API void free_gpu_activations(GpuActivations* gpu);
CUDA_API bool allocate_gpu_embeddings(GpuEmbeddings* gpu_emb, size_t vocab_size, size_t embedding_dim);
CUDA_API void free_gpu_embeddings(GpuEmbeddings* gpu_emb);
CUDA_API bool upload_embeddings(const float* h_embeddings, GpuEmbeddings* gpu_emb);
CUDA_API bool download_embeddings(float* h_embeddings, const GpuEmbeddings* gpu_emb);

CUDA_API void* pinned_alloc(size_t bytes);
CUDA_API void pinned_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // MEMORY_ALLOCATION_CUH
