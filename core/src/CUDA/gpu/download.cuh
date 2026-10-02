#ifndef DOWNLOAD_CUH
#define DOWNLOAD_CUH

#include "../util/gpu_types.cuh"

#ifdef __cplusplus
extern "C" {
#endif

CUDA_API bool download_from_gpu(void *h_context, size_t context_elements,
                                void *h_target,  size_t target_elements,
                                const GpuPortion *gpu, int buffer_index);

#ifdef __cplusplus
}
#endif

#endif // DOWNLOAD_CUH
