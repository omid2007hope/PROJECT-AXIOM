#ifndef CUDA_TELEMETRY_CUH
#define CUDA_TELEMETRY_CUH

#include "gpu_types.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// Fetches GPU VRAM usage.
CUDA_API void get_gpu_telemetry(size_t* free_mem, size_t* total_mem);

#ifdef __cplusplus
}
#endif

#endif // CUDA_TELEMETRY_CUH
