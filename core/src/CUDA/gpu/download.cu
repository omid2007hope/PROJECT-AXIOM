#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../util/cuda_utils.cuh"
#include "download.cuh"

extern "C" CUDA_API bool download_from_gpu(void* h_context,
					   size_t context_elements,
					   void* h_target,
					   size_t target_elements,
					   const GpuPortion* gpu,
					   int buffer_index) {
	if (!gpu)
		return false;

	if (h_context && gpu->d_context_window[buffer_index] &&
	    context_elements == gpu->context_elements) {
		if (!CUDA_CHECK(cudaMemcpyAsync(h_context,
						gpu->d_context_window[buffer_index],
						context_elements * sizeof(__nv_bfloat16),
						cudaMemcpyDeviceToHost,
						(cudaStream_t)gpu->stream[buffer_index])))
			return false;
	}

	if (h_target && gpu->d_target[buffer_index] && target_elements == gpu->target_elements) {
		if (!CUDA_CHECK(cudaMemcpyAsync(h_target,
						gpu->d_target[buffer_index],
						target_elements * sizeof(__nv_bfloat16),
						cudaMemcpyDeviceToHost,
						(cudaStream_t)gpu->stream[buffer_index])))
			return false;
	}

	return true;
}
