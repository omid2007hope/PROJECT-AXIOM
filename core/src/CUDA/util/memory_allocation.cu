#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>

#include "cuda_utils.cuh"
#include "memory_allocation.cuh"

__global__ static void float_to_bf16_array_kernel(const float* __restrict__ src,
						  __nv_bfloat16* __restrict__ dst,
						  size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		dst[i] = __float2bfloat16(src[i]);
	}
}

extern "C" CUDA_API void free_gpu(GpuPortion* gpu) {
	if (!gpu)
		return;
	for (size_t i = 0; i < gpu->context_ptr; ++i) {
		if (gpu->d_context_window[i]) {
			cudaFree(gpu->d_context_window[i]);
			gpu->d_context_window[i] = nullptr;
		}
		if (gpu->d_token_ids[i]) {
			cudaFree(gpu->d_token_ids[i]);
			gpu->d_token_ids[i] = nullptr;
		}
		if (gpu->d_dx[i]) {
			cudaFree(gpu->d_dx[i]);
			gpu->d_dx[i] = nullptr;
		}
	}
	gpu->context_elements = 0;
	gpu->token_count = 0;

	for (size_t i = 0; i < gpu->target_ptr; ++i) {
		if (gpu->d_target[i]) {
			cudaFree(gpu->d_target[i]);
			gpu->d_target[i] = nullptr;
		}
	}
	gpu->target_elements = 0;

	for (size_t i = 0; i < gpu->context_ptr; ++i) {
		if (gpu->stream[i]) {
			cudaStreamDestroy((cudaStream_t)gpu->stream[i]);
			gpu->stream[i] = nullptr;
		}
		if (gpu->compute_done[i]) {
			cudaEventDestroy((cudaEvent_t)gpu->compute_done[i]);
			gpu->compute_done[i] = nullptr;
		}
	}
}

extern "C" CUDA_API void free_gpu_weights(GpuWeights* gpu) {
	if (!gpu)
		return;
	if (gpu->d_weights) {
		cudaFree(gpu->d_weights);
		gpu->d_weights = nullptr;
	}
	if (gpu->d_bias) {
		cudaFree(gpu->d_bias);
		gpu->d_bias = nullptr;
	}
	if (gpu->d_weights_master) {
		cudaFree(gpu->d_weights_master);
		gpu->d_weights_master = nullptr;
	}
	if (gpu->d_bias_master) {
		cudaFree(gpu->d_bias_master);
		gpu->d_bias_master = nullptr;
	}
	if (gpu->d_weights_m) {
		cudaFree(gpu->d_weights_m);
		gpu->d_weights_m = nullptr;
	}
	if (gpu->d_weights_v) {
		cudaFree(gpu->d_weights_v);
		gpu->d_weights_v = nullptr;
	}
	if (gpu->d_bias_m) {
		cudaFree(gpu->d_bias_m);
		gpu->d_bias_m = nullptr;
	}
	if (gpu->d_bias_v) {
		cudaFree(gpu->d_bias_v);
		gpu->d_bias_v = nullptr;
	}
	gpu->weight_elements = 0;
	gpu->bias_elements = 0;
	gpu->step_count = 0;
}

extern "C" CUDA_API bool allocate_gpu_embeddings(GpuEmbeddings* gpu_emb,
						 size_t vocab_size,
						 size_t embedding_dim) {
	if (!gpu_emb || vocab_size == 0 || embedding_dim == 0)
		return false;

	gpu_emb->vocab_size = vocab_size;
	gpu_emb->embedding_dim = embedding_dim;
	size_t total_elements = vocab_size * embedding_dim;

	// Free any previous allocations
	free_gpu_embeddings(gpu_emb);

	// 1. Allocate bf16 device table (fast Tensor Core / gather reads)
	if (!CUDA_CHECK(cudaMalloc(&gpu_emb->d_table, total_elements * sizeof(__nv_bfloat16))))
		return false;

	// 2. Allocate FP32 master table (high-precision accumulation)
	if (!CUDA_CHECK(cudaMalloc(&gpu_emb->d_table_master, total_elements * sizeof(float))))
		return false;

	// 3. Allocate AdamW moment buffers
	if (!CUDA_CHECK(cudaMalloc(&gpu_emb->d_m, total_elements * sizeof(float))))
		return false;
	if (!CUDA_CHECK(cudaMalloc(&gpu_emb->d_v, total_elements * sizeof(float))))
		return false;
	cudaMemset(gpu_emb->d_m, 0, total_elements * sizeof(float));
	cudaMemset(gpu_emb->d_v, 0, total_elements * sizeof(float));

	// 4. Chaos initialization in [-1/sqrt(D), +1/sqrt(D)]
	float bound = 1.0f / sqrtf(static_cast<float>(embedding_dim));
	float* h_chaos = static_cast<float*>(malloc(total_elements * sizeof(float)));
	if (!h_chaos)
		return false;

	std::mt19937 gen(1337);
	std::uniform_real_distribution<float> dis(-bound, bound);
	for (size_t i = 0; i < total_elements; ++i) {
		h_chaos[i] = dis(gen);
	}

	cudaMemcpy(gpu_emb->d_table_master, h_chaos, total_elements * sizeof(float), cudaMemcpyHostToDevice);
	free(h_chaos);

	int threads = 256;
	int blocks = static_cast<int>((total_elements + threads - 1) / threads);
	float_to_bf16_array_kernel<<<blocks, threads>>>(
		static_cast<const float*>(gpu_emb->d_table_master),
		static_cast<__nv_bfloat16*>(gpu_emb->d_table),
		total_elements);
	cudaDeviceSynchronize();

	return true;
}

extern "C" CUDA_API void free_gpu_embeddings(GpuEmbeddings* gpu_emb) {
	if (!gpu_emb)
		return;
	if (gpu_emb->d_table) {
		cudaFree(gpu_emb->d_table);
		gpu_emb->d_table = nullptr;
	}
	if (gpu_emb->d_table_master) {
		cudaFree(gpu_emb->d_table_master);
		gpu_emb->d_table_master = nullptr;
	}
	if (gpu_emb->d_m) {
		cudaFree(gpu_emb->d_m);
		gpu_emb->d_m = nullptr;
	}
	if (gpu_emb->d_v) {
		cudaFree(gpu_emb->d_v);
		gpu_emb->d_v = nullptr;
	}
}

extern "C" CUDA_API bool upload_embeddings(const float* h_embeddings, GpuEmbeddings* gpu_emb) {
	if (!h_embeddings || !gpu_emb || !gpu_emb->d_table_master || !gpu_emb->d_table)
		return false;

	size_t total_elements = gpu_emb->vocab_size * gpu_emb->embedding_dim;
	if (!CUDA_CHECK(cudaMemcpy(gpu_emb->d_table_master,
				   h_embeddings,
				   total_elements * sizeof(float),
				   cudaMemcpyHostToDevice))) {
		return false;
	}

	int threads = 256;
	int blocks = static_cast<int>((total_elements + threads - 1) / threads);
	float_to_bf16_array_kernel<<<blocks, threads>>>(
		static_cast<const float*>(gpu_emb->d_table_master),
		static_cast<__nv_bfloat16*>(gpu_emb->d_table),
		total_elements);
	return CUDA_CHECK(cudaGetLastError());
}

extern "C" CUDA_API bool download_embeddings(float* h_embeddings, const GpuEmbeddings* gpu_emb) {
	if (!h_embeddings || !gpu_emb || !gpu_emb->d_table_master)
		return false;

	size_t total_elements = gpu_emb->vocab_size * gpu_emb->embedding_dim;
	return CUDA_CHECK(cudaMemcpy(h_embeddings,
				     gpu_emb->d_table_master,
				     total_elements * sizeof(float),
				     cudaMemcpyDeviceToHost));
}

extern "C" CUDA_API void free_gpu_activations(GpuActivations* gpu) {
	if (!gpu)
		return;
	if (gpu->d_output) {
		cudaFree(gpu->d_output);
		gpu->d_output = nullptr;
	}
	if (gpu->d_loss) {
		cudaFree(gpu->d_loss);
		gpu->d_loss = nullptr;
	}
	gpu->output_elements = 0;
}

extern "C" CUDA_API void* pinned_alloc(size_t bytes) {
	void* ptr = nullptr;
	cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocWriteCombined);
	if (err != cudaSuccess) {
		fprintf(stderr,
			"[CUDA] pinned_alloc failed (%zu bytes): %s\n",
			bytes,
			cudaGetErrorString(err));
		return nullptr;
	}
	return ptr;
}

extern "C" CUDA_API void pinned_free(void* ptr) {
	if (ptr)
		cudaFreeHost(ptr);
}
