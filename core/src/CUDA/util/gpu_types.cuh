#ifndef GPU_TYPES_CUH
#define GPU_TYPES_CUH

#include <cstddef>

#include "gpu_kernel.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// GPU-side mirror of the context window and target that live in Portion.
// All pointers are device memory — do NOT dereference on the CPU.
// Uses void* to hide __nv_bfloat16 from the CPU compiler.
struct GpuPortion {
	void* d_context_window[2] = {nullptr, nullptr}; // device ptr: X  (128 tokens * 4 dim)
	void* d_target[2] = {nullptr, nullptr};		// device ptr: Y  (1  token  * 4 dim)

	size_t context_ptr = sizeof(d_context_window) / sizeof(d_context_window[0]);
	size_t target_ptr = sizeof(d_target) / sizeof(d_target[0]);

	void* stream[2] = {nullptr, nullptr};

	size_t context_elements = 0; // total bfloat16 elements in X
	size_t target_elements = 0;  // total bfloat16 elements in Y
};

// GPU-side mirror of the AI weights and biases.
struct GpuWeights {
	void* d_weights = nullptr; // device ptr to __nv_bfloat16
	void* d_bias = nullptr;	   // device ptr to __nv_bfloat16
	size_t weight_elements = 0;
	size_t bias_elements = 0;
};

// GPU-side activations buffer
struct GpuActivations {
	void* d_output = nullptr; // device ptr to __nv_bfloat16
	size_t output_elements = 0;
};

#ifdef BUILDING_CUDA_DLL
#define CUDA_API __declspec(dllexport)
#else
#define CUDA_API __declspec(dllimport)
#endif

#ifdef __cplusplus
}
#endif

#endif // GPU_TYPES_CUH
