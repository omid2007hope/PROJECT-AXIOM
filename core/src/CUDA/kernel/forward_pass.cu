#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../util/cuda_utils.cuh"
#include "forward_pass.cuh"

// The CUTLASS visitor headers are not self-contained: gemm_universal.h must come
// first, so this order must not be re-sorted.
// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
// clang-format on

// =============================================================================
// Forward pass:  y = W x + b  as ONE fused CUTLASS kernel on the Tensor Cores
//
// The output layer is deliberately linear: the regression targets are signed
// (sin() in [-1, 1]), and a ReLU can neither output a negative value nor pass a
// gradient once it is at 0, so it permanently kills every neuron it clamps.
//
// The layer is a GEMM   Y[M x N] = X[M x K] * W^T[K x N]
//     M = 1         (one context window per step)
//     N = out_dim   (one output per neuron)
//     K = in_dim    (window elements)
//
// The multiply runs on the Tensor Cores (bf16 inputs, FP32 accumulation). The
// "epilogue" that normally needs separate kernels, each one writing to VRAM and
// reading it back, is an Epilogue Visitor Tree (EVT) executed on the accumulators
// while they are still in registers:
//
//              Store(bf16)
//                  |
//          Add( Accum , Bias )
//           /               \
//   VisitorAccFetch    VisitorRowBroadcast
// =============================================================================
namespace {

using ElementIn = cutlass::bfloat16_t; // layout-compatible with __nv_bfloat16
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutX = cutlass::layout::RowMajor;    // x: K contiguous
using LayoutW = cutlass::layout::ColumnMajor; // W is stored [out][in] == W^T column-major
using LayoutY = cutlass::layout::RowMajor;

constexpr int AlignmentIn = 8;	// 128-bit vector loads of bf16 (in_dim must be a multiple of 8)
constexpr int AlignmentOut = 1; // out_dim is tiny, so it is not forced to a multiple of 8

using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>; // bf16 mma.sync m16n8k16
constexpr int NumStages = 4;
constexpr int EpilogueStages = 1;

using OutputTileThreadMap = cutlass::epilogue::threadblock::
	OutputTileThreadLayout<ThreadblockShape, WarpShape, ElementIn, AlignmentOut, EpilogueStages>;

using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

// One bias value per output column, repeated for every row (stride 0 along M)
using Bias = cutlass::epilogue::threadblock::
	VisitorRowBroadcast<OutputTileThreadMap, ElementIn, cute::Stride<cute::_0, cute::_1, int32_t>>;

using AddBias = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus,
							       ElementCompute,
							       ElementCompute,
							       cutlass::FloatRoundStyle::round_to_nearest>;

using EvtAddBias = cutlass::epilogue::threadblock::Sm80EVT<AddBias, Accum, Bias>;

using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
	OutputTileThreadMap,
	ElementIn,
	cutlass::FloatRoundStyle::round_to_nearest,
	cute::Stride<int64_t, cute::_1, int64_t>>;

using EvtStore = cutlass::epilogue::threadblock::Sm80EVT<Store, EvtAddBias>;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
	ElementIn,
	LayoutX,
	cutlass::ComplexTransform::kNone,
	AlignmentIn,
	ElementIn,
	LayoutW,
	cutlass::ComplexTransform::kNone,
	AlignmentIn,
	ElementIn,
	LayoutY,
	AlignmentOut,
	ElementAccumulator,
	ElementCompute,
	cutlass::arch::OpClassTensorOp,
	cutlass::arch::Sm80, // Ada (sm_89) runs the Ampere Tensor Core kernels
	ThreadblockShape,
	WarpShape,
	InstructionShape,
	EvtStore,
	cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
	NumStages,
	cutlass::arch::OpMultiplyAdd,
	EpilogueStages>::GemmKernel;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

} // namespace

__global__ static void gather_embeddings_kernel(const uint32_t* __restrict__ token_ids,
						const __nv_bfloat16* __restrict__ table,
						__nv_bfloat16* __restrict__ context_window,
						size_t num_tokens,
						size_t embedding_dim,
						size_t vocab_size) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t total_elements = num_tokens * embedding_dim;
	if (idx < total_elements) {
		size_t token_idx = idx / embedding_dim;
		size_t dim_idx = idx % embedding_dim;
		uint32_t tok_id = token_ids[token_idx];
		if (tok_id == 0) {
			// PAD must contribute exactly 0.0 to X * W
			context_window[idx] = __float2bfloat16(0.0f);
			return;
		}
		if (tok_id >= vocab_size) {
			tok_id = 1; // UNK fallback
		}
		context_window[idx] = table[tok_id * embedding_dim + dim_idx];
	}
}

// Milestone 3: RMSNorm Forward Kernel (in-place normalization before Tensor Cores)
__global__ static void rms_norm_forward_kernel(__nv_bfloat16* __restrict__ x,
					       float* __restrict__ inv_rms_out,
					       size_t n,
					       float eps) {
	__shared__ float s_sum[32];
	float sum_sq = 0.0f;

	for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
		float val = __bfloat162float(x[i]);
		if (isfinite(val)) {
			sum_sq += val * val;
		}
	}

	for (int offset = 16; offset > 0; offset /= 2) {
		sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
	}

	int lane = threadIdx.x % 32;
	int wid = threadIdx.x / 32;
	if (lane == 0) {
		s_sum[wid] = sum_sq;
	}
	__syncthreads();

	if (wid == 0) {
		float warp_sum = (lane < (blockDim.x / 32)) ? s_sum[lane] : 0.0f;
		for (int offset = 16; offset > 0; offset /= 2) {
			warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
		}
		if (lane == 0) {
			float mean_sq = warp_sum / static_cast<float>(n > 0 ? n : 1);
			float inv_rms = rsqrtf(mean_sq + eps);
			if (!isfinite(inv_rms) || isinf(inv_rms)) {
				inv_rms = 1.0f;
			}
			s_sum[0] = inv_rms;
			if (inv_rms_out) {
				*inv_rms_out = inv_rms;
			}
		}
	}
	__syncthreads();

	float inv_rms = s_sum[0];
	for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
		float val = __bfloat162float(x[i]);
		float norm_val = val * inv_rms;
		norm_val = fmaxf(-10.0f, fminf(10.0f, norm_val));
		x[i] = __float2bfloat16(norm_val);
	}
}

// Milestone 3: Tied Logit Projection Kernel (Y * E^T)
__global__ static void compute_tied_logits_kernel(const __nv_bfloat16* __restrict__ y,
						  const __nv_bfloat16* __restrict__ table,
						  float* __restrict__ logits,
						  size_t vocab_size,
						  size_t emb_dim) {
	size_t v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
	if (v < vocab_size) {
		float dot = 0.0f;
		const __nv_bfloat16* emb_row = table + v * emb_dim;
		for (size_t d = 0; d < emb_dim; ++d) {
			dot += __bfloat162float(y[d]) * __bfloat162float(emb_row[d]);
		}
		logits[v] = (v == 0) ? -1e30f : dot; // PAD is never a prediction
	}
}

__global__ static void copy_bf16_to_float_kernel(const __nv_bfloat16* __restrict__ src,
						 float* __restrict__ dst,
						 size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		dst[i] = __bfloat162float(src[i]);
	}
}

// Milestone 3: Fused Softmax & Argmax Kernel on GPU
__global__ static void softmax_argmax_kernel(float* __restrict__ logits_and_probs,
					     uint32_t* __restrict__ pred_id,
					     size_t n) {
	__shared__ float s_max[32];
	__shared__ uint32_t s_arg[32];
	__shared__ float s_sum[32];

	float local_max = -1e30f;
	uint32_t local_arg = 0;

	for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
		float val = logits_and_probs[i];
		if (isfinite(val) && val > local_max) {
			local_max = val;
			local_arg = static_cast<uint32_t>(i);
		}
	}

	for (int offset = 16; offset > 0; offset /= 2) {
		float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
		uint32_t other_arg = __shfl_down_sync(0xffffffff, local_arg, offset);
		if (other_max > local_max) {
			local_max = other_max;
			local_arg = other_arg;
		}
	}

	int lane = threadIdx.x % 32;
	int wid = threadIdx.x / 32;
	if (lane == 0) {
		s_max[wid] = local_max;
		s_arg[wid] = local_arg;
	}
	__syncthreads();

	if (wid == 0) {
		float warp_max = (lane < (blockDim.x / 32)) ? s_max[lane] : -1e30f;
		uint32_t warp_arg = (lane < (blockDim.x / 32)) ? s_arg[lane] : 0;
		for (int offset = 16; offset > 0; offset /= 2) {
			float other_max = __shfl_down_sync(0xffffffff, warp_max, offset);
			uint32_t other_arg = __shfl_down_sync(0xffffffff, warp_arg, offset);
			if (other_max > warp_max) {
				warp_max = other_max;
				warp_arg = other_arg;
			}
		}
		if (lane == 0) {
			s_max[0] = warp_max;
			s_arg[0] = warp_arg;
			if (pred_id) {
				*pred_id = warp_arg;
			}
		}
	}
	__syncthreads();

	float global_max = s_max[0];

	float sum_exp = 0.0f;
	for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
		float exp_val = expf(fmaxf(-50.0f, fminf(50.0f, logits_and_probs[i] - global_max)));
		logits_and_probs[i] = exp_val;
		sum_exp += exp_val;
	}

	for (int offset = 16; offset > 0; offset /= 2) {
		sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
	}
	if (lane == 0) {
		s_sum[wid] = sum_exp;
	}
	__syncthreads();

	if (wid == 0) {
		float warp_sum = (lane < (blockDim.x / 32)) ? s_sum[lane] : 0.0f;
		for (int offset = 16; offset > 0; offset /= 2) {
			warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
		}
		if (lane == 0) {
			s_sum[0] = warp_sum > 1e-12f ? warp_sum : 1.0f;
		}
	}
	__syncthreads();

	float inv_sum = 1.0f / s_sum[0];
	for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
		logits_and_probs[i] *= inv_sum;
	}
}

extern "C" CUDA_API bool forward_pass_with_embeddings(const GpuPortion* gpu_portion,
						      const GpuEmbeddings* gpu_embeddings,
						      const GpuWeights* gpu_weights,
						      GpuActivations* gpu_activations,
						      size_t input_dim,
						      size_t output_dim,
						      int buffer_index) {

	// Safe Guard
	if (!gpu_portion || !gpu_weights || !gpu_activations ||
	    !gpu_weights->d_weights || !gpu_weights->d_bias || input_dim == 0 || output_dim == 0) {
		return false;
	}

	cudaStream_t stream = (cudaStream_t)gpu_portion->stream[buffer_index];

	// The two buffers share the weights and the output, so this step may only start
	// after the previous step (other buffer) finished updating them. Its upload is
	// already queued on this stream and still overlaps with that step.
	cudaEvent_t previous_step_done = (cudaEvent_t)gpu_portion->compute_done[1 - buffer_index];
	if (previous_step_done) {
		if (!CUDA_CHECK(cudaStreamWaitEvent(stream, previous_step_done, 0)))
			return false;
	}

	// Ensure d_context_window is allocated on GPU for the gather scratchpad or GEMM input
	GpuPortion* non_const_gpu = const_cast<GpuPortion*>(gpu_portion);
	for (size_t i = 0; i < non_const_gpu->context_ptr; ++i) {
		if (non_const_gpu->context_elements != input_dim || !non_const_gpu->d_context_window[i]) {
			if (non_const_gpu->d_context_window[i]) {
				CUDA_CHECK(cudaFree(non_const_gpu->d_context_window[i]));
				non_const_gpu->d_context_window[i] = nullptr;
			}
			if (!CUDA_CHECK(cudaMalloc(&non_const_gpu->d_context_window[i],
						   input_dim * sizeof(__nv_bfloat16))))
				return false;
		}
	}
	non_const_gpu->context_elements = input_dim;

	// Phase 4: Forward Gather Kernel & PCIe Bandwidth Optimization
	// If token IDs and embedding table are provided, branchlessly gather tokens into d_context_window on the GPU
	if (gpu_portion->token_ids_active[buffer_index] && gpu_embeddings && gpu_embeddings->d_table) {
		size_t emb_dim = gpu_embeddings->embedding_dim > 0 ? gpu_embeddings->embedding_dim : 32;
		size_t num_tok = gpu_portion->token_count > 0 ? gpu_portion->token_count : (input_dim / emb_dim);
		int threads = 256;
		int blocks = static_cast<int>((input_dim + threads - 1) / threads);

		gather_embeddings_kernel<<<blocks, threads, 0, stream>>>(
			static_cast<const uint32_t*>(gpu_portion->d_token_ids[buffer_index]),
			static_cast<const __nv_bfloat16*>(gpu_embeddings->d_table),
			static_cast<__nv_bfloat16*>(gpu_portion->d_context_window[buffer_index]),
			num_tok,
			emb_dim,
			gpu_embeddings->vocab_size);
		if (!CUDA_CHECK(cudaGetLastError()))
			return false;
	}

	if (!gpu_portion->d_context_window[buffer_index]) {
		return false;
	}

	// Milestone 3: RMSNorm Forward Pass (Normalizes variance of flattened input to 1.0)
	// Applied only when embeddings are active to protect large context windows from variance explosion
	if (gpu_embeddings && gpu_embeddings->d_table) {
		if (!gpu_activations->d_inv_rms) {
			if (!CUDA_CHECK(cudaMalloc(&gpu_activations->d_inv_rms, sizeof(float))))
				return false;
		}
		int norm_threads = 1024;
		rms_norm_forward_kernel<<<1, norm_threads, 0, stream>>>(
			static_cast<__nv_bfloat16*>(gpu_portion->d_context_window[buffer_index]),
			static_cast<float*>(gpu_activations->d_inv_rms),
			input_dim,
			1e-5f);
		if (!CUDA_CHECK(cudaGetLastError()))
			return false;
	} else if (gpu_activations->d_inv_rms) {
		float unity = 1.0f;
		cudaMemcpyAsync(gpu_activations->d_inv_rms, &unity, sizeof(float), cudaMemcpyHostToDevice, stream);
	}

	if (gpu_activations->output_elements != output_dim) {
		if (gpu_activations->d_output) {
			if (!CUDA_CHECK(cudaFree(gpu_activations->d_output)))
				return false;
			gpu_activations->d_output = nullptr;
		}

		if (!CUDA_CHECK(cudaMalloc(&gpu_activations->d_output,
					   output_dim * sizeof(__nv_bfloat16))))
			return false;
		gpu_activations->output_elements = output_dim;
	}

	// The loss accumulator is written by the backward pass; it starts at zero
	if (!gpu_activations->d_loss) {
		if (!CUDA_CHECK(cudaMalloc(&gpu_activations->d_loss, sizeof(float))))
			return false;
		if (!CUDA_CHECK(cudaMemset(gpu_activations->d_loss, 0, sizeof(float))))
			return false;
	}

	using cute::_0;
	using cute::_1;

	const int M = 1;
	const int N = static_cast<int>(output_dim);
	const int K = static_cast<int>(input_dim);

	// Visitor arguments are listed children first, then the node itself
	typename EvtStore::Arguments callback_args {
		{
			{}, // Accum
			{static_cast<const ElementIn*>(gpu_weights->d_bias),
			 ElementIn(0),
			 {_0 {}, _1 {}, int32_t(N)}}, // Bias
			{}			       // AddBias
		},				       // EvtAddBias
		{static_cast<ElementIn*>(gpu_activations->d_output),
		 {int64_t(N), _1 {}, int64_t(M) * N}} // Store
	};

	typename Gemm::Arguments arguments(
		cutlass::gemm::GemmUniversalMode::kGemm,
		cutlass::gemm::GemmCoord(M, N, K),
		1, // batch_count
		callback_args,
		gpu_portion->d_context_window[buffer_index], // ptr_A: x
		gpu_weights->d_weights,			     // ptr_B: W
		nullptr,				     // ptr_C (the EVT does the store)
		nullptr,				     // ptr_D
		int64_t(0),				     // batch_stride_A
		int64_t(0),				     // batch_stride_B
		int64_t(0),				     // batch_stride_C
		int64_t(0),				     // batch_stride_D
		int64_t(K),				     // lda
		int64_t(K),				     // ldb
		int64_t(0),				     // ldc
		int64_t(0));				     // ldd

	Gemm gemm_op;

	cutlass::Status status = gemm_op.can_implement(arguments);
	if (status != cutlass::Status::kSuccess) {
		fprintf(stderr,
			"[CUTLASS] forward_pass unsupported (in_dim=%zu must be a multiple of %d): %s\n",
			input_dim,
			AlignmentIn,
			cutlass::cutlassGetStatusString(status));
		return false;
	}

	// A single-slice GEMM needs no workspace
	if (Gemm::get_workspace_size(arguments) != 0) {
		fprintf(stderr, "[CUTLASS] forward_pass unexpectedly needs a workspace\n");
		return false;
	}

	status = gemm_op(arguments, nullptr, stream);
	if (status != cutlass::Status::kSuccess) {
		fprintf(stderr,
			"[CUTLASS] forward_pass failed: %s\n",
			cutlass::cutlassGetStatusString(status));
		return false;
	}

	// Milestone 3: Discrete Autoregressive Token Classification
	// Compute Softmax probabilities and Argmax winning token on the GPU
	size_t vocab_sz = (gpu_embeddings && gpu_embeddings->d_table)
				  ? (gpu_embeddings->active_vocab_size > 0 ? gpu_embeddings->active_vocab_size
									   : gpu_embeddings->vocab_size)
				  : output_dim;
	if (gpu_activations->vocab_elements != vocab_sz || !gpu_activations->d_probs) {
		if (gpu_activations->d_probs) {
			CUDA_CHECK(cudaFree(gpu_activations->d_probs));
			gpu_activations->d_probs = nullptr;
		}
		if (!CUDA_CHECK(cudaMalloc(&gpu_activations->d_probs, vocab_sz * sizeof(float))))
			return false;
		gpu_activations->vocab_elements = vocab_sz;
	}
	if (!gpu_activations->d_pred_id) {
		if (!CUDA_CHECK(cudaMalloc(&gpu_activations->d_pred_id, sizeof(uint32_t))))
			return false;
	}

	if (gpu_embeddings && gpu_embeddings->d_table) {
		size_t emb_dim = gpu_embeddings->embedding_dim > 0 ? gpu_embeddings->embedding_dim : output_dim;
		int l_threads = 256;
		int l_blocks = static_cast<int>((vocab_sz + l_threads - 1) / l_threads);
		compute_tied_logits_kernel<<<l_blocks, l_threads, 0, stream>>>(
			static_cast<const __nv_bfloat16*>(gpu_activations->d_output),
			static_cast<const __nv_bfloat16*>(gpu_embeddings->d_table),
			static_cast<float*>(gpu_activations->d_probs),
			vocab_sz,
			emb_dim);
	} else {
		int d_threads = 256;
		int d_blocks = static_cast<int>((output_dim + d_threads - 1) / d_threads);
		copy_bf16_to_float_kernel<<<d_blocks, d_threads, 0, stream>>>(
			static_cast<const __nv_bfloat16*>(gpu_activations->d_output),
			static_cast<float*>(gpu_activations->d_probs),
			output_dim);
	}

	softmax_argmax_kernel<<<1, 1024, 0, stream>>>(
		static_cast<float*>(gpu_activations->d_probs),
		static_cast<uint32_t*>(gpu_activations->d_pred_id),
		vocab_sz);

	return CUDA_CHECK(cudaGetLastError());
}

extern "C" CUDA_API bool forward_pass(const GpuPortion* gpu_portion,
				      const GpuWeights* gpu_weights,
				      GpuActivations* gpu_activations,
				      size_t input_dim,
				      size_t output_dim,
				      int buffer_index) {
	return forward_pass_with_embeddings(gpu_portion,
					    nullptr,
					    gpu_weights,
					    gpu_activations,
					    input_dim,
					    output_dim,
					    buffer_index);
}

extern "C" CUDA_API bool run_cutlass_saturation_benchmark(int M, int N, int K, int iterations,
							 double* out_tflops,
							 double* out_bandwidth_gb_s,
							 float* out_avg_ms) {
	if (M <= 0 || N <= 0 || K <= 0 || iterations <= 0) {
		return false;
	}

	if (K % AlignmentIn != 0) {
		return false;
	}

	using cute::_0;
	using cute::_1;

	size_t size_A = static_cast<size_t>(M) * K * sizeof(__nv_bfloat16);
	size_t size_B = static_cast<size_t>(K) * N * sizeof(__nv_bfloat16);
	size_t size_bias = static_cast<size_t>(N) * sizeof(__nv_bfloat16);
	size_t size_out = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);

	__nv_bfloat16* d_A = nullptr;
	__nv_bfloat16* d_B = nullptr;
	__nv_bfloat16* d_bias = nullptr;
	__nv_bfloat16* d_out = nullptr;

	if (cudaMalloc(&d_A, size_A) != cudaSuccess ||
	    cudaMalloc(&d_B, size_B) != cudaSuccess ||
	    cudaMalloc(&d_bias, size_bias) != cudaSuccess ||
	    cudaMalloc(&d_out, size_out) != cudaSuccess) {
		if (d_A) cudaFree(d_A);
		if (d_B) cudaFree(d_B);
		if (d_bias) cudaFree(d_bias);
		if (d_out) cudaFree(d_out);
		return false;
	}

	cudaMemset(d_A, 0x3c, size_A);
	cudaMemset(d_B, 0x3c, size_B);
	cudaMemset(d_bias, 0x00, size_bias);
	cudaMemset(d_out, 0x00, size_out);

	typename EvtStore::Arguments callback_args {
		{
			{}, // Accum
			{reinterpret_cast<const ElementIn*>(d_bias),
			 ElementIn(0),
			 {_0 {}, _1 {}, int32_t(N)}}, // Bias
			{}			       // AddBias
		},				       // EvtAddBias
		{reinterpret_cast<ElementIn*>(d_out),
		 {int64_t(N), _1 {}, int64_t(M) * N}} // Store
	};

	typename Gemm::Arguments arguments(
		cutlass::gemm::GemmUniversalMode::kGemm,
		cutlass::gemm::GemmCoord(M, N, K),
		1, // batch_count
		callback_args,
		reinterpret_cast<ElementIn*>(d_A),
		reinterpret_cast<ElementIn*>(d_B),
		nullptr,
		nullptr,
		int64_t(0),
		int64_t(0),
		int64_t(0),
		int64_t(0),
		int64_t(K),
		int64_t(K),
		int64_t(0),
		int64_t(0));

	Gemm gemm_op;

	cutlass::Status status = gemm_op.can_implement(arguments);
	if (status != cutlass::Status::kSuccess) {
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_bias);
		cudaFree(d_out);
		return false;
	}

	size_t workspace_size = Gemm::get_workspace_size(arguments);
	void* d_workspace = nullptr;
	if (workspace_size > 0) {
		if (cudaMalloc(&d_workspace, workspace_size) != cudaSuccess) {
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_bias);
			cudaFree(d_out);
			return false;
		}
	}

	cudaStream_t stream = nullptr;

	// Warm-up passes (locks Ada Lovelace Tensor Cores into P0 boost clock)
	for (int i = 0; i < 30; ++i) {
		gemm_op(arguments, d_workspace, stream);
	}
	cudaStreamSynchronize(stream);

	cudaEvent_t start = nullptr, stop = nullptr;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, stream);
	for (int i = 0; i < iterations; ++i) {
		gemm_op(arguments, d_workspace, stream);
	}
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float total_ms = 0.0f;
	cudaEventElapsedTime(&total_ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (d_workspace) cudaFree(d_workspace);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_bias);
	cudaFree(d_out);

	if (total_ms <= 0.0f) {
		return false;
	}

	float avg_ms = total_ms / static_cast<float>(iterations);
	if (out_avg_ms) *out_avg_ms = avg_ms;

	double flops_per_iter = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
	double total_flops = flops_per_iter * static_cast<double>(iterations);
	double achieved_tflops = (total_flops / 1e12) / (total_ms / 1000.0);
	if (out_tflops) *out_tflops = achieved_tflops;

	double bytes_per_iter = static_cast<double>(size_A + size_B + size_bias + size_out);
	double total_bytes = bytes_per_iter * static_cast<double>(iterations);
	double achieved_bw_gb_s = (total_bytes / 1e9) / (total_ms / 1000.0);
	if (out_bandwidth_gb_s) *out_bandwidth_gb_s = achieved_bw_gb_s;

	return true;
}

__global__ void vram_saturation_kernel(const uint4* __restrict__ src,
				       uint4* __restrict__ dst,
				       size_t num_elements) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (size_t i = idx; i < num_elements; i += stride) {
		dst[i] = src[i];
	}
}

extern "C" CUDA_API bool run_vram_saturation_benchmark(size_t safety_headroom_bytes,
						       size_t* out_allocated_bytes,
						       double* out_bandwidth_gb_s,
						       float* out_elapsed_ms) {
	size_t free_vram = 0, total_vram = 0;
	if (cudaMemGetInfo(&free_vram, &total_vram) != cudaSuccess) {
		return false;
	}

	if (free_vram <= safety_headroom_bytes) {
		return false;
	}

	size_t safe_budget = free_vram - safety_headroom_bytes;
	// Allocate 2 buffers: src and dst, 16-byte aligned
	size_t buf_size = (safe_budget / 2 / 16) * 16;
	constexpr size_t MAX_BUF_PER_SIDE = 3ULL * 1024 * 1024 * 1024; // 3 GB each = 6 GB total max
	if (buf_size > MAX_BUF_PER_SIDE) {
		buf_size = MAX_BUF_PER_SIDE;
	}

	uint4* d_src = nullptr;
	uint4* d_dst = nullptr;

	if (cudaMalloc(&d_src, buf_size) != cudaSuccess ||
	    cudaMalloc(&d_dst, buf_size) != cudaSuccess) {
		if (d_src) cudaFree(d_src);
		if (d_dst) cudaFree(d_dst);
		return false;
	}

	cudaMemset(d_src, 0x3c, buf_size);
	cudaMemset(d_dst, 0x00, buf_size);

	size_t num_elements = buf_size / sizeof(uint4);
	int threads = 256;
	int blocks = 58 * 8; // Saturate all 58 Ada SMs

	// Warmup
	vram_saturation_kernel<<<blocks, threads>>>(d_src, d_dst, num_elements);
	cudaDeviceSynchronize();

	cudaEvent_t start = nullptr, stop = nullptr;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	constexpr int passes = 15;
	cudaEventRecord(start);
	for (int p = 0; p < passes; ++p) {
		vram_saturation_kernel<<<blocks, threads>>>(d_src, d_dst, num_elements);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float total_ms = 0.0f;
	cudaEventElapsedTime(&total_ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_src);
	cudaFree(d_dst);

	if (total_ms <= 0.0f) {
		return false;
	}

	if (out_allocated_bytes) *out_allocated_bytes = buf_size * 2;
	if (out_elapsed_ms) *out_elapsed_ms = total_ms;

	double total_bytes_transferred = 2.0 * static_cast<double>(buf_size) * static_cast<double>(passes);
	double achieved_bw = (total_bytes_transferred / 1e9) / (total_ms / 1000.0);
	if (out_bandwidth_gb_s) *out_bandwidth_gb_s = achieved_bw;

	return true;
}


