#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../util/cuda_utils.cuh"
#include "backward_pass.cuh"

// =============================================================================
// =============================================================================
// Milestone 3: Discrete Autoregressive Token Classification Backward Pass
//
// 1. Cross-Entropy Loss: L = -log(P_{target})
// 2. Exact bounded logit gradient: dL/dz_i = P_i - 1[i == target]  in [-1.0, +1.0]
// 3. RMSNorm backpropagation scaling: dx_j = inv_rms * (dL/d_x_hat_j)
// =============================================================================

__global__ static void compute_loss_and_dz_kernel(
	const float* __restrict__ probs,
	const __nv_bfloat16* __restrict__ y,
	const __nv_bfloat16* __restrict__ target,
	const __nv_bfloat16* __restrict__ emb_table,
	float* __restrict__ dz,
	float* __restrict__ loss,
	size_t out_dim,
	size_t vocab_size,
	size_t emb_dim,
	bool is_token_training) {

	size_t row = blockIdx.x * (size_t)blockDim.x + threadIdx.x;

	if (is_token_training && probs != nullptr) {
		uint32_t target_tok_id = *reinterpret_cast<const uint32_t*>(target);
		if (target_tok_id >= vocab_size) {
			target_tok_id = 0;
		}

		if (row == 0) {
			float p_target = probs[target_tok_id];
			float ce_loss = -logf(fmaxf(p_target, 1e-12f));
			if (isfinite(ce_loss)) {
				atomicAdd(loss, ce_loss);
			}
		}

		if (row < out_dim) {
			float dz_val = 0.0f;
			if (emb_table != nullptr && emb_dim == out_dim) {
				for (size_t v = 0; v < vocab_size; ++v) {
					float p = probs[v];
					if (p > 1e-7f || v == target_tok_id) {
						float grad_logit = p - (v == target_tok_id ? 1.0f : 0.0f);
						float e_val = __bfloat162float(emb_table[v * emb_dim + row]);
						dz_val += grad_logit * e_val;
					}
				}
			} else {
				if (row < vocab_size) {
					dz_val = probs[row] - (row == target_tok_id ? 1.0f : 0.0f);
				}
			}

			dz_val = fmaxf(-1.0f, fminf(1.0f, dz_val));
			dz[row] = dz_val;
		}
	} else {
		if (row < out_dim) {
			float y_val = __bfloat162float(y[row]);
			float t_val = __bfloat162float(target[row]);
			float dy = y_val - t_val;
			if (isnan(dy) || isinf(dy)) {
				dy = 0.0f;
			}
			float dz_val = fmaxf(-1.0f, fminf(1.0f, dy));
			dz[row] = dz_val;

			if (row == 0) {
				float mse_loss = 0.0f;
				for (size_t r = 0; r < out_dim; ++r) {
					float y_r = __bfloat162float(y[r]);
					float t_r = __bfloat162float(target[r]);
					float d = y_r - t_r;
					if (isfinite(d)) {
						mse_loss += 0.5f * d * d;
					}
				}
				atomicAdd(loss, mse_loss);
			}
		}
	}
}

__global__ static void backward_pass_kernel(float* __restrict__ W_master,
					    __nv_bfloat16* __restrict__ W,
					    float* __restrict__ b_master,
					    __nv_bfloat16* __restrict__ b,
					    const __nv_bfloat16* __restrict__ x,
					    const float* __restrict__ dz_vector,
					    size_t in_dim,
					    size_t out_dim,
					    float learning_rate) {

	size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;

	if (i < out_dim * in_dim) {
		size_t row = i / in_dim;
		size_t col = i % in_dim;

		float dz = dz_vector[row];

		float x_val = __bfloat162float(x[col]);
		if (isnan(x_val) || isinf(x_val)) {
			x_val = 0.0f;
		}

		float w_val = W_master[i] - learning_rate * dz * x_val;
		if (isfinite(w_val)) {
			w_val = fmaxf(-10.0f, fminf(10.0f, w_val));
			W_master[i] = w_val;
			W[i] = __float2bfloat16(w_val);
		}

		if (col == 0) {
			float b_val = b_master[row] - learning_rate * dz;
			if (isfinite(b_val)) {
				b_val = fmaxf(-10.0f, fminf(10.0f, b_val));
				b_master[row] = b_val;
				b[row] = __float2bfloat16(b_val);
			}
		}
	}
}

// Phase 6.2: Fused AdamW Kernel in CUDA
__global__ static void backward_pass_adamw_kernel(float* __restrict__ W_master,
						  __nv_bfloat16* __restrict__ W,
						  float* __restrict__ W_m,
						  float* __restrict__ W_v,
						  float* __restrict__ b_master,
						  __nv_bfloat16* __restrict__ b,
						  float* __restrict__ b_m,
						  float* __restrict__ b_v,
						  const __nv_bfloat16* __restrict__ x,
						  const float* __restrict__ dz_vector,
						  size_t in_dim,
						  size_t out_dim,
						  float learning_rate,
						  uint64_t step,
						  float beta1,
						  float beta2,
						  float eps,
						  float weight_decay) {

	size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;

	if (i < out_dim * in_dim) {
		size_t row = i / in_dim;
		size_t col = i % in_dim;

		float dz = dz_vector[row];

		float x_val = __bfloat162float(x[col]);
		if (isnan(x_val) || isinf(x_val)) {
			x_val = 0.0f;
		}

		float g = dz * x_val;
		float m = beta1 * W_m[i] + (1.0f - beta1) * g;
		float v = beta2 * W_v[i] + (1.0f - beta2) * g * g;
		W_m[i] = m;
		W_v[i] = v;

		float step_f = static_cast<float>(step);
		float bc1 = 1.0f - powf(beta1, step_f);
		float bc2 = 1.0f - powf(beta2, step_f);
		float m_hat = m / (bc1 > 1e-7f ? bc1 : 1.0f);
		float v_hat = v / (bc2 > 1e-7f ? bc2 : 1.0f);

		float update = m_hat / (sqrtf(v_hat) + eps) + weight_decay * W_master[i];
		float w_val = W_master[i] - learning_rate * update;

		if (isfinite(w_val)) {
			w_val = fmaxf(-10.0f, fminf(10.0f, w_val));
			W_master[i] = w_val;
			W[i] = __float2bfloat16(w_val);
		}

		if (col == 0) {
			float bg = dz;
			float bm = beta1 * b_m[row] + (1.0f - beta1) * bg;
			float bv = beta2 * b_v[row] + (1.0f - beta2) * bg * bg;
			b_m[row] = bm;
			b_v[row] = bv;

			float bm_hat = bm / (bc1 > 1e-7f ? bc1 : 1.0f);
			float bv_hat = bv / (bc2 > 1e-7f ? bc2 : 1.0f);
			float b_update = bm_hat / (sqrtf(bv_hat) + eps);
			float b_val = b_master[row] - learning_rate * b_update;

			if (isfinite(b_val)) {
				b_val = fmaxf(-10.0f, fminf(10.0f, b_val));
				b_master[row] = b_val;
				b[row] = __float2bfloat16(b_val);
			}
		}
	}
}

// Phase 5: Backward Scatter Kernel & Input Gradient Calculation
__global__ static void backward_scatter_embeddings_kernel(
	const uint32_t* __restrict__ token_ids,
	float* __restrict__ table_master,
	__nv_bfloat16* __restrict__ table,
	const float* __restrict__ W_master,
	const float* __restrict__ dz_vector,
	const float* __restrict__ inv_rms_ptr,
	size_t num_tokens,
	size_t embedding_dim,
	size_t vocab_size,
	size_t out_dim,
	float learning_rate) {

	size_t in_dim = num_tokens * embedding_dim;
	size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;

	if (j < in_dim) {
		float dx_j = 0.0f;
		for (size_t row = 0; row < out_dim; ++row) {
			float dz = dz_vector[row];
			float w = W_master[row * in_dim + j];
			dx_j += dz * w;
		}

		// Milestone 3: Scale gradient through RMSNorm
		float inv_rms = (inv_rms_ptr && isfinite(*inv_rms_ptr)) ? *inv_rms_ptr : 1.0f;
		dx_j *= inv_rms;
		dx_j = fmaxf(-1.0f, fminf(1.0f, dx_j));

		// Accumulate gradient into FP32 master table with atomicAdd
		size_t token_idx = j / embedding_dim;
		size_t dim_idx = j % embedding_dim;
		uint32_t tok_id = token_ids[token_idx];

		// PAD (ID 0) is a fixed zero vector and never learns
		if (tok_id != 0 && tok_id < vocab_size) {
			size_t table_idx = tok_id * embedding_dim + dim_idx;
			atomicAdd(&table_master[table_idx], -learning_rate * dx_j);

			float new_val = table_master[table_idx];
			if (isfinite(new_val)) {
				new_val = fmaxf(-5.0f, fminf(5.0f, new_val));
				table_master[table_idx] = new_val;
				table[table_idx] = __float2bfloat16(new_val);
			}
		}
	}
}

extern "C" CUDA_API bool backward_pass_with_embeddings(const GpuPortion* gpu_portion,
						      GpuEmbeddings* gpu_embeddings,
						      GpuWeights* gpu_weights,
						      const GpuActivations* gpu_activations,
						      size_t input_dim,
						      size_t output_dim,
						      float learning_rate,
						      int buffer_index,
						      bool use_adamw) {

	// Safe Guard
	if (!gpu_portion || !gpu_weights || !gpu_activations ||
	    !gpu_portion->d_context_window[buffer_index] || !gpu_portion->d_target[buffer_index] ||
	    !gpu_weights->d_weights || !gpu_weights->d_bias || !gpu_weights->d_weights_master ||
	    !gpu_weights->d_bias_master || !gpu_activations->d_output || !gpu_activations->d_loss ||
	    input_dim == 0 || output_dim == 0) {
		return false;
	}

	cudaStream_t stream = (cudaStream_t)gpu_portion->stream[buffer_index];
	gpu_weights->step_count += 1;

	// Lazy allocation of AdamW momentum & variance buffers if requested
	if (use_adamw && (!gpu_weights->d_weights_m || !gpu_weights->d_weights_v)) {
		size_t w_elems = input_dim * output_dim;
		size_t b_elems = output_dim;
		if (CUDA_CHECK(cudaMalloc(&gpu_weights->d_weights_m, w_elems * sizeof(float))) &&
		    CUDA_CHECK(cudaMalloc(&gpu_weights->d_weights_v, w_elems * sizeof(float))) &&
		    CUDA_CHECK(cudaMalloc(&gpu_weights->d_bias_m, b_elems * sizeof(float))) &&
		    CUDA_CHECK(cudaMalloc(&gpu_weights->d_bias_v, b_elems * sizeof(float)))) {
			cudaMemsetAsync(gpu_weights->d_weights_m, 0, w_elems * sizeof(float), stream);
			cudaMemsetAsync(gpu_weights->d_weights_v, 0, w_elems * sizeof(float), stream);
			cudaMemsetAsync(gpu_weights->d_bias_m, 0, b_elems * sizeof(float), stream);
			cudaMemsetAsync(gpu_weights->d_bias_v, 0, b_elems * sizeof(float), stream);
		} else {
			use_adamw = false;
		}
	}

	// Milestone 3: Lazy allocation of dz gradient buffer
	GpuActivations* non_const_act = const_cast<GpuActivations*>(gpu_activations);
	if (!non_const_act->d_dz) {
		if (!CUDA_CHECK(cudaMalloc(&non_const_act->d_dz, output_dim * sizeof(float))))
			return false;
	}

	bool is_token_training = gpu_portion->token_ids_active[buffer_index];
	size_t vocab_sz = (gpu_embeddings && gpu_embeddings->d_table)
				  ? (gpu_embeddings->active_vocab_size > 0 ? gpu_embeddings->active_vocab_size
									   : gpu_embeddings->vocab_size)
				  : output_dim;
	size_t emb_dim = (gpu_embeddings && gpu_embeddings->d_table) ? gpu_embeddings->embedding_dim : output_dim;
	const __nv_bfloat16* emb_tbl = (gpu_embeddings && gpu_embeddings->d_table)
					       ? static_cast<const __nv_bfloat16*>(gpu_embeddings->d_table)
					       : nullptr;

	int dz_threads = 256;
	int dz_blocks = static_cast<int>((output_dim + dz_threads - 1) / dz_threads);
	compute_loss_and_dz_kernel<<<dz_blocks, dz_threads, 0, stream>>>(
		static_cast<const float*>(gpu_activations->d_probs),
		static_cast<const __nv_bfloat16*>(gpu_activations->d_output),
		static_cast<const __nv_bfloat16*>(gpu_portion->d_target[buffer_index]),
		emb_tbl,
		static_cast<float*>(non_const_act->d_dz),
		static_cast<float*>(non_const_act->d_loss),
		output_dim,
		vocab_sz,
		emb_dim,
		is_token_training);

	if (!CUDA_CHECK(cudaGetLastError()))
		return false;

	Grid grid(input_dim * output_dim);

	if (use_adamw && gpu_weights->d_weights_m && gpu_weights->d_weights_v) {
		constexpr float beta1 = 0.9f;
		constexpr float beta2 = 0.999f;
		constexpr float eps = 1e-8f;
		constexpr float weight_decay = 0.01f;

		backward_pass_adamw_kernel<<<grid.block, grid.thread, grid.index, stream>>>(
			static_cast<float*>(gpu_weights->d_weights_master),
			static_cast<__nv_bfloat16*>(gpu_weights->d_weights),
			static_cast<float*>(gpu_weights->d_weights_m),
			static_cast<float*>(gpu_weights->d_weights_v),
			static_cast<float*>(gpu_weights->d_bias_master),
			static_cast<__nv_bfloat16*>(gpu_weights->d_bias),
			static_cast<float*>(gpu_weights->d_bias_m),
			static_cast<float*>(gpu_weights->d_bias_v),
			static_cast<const __nv_bfloat16*>(gpu_portion->d_context_window[buffer_index]),
			static_cast<const float*>(non_const_act->d_dz),
			input_dim,
			output_dim,
			learning_rate,
			gpu_weights->step_count,
			beta1,
			beta2,
			eps,
			weight_decay);
	} else {
		backward_pass_kernel<<<grid.block, grid.thread, grid.index, stream>>>(
			static_cast<float*>(gpu_weights->d_weights_master),
			static_cast<__nv_bfloat16*>(gpu_weights->d_weights),
			static_cast<float*>(gpu_weights->d_bias_master),
			static_cast<__nv_bfloat16*>(gpu_weights->d_bias),
			static_cast<const __nv_bfloat16*>(gpu_portion->d_context_window[buffer_index]),
			static_cast<const float*>(non_const_act->d_dz),
			input_dim,
			output_dim,
			learning_rate);
	}

	if (!CUDA_CHECK(cudaGetLastError()))
		return false;

	// Phase 5: Backward Scatter Kernel for learnable embedding matrix
	if (gpu_portion->token_ids_active[buffer_index] && gpu_embeddings &&
	    gpu_embeddings->d_table_master && gpu_embeddings->d_table) {
		size_t num_tok = gpu_portion->token_count > 0 ? gpu_portion->token_count : (input_dim / emb_dim);
		int threads = 256;
		int blocks = static_cast<int>((input_dim + threads - 1) / threads);

		backward_scatter_embeddings_kernel<<<blocks, threads, 0, stream>>>(
			static_cast<const uint32_t*>(gpu_portion->d_token_ids[buffer_index]),
			static_cast<float*>(gpu_embeddings->d_table_master),
			static_cast<__nv_bfloat16*>(gpu_embeddings->d_table),
			static_cast<const float*>(gpu_weights->d_weights_master),
			static_cast<const float*>(non_const_act->d_dz),
			static_cast<const float*>(gpu_activations->d_inv_rms),
			num_tok,
			emb_dim,
			gpu_embeddings->vocab_size,
			output_dim,
			learning_rate);

		if (!CUDA_CHECK(cudaGetLastError()))
			return false;
	}

	// Releases the next step's forward pass (see forward_pass)
	return CUDA_CHECK(cudaEventRecord((cudaEvent_t)gpu_portion->compute_done[buffer_index], stream));
}

extern "C" CUDA_API bool backward_pass(const GpuPortion* gpu_portion,
				       GpuWeights* gpu_weights,
				       const GpuActivations* gpu_activations,
				       size_t input_dim,
				       size_t output_dim,
				       float learning_rate,
				       int buffer_index) {
	return backward_pass_with_embeddings(gpu_portion,
					    nullptr,
					    gpu_weights,
					    gpu_activations,
					    input_dim,
					    output_dim,
					    learning_rate,
					    buffer_index,
					    false);
}
