#include "../include/ai.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "../include/transfer.hpp"

AI::AI(size_t in_dim, size_t out_dim, float weight_min, float weight_max, Elements& e)
    : input_dim(in_dim), output_dim(out_dim), weight_min(weight_min), weight_max(weight_max) {

	// in_dim = token * dim = 512
	// out_dim = target * dim = 4

	// 512 * 4 = 2048
	e.weight_elements = in_dim * out_dim;

	// out_dim = 4
	e.bias_elements = out_dim;

	h_weights.resize(e.weight_elements);
	h_bias.resize(e.bias_elements);

	vocab_size = 65536;
	embedding_dim = in_dim / 128;
	if (embedding_dim == 0) embedding_dim = 4;

	h_embeddings.resize(vocab_size * embedding_dim);

	reset_weights(e);
	if (allocate_gpu_embeddings(&gpu_embeddings, vocab_size, embedding_dim)) {
		download_embeddings(h_embeddings.data(), &gpu_embeddings);
	}
}

void AI::reset_weights(Elements& e) {
	e.weight_elements = input_dim * output_dim;
	e.bias_elements = output_dim;

	// Initialize with Chaos (Symmetry Breaking)
	// We use the Mersenne Twister engine for high-quality random numbers
	std::random_device rd;
	std::mt19937 gen(rd());

	// Uniform distribution between weight_min and weight_max
	std::uniform_real_distribution<float> dis(weight_min, weight_max);

	for (size_t i = 0; i < e.weight_elements; ++i) {
		h_weights[i] = dis(gen);
	}

	for (size_t i = 0; i < e.bias_elements; ++i) {
		// Biases can often safely start at 0.0, because the chaotic weights
		// will do the heavy lifting to break the symmetry on the first pass.
		h_bias[i] = 0.0f;
	}

	float bound = 1.0f / sqrtf(static_cast<float>(embedding_dim));
	std::uniform_real_distribution<float> dis_emb(-bound, bound);
	for (float& val : h_embeddings) {
		val = dis_emb(gen);
	}

	if (gpu_embeddings.d_table_master != nullptr) {
		upload_embeddings(h_embeddings.data(), &gpu_embeddings);
	}

	std::cout << "[AI Brain] Weights, Biases, and Embeddings reset to chaotic state.\n" << std::flush;
}

AI::~AI() {
	free_gpu(&gpu_portion);
	free_gpu_weights(&gpu_weights);
	free_gpu_activations(&gpu_activations);
	free_gpu_embeddings(&gpu_embeddings);
}

bool AI::upload_weights_to_gpu() {
	bool ok = Transfer::upload_weights_to_gpu(
		h_weights, h_bias, pending_steps, gpu_weights, gpu_activations);
	if (!ok) return false;

	if (gpu_embeddings.d_table != nullptr && !h_embeddings.empty()) {
		if (!Transfer::upload_embeddings_to_gpu(h_embeddings, gpu_embeddings)) {
			return false;
		}
	}
	return true;
}

bool AI::download_weights_from_gpu() {
	bool ok = Transfer::download_weights_from_gpu(
		h_weights, h_bias, gpu_weights);
	if (!ok) return false;

	if (gpu_embeddings.d_table_master != nullptr && !h_embeddings.empty()) {
		if (!Transfer::download_embeddings_from_gpu(h_embeddings, gpu_embeddings)) {
			return false;
		}
	}
	return true;
}

bool AI::read_loss(double& mean_loss, size_t& steps) {
	float loss_sum = 0.0f;

	if (!download_loss(&gpu_activations, &loss_sum)) {
		return false;
	}

	steps = pending_steps;
	pending_steps = 0;
	mean_loss = (steps > 0) ? static_cast<double>(loss_sum) / static_cast<double>(steps) : 0.0;

	return true;
}

TrainStatus AI::train(Optimizer& optimizer, Portion& portion) {
	// Sync the stream to ensure the GPU is done with this buffer before we
	// overwrite it.
	synchronize_stream(&gpu_portion, portion.buffer_index);

	bool has_data = optimizer.ram_to_contextWindow(portion);

	if (!has_data) {
		return TrainStatus::NoData;
	}

	int b_idx = portion.buffer_index;
	float learning_rate = optimizer.learning_rate;

	if (portion.has_token_ids && portion.context_token_ids[b_idx] != nullptr &&
	    gpu_embeddings.d_table != nullptr) {
		// Transfer only 128 uint32_t token IDs across PCIe per context window (512 bytes!)
		if (!upload_tokens_to_gpu(portion.context_token_ids[b_idx],
					  portion.context_window_size,
					  portion.target_tokens[b_idx],
					  portion.target_elements,
					  &gpu_portion,
					  b_idx)) {
			std::cerr << "[AI Brain] Failed to upload tokens to GPU.\n";
			return TrainStatus::Failed;
		}

		// Launch CUDA gather kernel + CUTLASS EVT GEMM forward pass
		if (!forward_pass_with_embeddings(&gpu_portion,
						  &gpu_embeddings,
						  &gpu_weights,
						  &gpu_activations,
						  input_dim,
						  output_dim,
						  b_idx)) {
			std::cerr << "[AI Brain] Forward pass with embeddings failed.\n";
			return TrainStatus::Failed;
		}

		// Launch CUDA fused AdamW + backward scatter embedding update
		if (!backward_pass_with_embeddings(&gpu_portion,
						   &gpu_embeddings,
						   &gpu_weights,
						   &gpu_activations,
						   input_dim,
						   output_dim,
						   learning_rate,
						   b_idx,
						   true /* use_adamw */)) {
			std::cerr << "[AI Brain] Backward pass with embeddings failed.\n";
			return TrainStatus::Failed;
		}
	} else {
		// Fallback for float-based training
		if (!upload_to_gpu(portion.context_window_tokens[b_idx],
				   portion.all_elements,
				   portion.target_tokens[b_idx],
				   portion.target_elements,
				   &gpu_portion,
				   b_idx)) {
			std::cerr << "[AI Brain] Failed to upload portion to GPU.\n";
			return TrainStatus::Failed;
		}

		if (!forward_pass(&gpu_portion,
				  &gpu_weights,
				  &gpu_activations,
				  input_dim,
				  output_dim,
				  b_idx)) {
			std::cerr << "[AI Brain] Forward pass failed.\n";
			return TrainStatus::Failed;
		}

		if (!backward_pass(&gpu_portion,
				   &gpu_weights,
				   &gpu_activations,
				   input_dim,
				   output_dim,
				   learning_rate,
				   b_idx)) {
			std::cerr << "[AI Brain] Backward pass failed.\n";
			return TrainStatus::Failed;
		}
	}

	// Slide to the next buffer for the next training step
	portion.buffer_index = 1 - portion.buffer_index;
	++pending_steps;

	return TrainStatus::Trained;
}

bool AI::save_weights(const std::string& path) {
	// 1. Synchronize CUDA streams to ensure in-flight work completes
	synchronize_stream(&gpu_portion, 0);
	synchronize_stream(&gpu_portion, 1);

	// 2. Download latest trained weights and embeddings from GPU if allocated
	if (gpu_weights.d_weights_master != nullptr) {
		if (!download_weights_from_gpu()) {
			std::cerr << "[AI Brain] Failed to download weights from GPU before saving.\n";
			return false;
		}
	}

	// 3. Populate binary header
	AxiomWeightHeader header {};
	header.magic[0] = 'A';
	header.magic[1] = 'X';
	header.magic[2] = 'I';
	header.magic[3] = 'O';
	header.magic[4] = 'M';
	header.magic[5] = '_';
	header.magic[6] = 'V';
	header.magic[7] = '1';
	header.in_dim = input_dim;
	header.out_dim = output_dim;
	header.weight_count = h_weights.size();
	header.bias_count = h_bias.size();
	header.vocab_size = vocab_size;
	header.embedding_dim = embedding_dim;

	if (header.weight_count != input_dim * output_dim || header.bias_count != output_dim) {
		std::cerr << "[AI Brain] Weight or bias dimension mismatch during serialization.\n";
		return false;
	}

	for (float w : h_weights) {
		if (!std::isfinite(w)) {
			std::cerr << "[AI Brain] Refusing to save corrupt checkpoint: weights contain NaN or Inf.\n";
			return false;
		}
	}
	for (float b : h_bias) {
		if (!std::isfinite(b)) {
			std::cerr << "[AI Brain] Refusing to save corrupt checkpoint: biases contain NaN or Inf.\n";
			return false;
		}
	}
	for (float e_val : h_embeddings) {
		if (!std::isfinite(e_val)) {
			std::cerr << "[AI Brain] Refusing to save corrupt checkpoint: embeddings contain NaN or Inf.\n";
			return false;
		}
	}

	// 4. Atomic write: write to temporary file first (.tmp)
	std::filesystem::path target_p(path);
	if (!target_p.parent_path().empty()) {
		std::error_code dir_ec;
		std::filesystem::create_directories(target_p.parent_path(), dir_ec);
	}

	std::string temp_path = path + ".tmp";
	std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
	if (!out.is_open()) {
		std::cerr << "[AI Brain] Failed to open temporary file for writing: " << temp_path << "\n";
		return false;
	}

	out.write(reinterpret_cast<const char*>(&header), sizeof(header));
	out.write(reinterpret_cast<const char*>(h_weights.data()),
		  header.weight_count * sizeof(float));
	out.write(reinterpret_cast<const char*>(h_bias.data()),
		  header.bias_count * sizeof(float));
	out.write(reinterpret_cast<const char*>(h_embeddings.data()),
		  header.vocab_size * header.embedding_dim * sizeof(float));

	out.flush();
	if (!out.good()) {
		std::cerr << "[AI Brain] Stream error while writing weights to: " << temp_path << "\n";
		out.close();
		std::error_code ec;
		std::filesystem::remove(temp_path, ec);
		return false;
	}
	out.close();

	// 5. Atomically replace the target file
	std::error_code ec;
	if (std::filesystem::exists(path, ec)) {
		std::filesystem::remove(path, ec);
	}
	std::filesystem::rename(temp_path, path, ec);
	if (ec) {
		ec.clear();
		std::filesystem::copy_file(temp_path, path,
					   std::filesystem::copy_options::overwrite_existing, ec);
		std::filesystem::remove(temp_path, ec);
		if (ec) {
			std::cerr << "[AI Brain] Failed to rename/copy " << temp_path << " to " << path
				  << ": " << ec.message() << "\n";
			return false;
		}
	}

	std::cout << "[AI Brain] Saved " << header.weight_count << " weights, "
		  << header.bias_count << " biases, and " << (header.vocab_size * header.embedding_dim)
		  << " embeddings to " << path << "\n" << std::flush;
	return true;
}

bool AI::load_weights(const std::string& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in.is_open()) {
		std::cerr << "[AI Brain] Failed to open weight file: " << path << "\n";
		return false;
	}

	AxiomWeightHeader header {};
	in.read(reinterpret_cast<char*>(&header), sizeof(header));
	if (in.gcount() != static_cast<std::streamsize>(sizeof(header))) {
		std::cerr << "[AI Brain] Incomplete or corrupt header in weight file: " << path << "\n";
		return false;
	}

	// Validate magic bytes
	if (std::memcmp(header.magic, "AXIOM_V1", 8) != 0) {
		std::cerr << "[AI Brain] Invalid magic header in " << path << " (expected 'AXIOM_V1')\n";
		return false;
	}

	// Validate model dimensions
	if (header.in_dim != input_dim || header.out_dim != output_dim) {
		std::cerr << "[AI Brain] Dimension mismatch in " << path
			  << ": file has (" << header.in_dim << " -> " << header.out_dim
			  << "), model expects (" << input_dim << " -> " << output_dim << ")\n";
		return false;
	}

	if (header.weight_count != input_dim * output_dim || header.bias_count != output_dim) {
		std::cerr << "[AI Brain] Element count mismatch in " << path << "\n";
		return false;
	}

	std::vector<float> loaded_weights(header.weight_count);
	std::vector<float> loaded_bias(header.bias_count);
	std::vector<float> loaded_embeddings;

	in.read(reinterpret_cast<char*>(loaded_weights.data()),
		header.weight_count * sizeof(float));
	if (in.gcount() != static_cast<std::streamsize>(header.weight_count * sizeof(float))) {
		std::cerr << "[AI Brain] Incomplete weight data in " << path << "\n";
		return false;
	}

	in.read(reinterpret_cast<char*>(loaded_bias.data()),
		header.bias_count * sizeof(float));
	if (in.gcount() != static_cast<std::streamsize>(header.bias_count * sizeof(float))) {
		std::cerr << "[AI Brain] Incomplete bias data in " << path << "\n";
		return false;
	}

	if (header.vocab_size > 0 && header.embedding_dim > 0) {
		size_t emb_count = header.vocab_size * header.embedding_dim;
		loaded_embeddings.resize(emb_count);
		in.read(reinterpret_cast<char*>(loaded_embeddings.data()),
			emb_count * sizeof(float));
		if (in.gcount() != static_cast<std::streamsize>(emb_count * sizeof(float))) {
			std::cerr << "[AI Brain] Incomplete embedding data in " << path << "\n";
			return false;
		}
	}

	for (float w : loaded_weights) {
		if (!std::isfinite(w)) {
			std::cerr << "[AI Brain] Corrupt weights in " << path << ": contains NaN or Inf. Rejecting load.\n";
			return false;
		}
	}
	for (float b : loaded_bias) {
		if (!std::isfinite(b)) {
			std::cerr << "[AI Brain] Corrupt biases in " << path << ": contains NaN or Inf. Rejecting load.\n";
			return false;
		}
	}
	for (float e_val : loaded_embeddings) {
		if (!std::isfinite(e_val)) {
			std::cerr << "[AI Brain] Corrupt embeddings in " << path << ": contains NaN or Inf. Rejecting load.\n";
			return false;
		}
	}

	// Atomic update of host parameters
	h_weights = std::move(loaded_weights);
	h_bias = std::move(loaded_bias);
	if (!loaded_embeddings.empty()) {
		vocab_size = header.vocab_size;
		embedding_dim = header.embedding_dim;
		h_embeddings = std::move(loaded_embeddings);
	}

	// Synchronize and push directly to GPU VRAM
	if (!upload_weights_to_gpu()) {
		std::cerr << "[AI Brain] Failed to upload loaded weights to GPU VRAM.\n";
		return false;
	}

	std::cout << "[AI Brain] Successfully loaded weights and embeddings from " << path << " and synced to GPU.\n" << std::flush;
	return true;
}

bool AI::predict(const Portion& portion, std::vector<float>& output_prediction) {
	if (!portion.context_window_tokens[portion.buffer_index]) {
		std::cerr << "[AI Brain] Pinned context buffer is unallocated in portion.\n";
		return false;
	}

	synchronize_stream(&gpu_portion, portion.buffer_index);

	const void* tgt_ptr = portion.target_tokens[portion.buffer_index]
				      ? portion.target_tokens[portion.buffer_index]
				      : portion.context_window_tokens[portion.buffer_index];

	if (!upload_to_gpu(portion.context_window_tokens[portion.buffer_index],
			   portion.all_elements,
			   tgt_ptr,
			   portion.target_elements,
			   &gpu_portion,
			   portion.buffer_index)) {
		std::cerr << "[AI Brain] Failed to upload context window to GPU for prediction.\n";
		return false;
	}

	if (!forward_pass(&gpu_portion,
			  &gpu_weights,
			  &gpu_activations,
			  input_dim,
			  output_dim,
			  portion.buffer_index)) {
		std::cerr << "[AI Brain] Forward pass failed during prediction.\n";
		return false;
	}

	std::vector<std::bfloat16_t> out_bf16(output_dim);
	if (!download_output(&gpu_activations, out_bf16.data(), output_dim)) {
		std::cerr << "[AI Brain] Failed to download prediction activations from GPU.\n";
		return false;
	}

	output_prediction.resize(output_dim);
	for (size_t i = 0; i < output_dim; ++i) {
		output_prediction[i] = static_cast<float>(out_bf16[i]);
	}

	return true;
}

bool AI::predict(const std::vector<std::bfloat16_t>& context_window,
		 std::vector<float>& output_prediction) {
	if (context_window.size() != input_dim) {
		std::cerr << "[AI Brain] Input context dimension (" << context_window.size()
			  << ") does not match model in_dim (" << input_dim << ").\n";
		return false;
	}

	synchronize_stream(&gpu_portion, 0);

	std::vector<std::bfloat16_t> dummy_tgt(output_dim, static_cast<std::bfloat16_t>(0.0f));

	if (!upload_to_gpu(context_window.data(),
			   context_window.size(),
			   dummy_tgt.data(),
			   dummy_tgt.size(),
			   &gpu_portion,
			   0)) {
		std::cerr << "[AI Brain] Failed to upload context vector to GPU for prediction.\n";
		return false;
	}

	if (!forward_pass(&gpu_portion,
			  &gpu_weights,
			  &gpu_activations,
			  input_dim,
			  output_dim,
			  0)) {
		std::cerr << "[AI Brain] Forward pass failed during prediction.\n";
		return false;
	}

	std::vector<std::bfloat16_t> out_bf16(output_dim);
	if (!download_output(&gpu_activations, out_bf16.data(), output_dim)) {
		std::cerr << "[AI Brain] Failed to download prediction activations from GPU.\n";
		return false;
	}

	output_prediction.resize(output_dim);
	for (size_t i = 0; i < output_dim; ++i) {
		output_prediction[i] = static_cast<float>(out_bf16[i]);
	}

	return true;
}
