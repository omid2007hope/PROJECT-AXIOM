#include "../include/ai.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "../include/transfer.hpp"

AI::AI(size_t input_dimension, size_t output_dimension, float weight_min, float weight_max, Elements& elements)
    : input_dimension(input_dimension), output_dimension(output_dimension), weight_min(weight_min), weight_max(weight_max) {

	// input_dimension = token * dim = 512
	// output_dimension = target * dim = 4

	// 512 * 4 = 2048
	elements.weight_elements = input_dimension * output_dimension;

	// output_dimension = 4
	elements.bias_elements = output_dimension;

	host_weights.resize(elements.weight_elements);
	host_bias.resize(elements.bias_elements);

	vocab_size = 65536;
	embedding_dim = output_dimension;
	if (embedding_dim == 0)
		embedding_dim = 4;

	host_embeddings.resize(vocab_size * embedding_dim);

	reset_weights(elements);
	if (allocate_gpu_embeddings(&gpu_embeddings, vocab_size, embedding_dim)) {
		download_embeddings(host_embeddings.data(), &gpu_embeddings);
	}
}

void AI::reset_weights(Elements& elements) {
	elements.weight_elements = input_dimension * output_dimension;
	elements.bias_elements = output_dimension;

	// Initialize with Chaos (Symmetry Breaking)
	// We use the Mersenne Twister engine for high-quality random numbers
	std::random_device rd;
	std::mt19937 gen(rd());

	// Uniform distribution between weight_min and weight_max
	std::uniform_real_distribution<float> dis(weight_min, weight_max);

	for (size_t i = 0; i < elements.weight_elements; ++i) {
		host_weights[i] = dis(gen);
	}

	for (size_t i = 0; i < elements.bias_elements; ++i) {
		// Biases can often safely start at 0.0, because the chaotic weights
		// will do the heavy lifting to break the symmetry on the first pass.
		host_bias[i] = 0.0f;
	}

	float bound = 1.0f / sqrtf(static_cast<float>(embedding_dim));
	std::uniform_real_distribution<float> dis_emb(-bound, bound);
	for (float& val : host_embeddings) {
		val = dis_emb(gen);
	}

	if (gpu_embeddings.d_table_master != nullptr) {
		upload_embeddings(host_embeddings.data(), &gpu_embeddings);
	}

	std::cout << "[AI Brain] Weights, Biases, and Embeddings reset to chaotic state.\n"
		  << std::flush;
}

AI::~AI() {
	free_gpu(&gpu_portion);
	free_gpu_weights(&gpu_weights);
	free_gpu_activations(&gpu_activations);
	free_gpu_embeddings(&gpu_embeddings);
}

bool AI::upload_weights() {
	bool ok = Transfer::upload_weights(
		host_weights, host_bias, pending_steps, gpu_weights, gpu_activations);
	if (!ok)
		return false;

	if (gpu_embeddings.d_table != nullptr && !host_embeddings.empty()) {
		if (!Transfer::upload_embeddings(host_embeddings, gpu_embeddings)) {
			return false;
		}
	}
	return true;
}

bool AI::download_weights() {
	bool ok = Transfer::download_weights(host_weights, host_bias, gpu_weights);
	if (!ok)
		return false;

	if (gpu_embeddings.d_table_master != nullptr && !host_embeddings.empty()) {
		if (!Transfer::download_embeddings(host_embeddings, gpu_embeddings)) {
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

	bool has_data = optimizer.load_context_window(portion);

	if (!has_data) {
		return TrainStatus::NoData;
	}

	int buffer_index = portion.buffer_index;
	float learning_rate = optimizer.learning_rate;

	if (portion.has_token_ids && portion.context_token_ids[buffer_index] != nullptr &&
	    gpu_embeddings.d_table != nullptr) {
		// Transfer only 128 uint32_t token IDs across PCIe per context window (512 bytes!)
		if (!upload_tokens_to_gpu(portion.context_token_ids[buffer_index],
					  portion.context_window_size,
					  portion.target_tokens[buffer_index],
					  portion.target_elements,
					  &gpu_portion,
					  buffer_index)) {
			std::cerr << "[AI Brain] Failed to upload tokens to GPU.\n";
			return TrainStatus::Failed;
		}

		// Launch CUDA gather kernel + CUTLASS EVT GEMM forward pass
		gpu_embeddings.active_vocab_size = active_vocab_size;
		if (!forward_pass_with_embeddings(&gpu_portion,
						  &gpu_embeddings,
						  &gpu_weights,
						  &gpu_activations,
						  input_dimension,
						  output_dimension,
						  buffer_index)) {
			std::cerr << "[AI Brain] Forward pass with embeddings failed.\n";
			return TrainStatus::Failed;
		}

		// Launch CUDA fused AdamW + backward scatter embedding update
		if (!backward_pass_with_embeddings(&gpu_portion,
						   &gpu_embeddings,
						   &gpu_weights,
						   &gpu_activations,
						   input_dimension,
						   output_dimension,
						   learning_rate,
						   buffer_index,
						   true /* use_adamw */)) {
			std::cerr << "[AI Brain] Backward pass with embeddings failed.\n";
			return TrainStatus::Failed;
		}
	} else {
		// Fallback for float-based training
		if (!upload_to_gpu(portion.context_window_tokens[buffer_index],
				   portion.all_elements,
				   portion.target_tokens[buffer_index],
				   portion.target_elements,
				   &gpu_portion,
				   buffer_index)) {
			std::cerr << "[AI Brain] Failed to upload portion to GPU.\n";
			return TrainStatus::Failed;
		}

		if (!forward_pass(&gpu_portion,
				  &gpu_weights,
				  &gpu_activations,
				  input_dimension,
				  output_dimension,
				  buffer_index)) {
			std::cerr << "[AI Brain] Forward pass failed.\n";
			return TrainStatus::Failed;
		}

		if (!backward_pass(&gpu_portion,
				   &gpu_weights,
				   &gpu_activations,
				   input_dimension,
				   output_dimension,
				   learning_rate,
				   buffer_index)) {
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
		if (!download_weights()) {
			std::cerr
				<< "[AI Brain] Failed to download weights from GPU before saving.\n";
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
	header.input_dimension = input_dimension;
	header.output_dimension = output_dimension;
	header.weight_count = host_weights.size();
	header.bias_count = host_bias.size();
	header.vocab_size = vocab_size;
	header.embedding_dim = embedding_dim;

	if (header.weight_count != input_dimension * output_dimension || header.bias_count != output_dimension) {
		std::cerr << "[AI Brain] Weight or bias dimension mismatch during serialization.\n";
		return false;
	}

	for (float w : host_weights) {
		if (!std::isfinite(w)) {
			std::cerr
				<< "[AI Brain] Refusing to save corrupt checkpoint: weights contain NaN or Inf.\n";
			return false;
		}
	}
	for (float b : host_bias) {
		if (!std::isfinite(b)) {
			std::cerr
				<< "[AI Brain] Refusing to save corrupt checkpoint: biases contain NaN or Inf.\n";
			return false;
		}
	}
	for (float e_val : host_embeddings) {
		if (!std::isfinite(e_val)) {
			std::cerr
				<< "[AI Brain] Refusing to save corrupt checkpoint: embeddings contain NaN or Inf.\n";
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
		std::cerr << "[AI Brain] Failed to open temporary file for writing: " << temp_path
			  << "\n";
		return false;
	}

	out.write(reinterpret_cast<const char*>(&header), sizeof(header));
	out.write(reinterpret_cast<const char*>(host_weights.data()),
		  header.weight_count * sizeof(float));
	out.write(reinterpret_cast<const char*>(host_bias.data()), header.bias_count * sizeof(float));
	out.write(reinterpret_cast<const char*>(host_embeddings.data()),
		  header.vocab_size * header.embedding_dim * sizeof(float));

	out.flush();
	if (!out.good()) {
		std::cerr << "[AI Brain] Stream error while writing weights to: " << temp_path
			  << "\n";
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
		std::filesystem::copy_file(
			temp_path, path, std::filesystem::copy_options::overwrite_existing, ec);
		std::filesystem::remove(temp_path, ec);
		if (ec) {
			std::cerr << "[AI Brain] Failed to rename/copy " << temp_path << " to "
				  << path << ": " << ec.message() << "\n";
			return false;
		}
	}

	std::cout << "[AI Brain] Saved " << header.weight_count << " weights, " << header.bias_count
		  << " biases, and " << (header.vocab_size * header.embedding_dim)
		  << " embeddings to " << path << "\n"
		  << std::flush;
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
		std::cerr << "[AI Brain] Incomplete or corrupt header in weight file: " << path
			  << "\n";
		return false;
	}

	// Validate magic bytes
	if (std::memcmp(header.magic, "AXIOM_V1", 8) != 0) {
		std::cerr << "[AI Brain] Invalid magic header in " << path
			  << " (expected 'AXIOM_V1')\n";
		return false;
	}

	// Validate model dimensions
	if (header.input_dimension != input_dimension || header.output_dimension != output_dimension) {
		std::cerr << "[AI Brain] Dimension mismatch in " << path << ": file has ("
			  << header.input_dimension << " -> " << header.output_dimension << "), model expects ("
			  << input_dimension << " -> " << output_dimension << ")\n";
		return false;
	}

	if (header.weight_count != input_dimension * output_dimension || header.bias_count != output_dimension) {
		std::cerr << "[AI Brain] Element count mismatch in " << path << "\n";
		return false;
	}

	if (header.embedding_dim != 0 && header.embedding_dim != output_dimension) {
		std::cerr << "[AI Brain] Embedding dimension mismatch in " << path << ": file has "
			  << header.embedding_dim << ", model expects " << output_dimension << "\n";
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

	in.read(reinterpret_cast<char*>(loaded_bias.data()), header.bias_count * sizeof(float));
	if (in.gcount() != static_cast<std::streamsize>(header.bias_count * sizeof(float))) {
		std::cerr << "[AI Brain] Incomplete bias data in " << path << "\n";
		return false;
	}

	if (header.vocab_size > 0 && header.embedding_dim > 0) {
		size_t embedding_count = header.vocab_size * header.embedding_dim;
		loaded_embeddings.resize(embedding_count);
		in.read(reinterpret_cast<char*>(loaded_embeddings.data()),
			embedding_count * sizeof(float));
		if (in.gcount() != static_cast<std::streamsize>(embedding_count * sizeof(float))) {
			std::cerr << "[AI Brain] Incomplete embedding data in " << path << "\n";
			return false;
		}
	}

	for (float w : loaded_weights) {
		if (!std::isfinite(w)) {
			std::cerr << "[AI Brain] Corrupt weights in " << path
				  << ": contains NaN or Inf. Rejecting load.\n";
			return false;
		}
	}
	for (float b : loaded_bias) {
		if (!std::isfinite(b)) {
			std::cerr << "[AI Brain] Corrupt biases in " << path
				  << ": contains NaN or Inf. Rejecting load.\n";
			return false;
		}
	}
	for (float e_val : loaded_embeddings) {
		if (!std::isfinite(e_val)) {
			std::cerr << "[AI Brain] Corrupt embeddings in " << path
				  << ": contains NaN or Inf. Rejecting load.\n";
			return false;
		}
	}

	// Atomic update of host parameters
	host_weights = std::move(loaded_weights);
	host_bias = std::move(loaded_bias);
	if (!loaded_embeddings.empty()) {
		vocab_size = header.vocab_size;
		embedding_dim = header.embedding_dim;
		host_embeddings = std::move(loaded_embeddings);
	}

	// Synchronize and push directly to GPU VRAM
	if (!upload_weights()) {
		std::cerr << "[AI Brain] Failed to upload loaded weights to GPU VRAM.\n";
		return false;
	}

	std::cout << "[AI Brain] Successfully loaded weights and embeddings from " << path
		  << " and synced to GPU.\n"
		  << std::flush;
	return true;
}

bool AI::predict(const std::vector<std::bfloat16_t>& context_window,
		 std::vector<float>& output_prediction) {
	if (context_window.size() != input_dimension) {
		std::cerr << "[AI Brain] Input context dimension (" << context_window.size()
			  << ") does not match model in_dim (" << input_dimension << ").\n";
		return false;
	}

	synchronize_stream(&gpu_portion, 0);

	std::vector<std::bfloat16_t> dummy_target(output_dimension, static_cast<std::bfloat16_t>(0.0f));

	if (!upload_to_gpu(context_window.data(),
			   context_window.size(),
			   dummy_target.data(),
			   dummy_target.size(),
			   &gpu_portion,
			   0)) {
		std::cerr << "[AI Brain] Failed to upload context vector to GPU for prediction.\n";
		return false;
	}

	gpu_embeddings.active_vocab_size = active_vocab_size;
	const GpuEmbeddings* embeddings_pointer =
		(gpu_embeddings.d_table != nullptr) ? &gpu_embeddings : nullptr;
	if (!forward_pass_with_embeddings(&gpu_portion,
					  embeddings_pointer,
					  &gpu_weights,
					  &gpu_activations,
					  input_dimension,
					  output_dimension,
					  0)) {
		std::cerr << "[AI Brain] Forward pass failed during prediction.\n";
		return false;
	}

	std::vector<std::bfloat16_t> output_activations(output_dimension);
	if (!download_output(&gpu_activations, output_activations.data(), output_dimension)) {
		std::cerr << "[AI Brain] Failed to download prediction activations from GPU.\n";
		return false;
	}

	uint32_t gpu_token = 0;
	if (download_pred_token(&gpu_activations, &gpu_token)) {
		last_predicted_token_id = gpu_token;
	} else {
		last_predicted_token_id = 0;
	}

	output_prediction.resize(output_dimension);
	for (size_t i = 0; i < output_dimension; ++i) {
		output_prediction[i] = static_cast<float>(output_activations[i]);
	}

	return true;
}

bool AI::generate(const std::string prompt,
		  size_t max_new_tokens,
		  float temperature,
		  int top_k,
		  float top_d) {
	return false;
}
