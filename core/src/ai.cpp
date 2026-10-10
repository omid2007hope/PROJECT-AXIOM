#include "../include/ai.hpp"

#include <iostream>
#include <random>

#include "../include/logistic.hpp"
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

	reset_weights(e);
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

	std::cout << "[AI Brain] Weights and Biases reset to chaotic state.\n";
}

AI::~AI() {
	free_gpu(&gpu_portion);
	free_gpu_weights(&gpu_weights);
	free_gpu_activations(&gpu_activations);
}

bool AI::upload_weights_to_gpu() {
	return Transfer::upload_weights_to_gpu(
		h_weights, h_bias, pending_steps, gpu_weights, gpu_activations);
}

bool AI::download_weights_from_gpu() {
	return Transfer::download_weights_from_gpu(
		h_weights, h_bias, gpu_weights);
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

	// Transfer the pinned host memory to VRAM across the PCIe bus
	if (!upload_to_gpu(portion.context_window_tokens[portion.buffer_index],
			   portion.all_elements,
			   portion.target_tokens[portion.buffer_index],
			   portion.target_elements,
			   &gpu_portion,
			   portion.buffer_index)) {
		std::cerr << "[AI Brain] Failed to upload portion to GPU.\n";
		return TrainStatus::Failed;
	}

	// Launch CUDA kernel for Forward Pass
	if (!forward_pass(&gpu_portion,
			  &gpu_weights,
			  &gpu_activations,
			  input_dim,
			  output_dim,
			  portion.buffer_index)) {
		std::cerr << "[AI Brain] Forward pass failed.\n";
		return TrainStatus::Failed;
	}

	// Launch CUDA kernel for Backward pass (Gradient Descent)
	float learning_rate = optimizer.learning_rate;

	if (!backward_pass(&gpu_portion,
			   &gpu_weights,
			   &gpu_activations,
			   input_dim,
			   output_dim,
			   learning_rate,
			   portion.buffer_index)) {
		std::cerr << "[AI Brain] Backward pass failed.\n";
		return TrainStatus::Failed;
	}

	// Slide to the next buffer for the next training step
	portion.buffer_index = 1 - portion.buffer_index;
	++pending_steps;

	return TrainStatus::Trained;
}
