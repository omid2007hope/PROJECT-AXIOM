#include "ai.hpp"
#include <random>
#include <iostream>



AI::AI(size_t in_dim, size_t out_dim) : input_dim(in_dim), output_dim(out_dim) {
    size_t weight_elements = in_dim * out_dim;
    size_t bias_elements = out_dim;

    h_weights.resize(weight_elements);
    h_bias.resize(bias_elements);

    // Initialize with Chaos (Symmetry Breaking)
    // We use the Mersenne Twister engine for high-quality random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Uniform distribution between -0.1 and 0.1
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

    for (size_t i = 0; i < weight_elements; ++i) {
        h_weights[i] = dis(gen);
    }

    for (size_t i = 0; i < bias_elements; ++i) {
        // Biases can often safely start at 0.0, because the chaotic weights 
        // will do the heavy lifting to break the symmetry on the first pass.
        h_bias[i] = 0.0f; 
    }

    std::cout << "[AI Brain] Initialized with " << weight_elements << " chaotic weights.\n";
}

AI::~AI() {
    free_gpu(&gpu_portion);
    free_gpu_weights(&gpu_weights);
    free_gpu_activations(&gpu_activations);
}

bool AI::upload_weights_to_gpu() {
    // Pass the data down to the CUDA C-interface
    return upload_ai_weights(h_weights.data(), h_weights.size(),
                             h_bias.data(), h_bias.size(),
                             &gpu_weights);
}

void AI::train(Optimizer &optimizer, Portion &portion) {
    // Signal: we are now consuming a context window.
    this->training_in_use.store(true, std::memory_order_release);

    // Extract the value into a plain bool so we can pass it by reference
    // to ram_to_contextWindow (which still takes bool& for its internal guard).
    bool in_use = this->training_in_use.load(std::memory_order_acquire);
    bool has_data = optimizer.ram_to_contextWindow(portion, in_use);
    // Sync the atomic back from what the optimizer may have modified.
    this->training_in_use.store(in_use, std::memory_order_release);

    if (has_data) {
        // Transfer the pinned host memory to VRAM across the PCIe bus
        if (upload_to_gpu(portion.context_window_tokens, portion.all_elements,
                          portion.target_tokens, portion.target_elements,
                          &gpu_portion)) {
            // Launch CUDA kernel for Forward Pass
            forward_pass(&gpu_portion, &gpu_weights, &gpu_activations, input_dim, output_dim);
            
            // TODO: Launch CUDA kernel for Backward pass here!
        } else {
            std::cerr << "[AI Brain] Failed to upload portion to GPU.\n";
        }
    }
}
