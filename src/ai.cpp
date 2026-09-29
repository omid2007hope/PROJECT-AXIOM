#include "ai.hpp"
#include <random>
#include <iostream>

// External CUDA function (we will implement this in a .cu file soon)
extern "C" bool upload_ai_weights(const float* h_weights, size_t weight_elements,
                                  const float* h_bias, size_t bias_elements,
                                  GpuWeights* gpu);

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

bool AI::upload_weights_to_gpu() {
    // Pass the data down to the CUDA C-interface
    return upload_ai_weights(h_weights.data(), h_weights.size(),
                             h_bias.data(), h_bias.size(),
                             &gpu_weights);
}

void AI::train(Optimizer &optimizer, Portion &portion) {
    // Acquire the lock before requesting data
    this->training_in_use = true;

    // The Optimizer takes control, writes the context window, and then releases the lock!
    bool has_data = optimizer.ram_to_contextWindow(portion, this->training_in_use);
    
    if (has_data) {
        // std::cout << "[AI] Context window loaded. Ready for Matrix Multiplication!\n";
        // TODO: Launch CUDA kernels for Forward and Backward pass here!
    }
}
