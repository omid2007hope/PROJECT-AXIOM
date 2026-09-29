#ifndef INCLUDE_AI_HPP
#define INCLUDE_AI_HPP

#include "optimizer.hpp"
#include <vector>
#include <cstddef>

// Forward declaration of the GPU pointers struct
// (Keeps CUDA headers out of our pure C++ code)
struct GpuWeights {
    void* d_weights = nullptr; // Will be cast to __nv_bfloat16* in CUDA
    void* d_bias = nullptr;
    size_t weight_elements = 0;
    size_t bias_elements = 0;
};

class AI {
private:
    size_t input_dim;
    size_t output_dim;

public:
    // CPU Memory (High precision float for initialization and updates)
    std::vector<float> h_weights;
    std::vector<float> h_bias;

    // GPU Memory State (The pointers for the VRAM)
    GpuWeights gpu_weights;

    // Synchronization Lock
    bool training_in_use = false;

    // Constructor: Takes dimensions and initializes with Chaos (Randomness)
    AI(size_t in_dim, size_t out_dim);

    // Method to move memory to GPU
    bool upload_weights_to_gpu();

    // The Training Loop Orchestrator
    void train(Optimizer &optimizer, Portion &portion);
};

#endif
