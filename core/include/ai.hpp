#ifndef INCLUDE_AI_HPP
#define INCLUDE_AI_HPP

#include "optimizer.hpp"
#include <atomic>
#include <vector>
#include <cstddef>

#include "../src/CUDA/transfer.cuh"

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
    GpuActivations gpu_activations;
    GpuPortion gpu_portion; // Reused across iterations to prevent reallocation

    // Synchronization Lock
    // UPGRADED: bool → std::atomic<bool>
    // Prevents torn reads when the GPU thread and the data-loader thread
    // access this flag concurrently.
    std::atomic<bool> training_in_use{false};

    // Constructor: Takes dimensions and initializes with Chaos (Randomness)
    AI(size_t in_dim, size_t out_dim);
    ~AI();

    // Method to move memory to GPU
    bool upload_weights_to_gpu();
    
    // Method to reset all parameters/weights back to chaotic state
    void reset_weights();

    // The Training Loop Orchestrator
    void train(Optimizer &optimizer, Portion &portion);
};

#endif
