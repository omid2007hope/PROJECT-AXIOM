#ifndef INCLUDE_AI_HPP
#define INCLUDE_AI_HPP

#include "optimizer.hpp"
#include <atomic>
#include <cstddef>
#include <vector>

#include "../src/CUDA/transfer.cuh"

struct Elements {
  size_t weight_elements = 0;
  size_t bias_elements = 0;
};

class AI {
private:
  size_t input_dim;
  size_t output_dim;

  // Bounds of the uniform distribution used to initialize the weights
  float weight_min;
  float weight_max;

public:
  // CPU Memory (High precision float for initialization and updates)
  std::vector<float> h_weights;
  std::vector<float> h_bias;

  // GPU Memory State (The pointers for the VRAM)
  GpuWeights gpu_weights;
  GpuActivations gpu_activations;
  GpuPortion gpu_portion; // Reused across iterations to prevent reallocation

  // Constructor: Takes dimensions and initializes with Chaos (Randomness)
  // inside [weight_min, weight_max]
  AI(size_t in_dim, size_t out_dim, float weight_min, float weight_max,
     Elements &e);

  ~AI();

  // Method to move memory to GPU
  bool upload_weights_to_gpu();

  // Method to reset all parameters/weights back to chaotic state
  void reset_weights(Elements &e);

  // The Training Loop Orchestrator
  void train(Optimizer &optimizer, Portion &portion);
};

#endif
