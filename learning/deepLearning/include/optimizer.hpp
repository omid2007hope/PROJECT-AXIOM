#ifndef include_optimizer
#define include_optimizer

#include <cstddef>
#include <vector>

struct Portion {
  // ---------------------------------------------------------
  // Tier 1: The RAM Buffer (Disk -> RAM)
  // Optimal size: 100,000 to 1,000,000 tokens to keep CPU fed.
  // ---------------------------------------------------------
  size_t ram_buffer_size = 100000;

  // Will hold: ram_buffer_size * embedding_dim (e.g., 400,000 doubles)
  std::vector<double> ram_buffer_tokens = {};

  // ---------------------------------------------------------
  // Tier 2: The AI's Field of View (RAM -> Perceptron)
  // Optimal for basic CPU Perceptron: 64 or 128 tokens.
  // ---------------------------------------------------------
  size_t context_window_size = 128;

  // The Input (X): What the AI sees (128 tokens * 4 dim = 512 doubles)
  std::vector<double> context_window_tokens = {};

  // The Target (Y): The answer key (1 token * 4 dim = 4 doubles)
  std::vector<double> target_token = {};
};

class Optimizer {
public:
  void disk_to_ram(const Portion &portion);

  void ram_to_neuralNetwork(const Portion &portion);
};

#endif