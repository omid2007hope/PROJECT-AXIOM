#ifndef include_optimizer
#define include_optimizer

#include <cstddef>

// Defines the data sizes for the Two-Tiered Loading System
struct Portion {
  // Tier 2: The AI's Field of View (RAM -> Perceptron)
  // Optimal for basic CPU Perceptron: 64 or 128 tokens.
  size_t context_window = 128;

  // Tier 1: The RAM Buffer (Disk -> RAM)
  // Optimal size: 100,000 to 1,000,000 tokens to keep CPU cache fed without OOM.
  size_t ram_buffer = 100000;
};

class Optimizer {
public:
  Portion portion;
  
  Optimizer() = default;
};

#endif