#ifndef include_optimizer
#define include_optimizer

#include "parser.hpp"
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Architecture principle: Storage drives are hyper-optimized for sequential forward reads.

struct Portion {

  // State flag: locks the RAM buffer while reading data from disk
  bool buffer_in_use = false;
  // State flag: locks the context window while populating batches for the network
  bool context_in_use = false;

  // ---------------------------------------------------------
  // Tier 1: The RAM Buffer (Disk -> RAM)
  // Optimal size: 100,000 to 1,000,000 tokens to keep CPU fed.
  // ---------------------------------------------------------
  size_t ram_buffer_size = 100000;

  // Holds: ram_buffer_size * embedding_dim (e.g., 400,000 doubles)
  std::vector<double> ram_buffer_tokens = {};

  // ---------------------------------------------------------
  // Tier 2: The AI's Field of View (RAM -> Perceptron)
  // Optimal for basic CPU Perceptron: 64 or 128 tokens.
  // ---------------------------------------------------------
  size_t context_window_size = 128;

  // The Input (X): What the AI sees (128 tokens * 4 dim = 512 doubles)
  std::vector<double> context_window_tokens = {};

  // The Target (Y): The prediction ground-truth (1 token * 4 dim = 4 doubles)
  std::vector<double> target_tokens = {};
};

class Optimizer {
private:
  // Kept permanently open: ~8KB OS buffer cost, zero syscall overhead per chunk.
  std::ifstream file_stream;

public:
  void disk_to_ram(Portion &portion);
  void ram_to_contextWindow(Portion &portion);
};

#endif