#include "../include/optimizer.hpp"

// Tier 1: Stream chunks from disk into the RAM buffer
void Optimizer::disk_to_ram(Portion &portion) {

  // =========================================================================
  // CRITICAL SYNCHRONIZATION BARRIER — LIFE AND DEATH GUARANTEE
  // Verify buffer state locks before touching memory to prevent corruption!
  // =========================================================================
  if (portion.buffer_in_use == false && portion.context_in_use == false &&
      portion.ram_buffer_tokens.empty()) {
    portion.buffer_in_use = true;

    // Open the binary file stream once; keep it open across chunks to minimize
    // syscalls
    if (!file_stream.is_open()) {
      file_stream.open("database/embedded_data.bin", std::ios::binary);
      if (!file_stream) {
        std::cerr << "Failed to open database/embedded_data.bin\n";
        portion.buffer_in_use = false;
        return;
      }
    }

    // Reset stream position if EOF was reached on a previous pass (start a new
    // training epoch)
    if (file_stream.eof()) {
      file_stream.clear();
      file_stream.seekg(0, std::ios::beg);
    }

    // ----------------------------------------------------------------
    // Carry-Forward: Copy the last context_window_size tokens from the
    // previous chunk to the front of the new buffer.
    // This prevents "amnesia" at the seam between two chunks.
    // RAM cost: a few KB. CPU cost: negligible. Disk reads stay sequential.
    // ----------------------------------------------------------------
    std::vector<double> carry = {};
    if (portion.ram_buffer_tokens.size() >= portion.context_window_size) {
      auto it = portion.ram_buffer_tokens.end() -
                static_cast<std::ptrdiff_t>(portion.context_window_size);
      carry.assign(it, portion.ram_buffer_tokens.end());
    }

    // Clear old buffer and prepend the carry-forward tokens
    portion.ram_buffer_tokens.clear();
    portion.ram_buffer_tokens = carry;

    // Determine remaining doubles needed to fill the RAM buffer
    size_t doubles_needed =
        portion.ram_buffer_size - portion.ram_buffer_tokens.size();

    size_t prev_size = portion.ram_buffer_tokens.size();
    portion.ram_buffer_tokens.resize(prev_size + doubles_needed);

    // Read raw binary doubles sequentially from disk
    file_stream.read(
        reinterpret_cast<char *>(portion.ram_buffer_tokens.data() + prev_size),
        doubles_needed * sizeof(double));

    std::streamsize bytes_read = file_stream.gcount();
    size_t doubles_read = bytes_read / sizeof(double);

    // Truncate to actual doubles read (handles partial read at EOF)
    portion.ram_buffer_tokens.resize(prev_size + doubles_read);

    // Release RAM buffer lock
    portion.buffer_in_use = false;
  }
}

// Tier 2: Extract sliding context windows from RAM buffer to feed neural
// network
void Optimizer::ram_to_contextWindow(Portion &portion, bool &training_in_use) {
  // TODO: Implement sliding window over ram_buffer_tokens to populate
  // context_window_tokens

  // Whenever the training executes it makes the training_in_use true
  // Which means for sure the buffer is in use
  while (training_in_use == true) {
    portion.context_in_use = true;
    portion.buffer_in_use = true;

    std::cerr << "Warring:Learning";
  }

  // =========================================================================
  // CRITICAL SYNCHRONIZATION BARRIER — LIFE AND DEATH GUARANTEE
  // Verify context window state locks before populating neural network batches!
  // =========================================================================
  while (portion.context_in_use == false && portion.buffer_in_use == false &&
         portion.context_window_tokens.empty()) {

    // Buffer locked — conduct batch extraction

    while (portion.context_window_tokens.size() < portion.context_window_size) {
      double one_portion_128;
      portion.context_window_tokens.push_back(one_portion_128);
    }

    // Release context window lock
    portion.context_in_use = false;
  }
}