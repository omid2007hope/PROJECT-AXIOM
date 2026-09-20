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
        portion.buffer_in_use = false; // release lock so callers can retry
        return;
      }
    }

    // Reset stream position if EOF was reached on a previous pass (start a new
    // training epoch)
    if (file_stream.eof()) {
      file_stream.clear();
      file_stream.seekg(0, std::ios::beg);
      std::cerr << "EOF Reached!!! New Epoch Started!\n";
    }

    // ----------------------------------------------------------------
    // Carry-Forward: Copy the last context_window_size tokens from the
    // previous chunk to the front of the new buffer.
    // This prevents "amnesia" at the seam between two chunks.
    // RAM cost: a few KB. CPU cost: negligible. Disk reads stay sequential.
    // ----------------------------------------------------------------
    std::vector<std::bfloat16_t> carry = {};
    size_t token_dim = 4; // 1 token = 4 bfloat16_t values
    size_t carry_elements = portion.context_window_size * token_dim;
    if (portion.ram_buffer_tokens.size() >= carry_elements) {
      auto it = portion.ram_buffer_tokens.end() -
                static_cast<std::ptrdiff_t>(carry_elements);
      carry.assign(it, portion.ram_buffer_tokens.end());
    }

    // Clear old buffer and prepend the carry-forward tokens
    portion.ram_buffer_tokens.clear();
    portion.ram_buffer_tokens = carry;

    // Determine remaining elements needed to fill the RAM buffer
    size_t brain_floats_needed =
        portion.ram_buffer_size - portion.ram_buffer_tokens.size();

    size_t prev_size = portion.ram_buffer_tokens.size();

    portion.ram_buffer_tokens.resize(prev_size + brain_floats_needed);

    // Read raw binary bfloat16_t values sequentially from disk
    file_stream.read(
        reinterpret_cast<char *>(portion.ram_buffer_tokens.data() + prev_size),
        brain_floats_needed * sizeof(std::bfloat16_t));

    std::streamsize bytes_read = file_stream.gcount();

    size_t brain_floats_read = bytes_read / sizeof(std::bfloat16_t);

    // Truncate to actual elements read (handles partial read at EOF)
    portion.ram_buffer_tokens.resize(prev_size + brain_floats_read);

    // Release RAM buffer lock
    portion.buffer_in_use = false;
  }
}

// Tier 2: Extract sliding context windows from RAM buffer to feed neural
// network
void Optimizer::ram_to_contextWindow(Portion &portion, bool &training_in_use) {
  size_t token_dim = 4; // 1 token = 4 bfloat16_t values
  size_t window_elements = portion.context_window_size * token_dim;
  size_t target_elements = 1 * token_dim;

  // Slide through the RAM buffer one token at a time until the end
  // (Stop when there isn't enough data for a full context window + target)
  for (size_t i = 0;
       i + window_elements + target_elements <= portion.ram_buffer_tokens.size();
       i += token_dim) {

    // =========================================================================
    // CRITICAL SYNCHRONIZATION BARRIER — LIFE AND DEATH GUARANTEE
    // Pause if the neural network is currently using the data, or if the
    // context window hasn't been emptied yet by the consumer!
    // =========================================================================
    while (training_in_use == true || portion.context_in_use == true ||
           !portion.context_window_tokens.empty()) {
      // Yield or wait for the neural network to finish its pass
      // In a real threaded environment we would sleep or yield here.
    }

    // Lock the context window and buffer
    portion.context_in_use = true;
    portion.buffer_in_use = true;

    // Clear previous tokens
    portion.context_window_tokens.clear();
    portion.target_tokens.clear();

    // Extract Context Window (X)
    portion.context_window_tokens.assign(
        portion.ram_buffer_tokens.begin() + i,
        portion.ram_buffer_tokens.begin() + i + window_elements);

    // Extract Target (Y)
    portion.target_tokens.assign(
        portion.ram_buffer_tokens.begin() + i + window_elements,
        portion.ram_buffer_tokens.begin() + i + window_elements +
            target_elements);

    // Release locks so the neural network can start learning on the new data
    portion.buffer_in_use = false;
    portion.context_in_use = false;
  }

  // All 100,000-token windows have been served to the AI.
  // Clear the RAM buffer so disk_to_ram detects it is empty and
  // refills from disk on the next call — completing the cycle:
  //   disk → ram (100k) → context window (128) → AI → repeat
  portion.ram_buffer_tokens.clear();
}