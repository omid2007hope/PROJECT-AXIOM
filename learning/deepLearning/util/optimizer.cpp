#include "../include/optimizer.hpp"

// Tier 1: Stream chunks from disk into the RAM buffer
void Optimizer::disk_to_ram(Portion &portion) {

  // =========================================================================
  // CRITICAL SYNCHRONIZATION BARRIER — LIFE AND DEATH GUARANTEE
  // Verify buffer state locks before touching memory to prevent corruption!
  // =========================================================================
  while (portion.buffer_in_use == false && portion.context_in_use == false &&
         portion.ram_buffer_tokens.empty()) {
    portion.buffer_in_use = true;

    // Open the binary file stream once; keep it open across chunks to minimize
    // syscalls
    if (!file_stream.is_open()) {
      file_stream.open("database/embedded_data.bin", std::ios::binary);
      if (!file_stream) {
        std::cerr << "Failed to open database/embedded_data.bin\n";
        portion.buffer_in_use = true; // stop immediately
                                      ///
        /// IMPORTANT add : and preserve everything that is in the buffer
        ///
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

    // Clear old buffer and prepend the carry-forward tokens
    if ((!portion.ram_buffer_tokens.empty() &&
         portion.ram_buffer_tokens.size() < portion.ram_buffer_size) ||
        portion.ram_buffer_tokens.empty() &&
            portion.ram_buffer_tokens.size() == 0) {

      portion.ram_buffer_tokens.clear();

      // 100 000 >= 512
      if (file_stream.size() >= portion.ram_buffer_size) {

        // Grab 512 tokens from ram_buffer_tokens
        auto grabbed = file_stream.end() -
                       static_cast<std::ptrdiff_t>(portion.ram_buffer_size);

        portion.ram_buffer_tokens.assign(grabbed,
                                         portion.ram_buffer_tokens.end());
      }

    } else {
      std::cerr << "Ram Buffer has unused data, clear is not allowed";
    };

    // Determine remaining doubles needed to fill the RAM buffer
    size_t brain_floats_needed =
        portion.ram_buffer_size - portion.ram_buffer_tokens.size();

    size_t prev_size = portion.ram_buffer_tokens.size();

    portion.ram_buffer_tokens.resize(prev_size + brain_floats_needed);

    // Read raw binary doubles sequentially from disk
    file_stream.read(
        reinterpret_cast<char *>(portion.ram_buffer_tokens.data() + prev_size),
        brain_floats_needed * sizeof(std::bfloat16_t));

    std::streamsize bytes_read = file_stream.gcount();

    size_t brain_floats_read = bytes_read / sizeof(std::bfloat16_t);

    // Truncate to actual doubles read (handles partial read at EOF)
    portion.ram_buffer_tokens.resize(prev_size + brain_floats_read);

    // Release RAM buffer lock
    portion.buffer_in_use = false;
  }
}

// Tier 2: Extract sliding context windows from RAM buffer to feed neural
// network
void Optimizer::ram_to_contextWindow(Portion &portion, bool &training_in_use) {

  while (portion.context_in_use == false &&
         portion.context_window_tokens.empty() && training_in_use == false) {

    portion.buffer_in_use = true;
    portion.context_window_tokens = true;

    if ((!portion.context_window_tokens.empty() &&
         portion.context_window_tokens.size() < portion.context_window_size) ||
        portion.context_window_tokens.empty() &&
            portion.context_window_tokens.size() == 0) {

      portion.context_window_tokens.clear();

      // 100 000 >= 128
      if (portion.ram_buffer_tokens.size() >= portion.context_window_size) {

        // Grab 512 tokens from.context_window_tokens
        auto grabbed = portion.ram_buffer_tokens.end() -
                       static_cast<std::ptrdiff_t>(portion.context_window_size);

        portion.context_window_tokens.assign(
            grabbed, portion.context_window_tokens.end());
      }

    } else {
      std::cerr << "Context Window has unused data, clear is not allowed";
    };

    // Determine remaining doubles needed to fill the RAM buffer
    size_t brain_floats_needed =
        portion.context_window_size - portion.context_window_tokens.size();

    size_t prev_size = portion.context_window_tokens.size();

    portion.context_window_tokens.resize(prev_size + brain_floats_needed);

    portion.buffer_in_use = false;
    portion.context_window_tokens = false;
  }
}