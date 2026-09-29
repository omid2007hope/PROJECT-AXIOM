#include "../include/optimizer.hpp"
#include <cstring> // std::memcpy

// Tier 1: Stream chunks from disk into the RAM buffer
void Optimizer::disk_to_ram(Portion &portion) {

  // =========================================================================
  // CRITICAL SYNCHRONIZATION BARRIER — LIFE AND DEATH GUARANTEE
  // Verify buffer state locks before touching memory to prevent corruption!
  // =========================================================================
  // Atomic acquire: ensures we see all prior writes from other threads
  // before we decide to proceed.
  if (portion.buffer_in_use.load(std::memory_order_acquire) == false &&
      portion.context_in_use.load(std::memory_order_acquire) == false) {
    portion.buffer_in_use.store(true, std::memory_order_release);

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
    // disk_to_ram owns this: it reads carry BEFORE clearing, so the
    // tail of the old 100k window seeds the front of the new 100k window.
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

    // Atomic release: all prior writes are visible to any thread that
    // subsequently acquires this flag.
    portion.buffer_in_use.store(false, std::memory_order_release);
  }
}

// Tier 2: Extract sliding context windows from RAM buffer to feed neural
// network

bool Optimizer::ram_to_contextWindow(Portion &portion, bool &training_in_use) {

  if (portion.context_in_use.load(std::memory_order_acquire) == false &&
      portion.buffer_in_use.load(std::memory_order_acquire)  == false &&
      training_in_use == false) {

    // Lock with release semantics so the GPU thread sees the full buffer
    portion.context_in_use.store(true, std::memory_order_release);
    portion.buffer_in_use.store(true,  std::memory_order_release);

    if (portion.offset + portion.all_elements + portion.target_elements <=
        portion.ram_buffer_tokens.size()) {

      // Guard: pinned buffers must be allocated before we write to them.
      if (!portion.context_window_tokens || !portion.target_tokens) {
        std::cerr << "[Optimizer] Pinned memory not allocated. Call portion.allocate_pinned() first.\n";
        portion.buffer_in_use.store(false, std::memory_order_release);
        portion.context_in_use.store(false, std::memory_order_release);
        return false;
      }

      // Write X into the pinned context window buffer via memcpy.
      // The write-combining hardware coalesces these writes and pushes them
      // directly to the PCIe bus interface — no L1/L2 cache pollution.
      std::memcpy(
          portion.context_window_tokens,
          portion.ram_buffer_tokens.data() + portion.offset,
          portion.all_elements * sizeof(std::bfloat16_t));

      // Write Y into the pinned target buffer.
      std::memcpy(
          portion.target_tokens,
          portion.ram_buffer_tokens.data() + portion.offset + portion.all_elements,
          portion.target_elements * sizeof(std::bfloat16_t));

      // Move the bookmark forward
      portion.offset += portion.token_dim;

      // Release locks — caller trains on context_window_tokens / target_tokens
      portion.buffer_in_use.store(false,  std::memory_order_release);
      portion.context_in_use.store(false, std::memory_order_release);

      return true; // Successfully loaded a window
    } else {
      // Reached the end of the RAM buffer.
      // Reset the offset for the next disk load
      portion.offset = 0;

      // Release locks so disk_to_ram can run!
      portion.buffer_in_use.store(false,  std::memory_order_release);
      portion.context_in_use.store(false, std::memory_order_release);

      return false; // Out of data
    }
  }

  // Locks were busy
  return false;
}
