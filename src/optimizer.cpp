#include "optimizer.hpp"
#include <cstring>  // std::memcpy
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
// disk_to_ram
//
// Fills portion.ram_buffer_tokens from the binary embedded data file.
//
// THE SEAM (Carry-Forward Mechanism):
//   When called after the first load, the last `context_window_size` tokens
//   of the exhausted buffer are saved and prepended to the new chunk.
//   This prevents historical amnesia at chunk boundaries — the network always
//   sees a continuous stream of history with no cold starts.
//
// The file_stream stays permanently open (one syscall per chunk, not per token).
// On EOF the stream rewinds to position 0 so training loops indefinitely.
// ─────────────────────────────────────────────────────────────────────────────
void Optimizer::disk_to_ram(Portion &portion) {

  // Open the stream once on the very first call.
  if (!file_stream.is_open()) {
    file_stream.open(data_path, std::ios::binary);
    if (!file_stream) {
      std::cerr << "[Optimizer] Failed to open: " << data_path << "\n";
      return;
    }
    std::cout << "[Optimizer] Opened data file: " << data_path << "\n";
  }

  const size_t seam_tokens   = portion.context_window_size;          // 128
  const size_t seam_elements = seam_tokens * portion.token_dim;      // 512
  const size_t read_elements = portion.ram_buffer_size * portion.token_dim; // 400,000

  // ── Save the seam from the tail of the old buffer ─────────────────────────
  std::vector<std::bfloat16_t> seam;
  if (portion.ram_buffer_tokens.size() >= seam_elements) {
    seam.assign(
      portion.ram_buffer_tokens.end() - static_cast<std::ptrdiff_t>(seam_elements),
      portion.ram_buffer_tokens.end()
    );
  }

  // ── Allocate: seam up front + new chunk behind it ─────────────────────────
  const size_t prefix = seam.size(); // 0 on first call, seam_elements after
  portion.ram_buffer_tokens.resize(prefix + read_elements);

  // ── Prepend the seam ──────────────────────────────────────────────────────
  if (prefix > 0) {
    std::copy(seam.begin(), seam.end(), portion.ram_buffer_tokens.begin());
  }

  // ── Stream new data from disk ─────────────────────────────────────────────
  file_stream.read(
    reinterpret_cast<char *>(portion.ram_buffer_tokens.data() + prefix),
    read_elements * sizeof(std::bfloat16_t)
  );

  const size_t elements_read =
    static_cast<size_t>(file_stream.gcount()) / sizeof(std::bfloat16_t);

  // Trim to exactly what was written
  portion.ram_buffer_tokens.resize(prefix + elements_read);

  // ── Rewind on EOF so training loops indefinitely ──────────────────────────
  if (file_stream.eof()) {
    file_stream.clear();
    file_stream.seekg(0, std::ios::beg);
    std::cout << "[Optimizer] EOF reached — rewinding to start of file.\n";
  }

  // ── Reset offset: skip past the seam, it was already processed ───────────
  portion.offset = prefix;

  std::cout << "[Optimizer] Loaded " << elements_read / portion.token_dim
            << " new tokens (" << elements_read << " elements).\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// ram_to_contextWindow
//
// Copies the next sliding window from the RAM buffer into the pinned host
// buffers (context_window_tokens = X, target_tokens = Y).
//
// The window slides by exactly 1 token per call (token_dim elements), so
// every consecutive call gives 127 tokens of shared history + 1 new token.
//
// If the buffer is exhausted, disk_to_ram() is called automatically.
// Returns false (and sets training_in_use = false) if no data is available.
// ─────────────────────────────────────────────────────────────────────────────
bool Optimizer::ram_to_contextWindow(Portion &portion, bool &training_in_use) {

  // Elements needed: full context window + one target token
  const size_t needed = portion.all_elements + portion.target_elements; // 516

  // Refill from disk if the buffer is empty or too close to the end
  if (portion.ram_buffer_tokens.empty() ||
      portion.offset + needed > portion.ram_buffer_tokens.size()) {
    disk_to_ram(portion);

    // Still not enough data after a fresh load — give up.
    if (portion.offset + needed > portion.ram_buffer_tokens.size()) {
      training_in_use = false;
      return false;
    }
  }

  // ── Copy X: 128 tokens → pinned context_window_tokens ────────────────────
  std::memcpy(
    portion.context_window_tokens,
    portion.ram_buffer_tokens.data() + portion.offset,
    portion.all_elements * sizeof(std::bfloat16_t)
  );

  // ── Copy Y: 1 token immediately after the window → pinned target_tokens ──
  std::memcpy(
    portion.target_tokens,
    portion.ram_buffer_tokens.data() + portion.offset + portion.all_elements,
    portion.target_elements * sizeof(std::bfloat16_t)
  );

  // ── Slide the window forward by exactly 1 token ───────────────────────────
  portion.offset += portion.token_dim;

  return true;
}
