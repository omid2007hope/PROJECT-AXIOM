#include "../include/optimizer.hpp"
#include "../include/parser.hpp"
#include <iostream>

// ─── Minimal assertion helper ────────────────────────────────────────────────
static int s_pass = 0, s_fail = 0;

template <typename A, typename B>
static void check_eq(A actual, B expected, const char *label) {
  if (static_cast<B>(actual) == expected) {
    std::cout << "  PASS [" << label << "]: " << actual << "\n";
    ++s_pass;
  } else {
    std::cerr << "  FAIL [" << label << "]: got " << actual << ", expected "
              << expected << "\n";
    ++s_fail;
  }
}
static void check(bool cond, const char *label) {
  if (cond) {
    std::cout << "  PASS [" << label << "]\n"; ++s_pass;
  } else {
    std::cerr << "  FAIL [" << label << "]\n"; ++s_fail;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void test_optimizer() {
  std::cout << "\n=== Optimizer Test Suite ===\n";

  // Constants that match optimizer.cpp internals
  const size_t TOKEN_DIM = 4; // 1 token = 4 bfloat16_t values

  // Use small buffer so tests are fast yet cover all code paths.
  // ram_buffer_size = 100 floats = 25 tokens
  // context_window_size = 4 tokens → window=16 floats, target=4 floats
  const size_t RAM_BUF   = 100;
  const size_t CTX_WIN   = 4;
  const size_t WIN_ELEMS = CTX_WIN * TOKEN_DIM;  // 16
  const size_t TGT_ELEMS = 1 * TOKEN_DIM;        //  4

  // Expected windows per chunk: floor((RAM_BUF - WIN_ELEMS - TGT_ELEMS) / TOKEN_DIM) + 1
  const size_t EXPECTED_WINDOWS =
      (RAM_BUF - WIN_ELEMS - TGT_ELEMS) / TOKEN_DIM + 1; // = 21

  // ── [1] Initial Portion state ────────────────────────────────────────────
  std::cout << "\n[1] Initial Portion state\n";
  Portion portion;
  check_eq(portion.ram_buffer_tokens.size(),     size_t{0}, "ram_buffer empty");
  check_eq(portion.context_window_tokens.size(), size_t{0}, "context_window empty");
  check_eq(portion.target_tokens.size(),         size_t{0}, "target empty");
  check(portion.buffer_in_use  == false, "buffer_in_use == false");
  check(portion.context_in_use == false, "context_in_use == false");

  portion.ram_buffer_size    = RAM_BUF;
  portion.context_window_size = CTX_WIN;

  Optimizer optimizer;

  // ── [2] disk_to_ram — first load ─────────────────────────────────────────
  std::cout << "\n[2] disk_to_ram — first load\n";
  optimizer.disk_to_ram(portion);
  check_eq(portion.ram_buffer_tokens.size(), RAM_BUF,
           "ram_buffer filled to RAM_BUF");
  check(portion.buffer_in_use == false, "buffer lock released after load");

  // ── [3] ram_to_contextWindow — slide all windows ──────────────────────────
  std::cout << "\n[3] ram_to_contextWindow — slide " << EXPECTED_WINDOWS
            << " windows\n";
  bool training_in_use = false;
  optimizer.ram_to_contextWindow(portion, training_in_use);

  // After the loop the last window must be correctly sized
  check_eq(portion.context_window_tokens.size(), WIN_ELEMS,
           "last context_window_tokens size == WIN_ELEMS");
  check_eq(portion.target_tokens.size(), TGT_ELEMS,
           "last target_tokens size == TGT_ELEMS");
  check(portion.buffer_in_use  == false, "buffer lock released after windows");
  check(portion.context_in_use == false, "context lock released after windows");

  // ── [4] disk_to_ram — second load (carry-forward) ────────────────────────
  std::cout << "\n[4] disk_to_ram — second load (carry-forward)\n";
  // ram_buffer_tokens still holds the old 100 elements.
  // disk_to_ram must: read carry (last WIN_ELEMS=16), clear, prepend carry,
  // then fill so total == RAM_BUF again.
  check_eq(portion.ram_buffer_tokens.size(), RAM_BUF,
           "buffer still holds data (not cleared by ram_to_contextWindow)");
  optimizer.disk_to_ram(portion);
  check_eq(portion.ram_buffer_tokens.size(), RAM_BUF,
           "ram_buffer refilled to RAM_BUF on 2nd call");
  check(portion.buffer_in_use == false, "buffer lock released 2nd load");

  // ── [5] Carry-forward correctness ────────────────────────────────────────
  std::cout << "\n[5] Carry-forward sanity\n";
  // The first WIN_ELEMS elements of the new buffer came from the tail of the
  // old chunk. We can't check their exact values without storing the old buffer,
  // but we can verify the buffer is full and non-empty at the front.
  check(portion.ram_buffer_tokens.size() == RAM_BUF,
        "carry prepended + disk fill = RAM_BUF");
  // Every element at the carry positions should be a valid BF16 value
  bool carry_nonzero = false;
  for (size_t i = 0; i < WIN_ELEMS; ++i) {
    if (static_cast<double>(portion.ram_buffer_tokens[i]) != 0.0) {
      carry_nonzero = true; break;
    }
  }
  check(carry_nonzero, "carry-forward region contains non-zero values");

  // ── [6] Lock safety — buffer_in_use blocks disk_to_ram ───────────────────
  std::cout << "\n[6] Lock safety\n";
  {
    Portion locked;
    locked.buffer_in_use = true;
    locked.ram_buffer_size = 64;
    Optimizer opt2;
    opt2.disk_to_ram(locked); // must be a no-op
    check_eq(locked.ram_buffer_tokens.size(), size_t{0},
             "disk_to_ram skipped when buffer_in_use=true");
  }
  {
    Portion locked;
    locked.context_in_use = true;
    locked.ram_buffer_size = 64;
    Optimizer opt3;
    opt3.disk_to_ram(locked); // must be a no-op
    check_eq(locked.ram_buffer_tokens.size(), size_t{0},
             "disk_to_ram skipped when context_in_use=true");
  }

  // ── [7] Full 3-cycle training loop ───────────────────────────────────────
  std::cout << "\n[7] Full training cycle — 3 consecutive cycles\n";
  for (int cycle = 0; cycle < 3; ++cycle) {
    optimizer.disk_to_ram(portion);
    size_t buf_size = portion.ram_buffer_tokens.size();
    optimizer.ram_to_contextWindow(portion, training_in_use);

    std::cout << "  cycle " << cycle + 1
              << ": ram=" << buf_size
              << " → ctx=" << portion.context_window_tokens.size()
              << " tgt=" << portion.target_tokens.size() << "\n";

    check_eq(portion.context_window_tokens.size(), WIN_ELEMS,
             "ctx size correct");
    check_eq(portion.target_tokens.size(), TGT_ELEMS,
             "tgt size correct");
    check(portion.buffer_in_use  == false, "buffer unlocked after cycle");
    check(portion.context_in_use == false, "context unlocked after cycle");
  }

  std::cout << "\n=== Optimizer Test Suite: " << s_pass << " passed, "
            << s_fail << " failed ===\n";
  s_pass = s_fail = 0;
}
