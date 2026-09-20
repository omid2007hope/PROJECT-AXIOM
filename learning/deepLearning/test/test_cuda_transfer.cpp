// ─────────────────────────────────────────────────────────────────────────────
// test_cuda_transfer.cpp
//
// Tests are split into two sections:
//
//  Section 1 — CPU-side guarantees (always compiled and run with g++)
//    Verifies the bfloat16 type properties that the CPU→GPU transfer relies on.
//    These must hold for upload_to_gpu() to be correct.
//
//  Section 2 — GPU transfer tests (compiled only when CUDA is linked)
//    Verifies cudaMalloc, cudaMemcpy H→D, cudaFree, reuse, realloc, double-free.
//    Skipped with a clear message until MSVC Build Tools are installed.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdfloat>

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
// Section 1 — CPU-side bfloat16 guarantees
// ─────────────────────────────────────────────────────────────────────────────

static void test_bf16_type_size() {
  // std::bfloat16_t and __nv_bfloat16 both use the same 16-bit BF16 layout.
  // If sizeof != 2 the raw pointer cast in upload_to_gpu() is undefined behaviour.
  check_eq(sizeof(std::bfloat16_t), size_t{2},
           "sizeof(std::bfloat16_t) == 2 bytes");
}

static void test_bf16_precision() {
  // BF16 keeps 8 bits of exponent (same as float32) and 7 mantissa bits.
  // Precision: ~2–3 significant decimal digits.

  // sin(1.0) = 0.8414709848...
  // Nearest BF16: 0.8398 or 0.8438 — within 0.3% of true value.
  std::bfloat16_t v = static_cast<std::bfloat16_t>(0.8414709848);
  double back = static_cast<double>(v);
  check(back >= 0.83 && back <= 0.85,
        "BF16(sin(1.0)) round-trips within ±0.01");
  std::cout << "    value stored: " << back
            << "  original: 0.8414709848\n";

  // Large integer: BF16 can represent 256 exactly (8-bit exponent range).
  std::bfloat16_t big = static_cast<std::bfloat16_t>(256.0);
  check_eq(static_cast<double>(big), 256.0, "BF16(256.0) exact round-trip");

  // Zero is always exact.
  std::bfloat16_t zero = static_cast<std::bfloat16_t>(0.0);
  check_eq(static_cast<double>(zero), 0.0, "BF16(0.0) exact");
}

static void test_bf16_memory_layout() {
  // Verify that std::bfloat16_t has the same binary layout as a 2-byte integer
  // (no padding, trivially copyable). This guarantees that reading BF16 values
  // written by the CPU with cudaMemcpy on the GPU is valid.
  check(std::is_trivially_copyable_v<std::bfloat16_t>,
        "std::bfloat16_t is trivially copyable (safe for memcpy/cudaMemcpy)");
  check(alignof(std::bfloat16_t) <= 2,
        "alignof(std::bfloat16_t) <= 2 (no surprise padding)");

  // Verify a vector of BF16 values is contiguous in memory.
  std::bfloat16_t arr[4] = {
      static_cast<std::bfloat16_t>(1.0f),
      static_cast<std::bfloat16_t>(2.0f),
      static_cast<std::bfloat16_t>(3.0f),
      static_cast<std::bfloat16_t>(4.0f),
  };
  const char *p0 = reinterpret_cast<const char *>(&arr[0]);
  const char *p1 = reinterpret_cast<const char *>(&arr[1]);
  check_eq(p1 - p0, ptrdiff_t{2},
           "consecutive BF16 elements are 2 bytes apart (contiguous)");
}

static void test_gpu_portion_mock() {
  // CPU mock of GpuPortion — verifies default init (no CUDA needed).
  struct MockGpuPortion {
    void  *d_context_window = nullptr;
    void  *d_target         = nullptr;
    size_t context_elements = 0;
    size_t target_elements  = 0;
  };

  MockGpuPortion g;
  check(g.d_context_window == nullptr, "GpuPortion d_context_window init nullptr");
  check(g.d_target         == nullptr, "GpuPortion d_target init nullptr");
  check_eq(g.context_elements, size_t{0}, "GpuPortion context_elements init 0");
  check_eq(g.target_elements,  size_t{0}, "GpuPortion target_elements init 0");

  // Simulate context window for 128 tokens * 4 dim
  const size_t ctx_bytes = 128 * 4 * sizeof(std::bfloat16_t);
  check_eq(ctx_bytes, size_t{1024},
           "context window byte size == 1024 (128 tokens * 4 dim * 2 bytes)");

  // Simulate target for 1 token * 4 dim
  const size_t tgt_bytes = 1 * 4 * sizeof(std::bfloat16_t);
  check_eq(tgt_bytes, size_t{8},
           "target byte size == 8 (1 token * 4 dim * 2 bytes)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2 — GPU tests (placeholder until MSVC + nvcc are available)
// ─────────────────────────────────────────────────────────────────────────────
static void gpu_test_summary() {
  std::cout << "\n[GPU] The following tests will run once MSVC Build Tools\n"
            << "      are installed and nvcc can compile transfer.cu:\n\n"
            << "  PENDING upload_to_gpu() — first call: cudaMalloc + cudaMemcpy\n"
            << "  PENDING upload_to_gpu() — second call same size: reuse buffer\n"
            << "  PENDING upload_to_gpu() — size change: free + realloc\n"
            << "  PENDING upload_to_gpu() — verify H→D values match (cudaMemcpy D→H)\n"
            << "  PENDING free_gpu()      — releases both device pointers\n"
            << "  PENDING free_gpu()      — safe to call twice (no double-free)\n"
            << "  PENDING GpuPortion      — d_context_window byte count == 1024\n"
            << "  PENDING GpuPortion      — d_target byte count == 8\n\n"
            << "  Install: vs_installer.exe modify --add Microsoft.VisualStudio.Workload.VCTools\n";
}

// ─────────────────────────────────────────────────────────────────────────────
void test_cuda_transfer() {
  std::cout << "\n=== CUDA Transfer Test Suite ===\n";

  std::cout << "\n[1] BF16 type size\n";
  test_bf16_type_size();

  std::cout << "\n[2] BF16 precision\n";
  test_bf16_precision();

  std::cout << "\n[3] BF16 memory layout (safe for cudaMemcpy)\n";
  test_bf16_memory_layout();

  std::cout << "\n[4] GpuPortion CPU mock\n";
  test_gpu_portion_mock();

  std::cout << "\n[5] GPU transfer tests\n";
  gpu_test_summary();

  std::cout << "=== CUDA Transfer Test Suite: " << s_pass << " passed, "
            << s_fail << " failed, "
            << "8 GPU tests PENDING ===\n";
  s_pass = s_fail = 0;
}
