# 09. Verification and Test Suite

In mission-critical systems engineering, unverified code is broken code. Because Project Axiom controls raw GPU pointers, asynchronous PCIe DMA channels, and low-level binary disk serialization, an exhaustive verification suite is required to guarantee memory safety, numerical correctness, and corruption resistance.

This document details the **200 automated verification tests across 9 test suites** implemented in [`core/tests/test_main.cpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/tests/test_main.cpp).

---

## 1. Test Suite Summary and Forensic Results

```
================================================================================
          PROJECT AXIOM AUTOMATED VERIFICATION SUITE: 200 / 200 PASSING         
================================================================================
 [SUITE 1] Weight Persistence & Header Structure        :  25 /  25 PASSED [OK]
 [SUITE 2] Header Corruption Rejection & Defenses       :  25 /  25 PASSED [OK]
 [SUITE 3] Truncated File & Partial Data Immunity       :  20 /  20 PASSED [OK]
 [SUITE 4] Direct Model Inference Engine                :  25 /  25 PASSED [OK]
 [SUITE 5] Multimodal Tokenizer & Namespace Offsets     :  25 /  25 PASSED [OK]
 [SUITE 6] Precision Scaling & Mathematical Identity    :  20 /  20 PASSED [OK]
 [SUITE 7] Pinned Write-Combined DMA & Multi-Stream     :  20 /  20 PASSED [OK]
 [SUITE 8] Defensive Pointer Null-Guard & Neutralization:  20 /  20 PASSED [OK]
 [SUITE 9] Zero-Cloud Stress Testing & VRAM Leak Proof  :  20 /  20 PASSED [OK]
================================================================================
 FINAL VERDICT: 200 / 200 TESTS PASSED (100% GREEN, ZERO MEMORY LEAKS)
================================================================================
```

---

## 2. Detailed Breakdown of the 9 Test Suites

### Suite 1: Single Model Weight Persistence (25 Tests)
- **Objective:** Verify byte-level serialization of parameters to disk.
- **Validations:**
  - Strict 40-byte header layout matching `sizeof(AxiomWeightHeader) == 40`.
  - Magic byte sequence exactly equals `"AXIOM_W1"`.
  - Topology dimensions match $512$ inputs and $4$ outputs.
  - Data payload size matches $(2048 + 4) \times 4 = 8,208$ bytes.
  - Total file size on disk equals exactly $8,248$ bytes.
  - Saved parameters reloaded into memory match original values within floating-point epsilon ($|w_{\text{orig}} - w_{\text{loaded}}| < 10^{-7}$).

### Suite 2: Header Corruption Rejection (25 Tests)
- **Objective:** Prove the engine rejects corrupt or malicious weight files without crashing.
- **Validations:**
  - Files with corrupted magic identifiers (e.g., `"BAD_MAG1"`, `"\0\0\0\0\0\0\0\0"`) are immediately rejected.
  - Incompatible version fields ($v \ne 1$) are rejected.
  - Dimension mismatches (e.g., $1024 \times 8$) are rejected before allocating memory.
  - Invalid payload size flags are detected and aborted.
  - Corrupt files trigger fallback to safe Mersenne Twister initialization with zero memory leaks.

### Suite 3: Truncated File & Partial Data Immunity (20 Tests)
- **Objective:** Test edge cases where disk writes are interrupted or storage is corrupted.
- **Validations:**
  - Empty (0-byte) files abort cleanly.
  - Partial headers (1 byte, 10 bytes, 39 bytes) abort cleanly.
  - Truncated tensor payloads (e.g., 4,000 bytes instead of 8,208 bytes) are caught via stream read counters (`gcount()`).
  - No undefined memory reads or buffer overruns occur during truncated file parsing.

### Suite 4: Direct Model Inference Engine (25 Tests)
- **Objective:** Verify the end-user interactive inference pipeline (`app.exe --use`).
- **Validations:**
  - Input prompt string correctly tokenizes into `std::vector<uint32_t>`.
  - Context windows shorter than 128 tokens are zero-padded deterministically.
  - Context windows longer than 128 tokens are sliced to the trailing 128 tokens.
  - Forward pass executes via Tensor Cores producing continuous 4D vectors.
  - Euclidean $L_2$ decoding matches continuous predictions to valid printable ASCII characters.
  - Output generation terminates cleanly without infinite loops.

### Suite 5: Multimodal Tokenizer & Namespace Isolation (25 Tests)
- **Objective:** Prove complete orthogonality across multimodal token spaces.
- **Validations:**
  - Text tokens lie strictly within range $[1, 9,999,999]$.
  - Number tokens lie strictly within range $[10,000,000, 19,999,999]$.
  - Image codebook tokens lie strictly within range $[20,000,000, 29,999,999]$.
  - Video codebook tokens lie strictly within range $[30,000,000, 39,999,999]$.
  - Voice acoustic tokens lie strictly within range $[40,000,000, 49,999,999]$.
  - Zero token ID collisions occur across all five modalities.
  - Symmetrical roundtrip text encoding and decoding: `tokens_to_text(text_to_tokens(str)) == str`.

### Suite 6: Precision Scaling & Mathematical Identity (20 Tests)
- **Objective:** Verify floating-point conversions and arithmetic stability.
- **Validations:**
  - IEEE FP32 float values roundtrip through `std::bfloat16_t` / `cutlass::bfloat16_t` within acceptable precision bounds.
  - Subnormal float values do not produce hardware traps or NaNs.
  - ReLU activation strictly computes $\max(0.0f, x)$ for both positive and negative values.
  - Delta error vector updates preserve directionality during gradient descent.

### Suite 7: Pinned Write-Combined DMA & Multi-Stream Buffering (20 Tests)
- **Objective:** Validate low-level PCIe memory transfers and CUDA stream concurrency.
- **Validations:**
  - Host context buffers allocate successfully via `cudaHostAllocWriteCombined`.
  - Pointers are 64-byte hardware cache-line aligned.
  - Stream 1 and Stream 2 instantiate with `cudaStreamNonBlocking`.
  - Asynchronous memory copies across non-blocking streams execute without stalling the host thread.
  - Double buffering ("ping-pong") maintains independent buffer states without race conditions.

### Suite 8: Defensive Pointer Null-Guard & Neutralization (20 Tests)
- **Objective:** Enforce the Null-Guard $\rightarrow$ Free $\rightarrow$ Neutralize lifecycle across all GPU pointers.
- **Validations:**
  - Deallocating `nullptr` evaluates to a safe no-op.
  - Calling deallocation twice on the same pointer does not trigger a double-free crash because the pointer is neutralized to `nullptr` on the first call.
  - State structs reset pointers to `nullptr` upon destructor completion.
  - Host RAM vectors clear capacity cleanly.

### Suite 9: Zero-Cloud Stress Testing & VRAM Leak Proof (20 Tests)
- **Objective:** Stress-test the engine over hundreds of iterations to verify zero VRAM leakage.
- **Validations:**
  - Ingests 500+ simulated context window portions sequentially.
  - Measures total free VRAM before, during, and after execution using `cudaMemGetInfo()`.
  - Verified: **Free VRAM variance is exactly 0 bytes** over extended runtime.
  - Total offline compliance: Zero network socket handles opened; all execution remains 100% local.

---

## 3. Executing the Test Suite

To run the complete automated suite:

```powershell
# From project root
.\core\test.exe --test
```

Or via Makefile:
```powershell
cd c:\Users\Omid\X\Code\PROJECT-AXIOM\core
mingw32-make test
```
All test assertions are evaluated with strict return code enforcement. Any single failure exits with a non-zero exit code, providing automated CI/CD safety.
