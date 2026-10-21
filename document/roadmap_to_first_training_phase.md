# Roadmap to First Training Phase: The Assembly God Blueprint

This document specifies the technical implementation sequence, engineering prerequisites, and compute budget required to transition **Project Axiom** from its current single-layer perceptron into a domain-specific **Bare-Metal x86_64 Assembly Model** running on a single **NVIDIA GeForce RTX 4080 Laptop GPU** (12 GB VRAM).

---

## 1. Executive Summary & Feasibility Envelope

The architectural goal is to train a silicon-whisperer that operates directly on CPU registers (`rax`, `rsp`, `xmm0`..`xmm31`) and machine opcodes without runtime compiler layers.

```
+-----------------------------------------------------------------------------------------+
|                  TRAINING DURATION COMPARISON (100M TOKENS x 5 EPOCHS)                  |
+-----------------------------------------------------------------------------------------+
| Configuration                  Throughput       1 Epoch (100M tok)    5 Epochs (Full Run)|
+-----------------------------------------------------------------------------------------+
| Current Unbatched Engine (M=1)  300 tok/s        92.6 hours (3.8 d)    463.0 hours (19.3 d)
| Target Batched Engine (M=32)  4,000 tok/s         6.9 hours (0.3 d)     34.7 hours (1.4 d) |
+-----------------------------------------------------------------------------------------+
```

### Hardware Envelope (RTX 4080 Laptop GPU)
* **Compute Capacity:** 61.3 sustained BF16 Dense TFLOPs (90.5% MFU measured).
* **Memory Bandwidth:** 284.7 GB/s GDDR6 bus bandwidth (65.9% of physical bus limit).
* **Physical VRAM Ceiling:** 12,281 MB (12.0 GB).
* **Estimated Working Set:** **~4.5 GB total VRAM** (Weights, AdamW states, $M=32$ activation buffers).
* **Safe OS Headroom:** **~7.7 GB free VRAM** (prevents Windows Desktop Window Manager / WDDM timeout detection and recovery (TDR) crashes while GPU runs at 100% compute load).

---

## 2. The 4 Engineering Gaps (Execution Order)

To execute this training run without token collapse, context starvation, or memory leaks, the 4 gaps must be addressed in strict chronological dependency:

```
+-----------------------------------------------------------------------------------+
|               PHASE 1: TOKENIZER INJECTION (GAP 2) - HOST C++                     |
|   Pre-seed x86_64 registers & opcodes in vocab.axiom (60% context compression)    |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|               PHASE 2: BATCHED CUDA DISPATCH (GAP 3) - GPU KERNELS                |
|   2D CUDA Grid (M=32) for gather_embeddings & CUTLASS EVT (300 -> 4,000 tok/s)    |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|               PHASE 3: MULTI-HEAD SELF-ATTENTION (GAP 1) - ARCHITECTURE           |
|   Fused QKV GEMM (1024 -> 3072) + Causal Masking for register lifecycle tracking  |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|               PHASE 4: HIGH-SIGNAL DATASET (GAP 4) - DATA PIPELINE                |
|   150 MB–300 MB pure x86_64 NASM with ; [INTENT] and ; [VAR] semantic bridges    |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|               PHASE 5: MULTI-EPOCH BULK TRAINING EXECUTION                        |
|   5 Epochs (~34.7 hours total compute) -> weights.axiom Assembly Checkpoint       |
+-----------------------------------------------------------------------------------+
```

---

## 3. Phase 1: The Tokenizer Injection (Gap 2)

### The Problem
Standard whitespace and byte tokenizers fragment assembly instructions. An AVX instruction like `vfmadd231ps ymm0, ymm1, [rax + 32]` splinters into 8 disjointed tokens (`vf`, `madd`, `231`, `ps`, `ymm0`, etc.), consuming 3% of the entire 256-token context window on a single line.

### The Solution: Pre-Seed Atomic Register and Opcode IDs
Before processing training files, manually inject core x86_64 tokens into `main_dictionary` in [`parser.cpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/parser.cpp):

1. **Reserve IDs `4` through `~500`** specifically for the x86_64 instruction set and register bank:
   * General-Purpose Registers: `rax`, `rbx`, `rcx`, `rdx`, `rsi`, `rdi`, `rbp`, `rsp`, `r8`–`r15`
   * 32-bit / 16-bit / 8-bit slices: `eax`, `ax`, `al`, etc.
   * Vector Registers: `xmm0`–`xmm31`, `ymm0`–`ymm31`, `zmm0`–`zmm31`
   * Core Opcodes: `mov`, `lea`, `add`, `sub`, `xor`, `cmp`, `test`, `jmp`, `je`, `jne`, `call`, `ret`, `push`, `pop`, `syscall`, `vfmadd231ps`, etc.
2. **Greedy Longest-Prefix Matching:**
   * Tokenizer matches complete strings like `vfmadd231ps` *before* attempting sub-string splits.
3. **Impact:** Instantly compresses assembly sequence length by **~60%**, allowing complete 80–120 instruction functions to fit cleanly within the 256-token context window.

---

## 4. Phase 2: Batched CUDA Dispatch ($M = 32$) (Gap 3)

### The Problem
At batch size $M = 1$, Tensor Cores starve (yielding only ~300 tok/s) because single-vector GEMMs do not fill the Ada Lovelace warp MMA tiles ($16 \times 16 \times 16$ or $32 \times 8 \times 16$).

### The Solution: 2D Grid Launch Layout
Upgrade [`gather_embeddings_kernel`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/kernel/forward_pass.cu) and the CUTLASS EVT launcher to use a 2D grid:

1. **Host Launch Grid:**
   ```cpp
   constexpr int BATCH_SIZE = 32;
   dim3 block(256);
   dim3 grid((context_window_size * embedding_dim + 255) / 256, BATCH_SIZE);
   ```
2. **Kernel Global Offset Calculation:**
   Inside `gather_embeddings_kernel`, use `blockIdx.y` to partition parallel sliding windows:
   ```cuda
   size_t batch_idx = blockIdx.y;
   size_t global_offset = batch_idx * (context_window_size * embedding_dim);
   ```
3. **CUTLASS GEMM Parameters:**
   Configure CUTLASS with dimension $M = 32$, $N = 1024$, $K = 262,144$ (or per-token attention matrix).
4. **Impact:** Elevates training throughput from **300 tok/s $\rightarrow$ 3,500–5,000 tok/s** with near-zero additional wall-clock overhead.

---

## 5. Phase 3: Horizon 2 Multi-Head Self-Attention (Gap 1)

### The Problem
A single-layer perceptron computes an un-gated static sum ($Y = X \cdot W + B$). It cannot establish causal relationships across time, such as tracking when a register pushed to the stack is popped, or identifying jump target labels.

### The Solution: Fused QKV GEMM & Causal Masking
1. **Fused $Q, K, V$ Projection:**
   * Avoid launching three independent GEMMs. Launch a single CUTLASS GEMM multiplying input matrix $X$ by a concatenated weight tensor $W_{QKV}$ of shape $[1024 \times 3072]$.
2. **Scaled Dot-Product Attention:**
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$
   Where $d_k = 128$ (8 heads across 1,024 dimensions).
3. **Causal Masking ($M_{\text{causal}}$):**
   * Overwrite upper-triangle attention scores ($j > i$) to $-\infty$ (`-1e30f`) before softmax. This enforces strict causal autoregression so tokens can only attend to preceding instructions.

---

## 6. Phase 4: High-Signal Dataset & Semantic Bridges (Gap 4)

Because a ~130M parameter model cannot infer abstract algorithms from bare opcodes alone, the training corpus must incorporate structured semantic intent headers.

### Corpus Formatting Standard

```nasm
; [INTENT] Fast inverse square root calculation (Quake III algorithm)
; [VAR] xmm0 = input float (number)
; [VAR] xmm1 = threehalfs constant (1.5)
; [REG] eax  = bit-level integer representation
movss   xmm1, dword [rel magic_number]  ; Load 0x5f3759df
movd    eax, xmm0                       ; Bit-cast float to int
sar     eax, 1                          ; eax >> 1
sub     eax, 0x5f3759df                 ; Magic subtract
movd    xmm0, eax                       ; Bit-cast back to float
ret
```

### Ingestion Target
* **Volume:** 150 MB to 300 MB of pure `.nasm` / `.asm` text (~50M to 100M tokens).
* **Sources:**
  1. Linux kernel x86_64 boot and math subroutines.
  2. Musl libc and glibc assembly primitives (`memcpy`, `memmove`, `strlen`, `math`).
  3. Agner Fog’s vector assembly library.
  4. Curated GitHub repositories containing standalone x86_64 NASM algorithms.

---

## 7. Compute & Hardware Budget Summary

| Parameter | Value |
| :--- | :--- |
| **Model Parameters** | ~120 Million to 135 Million parameters |
| **Active Vocabulary Size ($V$)** | 65,536 tokens |
| **Embedding Dimension ($D$)** | 1,024 channels |
| **Context Window ($K$)** | 256 tokens (expandable to 512) |
| **Batch Size ($M$)** | 32 parallel sliding windows |
| **Dataset Volume** | 100 Million high-signal tokens (150–300 MB) |
| **Target Epochs** | 5 Epochs (shuffled sliding window order) |
| **Total Cumulative Tokens** | 500 Million tokens processed |
| **Hardware Throughput** | ~4,000 tokens/second |
| **Epoch Duration** | ~6.9 hours |
| **Full 5-Epoch Duration** | **~34.7 hours (Single Weekend Run)** |
| **VRAM Consumption** | **~4.5 GB / 12.0 GB (leaves ~7.7 GB safe headroom)** |

---

## 8. Actionable Checklist to Start Training

- [ ] **Milestone 1:** Pre-seed registers and instructions in `parser.cpp` and generate updated `vocab.axiom`.
- [ ] **Milestone 2:** Implement 2D grid gathering (`blockIdx.y`) and $M=32$ CUTLASS EVT launch in `forward_pass.cu`.
- [ ] **Milestone 3:** Implement fused $W_{QKV}$ projection and causal attention kernel in `attention.cu`.
- [ ] **Milestone 4:** Run scraping and annotation pass to assemble 150 MB of commented NASM files into `core/asset_sample/`.
- [ ] **Milestone 5:** Launch `app.exe --bulk-train` over 5 epochs and verify continuous loss convergence on the terminal HUD.
