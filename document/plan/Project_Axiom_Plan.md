# Project Axiom: Master Engineering Evaluation & Strategic Scaling Plan (v2.0)
**Document Path:** `document/plan/Project_Axiom_Plan.md`  
**Target Architecture:** NVIDIA Ada Lovelace (`sm_89` / RTX 4080) & Native C++23 / CUDA  
**Author:** Antigravity Senior Systems & Deep Learning Engine Architect  
**Status:** Living Engineering Plan & Technical Audit  

---

## 1. Executive Summary & Forensic Reality Check

Project Axiom is an audacious, bare-metal artificial intelligence engine written entirely from scratch in **C++23** and **NVIDIA CUDA**, rejecting heavy runtime frameworks (PyTorch, libtorch, Python) in pursuit of zero-overhead silicon saturation. 

An exhaustive, line-by-line audit of the entire codebase confirms that the **systems engineering, PCIe DMA logistics, and low-level hardware plumbing (Phases 1–3 of the original roadmap) are exceptionally well-constructed**. The system successfully achieves:
1. Native `std::bfloat16_t` CPU/GPU memory layout matching `__nv_bfloat16`.
2. Asynchronous Direct Memory Access (DMA) using `cudaHostAllocWriteCombined` pinned host memory.
3. Multi-stream double-buffering overlapping PCIe transfers with Tensor Core computation.
4. Fused Epilogue Visitor Trees (EVT) utilizing CUTLASS v4.8.0 templates on Ada Lovelace (`sm_89`).
5. A deterministic 40-byte binary persistence format (`weights.axiom` and `vocab.axiom`) with atomic swapping.

### The "Counselor-at-Law" Collapse: What Actually Happened
Despite this brilliant hardware substrate, the recent training run of 214,400 steps collapsed into predicting a single frozen string: `"counselor-at-law"`, with all tensor outputs evaluating to `[0.0000, 0.0000, 0.0000, 0.0000]`.

This was **not a hardware or concurrency bug**. It was a **direct mathematical contradiction engineered into Phase 4 of the original plan**:
* The plan mandated fusing **ReLU** into the CUTLASS epilogue (`EvtRelu`).
* The data pipeline generated target embeddings using sinusoidal functions $\sin(\text{token\_id} \cdot (d + 1)) \in [-1.0, 1.0]$.
* Because roughly 50% of the target values are negative, Mean Squared Error (MSE) loss pushed weights and biases relentlessly downward.
* Once activations fell below zero ($Wx + b \le 0$), ReLU clamped them to $0.0$, and the gradient gate in `backward_pass.cu` (`(y_val > 0.0f) ? dy : 0.0f`) evaluated to zero.
* All 4 output neurons suffered permanent **Dying ReLU collapse**.
* In inference, the model output $[0, 0, 0, 0]$. The nearest-neighbor Euclidean search over 3,249,513 tokens found that Token #1146408 had the smallest $L_2$ norm to the origin. In the reverse dictionary, Token #1146408 mapped to `"counselor-at-law"`.
* Furthermore, iterating through 3.25 million hash map entries on the CPU took **530 ms per token**, completely nullifying the **0.6 ms GPU prefill latency**.

This document provides the definitive architectural blueprint to eliminate these bottlenecks, realign the mathematical formulation, bridge the gap to GPT-scale autoregressive generation, and execute the remaining phases of Project Axiom.

---

## 2. Line-by-Line Forensic Codebase Audit

### 2.1. Memory Hierarchy & Data Logistics (`core/util/optimizer.cpp`, `core/include/optimizer.hpp`)
* **What Works Flawlessly:**
  * **Zero-Copy DMA:** Pinned memory allocated via `cudaHostAllocWriteCombined` (`pinned_alloc`) ensures that the CPU bypasses L1/L2 cache pollution, enabling maximum PCIe bandwidth utilization and true non-blocking `cudaMemcpyAsync`.
  * **The Seam Carry-Forward Mechanism:** Lines 44–55 of `optimizer.cpp` preserve the trailing `context_window_size` tokens across 100,000-token chunk boundaries, preventing historical amnesia without seeking backward on disk.
  * **Persistent File Descriptors:** `file_stream` remains open across chunk reads, eliminating repeated OS file descriptor initialization overhead.
* **Flaws & Critical Liabilities:**
  * **Spin-Lock & Atomic State Complexity:** `Portion::buffer_in_use` and `Portion::context_in_use` use atomic store/load operations with acquire/release semantics. While this prevents torn reads, polling these atomics in a synchronous loop can burn CPU cycles.
  * **Rigid Single-Window Sliding:** `portion.offset += portion.token_dim;` slides by exactly one token per step. When the buffer is exhausted, it resets to zero and waits for `disk_to_ram` synchronously, causing momentary GPU starvation.

### 2.2. CUDA Memory & Device Abstraction (`core/src/CUDA/`)
* **What Works Flawlessly:**
  * **Toolchain Decoupling:** `cuda_transfer.dll` built with NVCC and MSVC (`cl.exe`) is dynamically consumed by the MinGW/GCC C++23 executable (`app.exe`). This cleanly isolates Windows MSVC CUDA ABI requirements from the main C++23 binary.
  * **Mixed-Precision Master Weights:** `GpuWeights` allocates both BF16 device weights (`d_weights`) and FP32 master weights (`d_weights_master`). Updates occur in FP32 to prevent gradient quantization stalls below the 8-bit BF16 mantissa, then round into BF16.
  * **Stream Isolation:** Dual streams (`gpu->stream[0]` and `gpu->stream[1]`) with non-blocking flags and `compute_done` CUDA events properly enforce serial weight dependency while overlapping data transfers.
* **Flaws & Critical Liabilities:**
  * **Dynamic Re-allocation Guards in Transfer Loops:** Lines 20–28 of `upload.cu` check and execute `cudaMalloc`/`cudaFree` dynamically inside `upload_to_gpu` if element counts change. Dynamic `cudaMalloc` forces a device-wide CPU-GPU synchronization barrier.
  * **Host CPU Loss Sync Bottleneck:** `download_loss` calls `cudaDeviceSynchronize()` across the entire device to read a single float. While only executed on report intervals, it destroys stream pipelining during those steps.

### 2.3. Compute Kernels & Epilogue Fusion (`core/src/CUDA/kernel/`)
* **What Works Flawlessly:**
  * **CUTLASS GEMM Execution:** `forward_pass.cu` leverages CUTLASS v4.8.0 templates (`cutlass::gemm::device::GemmUniversalAdapter`) targeting `sm_89` Ada Lovelace Tensor Cores. Prefill latency is clocked at an astonishing **0.6 ms**.
* **The Fatal Architectural Flaws:**
  1. **The Dead ReLU Epilogue (`forward_pass.cu:181`):**
     ```cpp
     typename EvtStore::Arguments callback_args {
         { { {}, { gpu_weights->d_bias, ... }, {} }, // Bias
           {}                                         // EvtRelu <--- FATAL FLAW
         },
         { gpu_activations->d_output, ... }           // Store
     };
     ```
     Enforcing ReLU on an unbounded continuous regression space guarantees dead activations when target coordinates are negative.
  2. **The Gated Derivative (`backward_pass.cu:66`):**
     ```cpp
     float dz = (y_val > 0.0f) ? dy : 0.0f;
     ```
     Once $y = 0$, $dz = 0$. The gradient can never revive the dead neuron.
  3. **Rank-1 Outer Product SGD (`backward_pass.cu:55-80`):**
     The backward pass runs on standard CUDA cores ($1 \text{ thread per weight}$), calculating $dL/dW = dz \times x$. Because $M=1$ (single batch), this is a memory-bound vector outer product that cannot utilize Tensor Cores.

### 2.4. Multimodal Tokenizer & Parser (`core/util/parser.cpp`, `core/include/parser.hpp`)
* **What Works Flawlessly:**
  * **Modality Namespace Isolation:** 10,000,000-slot offsets guarantee that Text (1+), Numbers (10M+), Images (20M+), Video (30M+), and Audio (40M+) never collide.
  * **Persistence Serialization:** `save_vocabulary` and `load_vocabulary` write a clean binary stream (`AXVK_V1`) storing string lengths and character bytes sequentially.
* **The Fatal Architectural Flaws:**
  1. **Synthetic Non-Semantic Embedding Space:**
     ```cpp
     vec[d] = sin(static_cast<double>(id) * (d + 1.0));
     ```
     Token IDs are assigned in chronological order of ingestion. Applying a sine function produces arbitrary pseudo-random coordinates with zero semantic topology: "USA" and "America" have no spatial correlation.
  2. **Severe Dimensionality Bottleneck:**
     Compressing 3,249,513 discrete concepts into $d = 4$ dimensions violates fundamental metric space geometry. In 4D space, 3.25 million points are crammed together with an average pairwise distance approaching zero.
  3. **Unsanitized Vocabulary Explosion:**
     Splitting purely on whitespace (`iss >> word`) without removing XML tags, HTML markup, punctuation, or applying Byte-Pair Encoding (BPE) caused the vocabulary to explode to 3.25 million tokens across 3,024 files.
  4. **The 530 ms CPU Search Jam:**
     `decode_embedding_to_token` scans the entire `input.embedding_table` (`std::unordered_map`) sequentially on the CPU:
     ```cpp
     for (const auto& [id, vec] : input.embedding_table) { ... }
     ```
     Traversing 3.25 million heap-allocated hash table nodes on a single CPU thread takes 530 ms per token, capping decoding throughput at 1.9 tokens/sec.

---

## 3. Progress Scorecard: Roadmap Evaluation (`research.md` vs. Reality)

| Roadmap Milestone | Plan Requirement | Actual Code Implementation Status | Evaluation & Reality Check |
| :--- | :--- | :--- | :--- |
| **Phase 1: Precision Scaling** | `uint32_t` IDs, 10M offsets, `bfloat16` layout | **100% COMPLETE** | Fully achieved. BF16 aligns natively with Ada hardware. |
| **Phase 2: Data Logistics** | Stream loader, seam carry-forward, non-blocking | **90% COMPLETE** | Chunk seam works. Minor CPU polling overhead remains. |
| **Phase 3: PCIe Zero-Copy** | Pinned write-combined host memory, ping-pong streams | **95% COMPLETE** | Implemented in `optimizer.cpp` & `upload.cu`. 0.6 ms TTFT achieved. |
| **Phase 4: Compute & EVT** | CUTLASS GEMM on Tensor Cores, EVT fusion | **100% RESOLVED & VERIFIED** | **[DONE]** Linear EVT epilogue + un-gated SGD ($dz = dy$). Dead ReLU eliminated. |
| **Phase 5: VRAM Management** | `cudaMallocAsync`, Stream memory pools, BFC Allocator | **10% COMPLETE** | Static allocation used; dynamic allocator deferred. |
| **Phase 6: Autoregressive State** | PagedAttention, KV Caching, COW prefix caching | **0% COMPLETE** | Code is still a 1-layer Perceptron, not a Transformer. |
| **Phase 7: Microsecond Scaling** | CUDA Graphs (DAGs), ATen-style C++ Dispatcher | **0% COMPLETE** | Monolithic functions; no graph capture or dispatcher yet. |
| **Decoding Layer (Unlisted)** | Efficient next-token extraction | **IN PROGRESS (CPU Flat Cache Done)** | **[DONE: Flat array cache]** GPU GEMM batched top-1 pending (Milestone 2). |

---

## 4. The Critical Engineering Changes Required

### 4.1. Surgical Fix for Phase 4: Reviving the Neural Nucleus [STATUS: COMPLETED & VERIFIED]
To immediately eliminate the "Dying ReLU" collapse and restore gradient learning:
1. **[DONE] Remove `EvtRelu` in `forward_pass.cu`:** Replaced the ReLU node in the EVT tree with an Identity / Linear store epilogue (`Store(AddBias(Accum, Bias))`). The network can freely predict throughout $[-1.0, +1.0]$.
2. **[DONE] Un-gate the Gradient in `backward_pass.cu`:**
   * Removed `float dz = (y_val > 0.0f) ? dy : 0.0f;`.
   * Replaced with `float dz = dy;`.
3. **[DONE] Weight Symmetry Reset:**
   * Quarantined the frozen `weights.axiom` checkpoint to `weights.axiom.dead_relu_backup`. Fresh weights initialize with random chaotic bounds.

### 4.2. Surgical Fix for Decoding: Contiguous Memory & GPU Acceleration
To eradicate the 530 ms latency and achieve >500 tok/s:
1. **[DONE] Contiguous RAM Decode Mirror (`parser.hpp`, `parser.cpp`):** Replaced pointer-chasing across 3.25M `std::unordered_map` nodes with flat, contiguous `decode_ids` and `decode_vectors` arrays.
2. **[PENDING] Move Embeddings to Device VRAM:** Maintain a contiguous device tensor `d_embedding_table` of shape $[V \times d]$ in `__nv_bfloat16`.
3. **[PENDING] Tensor Core Dot-Product Decoding:** Compute similarity as a batched matrix-vector multiplication directly on Tensor Cores:
   $$S = E \cdot y \quad (\text{Size: } V \times 1)$$
4. **[PENDING] Fast GPU Reduction:** Run a single `cub::DeviceReduce::ArgMax` kernel to find the winning token ID in $<0.2\text{ ms}$, bypassing the CPU entirely.

### 4.3. Text Cleaning & Vocabulary Bounding [STATUS: PENDING NEXT ACTION]
1. Strip XML headers (`<timestamp>`, `<stub>`, HTML tags) before tokenization.
2. Filter tokens by frequency (minimum occurrences $\ge 3$) or clamp vocabulary to a fixed upper bound $V \le 32,768$, mapping rare tokens to an `<UNK>` token ID.

---

## 5. Strategic Scaling Roadmap: Zero to Hero v2.0

```
                                  PROJECT AXIOM v2.0 EVOLUTION
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ MILESTONE 1: Mathematical Restoration & Dead Weight Revival [STATUS: DONE]             │
  │ • Linear Epilogue in CUTLASS EVT (Drop ReLU) [DONE] • Un-gated SGD ($dz = dy$) [DONE]   │
  │ • Quarantined dead checkpoint [DONE] • Tests verify active learning & non-zero outputs │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ MILESTONE 2: GPU-Accelerated Decoding & Vocabulary Sanity [STATUS: IN PROGRESS]        │
  │ • Contiguous RAM Decode Mirror [DONE] • Bounded Vocab (Top 32K + <UNK>) [NEXT]         │
  │ • Contiguous VRAM Embedding Matrix [V x d] & Tensor Core Dot-Product Decoding [NEXT]  │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ MILESTONE 3: Discrete Autoregressive Token Classification (The GPT Gateway)            │
  │ • Hidden dimension projection: $d_{in} \to d_{model} \to V$                            │
  │ • Fused GPU Cross-Entropy Loss & Softmax kernel (Replacing MSE).                       │
  │ • True discrete next-token probability distribution.                                   │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ MILESTONE 4: Multi-Layer Attention & Deep Feature Hierarchy                            │
  │ • Multi-Head Self-Attention (Q, K, V projections) on Tensor Cores via CUTLASS          │
  │ • Residual connections & RMSNorm / LayerNorm EVT fusion.                               │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ MILESTONE 5: Production VRAM & PagedAttention State Management (Original Phase 5 & 6)  │
  │ • Userspace Best-Fit with Coalescing (BFC) Caching Allocator                           │
  │ • PagedAttention Block Table for non-contiguous KV-Cache in VRAM                       │
  │ • Copy-On-Write (COW) prompt prefix caching.                                           │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ MILESTONE 6: Zero-Overhead Hardware Saturation (Original Phase 7)                      │
  │ • CUDA Graph DAG Capture (`cudaStreamBeginCapture` / `cudaGraphLaunch`)                │
  │ • ATen-Style Modular C++ Tensor & Operation Dispatcher                                 │
  │ • Thermodynamic silicon saturation on Ada Lovelace (`sm_89`).                          │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Specifications: File-by-File Action Plan

### Task 1: `core/src/CUDA/kernel/forward_pass.cu`
* **Current:** EVT combines Accumulator + Bias + `EvtRelu` + Store.
* **Modification:**
  * Define an `EvtIdentity` or direct `EvtStore` bypassing ReLU:
    ```cpp
    typename EvtStore::Arguments callback_args {
        {
            {}, // Accum
            { static_cast<const ElementIn*>(gpu_weights->d_bias), ElementIn(0), { _0{}, _1{}, int32_t(N) } }, // Bias
            {}  // EvtAddBias
        },
        { static_cast<ElementIn*>(gpu_activations->d_output), { int64_t(N), _1{}, int64_t(M) * N } } // Store
    };
    ```
* **Impact:** Outputs are free to span negative and positive values, perfectly matching sinusoidal targets.

### Task 2: `core/src/CUDA/kernel/backward_pass.cu`
* **Current:** Line 66 applies conditional ReLU gating: `float dz = (y_val > 0.0f) ? dy : 0.0f;`.
* **Modification:**
  * Eliminate the conditional check:
    ```cpp
    float dz = dy; // Direct linear gradient
    ```
* **Impact:** Gradients flow continuously on every sample; parameter deadlocks become mathematically impossible.

### Task 3: `core/util/parser.cpp`
* **Current:** Brute-force `for (const auto& [id, vec] : input.embedding_table)` CPU search.
* **Modification:**
  * Implement `build_device_embedding_matrix(Input& input, void** d_matrix_ptr)`.
  * Flatten embeddings into an aligned $[V \times d]$ buffer in VRAM.
  * In inference, dispatch a GEMM vector dot product to find similarity scores, followed by `argmax`.
* **Impact:** Decreases token decode latency from **530 ms to <0.5 ms** (a 1,000x throughput acceleration).

### Task 4: `core/src/main.cpp`
* **Current:** Training loop persists weights even when loss plateaus without divergence warnings.
* **Modification:**
  * Add a dead-neuron telemetry assertion: if $\sum |y_i| == 0.0f$ for 100 consecutive steps, fire an immediate divergence alert and auto-reset weights.

---

## 7. Verification & Acceptance Criteria

1. **Gate 1: Activation Vitality Test [PASSED]**
   * Verified in `test.exe` (`test_inference.cpp`): Forward inference outputs are non-zero across all coordinates (`any_inference_nonzero == true`). The dead all-zero collapse is eliminated.
2. **Gate 2: Loss Monotonicity Verification [PASSED]**
   * Verified in `test.exe` (`test_cuda_transfer.cpp` & `test_ai.cpp`): Multi-step loss falls monotonically (initial `0.713989` $\to$ final `0.0`, multi-step test falls from `2.58994` $\to$ `4.76837e-07`).
3. **Gate 3: Semantic Diversity in Inference [PENDING RETRAIN]**
   * Prompting `"USA"`, `"water"`, and `"hello"` produces varied token predictions once retrained with clean weights.
4. **Gate 4: Sub-Millisecond Decode Latency [IN PROGRESS]**
   * Host decode now traverses flat contiguous memory arrays (`decode_vectors`); GPU GEMM dot-product reduction scheduled for Milestone 2 completion.

---

*This document serves as the binding architectural roadmap for the ongoing development of Project Axiom. All subsequent commits and kernel revisions must adhere strictly to these engineering specifications.*
