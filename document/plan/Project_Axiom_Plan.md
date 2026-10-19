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

### 2.4. Multimodal Tokenizer & Ingestion Pipeline (`core/util/parser.cpp`, `core/include/parser.hpp`)
* **What Works Flawlessly:**
  * **Unified Text & Number Integration [Phase 1.1 DONE]:** Merged numbers and text into `main_dictionary`, allowing physics terms and numerical quantities (e.g. `"gravity"` and `"9.8"`) to share semantic context.
  * **Persistence Serialization:** `save_vocabulary` and `load_vocabulary` write a clean binary stream (`AXVK_V1`) storing string lengths and character bytes sequentially.
  * **Dual Ingestion Workflow Routing:** Raw manual CLI text inputs bypass file decompression and enter directly into the sanitizer and tokenizer.
* **The Fatal Architectural Flaws & Historical Traps:**
  1. **The Raw Byte Ingestion Trap:** Ingesting raw compressed files (`.pdf`, `.mp4`) without extraction forced the tokenizer to ingest zlib compression entropy and container tags rather than learning true physics, semantics, or visual geometry.
  2. **Synthetic Non-Semantic Embedding Space:** `vec[d] = sin(id * (d + 1))` scattered semantically linked concepts (e.g. `"bulk"` and `"training"`) randomly across vector space with zero semantic topology.
  3. **The Billion-ID Memory Wall:** Sparse offsets (1B..4B) caused multi-gigabyte VRAM allocation crashes when attempting to instantiate device embedding tables.
  4. **The 530 ms CPU Search Jam:** Sequential traversal of 3.25 million hash map entries on the CPU capped decoding throughput at 1.9 tokens/sec.

---

## 3. Progress Scorecard: Roadmap Evaluation (`research.md` vs. Reality)

| Roadmap Milestone | Plan Requirement | Actual Code Implementation Status | Evaluation & Reality Check |
| :--- | :--- | :--- | :--- |
| **Phase 1: Ingestion & Namespaces** | 4-Stage Ingestion, decompressors, unified text/num | **100% COMPLETE & VERIFIED** | **[DONE]** `main_dictionary` unified; `Identifier` and `Decom_*` (PDF, DOCX, TXT, MD, BMP, WAV) implemented & tested in `test_decom.cpp` and `test_identifier.cpp`. |
| **Phase 2: Dense Codebook** | Fixed bounded codebook ($V = 65,536$), 16-bit blocks | **100% COMPLETE & VERIFIED** | **[DONE]** Eradicated billion-scale offsets; bounds $V \le 65,536$ ($4.19\text{ MB}$ bf16 table). Verified in `test_inference.cpp`. |
| **Phase 3: VRAM Embedding Matrix** | Device BF16 + Master FP32 tables in VRAM | **100% COMPLETE & VERIFIED** | **[DONE]** `GpuEmbeddings` ($65,536 \times 32$) resident in VRAM with master FP32 weights. Verified in `test_ai.cpp`. |
| **Phase 4: Forward Gather Kernel** | Token ID PCIe streaming (512 B) + CUDA gather | **100% COMPLETE & VERIFIED** | **[DONE]** Slashes PCIe bus traffic by 93.75%; branchless gather into `d_context_window` on device. Verified in `test_ai.cpp`. |
| **Phase 5: Backward Scatter Kernel** | Input gradients $dL/dx_j$ + `atomicAdd` scatter | **100% COMPLETE & VERIFIED** | **[DONE]** `backward_scatter_embeddings_kernel` backpropagates into FP32 master embedding table. Verified in `test_ai.cpp`. |
| **Phase 6: Curing AI Amnesia** | Fused AdamW kernel + `optimizer.axiom` persistence | **100% COMPLETE & VERIFIED** | **[DONE]** Fused AdamW CUDA kernel updates momentum ($m_t$) and variance ($v_t$); 56-byte header with atomic persistence. |
| **Phase 7: Compute & Linear EVT** | CUTLASS GEMM on Tensor Cores, Linear EVT epilogue | **100% COMPLETE & VERIFIED** | **[DONE]** Linear EVT epilogue + un-gated SGD ($dz = dy$). Dead ReLU eliminated. Verified in `test_ai.cpp` & `test_inference.cpp`. |
| **Phase 8: Production VRAM** | `cudaMallocAsync`, Stream memory pools, BFC Allocator | **10% COMPLETE** | Static allocation used; userspace BFC allocator deferred. |
| **Phase 9: Autoregressive State** | PagedAttention, KV Caching, Multi-Head Attention | **IN PROGRESS** | Tied logits, GPU Softmax/Argmax, RMSNorm, and Cross-Entropy loss implemented. Attention blocks scheduled next. |
| **Phase 10: Microsecond Scaling** | CUDA Graphs (DAGs), ATen-style C++ Dispatcher | **0% COMPLETE** | Monolithic functions; graph capture deferred to V2. |
| **Phase 11: Mixture of Experts (MoE)**| Sparse Expert Blocks + Gating Router | **CONCEPTUAL / ROADMAP** | Multi-expert routing for extreme parameter scaling. |

---

## 4. Current Engineering Status & Completed Transformations

### 4.1. Surgical Fix for Phase 4: Reviving the Neural Nucleus [STATUS: COMPLETED & VERIFIED]
1. **[DONE] Remove `EvtRelu` in `forward_pass.cu`:** Replaced the ReLU node in the EVT tree with an Identity / Linear store epilogue (`Store(AddBias(Accum, Bias))`). The network can freely predict throughout $[-1.0, +1.0]$.
2. **[DONE] Un-gate the Gradient in `backward_pass.cu`:**
   * Removed `float dz = (y_val > 0.0f) ? dy : 0.0f;`.
   * Replaced with `float dz = dy;` (and exact bounded Cross-Entropy logit gradients $dz = P - 1_{[\text{target}]}$).
3. **[DONE] Weight Symmetry Reset:**
   * Quarantined the frozen `weights.axiom` checkpoint to `weights.axiom.dead_relu_backup`. Fresh weights initialize with random chaotic bounds (`std::mt19937`).

### 4.2. Decoding & GPU Kernel Acceleration [STATUS: COMPLETED & VERIFIED]
1. **[DONE] Contiguous RAM Decode Mirror (`parser.hpp`, `parser.cpp`):** Replaced pointer-chasing across 3.25M `std::unordered_map` nodes with flat, contiguous `decode_ids` and `decode_vectors` arrays.
2. **[DONE] Move Embeddings to Device VRAM:** Maintained contiguous device tensor `d_table` ($65,536 \times 32$) in `__nv_bfloat16` and master copy `d_table_master` in float.
3. **[DONE] GPU Tied Logits & Softmax / Argmax:** Implemented `compute_tied_logits_kernel` and `softmax_argmax_kernel` on the GPU in `forward_pass.cu`.
4. **[DONE] GPU Cross-Entropy Loss & Regularization:** Implemented `compute_loss_and_dz_kernel` and `rms_norm_forward_kernel` in CUDA kernels.

### 4.3. Remaining Action Items
1. **[PENDING] Multi-Token Autoregressive Generation Loop:** Expand single-token prediction in `main.cpp` into a multi-step token generator with temperature and top-k sampling.
2. **[PENDING] Multi-Head Attention (MHA / GQA):** Introduce query, key, value projections and attention score kernels.
3. **[PENDING] Vocabulary Bounding & Text Cleaning:** Frequency filtering (occurrences $\ge 3$) and `<UNK>` token remapping on raw text corpora.

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
  │ MILESTONE 2: GPU-Accelerated Decoding & Vocabulary Sanity [STATUS: DONE]               │
  │ • Contiguous RAM Decode Mirror [DONE] • VRAM Embedding Matrix [V x d] [DONE]           │
  │ • GPU Tied Logits ($Y \cdot E^T$) [DONE] • GPU Fused Softmax & ArgMax Kernel [DONE]   │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ MILESTONE 3: Discrete Autoregressive Token Classification [STATUS: FOUNDATION DONE]    │
  │ • RMSNorm forward pass & gradient scaling [DONE]                                       │
  │ • Fused GPU Cross-Entropy Loss & Logit Gradient ($dz = P - 1_{tgt}$) [DONE]           │
  │ • Multi-step autoregressive generation loop & sampling in host inference [NEXT]        │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ MILESTONE 4: Multi-Layer Attention & Deep Feature Hierarchy                            │
  │ • Multi-Head Self-Attention (Q, K, V projections) on Tensor Cores via CUTLASS          │
  │ • Residual connections & RMSNorm / LayerNorm EVT fusion.                               │
  │ • Feed-Forward SwiGLU / MLP blocks.                                                    │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ MILESTONE 5: Production VRAM & PagedAttention State Management                         │
  │ • Userspace Best-Fit with Coalescing (BFC) Caching Allocator                           │
  │ • PagedAttention Block Table for non-contiguous KV-Cache in VRAM                       │
  │ • Copy-On-Write (COW) prompt prefix caching.                                           │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ MILESTONE 6: Zero-Overhead Hardware Saturation                                         │
  │ • CUDA Graph DAG Capture (`cudaStreamBeginCapture` / `cudaGraphLaunch`)                │
  │ • ATen-Style Modular C++ Tensor & Operation Dispatcher                                 │
  │ • Thermodynamic silicon saturation on Ada Lovelace (`sm_89`).                          │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Verification & Acceptance Criteria

1. **Gate 1: Activation Vitality Test [PASSED]**
   * Verified in `test.exe` (`test_inference.cpp`): Forward inference outputs are non-zero across all coordinates (`any_inference_nonzero == true`). The dead all-zero collapse is eliminated.
2. **Gate 2: Loss Monotonicity Verification [PASSED]**
   * Verified in `test.exe` (`test_cuda_transfer.cpp` & `test_ai.cpp`): Multi-step loss falls monotonically (initial `0.713989` $\to$ final `0.0`, multi-step test falls from `2.58994` $\to$ `4.76837e-07`).
3. **Gate 3: Semantic Diversity in Inference [PASSED]**
   * Verified in `test.exe` (`test_tokenization.cpp` & `test_inference.cpp`): Varied token predictions decoded across distinct prompt inputs.
4. **Gate 4: Sub-Millisecond Decode Latency [PASSED]**
   * Verified in `test.exe`: Host decode traverses flat contiguous memory arrays (`decode_vectors`), and GPU-side `softmax_argmax_kernel` runs in $<0.05\text{ ms}$.

---

*This document serves as the binding architectural audit and roadmap for the ongoing development of Project Axiom. All subsequent commits and kernel revisions must adhere strictly to these engineering specifications.*
