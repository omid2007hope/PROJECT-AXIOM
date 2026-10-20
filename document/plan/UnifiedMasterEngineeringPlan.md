# Project Axiom: Unified Master Engineering Plan

```
   █████╗ ██╗  ██╗██╗ ██████╗ ███╗   ███╗
  ██╔══██╗╚██╗██╔╝██║██╔═══██╗████╗ ████║
  ███████║ ╚███╔╝ ██║██║   ██║██╔████╔██║
  ██╔══██║ ██╔██╗ ██║██║   ██║██║╚██╔╝██║
  ██║  ██║██╔╝ ██╗██║╚██████╔╝██║ ╚═╝ ██║
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝     ╚═╝
  BARE-METAL C++23 & CUDA DEEP LEARNING ENGINE
```

| Specification | Value |
| :--- | :--- |
| **Document Path** | `document/plan/UnifiedMasterEngineeringPlan.md` |
| **Target Architecture** | NVIDIA Ada Lovelace (`sm_89` / RTX 4080 Laptop GPU) |
| **Toolchains & Standards** | Native C++23 (`-std=c++23`, GCC/MinGW 14.2), CUDA 13.3 (`nvcc`, MSVC `cl.exe`), CUTLASS v4.8.0 |
| **Baseline Status** | **Foundations Locked (146/146 Unit & Stress Tests Passing — 100% Green)** |
| **Execution Directive** | Single Source of Truth — Active Scaling & Multi-Token Generative Roadmap |

---

## Executive Summary

This **Unified Master Engineering Plan** consolidates all architectural findings from the forensic codebase audit, `Project_Axiom_Plan.md`, `research.md`, and `plan.md` into a single, authoritative engineering blueprint. 

It formally locks the verified hardware substrate and mathematical foundation, integrates the surgical codebase enhancements uncovered during forensic analysis, and sequences the six future execution phases in strict order of dependency.

---

## Table of Contents

1. [Verified Baseline (Locked Foundations — Do Not Re-Open)](#1-verified-baseline-locked-foundations--do-not-re-open)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Unified Execution Roadmap](#3-unified-execution-roadmap)
   - [Phase 1: Multi-Token Autoregressive Generation, Sampling & Corpus Hygiene](#phase-1-multi-token-autoregressive-generation-sampling--corpus-hygiene-immediate-priority)
   - [Phase 2: Surgical Sync Eradication & Production VRAM Management](#phase-2-surgical-sync-eradication--production-vram-management)
   - [Phase 3: Multi-Head Self-Attention (MHA / GQA) & Transformer Blocks](#phase-3-multi-head-self-attention-mha--gqa--transformer-blocks)
   - [Phase 4: PagedAttention KV-Cache & Prefix State Management](#phase-4-pagedattention-kv-cache--prefix-state-management)
   - [Phase 5: Zero-Overhead Hardware Saturation & ATen Dispatcher](#phase-5-zero-overhead-hardware-saturation--aten-dispatcher)
   - [Phase 6: Sparse Mixture of Experts (MoE) & Multimodal Output Codecs](#phase-6-sparse-mixture-of-experts-moe--multimodal-output-codecs)
4. [Unified Verification Gates & Acceptance Protocol](#4-unified-verification-gates--acceptance-protocol)

---

## 1. Verified Baseline (Locked Foundations — Do Not Re-Open)

> [!IMPORTANT]
> All foundational systems plumbing, memory logistics, and single-layer mathematical operations listed below are **100% implemented, verified, and passing (146/146 tests green)** across `test_ai`, `test_inference`, `test_identifier`, and `test_decompressor`. These components are locked and must not be reopened or refactored arbitrarily.

| Subsystem | Verified Implementation Architecture | Key Code Locations |
| :--- | :--- | :--- |
| **Multimodal Ingestion** | Four-Stage Pipeline (`Identifier` $\to$ `Decompressor` $\to$ `Sanitizer` $\to$ `Tokenizer` $\to$ `Embedder`); unified text and numbers in `main_dictionary`; raw manual CLI text bypasses file decompression directly into tokenization. | [`Identifier.hpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/Identifier.hpp), [`identifier.cpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/util/identifier.cpp), [`decompress_*.cpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/util/decompressor/), [`parser.cpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/util/parser.cpp) |
| **Dense Bounded Codebook** | Fixed 16-bit address space ($V = 65,536$) partitioned into Special tokens (`[0..3]`), Text/Numbers (`[4..64,511]`), Visual bytes (`[64,512..64,767]`), and Audio bytes (`[64,768..65,023]`). Sparse billion-scale offsets completely eradicated. | [`parser.hpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/parser.hpp), [`parser.cpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/util/parser.cpp) |
| **Zero-Copy PCIe DMA** | Pinned `cudaHostAllocWriteCombined` buffers, dual non-blocking streams (`stream[0]`, `stream[1]`), seam carry-forward across 100k-token chunks, and 512-byte token ID PCIe streaming (93.75% bus reduction). | [`optimizer.hpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/optimizer.hpp), [`optimizer.cpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/util/optimizer.cpp), [`upload.cu`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/gpu/upload.cu) |
| **VRAM Embeddings** | VRAM-resident `__nv_bfloat16` table (`d_table`) and FP32 master weights (`d_table_master`); branchless `gather_embeddings_kernel` and `backward_scatter_embeddings_kernel` with `atomicAdd`. | [`gpu_types.cuh`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/util/gpu_types.cuh), [`forward_pass.cu`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/kernel/forward_pass.cu), [`backward_pass.cu`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/kernel/backward_pass.cu) |
| **Tensor Core Compute** | CUTLASS v4.8.0 GEMM (`sm_89`) with Linear Identity EVT epilogue (`Store(AddBias(Accum, Bias))`) and un-gated backward pass ($dz = dy$), permanently curing the Dying ReLU collapse. | [`forward_pass.cu`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/kernel/forward_pass.cu), [`backward_pass.cu`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/kernel/backward_pass.cu) |
| **Autoregressive Kernels** | Contiguous RAM decode mirror (`decode_ids`, `decode_vectors`), GPU `rms_norm_forward_kernel`, `compute_tied_logits_kernel` ($Y \cdot E^T$), `softmax_argmax_kernel`, and fused Cross-Entropy (`compute_loss_and_dz_kernel`). | [`forward_pass.cu`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/kernel/forward_pass.cu), [`backward_pass.cu`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/kernel/backward_pass.cu), [`parser.cpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/util/parser.cpp) |
| **State Persistence** | Fused `backward_pass_adamw_kernel` maintaining momentum ($m_t$) and variance ($v_t$); deterministic 56-byte `AxiomWeightHeader` atomic swaps for `weights.axiom`, `vocab.axiom`, and `optimizer.axiom`. | [`ai.cpp`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp), [`backward_pass.cu`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/CUDA/kernel/backward_pass.cu) |

---

## 2. End-to-End System Architecture

```
                                      PROJECT AXIOM ARCHITECTURE
                                      
  [RAW ASSETS]                [STAGE 1: IDENTIFIER]           [STAGE 2: DECOMPRESSOR]
  ┌───────────────┐           ┌───────────────────┐           ┌─────────────────────┐
  │ PDF, DOCX, TXT│ ────────> │ Magic Byte & Ext  │ ────────> │ Decom_doc / Image / │
  │ PNG, JPG, MP4 │           │ Switchboard Check │           │ Audio / Video Frame │
  └───────────────┘           └───────────────────┘           └──────────┬──────────┘
                                                                         │ Pure UTF-8 / Uncompressed Bytes
  [RAW CLI PROMPT] ──────────────────────────────────────────────────────┘
         │
         ▼
  [STAGE 3: SANITIZER & TOKENIZER]
  ┌────────────────────────────────────────────────────────┐
  │ • Lowercase ASCII & punctuation stripping              │
  │ • Dense Early-Fusion mapping into V = 65,536 codebook  │
  └──────────────────────────────┬─────────────────────────┘
                                 │ 128 uint32_t Token IDs (512 Bytes)
                                 ▼
  [HOST-DEVICE PCIe DMA (STREAM 0 & 1 PING-PONG)]
  ┌────────────────────────────────────────────────────────┐
  │ • Allocated via cudaHostAllocWriteCombined             │
  │ • Bypasses L1/L2 host CPU cache directly to PCIe bus   │
  └──────────────────────────────┬─────────────────────────┘
                                 │ Non-blocking DMA Transfer
                                 ▼
  [DEVICE VRAM TENSOR ENGINE (ADA LOVELACE sm_89)]
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. Branchless Gather Kernel: Expand 128 Token IDs -> [128 x 32] via d_embedding_table   │
  │ 2. GPU RMSNorm: Normalize variance to 1.0 with variance clamp [-10.0, 10.0]           │
  │ 3. CUTLASS v4.8.0 GEMM: Tensor Cores evaluate Y = X * W^T + b via Linear Identity EVT  │
  │ 4. Tied Logits Projection: Compute logits = Y * E^T across active vocabulary           │
  │ 5. Fused GPU Softmax & ArgMax: Compute probability distribution & winning token ID     │
  │ 6. Cross-Entropy Backward: Exact bounded logit gradient dz = P - 1[target]             │
  │ 7. Fused CUDA AdamW: Update master weights & biases with bias-corrected momentum       │
  │ 8. Backward Scatter: atomicAdd gradient updates back into master embedding table       │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Unified Execution Roadmap

### Phase 1: Multi-Token Autoregressive Generation, Sampling & Corpus Hygiene (Immediate Priority)

> [!NOTE]
> **Priority Target:** Transition from single-token prompt evaluation to full, continuous generative streaming with temperature and nucleus sampling.

#### 1.1. Host Autoregressive Generation Loop (`core/src/ai.cpp`, `core/src/main.cpp`)
* **Objective:** Upgrade the single-token `predict()` pathway into an iterative, streaming autoregressive loop.
* **Interface Signature:**
  ```cpp
  bool AI::generate(const std::string& prompt,
                    size_t max_new_tokens,
                    float temperature,
                    int top_k,
                    float top_p,
                    std::function<void(const std::string&)> token_callback);
  ```
* **Execution Steps:**
  1. Sanitize and tokenize the input prompt into initial token IDs.
  2. Populate the 128-token sliding context window buffer (`tokens_to_context_window` or token ID buffer).
  3. Run the forward pass with RMSNorm and tied logits projection ($Y \cdot E^T$).
  4. Sample the next token ID from GPU probability distribution, decode via `token_to_string()`, and invoke `token_callback` for real-time CLI terminal streaming in `main.cpp`.
  5. Append the generated token ID, slide the context window forward by 1 position (FIFO roll), and repeat until `vocab::EOS` is emitted or `max_new_tokens` is reached.

#### 1.2. Stochastic Sampling Engine (`core/src/CUDA/kernel/forward_pass.cu`, `core/src/ai.cpp`)
* **Temperature Scaling:** Apply logit scaling prior to probability normalization:
  $$\hat{z}_i = \frac{z_i}{T} \quad (T \in (0.0, 2.0])$$
* **Top-K Filtering:** Isolate the $K$ highest logits and mask all remaining indices to $-\infty$.
* **Top-P (Nucleus) Sampling:** Sort probabilities in descending order, compute the cumulative distribution, mask tokens once cumulative probability exceeds $P$, and re-normalize.
* **Greedy Fallback:** When $T \to 0$, bypass stochastic sampling and execute deterministic ArgMax directly via the existing `softmax_argmax_kernel`.

#### 1.3. Vocabulary Bounding & Corpus Hygiene (`core/util/parser.cpp`)
* **Frequency Filtering:** Enforce minimum occurrence thresholds ($\text{count} \ge 3$) during bulk corpus ingestion so rare typos and noise strings do not consume finite slots in the 64,508-word `main_dictionary` partition.
* **Fallback Routing:** Automatically remap low-frequency outliers and out-of-bounds indices to `vocab::UNK` (ID `1`).

---

### Phase 2: Surgical Sync Eradication & Production VRAM Management

> [!WARNING]
> **Forensic Audit Liability Notice:** The forensic codebase audit identified three runtime synchronization and allocation bottlenecks in the existing codebase. Resolving these prior to multi-layer scaling prevents catastrophic throughput stalls in subsequent phases.

#### 2.1. Eradicate Dynamic `cudaMalloc` in Transfer Loops (`core/src/CUDA/gpu/upload.cu`)
* **Audit Liability:** Lines 20–28 of `upload.cu` dynamically check element sizes and invoke `cudaMalloc`/`cudaFree` inside `upload_to_gpu`, triggering a device-wide CPU-GPU synchronization barrier.
* **Surgical Fix:** Remove runtime allocation guards from the active transfer loop. Transition temporary buffers to stream-ordered memory pools (`cudaMallocAsync` / `cudaFreeAsync` bound to `gpu_portion->stream[i]`) and implement a userspace **Best-Fit with Coalescing (BFC) Caching Allocator** that carves tensor pointers from a pre-allocated multi-gigabyte VRAM arena at startup.

#### 2.2. Non-Blocking Loss Telemetry (`core/src/CUDA/gpu/download.cu`)
* **Audit Liability:** `download_loss` invokes `cudaDeviceSynchronize()`, stalling both asynchronous streams across the entire GPU just to read a single `float`.
* **Surgical Fix:** Copy the scalar loss asynchronously via `cudaMemcpyAsync` into a dedicated 4-byte pinned host slot and poll/query a stream-specific `cudaEvent_t` (`cudaEventQuery`), preserving uninterrupted double-buffered compute.

#### 2.3. Lock-Free SPSC Ring Buffer & Async Disk Prefetch (`core/util/optimizer.cpp`)
* **Audit Liability:** `Portion::buffer_in_use` and `Portion::context_in_use` spin-wait in synchronous polling loops, and exhausting the 100,000-token buffer causes momentary GPU starvation while waiting synchronously for `disk_to_ram`.
* **Surgical Fix:** Replace busy-wait spin-locks and synchronous chunk reloads with a lock-free Single-Producer Single-Consumer (SPSC) ring buffer backed by a background producer thread that prefetches the next 100,000-token chunk before the active chunk drains.

---

### Phase 3: Multi-Head Self-Attention (MHA / GQA) & Transformer Blocks

#### 3.1. Batched Tensor Core Projections & Backward Pass Upgrade
* **Forward Projections:** Project normalized hidden states into Query, Key, and Value tensors using batched CUTLASS v4.8.0 GEMMs in `__nv_bfloat16`:
  $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad \left(W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\right)$$
* **Backward Pass Upgrade (`core/src/CUDA/kernel/backward_pass.cu`):** Replace the memory-bound Rank-1 outer product kernel ($M = 1$ on standard CUDA cores) with batched CUTLASS GEMM backward passes across the full sequence/batch dimension so gradient computation fully saturates Ada Lovelace Tensor Cores.

#### 3.2. Fused Scaled Dot-Product Attention (SDPA) & Causal Masking
* Implement a fused FlashAttention-style CUDA kernel computing:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}} + M\right) V$$
  where causal mask $M_{i, j} = 0$ for $j \le i$ and $-\infty$ for $j > i$, preventing future token leakage across the sequence.

#### 3.3. Transformer Block Composition & SwiGLU FFN
* **Pre-Norm Residual Stream:**
  $$X_{\text{mid}} = X + \text{Attention}(\text{RMSNorm}(X))$$
* **SwiGLU Feed-Forward Network:**
  $$\text{FFN}(x) = \left(\text{Swish}(x W_{\text{gate}}) \odot x W_{\text{up}}\right) W_{\text{down}}$$
* **Block Output:**
  $$X_{\text{out}} = X_{\text{mid}} + \text{FFN}(\text{RMSNorm}(X_{\text{mid}}))$$

---

### Phase 4: PagedAttention KV-Cache & Prefix State Management

#### 4.1. Fixed-Size Physical KV Blocks
* Partition VRAM Key and Value caches into uniform physical pages holding 16 tokens per block, eliminating internal and external VRAM fragmentation during variable-length generation.

#### 4.2. CPU Block Table & Copy-On-Write (COW) Prefix Caching
* Maintain a CPU-side Block Table mapping logical sequence positions $(0 \dots T-1)$ to physical VRAM block pointers.
* Implement Copy-On-Write semantics so shared system prompts and multi-turn conversation histories reference identical physical KV blocks in VRAM, duplicating a page only when token trajectories diverge.

---

### Phase 5: Zero-Overhead Hardware Saturation & ATen Dispatcher

#### 5.1. Static DAG Execution via CUDA Graphs
* With dynamic `cudaMalloc` calls eliminated in Phase 2 and static buffer topologies locked, capture the entire forward/backward pipeline into a single hardware Directed Acyclic Graph (DAG):
  ```cpp
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  // Enqueue gather, RMSNorm, Attention, SwiGLU, Tied Logits, Loss, Backward, AdamW
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  ```
* Execute complete iterations via a single `cudaGraphLaunch(graph_exec, stream)` call, reducing host driver launch overhead below $0.05\text{ ms}$.

#### 5.2. Modular C++ ATen Dispatcher & Autograd Tape
* **Tensor Handle Decoupling:** Separate lightweight `Tensor` handles from reference-counted `TensorImpl` storage containing shape, stride, and device metadata (`Device::CPU`, `Device::CUDA`) for zero-copy views and transpositions.
* **Central Dispatcher:** Route all tensor operations through a `DispatchKeySet` table that intercepts `Autograd` keys, records operation nodes onto a reverse-mode computational tape, and dispatches down to the underlying CUTLASS/CUDA kernels.

---

### Phase 6: Sparse Mixture of Experts (MoE) & Multimodal Output Codecs

#### 6.1. Sparse Mixture of Experts (MoE) Layers
* Partition dense SwiGLU FFN layers into $N = 8$ discrete expert weight stacks in contiguous VRAM.
* Implement a lightweight gating router network:
  $$G(x) = \text{TopK}(\text{Softmax}(x W_g), k=2)$$
* Route each token exclusively to its top 2 experts so inactive experts consume zero Tensor Core arithmetic FLOPs.

#### 6.2. Multimodal Output Codecs (`core/util/decompressor/` & Compressor Mirrors)
* Complete the output synthesis counterparts to the Phase 1 decompressors:
  * **Visual Codec:** Unroll generated discrete visual tokens (`[64,512..64,767]`) into spatial RGB pixel buffers and serialize to BMP/MP4 containers.
  * **Acoustic Codec:** Unroll generated discrete acoustic tokens (`[64,768..65,023]`) into linear PCM waveforms and wrap with standard RIFF `.wav` headers.

---

## 4. Unified Verification Gates & Acceptance Protocol

> [!TIP]
> Every phase must pass its corresponding verification gate in `mingw32-make test` before subsequent phases are unlocked:

| Phase | Milestone Name | Target Success Metric | Primary Verification Method | Status |
| :---: | :--- | :--- | :--- | :---: |
| **0** | **Foundations Locked** | 146/146 unit & stress tests passing | `mingw32-make test` (`test.exe --test`) | **[PASSED]** |
| **1** | **Multi-Token Streaming & Sampling** | $>50\text{ tokens/sec}$; $T > 0$ produces diverse, coherent text without `<UNK>` saturation | Interactive CLI streaming in `main.cpp` & deterministic seed tests in `test_inference.cpp` | **[NEXT]** |
| **2** | **Zero Runtime Syncs & Allocations** | Exactly $0$ calls to `cudaMalloc` or `cudaDeviceSynchronize` during active loops | Code audit of `upload.cu`/`download.cu` & Nsight Systems timeline trace | **[PENDING]** |
| **3** | **Transformer Block Convergence** | Monotonic loss reduction on causal multi-head sequence modeling; Tensor Core backward GEMM active | Multi-step convergence suite in `test_ai.cpp` | **[PENDING]** |
| **4** | **Zero KV Fragmentation** | $100\%$ block reuse across multi-turn prompts via COW prefix sharing | PagedAttention block allocation unit tests | **[PENDING]** |
| **5** | **Sub-Microsecond Graph Dispatch** | $<0.05\text{ ms}$ host launch latency per full step; automated tape backprop matches manual gradients | `cudaEventElapsedTime` profiling & autograd parity tests | **[PENDING]** |
| **6** | **MoE Routing & Multimodal Output** | Top-2 expert activation verified; valid `.bmp` and `.wav` files reconstructed from token streams | Router load-balance test & round-trip codec verification in `test_decompressor.cpp` | **[PENDING]** |
