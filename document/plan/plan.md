# Project Axiom: Master Engineering Roadmap (As of 19 October 2026)

**Document Path:** `document/plan/plan.md`  
**Target Hardware:** NVIDIA Ada Lovelace (`sm_89` / RTX 4080) & Native C++23 / CUDA  
**Status:** Active Engineering Directive — Future Work Exclusively  
**Historical Archive:** `document/plan/archive_plan_pre_19_oct.md` & `document/plan/Project_Axiom_Plan.md`

---

## 1. Executive Baseline & Scope Directive

All foundational Phase 1–7 systems plumbing—including zero-copy pinned DMA transfers, multi-stream double buffering, dense early-fusion codebook ($V = 65,536$), CUTLASS v4.8.0 Tensor Core execution, linear EVT epilogues, branchless VRAM embedding gather/scatter kernels, and atomic AdamW state persistence—is **100% completed, tested, and verified**.

This document defines the **active development roadmap moving forward**. All completed tasks have been removed. The engineering focus is strictly on transforming the single-layer neural nucleus into a production-grade, multi-token autoregressive multimodal generative engine.

---

## 2. Strategic Execution Horizons

```
                            PROJECT AXIOM — FUTURE EXECUTION PHASES
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ HORIZON 1: Multi-Token Autoregressive Generation & Sampling Loop [IMMEDIATE NEXT STEP]  │
  │ • Host generation loop: continuous autoregressive token feedback                       │
  │ • Sampling engine: Greedy ArgMax, Temperature scaling, and Top-K / Top-P sampling      │
  │ • Token streaming interface with <EOS> / length termination                            │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ HORIZON 2: Multi-Head Self-Attention (MHA) & Transformer Blocks                        │
  │ • Linear Q, K, V projections on Tensor Cores via CUTLASS GEMM                          │
  │ • Fused Scaled Dot-Product Attention (SDPA) with causal sequence masking               │
  │ • Residual skip connections & SwiGLU / MLP feed-forward blocks                         │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ HORIZON 3: Production VRAM & PagedAttention KV-Cache State Management                  │
  │ • Zero-allocation runtime: Stream memory pools (`cudaMallocAsync`) & BFC Allocator     │
  │ • PagedAttention: CPU-managed Block Table mapping non-contiguous VRAM blocks           │
  │ • Copy-On-Write (COW) prompt prefix caching for multi-turn conversations               │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ HORIZON 4: Zero-Overhead Hardware Saturation via CUDA Graphs                           │
  │ • Static Directed Acyclic Graph (DAG) capture with `cudaStreamBeginCapture`            │
  │ • Single-driver launch via `cudaGraphLaunch` (eliminating CPU scheduling latency)      │
  │ • Thermodynamic compute and memory bus saturation on sm_89                             │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ HORIZON 5: Modular C++ ATen Dispatcher & Autograd Tape Recording                       │
  │ • Decouple `Tensor` handles from underlying `TensorImpl` memory storage                │
  │ • Central C++ Dispatcher based on `DispatchKeySet` (CPU, CUDA, Autograd)               │
  │ • Automated reverse-mode computational graph construction                              │
  └───────────────────────────────────────────────────┬────────────────────────────────────┘
                                                      │
  ┌───────────────────────────────────────────────────▼────────────────────────────────────┐
  │ HORIZON 6: Sparse Mixture of Experts (MoE) Architecture                                │
  │ • Partition dense feed-forward blocks into 8 specialized expert weight stacks          │
  │ • Lightweight Router Network dispatching incoming tokens to Top-1 / Top-2 experts      │
  │ • Zero-FLOP inactive expert pruning for sub-millisecond multi-billion scaling          │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Horizon Specifications & Engineering Tasks

### Horizon 1: Multi-Token Autoregressive Generation & Sampling Loop (Immediate Priority)

#### 1.1. Host Generation Loop (`core/src/ai.cpp`, `core/src/main.cpp`)
* **Objective:** Expand the current single-token `predict()` function into an iterative streaming generator:
  ```cpp
  bool AI::generate(const std::string& prompt,
                    size_t max_new_tokens,
                    float temperature,
                    int top_k,
                    float top_p,
                    std::function<void(const std::string&)> token_callback);
  ```
* **Execution Flow:**
  1. Tokenize prompt into initial token ID sequence.
  2. Format into sliding 128-token context window (`tokens_to_context_window` or token ID buffer).
  3. Execute forward pass with tied logits ($Y \cdot E^T$).
  4. Fetch winning or sampled token ID from GPU.
  5. Decode token ID to string via `token_to_string()` and stream to caller via callback.
  6. Append token to history, slide context window by 1 position, and iterate until `vocab::EOS` or `max_new_tokens`.

#### 1.2. Stochastic Sampling Engine
* **Temperature Scaling:** Scale logits prior to softmax:
  $$\hat{z}_i = \frac{z_i}{T} \quad (T \in (0.0, 2.0])$$
* **Top-K Truncation:** Retain only the $K$ highest logit values; set all others to $-\infty$.
* **Top-P (Nucleus) Truncation:** Sort probabilities descending, find cutoff index where cumulative sum exceeds $P$, and re-normalize.
* **Deterministic Fallback:** $T \to 0$ collapses into greedy ArgMax via the existing GPU `softmax_argmax_kernel`.

---

### Horizon 2: Multi-Head Self-Attention (MHA) & Transformer Blocks

#### 2.1. Linear Projections on Tensor Cores
* Transform hidden states into Query, Key, and Value representations:
  $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
  where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.
* Execute on Ada Lovelace Tensor Cores via batched CUTLASS GEMMs in `__nv_bfloat16`.

#### 2.2. Scaled Dot-Product Attention & Causal Masking
* Implement fused attention kernel:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}} + M\right) V$$
  where $M_{i, j} = 0$ for $j \le i$, and $-\infty$ for $j > i$ (preventing attention to future tokens).

#### 2.3. Transformer Layer Composition
* **Residual Connections:** $X_{\text{mid}} = X + \text{Attention}(\text{RMSNorm}(X))$.
* **Feed-Forward SwiGLU Block:**
  $$\text{FFN}(x) = (\text{Swish}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$$
* **Layer Output:** $X_{\text{out}} = X_{\text{mid}} + \text{FFN}(\text{RMSNorm}(X_{\text{mid}}))$.

---

### Horizon 3: Production VRAM & PagedAttention State Management

#### 3.1. Elimination of Runtime Allocations
* Eliminate all dynamic `cudaMalloc` and `cudaFree` invocations during forward/backward execution loops.
* **Stream-Ordered Allocation:** Utilize `cudaMallocAsync` and `cudaFreeAsync` bound to `gpu_portion->stream[i]` to recycle memory within CUDA-managed pools without host synchronization stalls.
* **Userspace BFC Allocator:** Pre-allocate fixed multi-gigabyte VRAM blocks at engine startup; service tensor allocations via Best-Fit with Coalescing in userspace.

#### 3.2. PagedAttention KV-Cache Architecture
* Partition Key and Value cache memory into fixed-size physical blocks (e.g., 16 tokens per block).
* Implement CPU-side **Block Table** mapping logical token indices $(0 \dots T-1)$ to physical VRAM blocks.
* Enable **Copy-On-Write (COW)** prefix caching: common system prompts and multi-turn conversational histories share identical physical blocks in VRAM, eliminating redundant memory consumption.

---

### Horizon 4: Zero-Overhead Hardware Saturation via CUDA Graphs

#### 4.1. DAG Graph Capture
* Once memory allocations are static and pointer topologies are fixed:
  ```cpp
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  // Launch forward gather, CUTLASS GEMM, Softmax, Loss, and Backward pass
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  ```

#### 4.2. Host Driver Overhead Eradication
* Dispatch entire training/inference iterations via single `cudaGraphLaunch(graph_exec, stream)` calls.
* Offload intra-kernel scheduling directly to hardware work-distributors on Ada Lovelace (`sm_89`), achieving sub-microsecond launch latencies.

---

### Horizon 5: Modular C++ ATen Dispatcher & Autograd Tape

#### 5.1. Tensor Abstraction
* Refactor monolithic data arrays into clean `Tensor` handles wrapping reference-counted `TensorImpl` buffers.
* Store stride metadata, shapes, and device tags (`Device::CPU`, `Device::CUDA`) to enable zero-copy slicing, views, and transpositions.

#### 5.2. Dispatch Architecture & Tape Recording
* Route mathematical operations through a central C++ dispatcher keyed by `DispatchKeySet`.
* Autograd interceptor records operation nodes and saved tensors on a reverse computational graph tape during forward execution, enabling automated multi-layer backpropagation.

---

### Horizon 6: Sparse Mixture of Experts (MoE) Architecture

#### 6.1. Expert Partitioning
* Decompose dense FFN layers into $N$ discrete Expert weight stacks (e.g., $N = 8$).
* Maintain independent weight parameters for each expert in contiguous VRAM regions.

#### 6.2. Gating & Router Network
* Implement a lightweight routing layer:
  $$G(x) = \text{TopK}(\text{Softmax}(x W_g), k=2)$$
* Incoming token representations route exclusively to the top 2 ranked experts.
* Inactive experts consume zero Tensor Core arithmetic FLOPs, providing multi-billion parameter capacity while maintaining sub-millisecond per-token latency.

---

## 4. Verification Gates & Acceptance Criteria

Every milestone implementation must pass the following non-negotiable verification gates:

| Verification Gate | Target Metric | Verification Method |
| :--- | :--- | :--- |
| **Gate 1: Multi-Token Generation** | $>50\text{ tokens/sec}$ continuous streaming | Interactive prompt CLI test in `main.cpp` |
| **Gate 2: Sampling Entropy** | $T > 0$ yields diverse, grammatically valid continuations | Deterministic seed tests across varied temperatures |
| **Gate 3: Zero Runtime Allocations** | Exactly $0$ calls to `cudaMalloc` during active loops | NVIDIA Nsight Systems trace & CUDA memory profiler |
| **Gate 4: Attention Layer Convergence** | Monotonic loss reduction on causal language modeling | Multi-step loss validation in `test.exe` |
| **Gate 5: CUDA Graph Latency** | $<0.05\text{ ms}$ host scheduling overhead per step | GPU event timing (`cudaEventElapsedTime`) |
