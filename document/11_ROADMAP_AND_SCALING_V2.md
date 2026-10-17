# 11. Roadmap and Scaling Blueprint (Version 2.0)

With the completion of **Version 1.0**, Project Axiom has achieved foundational silicon sovereignty: bare-metal C++23/CUDA execution, 100% Tensor Core saturation via CUTLASS v4.8.0 EVT, atomic binary weight persistence (`weights.axiom`), real-time offline inference, and a 200-test verification harness with zero VRAM leaks.

This document outlines the strategic engineering roadmap to scale Project Axiom into a production-grade, modular deep learning framework capable of serving multi-billion parameter autoregressive omni-models.

---

## 1. Phase I: Ingestion & Modality Unification (The Four-Stage Pipeline)

### The Problem
Ingesting raw compressed bytes (zlib in PDFs, MP4 container headers) directly into a tokenizer is a critical trap: the neural network burns its capacity attempting to learn decompression algorithms rather than actual physics, semantics, or visual geometry. Furthermore, isolating numbers from text with billion-scale offsets fractures linguistic context.

### The Solution: Specialized Decompressors & Unified Namespaces
1. **Four-Stage Multimodal Pipeline:**
   $$\text{Identifier (Magic Bytes)} \longrightarrow \text{Decompressor (Strip Wrapper)} \longrightarrow \text{Sanitizer} \longrightarrow \text{Tokenizer} \longrightarrow \text{Embedder}$$
2. **Specialized Decompressor Modules:**
   - [`decom_doc`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/decompressor/decom_doc.hpp): Extracts clean UTF-8 text and equations from `.pdf` and `.docx`.
   - [`decom_image`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/decompressor/decom_image.hpp): Decodes container formats into raw uncompressed RGB pixel arrays.
   - [`decom_video`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/decompressor/decom_video.hpp): Strips container headers and frame encodings into raw RGB frame matrices.
   - [`decom_audio`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/decompressor/decom_audio.hpp): Strips container wrappers into raw PCM audio waveforms.
3. **Dual Ingestion Workflows:** Raw manual CLI text inputs completely bypass the Identifier and Decompressor, flowing straight into character sanitization and tokenization.
4. **Unified Text & Number Codebook:** Eliminates numerical offsets; words and numbers reside together in `main_dictionary`, allowing the network to learn relationships between concepts and quantities (e.g. `"gravity"` and `"9.8"`).

---

## 2. Phase II: Dense Contiguous Early-Fusion Indexing ($V = 65,536$)

### The Problem
Sparse multi-billion offsets (e.g., $1\text{B}$ for numbers, $2\text{B}$ for images) trigger multi-gigabyte VRAM allocation crashes when attempting to index dense device embedding matrices.

### The Solution: Contiguous 16-Bit Modality Blocks
Axiom constrains the entire multimodal vocabulary to a dense 16-bit addressable range $[0, 65535]$:
- **`[0 .. 3]`:** Special reserved tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).
- **`[4 .. 64,511]`:** Unified language, numbers, math, and subwords (`main_dictionary`).
- **`[64,512 .. 64,767]`:** Discrete visual byte tokens (`0x00 .. 0xFF`).
- **`[64,768 .. 65,023]`:** Discrete acoustic byte tokens (`0x00 .. 0xFF`).
- **`[65,024 .. 65,535]`:** Reserved expansion slots for future modality codebooks.

This caps the VRAM embedding table footprint to a fixed **$4.19\text{ MB}$** in `__nv_bfloat16`.

---

## 3. Phase III: Learnable VRAM Embedding Matrix & CUDA Kernels

### The Problem
Synthetic trigonometric formulas (`sin(id * (d+1))`) produce static, arbitrary coordinates where semantically linked terms (e.g. `"bulk"` and `"training"`) are scattered randomly with zero geometric correlation.

### The Solution: GPU Gather & Scatter Optimization
1. **Device Allocation:** Allocate `d_embedding_table` ($65,536 \times 32$ in `__nv_bfloat16`, $4.19\text{ MB}$) and `d_table_master` ($65,536 \times 32$ in `float`, $8.39\text{ MB}$).
2. **Branchless Forward Gather Kernel:** Host uploads 128 integer token IDs (512 bytes) across PCIe. The GPU gather kernel expands them directly inside VRAM, cutting PCIe bus bandwidth consumption by **93.75%**.
3. **Atomic Backward Scatter Kernel:** Propagates gradients $dL/dx_j$ backward into `d_table_master` via `atomicAdd()`, pulling semantically related tokens into neighboring regions of 32D hyperspace over time.

---

## 4. Phase IV: Curing AI Amnesia (The AdamW Optimizer State)

### The Problem
A standard neural network engine that saves only weights suffers from total amnesia upon restart: it loses the directional momentum and curvature velocity built over thousands of steps, forcing it to slowly rediscover gradient slopes.

### The Solution: Binary Persistence of $m_t$ and $v_t$
1. **Dynamic Moments:** Track first moment $m_t$ (momentum/velocity) and second moment $v_t$ (variance/curvature).
2. **Fused CUDA Kernel:** Execute bias-corrected AdamW parameter updates on the GPU.
3. **`optimizer.axiom` Persistence:** Atomically serialize $m_t$ and $v_t$ tensors alongside weights on checkpointing (`[S]`). Resuming training reloads this experience directly into VRAM, resuming optimization at full speed.

---

## 5. Phase V: Output Codecs / Detokenizers (The Compressor)

### The Problem
Generative models output mathematical sequences of token probabilities, not human-consumable video files or formatted documents.

### The Solution: Inverted Output Codecs
1. **Text Detokenizer:** Maps token IDs $[4, 64511]$ back to UTF-8 strings via `reverse_main_dictionary`.
2. **Visual Output Codec:** Maps discrete visual tokens $[64512, 64767]$ to RGB pixel grids, compressing them into standard formats (BMP, PNG, MP4).
3. **Acoustic Output Codec:** Maps discrete audio tokens $[64768, 65023]$ to PCM waveforms and wraps them in RIFF/WAV audio containers.

---

## 6. Phase VI: Userspace Best-Fit with Coalescing (BFC) Caching Allocator

### The Problem
In scaled multi-layer networks, allocating temporary activation tensors via standard `cudaMalloc()` during forward and backward passes introduces severe latency penalties. `cudaMalloc()` invokes a host operating system kernel call that serializes all active CUDA streams and locks GPU page tables, causing the hardware to idle for microseconds.

```
[Standard cudaMalloc]  ───► OS Kernel Trap ───► Locks Device Page Tables ───► 10-50 us Latency Stall
[BFC Userspace Pool]   ───► Pointer Offset ───► Zero OS Kernel Calls     ───► 5-10 ns Pure Speed
```

### The Solution: Userspace BFC Allocator
1. **Upfront VRAM Reservation:** The engine allocates large contiguous blocks of VRAM (e.g., 8 GB to 14 GB on an RTX 4080) at startup.
2. **Best-Fit with Coalescing (BFC):** Incoming tensor allocation requests are serviced by carving pointers out of the pre-allocated block in userspace, dropping allocation latency from microseconds to nanoseconds.
3. **Neighbor Coalescing:** When a tensor is freed, adjacent free blocks are immediately merged back into a single continuous span, eradicating external heap fragmentation.
4. **`max_split_size_mb` Constraint:** Enforces maximum split size limits to prevent large blocks from shattering into tiny, unusable fragments.

---

## 7. Phase VII: PagedAttention for KV-Cache Scalability

### The Problem
When expanding from fixed context perceptrons to full Generative Pre-trained Transformers (GPTs), autoregressive decoding requires caching Key and Value (KV) projection tensors for every past token. Naively allocating continuous VRAM blocks for the maximum sequence length wastes up to **80% of VRAM** due to internal fragmentation.

### The Solution: Virtual Memory PagedAttention
Axiom will implement **PagedAttention**, treating GPU VRAM identically to an operating system virtual memory manager:

```
+-----------------------------------------------------------------------------------+
|                        PAGEDATTENTION LOGICAL-TO-PHYSICAL MAP                     |
|                                                                                   |
|  Logical Sequence: [Token 0 ... 15]   [Token 16 ... 31]   [Token 32 ... 47]       |
|                            │                   │                   │              |
|  CPU Block Table:     Physical Page 12    Physical Page 5     Physical Page 89    |
|                            │                   │                   │              |
|  Physical VRAM:       [Page 12: 16 T]     [Page 5: 16 T]      [Page 89: 16 T]     |
+-----------------------------------------------------------------------------------+
```

1. **Fixed Physical Pages:** The KV-cache is partitioned into small, uniform physical pages (e.g., 16 tokens per page).
2. **Dynamic Block Table:** A CPU-side table maps logical sequence positions to non-contiguous physical pages in VRAM.
3. **Zero Fragmentation:** Any free page can be assigned to any active sequence, completely eliminating external fragmentation and allowing batch sizes to scale by up to $3.5\times$.
4. **Copy-On-Write (COW) Prefix Caching:** Prompts sharing identical system instructions reference identical physical pages in VRAM until sequences diverge, slashing Time-to-First-Token (TTFT) latency.

---

## 8. Phase VIII: Microsecond Overhead Eradication with CUDA Graphs

### The Problem
As individual Tensor Core operations are optimized to compute in sub-microsecond intervals, the time required for the host CPU to launch kernels across the PCIe bus and traverse the NVIDIA driver stack becomes the dominant system bottleneck.

### The Solution: Directed Acyclic Graph (DAG) Execution
Using the CUDA Graphs API, the engine captures entire iterations into hardware DAGs:

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;

// 1. Record the execution topology once
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
run_forward_pass_kernels();
run_backward_pass_kernels();
cudaStreamEndCapture(stream, &graph);

// 2. Instantiate hardware DAG
cudaGraphInstantiate(&instance, graph, nullptr, 0);

// 3. Ultra-low-latency single hardware launch
cudaGraphLaunch(instance, stream);
```

- **Single Driver Dispatch:** The entire iteration (transfers, GEMM kernels, bias additions, updates) is submitted in a single driver call.
- **Hardware Scheduling:** The GPU's onboard hardware work distributor manages dependencies between kernels with zero CPU round-trips.

---

## 9. Phase IX: Modular Framework Architecture via ATen Dispatcher

To scale from a bespoke single-model engine to an extensible deep learning framework capable of executing arbitrary computational graphs, Project Axiom will adopt an **ATen-style Dispatch Architecture**:

```
+-------------------------------------------------------------------+
|                     User Tensor Handle (Tensor)                   |
|                     - Shape, Strides, Device Type                 |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                       CENTRAL C++ DISPATCHER                      |
|                  Evaluates DispatchKeySet at runtime              |
+-------------------------------------------------------------------+
               │                                      │
               ▼                                      ▼
     [Key == Backend::CUDA]                 [Key == Backend::CPU]
+------------------------------+       +----------------------------+
| CUTLASS EVT Hardware Kernel  |       | AVX-512 Host C++ Kernel    |
+------------------------------+       +----------------------------+
```

1. **Intrusive Handle Separation:** Lightweight `Tensor` handles wrap reference-counted `TensorImpl` storage objects, enabling zero-copy slicing, transposing, and view transformations.
2. **Dynamic Operator Dispatch:** Mathematical operations (`matmul`, `add`, `layer_norm`) route through a centralized C++ dispatcher based on `DispatchKeySet`.
3. **Native Automatic Differentiation (Autograd):** When a tensor carries the `Autograd` dispatch key, the dispatcher transparently intercepts the call, records backward nodes in the execution tape, and dispatches the forward pass to silicon.

---

## 10. Phase X: Sparse Mixture of Experts (MoE) Architecture

As models scale to billions of parameters, executing dense GEMM passes where every token activates every parameter becomes computationally prohibitive.

Axiom will implement **Sparse Mixture of Experts (MoE)**:
1. **Expert Blocks:** Dense weight matrices are split into multiple specialized stacks of weights (e.g. 8 or 16 experts).
2. **Top-k Gating Router:** A lightweight gating network routes incoming tokens to only the top-1 or top-2 most relevant experts.
3. **Conditional Execution:** Compute cores skip inactive experts, enabling multi-billion parameter model capacities with small, fixed per-token compute costs.

---

## 11. Ecosystem & UI/UX Expansion

While Version 1.0 remains strictly a pure terminal command-line binary ([`core/app.exe`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/app.exe)), Version 2.0 will introduce dedicated visualization layers:

1. **Native Diagnostic GUI:** A hardware-accelerated desktop telemetry dashboard built with C++ and Dear ImGui / DirectX 12, visualizing live SM occupancy, PCIe DMA transfer rates, loss convergence curves, and VRAM memory maps in real time.
2. **Web Multimodal Workspace:** A decoupled web dashboard pairing the high-performance C++23 backend with a modern React, TypeScript, and Tailwind CSS frontend for interactive prompt engineering, multimedia inspection, and model training management.
