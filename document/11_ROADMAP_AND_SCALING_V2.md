# 11. Roadmap and Scaling Blueprint (Version 2.0)

With the completion of **Version 1.0**, Project Axiom has achieved foundational silicon sovereignty: bare-metal C++23/CUDA execution, 100% Tensor Core saturation via CUTLASS v4.8.0 EVT, atomic binary weight persistence (`weights.axiom`), real-time offline inference, and a 200-test verification harness with zero VRAM leaks.

This document outlines the strategic engineering roadmap to scale Project Axiom into a production-grade, modular deep learning framework capable of serving multi-billion parameter autoregressive transformers.

---

## 1. Phase IV: Userspace Best-Fit with Coalescing (BFC) Caching Allocator

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

## 2. Phase V: PagedAttention for KV-Cache Scalability

### The Problem
When expanding from fixed context perceptrons to full Generative Pre-trained Transformers (GPTs), autoregressive decoding requires caching Key and Value (KV) projection tensors for every past token. Naively allocating continuous VRAM blocks for the maximum sequence length wastes up to **80% of VRAM** due to internal fragmentation (short prompts reserving long buffers).

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

## 3. Phase VI: Microsecond Overhead Eradication with CUDA Graphs

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

## 4. Phase VII: Modular Framework Architecture via ATen Dispatcher

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

## 5. Ecosystem & UI/UX Expansion

While Version 1.0 remains strictly a pure terminal command-line binary ([`core/app.exe`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/app.exe)), Version 2.0 will introduce dedicated visualization layers:

1. **Native Diagnostic GUI:** A hardware-accelerated desktop telemetry dashboard built with C++ and Dear ImGui / DirectX 12, visualizing live SM occupancy, PCIe DMA transfer rates, loss convergence curves, and VRAM memory maps in real time.
2. **Web Multimodal Workspace:** A decoupled web dashboard pairing the high-performance C++23 backend with a modern React, TypeScript, and Tailwind CSS frontend for interactive prompt engineering, multimedia inspection, and model training management.
