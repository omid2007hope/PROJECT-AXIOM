# 04. CUDA and Tensor Cores Compute

Graphics processing units derive their computational power not from raw single-thread clock speed, but from massive, highly organized parallelism. To harness this silicon, software must be structured around the physical execution topology of NVIDIA hardware and the specialized matrix multiplication pipelines of **Tensor Cores**.

This document details the CUDA execution model, warp scheduling mechanics, and the implementation of **NVIDIA CUTLASS v4.8.0** with **Epilogue Visitor Trees (EVT)**.

---

## 1. The CUDA Software-to-Hardware Execution Hierarchy

A CUDA program organizes work through a multi-tiered hierarchy that maps directly onto physical silicon components:

| Software Abstraction | Physical Hardware Unit | Hardware Scope & Resources |
| :--- | :--- | :--- |
| **Thread** | **CUDA Core / ALU** | Executes a single instruction stream on private registers. |
| **Warp (32 Threads)** | **SM Sub-Partition** | Executes instructions in strict lockstep (SIMD / SIMT). |
| **Thread Block** | **Streaming Multiprocessor (SM)** | Shares L1 Cache, Shared Memory (SRAM), and barrier synchronization (`__syncthreads()`). |
| **Grid** | **Entire Graphics Processor (GPU)** | Global execution domain spanning all available SMs on the chip. |

```
+-----------------------------------------------------------------------------------------+
|                                    CUDA GRID (Device Scope)                             |
|                                                                                         |
|  +----------------------------------+          +----------------------------------+     |
|  |       THREAD BLOCK (SM Scope)    |          |       THREAD BLOCK (SM Scope)    |     |
|  |  +----------------------------+  |          |  +----------------------------+  |     |
|  |  |   WARP 0 (32 Threads Lockstep) |  |   ...    |  |   WARP 0 (32 Threads Lockstep) |  |     |
|  |  +----------------------------+  |          |  +----------------------------+  |     |
|  |  |   WARP 1 (32 Threads Lockstep) |  |          |  |   WARP 1 (32 Threads Lockstep) |  |     |
|  |  +----------------------------+  |          |  +----------------------------+  |     |
|  |  [Shared Memory / L1 Cache]       |          |  [Shared Memory / L1 Cache]       |     |
|  +----------------------------------+          +----------------------------------+     |
+-----------------------------------------------------------------------------------------+
```

### The Warp Unit (32 Threads)
Hardware schedulers inside each SM sub-partition dispatch instructions in discrete groups of exactly **32 threads called Warps**. All 32 threads in a warp execute the identical instruction at the exact same clock cycle. 
- Thread Block dimensions must always be configured as multiples of 32 (e.g., 128, 256, 512).
- Non-aligned thread counts leave hardware execution lanes masked off (wasted execution cycles).

### Warp Scheduling and 100% SM Occupancy
When a warp executes a global VRAM load, it pauses for hundreds of clock cycles waiting for data to arrive across the memory bus. If the SM has sufficient **Active Warps** resident on silicon (high SM occupancy), the hardware warp scheduler instantly switches execution to another ready warp in zero clock cycles. 

Maintaining high occupancy allows the GPU to hide memory latency behind concurrent computation, unlocking peak TFLOP throughput.

---

## 2. The Tensor Core Paradigm: Ada Lovelace (`sm_89`)

Standard CUDA cores perform one floating-point multiply-accumulate ($a \times b + c$) per instruction per cycle. In contrast, NVIDIA **Tensor Cores** are dedicated hardware matrix accelerators built directly into the Streaming Multiprocessor.

```
       Matrix A               Matrix B
    [ m16 x k16 ]    x     [ k16 x n8 ]
         \                     /
          \                   /
           ▼                 ▼
     +------------------------------+
     |   HARDWARE TENSOR CORE ALU   |  ---> Executes in a single instruction:
     |   mma.sync.aligned.m16n8k16  |       D = A * B + C
     +------------------------------+
                   │
                   ▼
               Matrix D
             [ m16 x n8 ]
```

### Hardware Instruction: `mma.sync`
On the NVIDIA Ada Lovelace architecture (Compute Capability `sm_89`, GeForce RTX 40-series), 4th-generation Tensor Cores execute low-level PTX matrix instructions such as `mma.sync.aligned.m16n8k16`:
- Simultaneously multiplies a $16 \times 16$ matrix of 16-bit floating-point values by a $16 \times 8$ matrix.
- Accumulates the product into single-precision 32-bit registers.
- Delivers up to **5x the throughput** of legacy FP32 CUDA cores.

---

## 3. NVIDIA CUTLASS v4.8.0 Integration

Writing raw CUDA kernels with nested loops will cause the compiler (`nvcc`) to target standard scalar CUDA cores, utilizing less than 5% of the RTX 4080's compute potential.

Project Axiom integrates **NVIDIA CUTLASS v4.8.0**—a high-performance header-only C++ template library that instantiates optimized Tensor Core GEMM pipelines:

```cpp
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/bfloat16.h"

// Define CUTLASS GEMM configuration targeting sm_89
using ElementInputA = cutlass::bfloat16_t;
using ElementInputB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float; // FP32 accumulation prevents underflow

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, cutlass::layout::RowMajor,
    ElementInputB, cutlass::layout::RowMajor,
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89
>;
```

### Key Architectural Advantages
1. **Brain Float Precision (`bfloat16`):** Matches the 8-bit dynamic range of IEEE FP32 while reducing memory footprint to 16 bits, preventing gradient explosion or underflow.
2. **Double-Buffered Shared Memory Tiling:** CUTLASS tiles matrices into SM shared memory (SRAM), loading the next tile asynchronously using hardware asynchronous copy (`cp.async`) instructions while Tensor Cores multiply the current tile.
3. **Hardware Alignment:** Guarantees 128-bit coalesced memory reads from global VRAM.

---

## 4. Epilogue Visitor Trees (EVT): Fused Kernel Execution

The forward pass of a neural network layer computes linear matrix multiplication followed by bias addition and epilogue operations:

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

### The Traditional Multi-Kernel Bottleneck
In naive frameworks, this requires multiple separate CUDA kernel dispatches:
1. **Kernel 1 (GEMM):** $\mathbf{Z} = \mathbf{W}\mathbf{x}$ $\rightarrow$ Writes intermediate matrix $\mathbf{Z}$ to global VRAM.
2. **Kernel 2 (Bias Addition):** $\mathbf{Z}' = \mathbf{Z} + \mathbf{b}$ $\rightarrow$ Reads $\mathbf{Z}$ from VRAM, adds $\mathbf{b}$, writes $\mathbf{Z}'$ to VRAM.
3. **Kernel 3 (Activation/Store):** Writes $\mathbf{y}$ to VRAM.

Because global VRAM access requires 200–400 clock cycles, this naive pipeline wastes 80% of execution time continuously transferring intermediate data back and forth across the memory bus.

### The Axiom Linear EVT Solution (Eliminating Dying ReLU)
Axiom leverages CUTLASS **Epilogue Visitor Trees (EVT)** to fuse the entire operation into a single hardware pass:

```
+-----------------------------------------------------------------------------------+
|                        FUSED CUTLASS EVT KERNEL EXECUTION                         |
|                                                                                   |
|  +--------------------+                                                           |
|  | GEMM Multiplication|                                                           |
|  |   (Tensor Cores)   |                                                           |
|  +--------------------+                                                           |
|           │                                                                       |
|           ▼ [Accumulator Registers]                                               |
|  +--------------------+                                                           |
|  |   Bias Addition    | ===> Executed directly within register space!             |
|  |     (in SRAM)      |                                                           |
|  +--------------------+                                                           |
|           │                                                                       |
|           ▼ [Shared Memory]                                                       |
|  +--------------------+                                                           |
|  | Linear / Identity  | ===> Direct un-gated linear store epilogue                |
|  |   Store Epilogue   |      (Allows full signed coordinate space [-1.0, +1.0])    |
|  +--------------------+                                                           |
|           │                                                                       |
|           ▼                                                                       |
|  +--------------------+                                                           |
|  | Write Final Output | ===> SINGLE write to global VRAM.                         |
|  |  to Device VRAM    |      (Zero intermediate memory round-trips!)              |
|  +--------------------+                                                           |
+-----------------------------------------------------------------------------------+
```

By performing bias addition inside accumulator registers and shared memory while data is already hot on the chip, EVT completely eliminates intermediate global memory round-trips. Furthermore, by adopting a linear store epilogue rather than a hard ReLU gate, Axiom prevents Dying ReLU deadlocks on signed continuous regression targets.

---

## 5. Branchless Embedding Gather Kernel

To support the learnable VRAM embedding matrix without CPU-side float overhead, Axiom implements a fast CUDA gather kernel:

```cpp
__global__ void gather_embeddings_kernel(
    const uint32_t* __restrict__ d_tokens,
    const __nv_bfloat16* __restrict__ d_embedding_table,
    __nv_bfloat16* __restrict__ d_context_window,
    int num_tokens,
    int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * embedding_dim;
    if (idx < total_elements) {
        int token_pos = idx / embedding_dim;
        int dim_idx = idx % embedding_dim;
        uint32_t token_id = d_tokens[token_pos];
        d_context_window[idx] = d_embedding_table[token_id * embedding_dim + dim_idx];
    }
}
```

### Hardware Efficiency Highlights
- **Zero Warp Divergence:** All threads in a warp follow identical linear index math.
- **Coalesced Global Memory Reads:** Adjacent threads read consecutive dimension slots along memory lines, saturating the GPU L2 cache line width.
- **Latency Hiding:** Executed in the compute stream immediately preceding the CUTLASS GEMM kernel.

---

## 6. Frontier Horizon: Sparse Mixture of Experts (MoE)

In dense neural networks, every token is multiplied against *every parameter* in the model—including weights specialized for audio when the input is purely text.

Axiom's roadmap incorporates **Sparse Mixture of Experts (MoE)**:
1. **Expert Blocks:** The monolithic weight matrix is partitioned into distinct, specialized sub-networks (e.g. 8 experts).
2. **The Router Network:** A lightweight gating layer inspects incoming tokens and computes a sparse routing vector.
3. **Selective Hardware Dispatch:** If a token represents visual data, it routes exclusively to Expert #4; the other 7 experts are completely bypassed on the GPU.

MoE expands the total parameter capacity to trillions of weights while keeping per-token compute latency within milliseconds.
