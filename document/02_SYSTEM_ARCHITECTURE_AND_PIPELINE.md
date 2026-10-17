# 02. System Architecture and Pipeline

The primary architectural challenge in high-throughput machine learning systems is **data starvation**. A graphics card equipped with thousands of computing cores can perform trillions of floating-point operations per second. If the host subsystem cannot deliver input tokens across the system buses quickly enough, the graphics processor stalls, dropping arithmetic efficiency to near zero.

Project Axiom addresses this bottleneck through an event-driven, four-tier data logistics pipeline designed to treat memory as an uninterrupted conveyor belt.

---

## 1. The 4-Tier Memory Logistics Conveyor Belt

```
+---------------------------------------------------------------------------------+
|                        TIER 0: PERSISTENT STORAGE (NVMe SSD)                    |
|                        Raw multimodal files (Gigabytes / Terabytes)             |
+---------------------------------------------------------------------------------+
                                        │
                         [Sequential Buffered Streaming]
                                        ▼
+---------------------------------------------------------------------------------+
|                        TIER 1: SYSTEM RAM BUFFER                                |
|                        100,000 Categorical uint32_t Tokens                      |
|                        Decouples slow disk I/O from compute                     |
+---------------------------------------------------------------------------------+
                                        │
                           [Overlapping Sliding Window]
                                        ▼
+---------------------------------------------------------------------------------+
|                        TIER 2: PINNED CONTEXT WINDOW                            |
|                        Portion Struct: 128 Inputs (X) + 1 Target (Y)            |
|                        Allocated via cudaHostAllocWriteCombined                 |
+---------------------------------------------------------------------------------+
                                        │
                            [Zero-Copy PCIe DMA Stream]
                                        ▼
+---------------------------------------------------------------------------------+
|                        TIER 3: GRAPHICS VRAM (RTX 4080)                         |
|                        Continuous GPU VRAM: Device Weights & Activations        |
|                        Saturated by NVIDIA CUTLASS v4.8.0 Tensor Cores          |
+---------------------------------------------------------------------------------+
```

### Tier 0: Persistent Storage (NVMe SSD)
Raw training corpora and model assets reside permanently on the disk. File access operations are orchestrated via persistent file streams (`std::ifstream`) that remain continuously open throughout the application lifecycle. This prevents repeated operating system kernel context switches and file descriptor spin-up costs.

### Tier 1: Host System RAM (The 100,000-Token Buffer)
To prevent the operating system from thrashing physical storage or exhausting host memory on massive multi-gigabyte datasets, [`DataLoader`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/loader.h) maintains a dedicated circular buffer holding exactly 100,000 categorical tokens (`uint32_t`). When the neural network consumes this buffer, the loader refreshes it in a single sequential I/O read.

### Tier 2: Pinned Context Windows (The Portion Structure)
The neural network operates on discrete sliding context windows of 128 tokens. [`DataLoader::ram_to_contextWindow()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/loader.cpp) slices the 100,000-token RAM array into [`Portion`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/loader.h) structures:
- **Input Tensor ($X$):** 128 context token IDs (`uint32_t` integers = 512 bytes). Uploading integer token IDs across PCIe rather than high-dimensional floating-point vectors slashes bus traffic by over **93%**.
- **Target Tensor ($Y$):** The 129th target token serving as the self-supervised ground truth.

Crucially, these buffers are allocated in page-locked, write-combined physical memory (`cudaHostAllocWriteCombined`) to enable true asynchronous hardware Direct Memory Access (DMA) without CPU intervention or cache pollution.

### Tier 3: GPU Video Memory (VRAM)
Data arriving over the PCIe bus lands in dedicated VRAM allocated via [`allocate_gpu_memory()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/tensor_cores.cu). Here:
1. **Branchless Gather Kernel:** Maps the 128 integer token IDs directly into 32-dimensional vectors from the learnable VRAM embedding table `d_embedding_table` ($65,536 \times 32$ in `__nv_bfloat16`).
2. **Tensor Core GEMM:** CUTLASS v4.8.0 templates execute the matrix multiplication directly within high-speed shared memory registers before linear EVT write-back.

---

## 2. "The Seam": Eradicating Boundary Amnesia

In naive chunked data loaders, reading discrete blocks of tokens from disk creates catastrophic boundary discontinuities. When Chunk 1 (tokens $1$ through $100,000$) is exhausted and Chunk 2 (tokens $100,001$ through $200,000$) is loaded, the context window is severed:

```
[Naive Chunk 1: Tokens 99,873 ... 100,000] || [Naive Chunk 2: Tokens 100,001 ... 100,128]
                                           ^^
                                   BOUNDARY DISCONTINUITY
                        (Historical sequence severed, network suffers amnesia)
```

If the loader seeks backwards on the disk to recover the overlapping tokens, it invalidates the operating system's sequential file cache, severely degrading storage read throughput.

### The Axiom Seam Solution
Axiom solves this through **The Seam**—an in-memory carry-forward mechanism implemented in [`DataLoader::file_to_ram()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/loader.cpp):

```cpp
// If the buffer already contained data, preserve the final 128 tokens
std::vector<uint32_t> carry_forward;
if (buffer.size() >= 128) {
    carry_forward.assign(buffer.end() - 128, buffer.end());
}

// Clear the exhausted buffer
buffer.clear();

// Prepend the carried-forward tokens to the front of the new buffer
buffer.insert(buffer.end(), carry_forward.begin(), carry_forward.end());

// Stream the next chunk from disk sequentially
uint32_t token;
while (buffer.size() < 100000 && stream >> token) {
    buffer.push_back(token);
}
```

```
+---------------------------------------------------------------------------------+
| OLD BUFFER: ... [Token 99,873] ... [Token 100,000]                              |
+---------------------------------------------------------------------------------+
                         │
                         ▼ Sliced (Last 128 tokens preserved)
+---------------------------------------------------------------------------------+
| NEW BUFFER: [Carried 128 Tokens] + [New Tokens 100,001 ... 200,000]             |
+---------------------------------------------------------------------------------+
```

By prepending the trailing 128 tokens to the head of the new memory buffer, the sliding context window flows over the boundary with zero interruption. The neural network experiences a continuous stream of consciousness without a single backward disk seek.

---

## 3. The Library, Book, and Page Operational Model

To conceptualize the memory hierarchy and concurrency flow, Project Axiom implements the **Library, Book, and Page** architectural analogy:

| Analogy | Physical Hardware | Axiom Component | Scale / Scope |
| :--- | :--- | :--- | :--- |
| **The Library** | NVMe Solid-State Disk | Raw Storage Corpora | Terabytes of uncompressed text and multimedia. |
| **The Book** | Host System RAM | [`DataLoader::buffer`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/loader.h) | 100,000 categorical tokens held in volatile physical memory. |
| **The Page** | Pinned Host RAM / VRAM | [`Portion`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/loader.h) Context Window | 128 token IDs uploaded via PCIe $\rightarrow$ gathered into $128 \times 32$ VRAM matrix for Tensor Cores. |

- You do not bring the entire **Library** onto your desk (this causes an Out-Of-Memory crash).
- You take a single **Book** (100,000 tokens) off the shelf and place it on your desk (System RAM).
- You read the book **Page by Page** (128-token context windows) using your eyes (the GPU Tensor Cores).
- When you reach the final page of the Book, you remember the last sentence (**The Seam**) so you can seamlessly begin the next volume without losing the story.

---

## 4. Primary System Modules and Boundaries

The engine is partitioned into four decoupled subsystems:

```
+-------------------------------------------------------------------+
|                           app.exe (Main)                          |
+-------------------------------------------------------------------+
        │                     │                        │
        ▼                     ▼                        ▼
+---------------+     +----------------+     +---------------------+
|    Parser     |     |   DataLoader   |     |         AI          |
|  (parser.cpp) |     |  (loader.cpp)  |     |      (ai.cpp)       |
+---------------+     +----------------+     +---------------------+
                                                       │
                                            [C ABI Boundary]
                                            extern "C" functions
                                                       ▼
                                             +--------------------+
                                             |    TensorCores     |
                                             | (cuda_transfer.dll)|
                                             +--------------------+
```

1. **[`Parser`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/parser.h) & Decompressors (`core/include/decompressor/`):** Orchestrates the Four-Stage Ingestion Pipeline. Detects file formats, strips compression/container wrappers (`decom_doc`, `decom_image`, `decom_audio`, `decom_video`), sanitizes text, and quantizes tokens into the dense early-fusion codebook ($V = 65,536$).
2. **[`DataLoader`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/loader.h):** Manages disk-to-RAM streaming, the 100,000-token circular buffer, the seam carry-forward window, and pinned write-combined context allocations.
3. **[`AI`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/ai.h):** Manages master weights, the master learnable embedding matrix, binary persistence (`weights.axiom` and `optimizer.axiom`), forward pass predictions, loss evaluations, and interactive inference.
4. **[`TensorCores`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/tensor_cores.h):** Compiled into [`core/cuda_transfer.dll`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/cuda_transfer.dll) behind an un-mangled `extern "C"` ABI boundary. Manages physical VRAM buffers, the branchless embedding gather kernel, CUTLASS v4.8.0 EVT matrix multiplications, and dual non-blocking CUDA streams.

---

## 5. The Four-Stage Multimodal Ingestion Pipeline

Project Axiom implements the exact ingestion architecture mandated by modern frontier omni-models (such as GPT-4o and Gemini). Reading raw compressed bytes directly into a tokenizer is an architectural trap: the model wastes its capacity reverse-engineering zlib compression headers or MP4 container framing instead of learning semantics.

Axiom enforces a strict four-stage pipeline:

```
  Stage 1: THE IDENTIFIER (The Switchboard)
  • Inspects file extensions & magic bytes (e.g. 0xFF 0xD8 for JPEG, %PDF for PDF, RIFF for WAV)
  • Routes raw file streams to the appropriate specialized decompressor
                       │
                       ▼
  Stage 2: THE DECOMPRESSOR (The Extractor)
  • Strips container wrappers to expose the raw semantic payload:
    - .pdf / .docx  ──► Clean UTF-8 text & equations (zlib stripped via decom_doc)
    - .mp4 / .mov   ──► Raw RGB pixel arrays (h.264 stripped via decom_video)
    - .wav / .ogg   ──► Raw PCM audio amplitudes (RIFF headers stripped via decom_audio)
    - .png / .jpg   ──► Uncompressed RGB bitmaps (decom_image)
                       │
                       ▼
  Stage 3: THE SANITIZER & TOKENIZER (The Quantizer)
  • Lowercases characters, strips punctuation, and maps clean tokens to dense IDs
  • Unified Contiguous Codebook: V = 65,536 (Words, numbers, pixels, and audio share index space)
                       │
                       ▼
  Stage 4: THE EMBEDDER (The VRAM Matrix)
  • Transmits 128 integer token IDs (512 bytes) across PCIe to GPU
  • Fast CUDA gather kernel pulls 32D geometric vectors from d_embedding_table in VRAM
```

### The Dual Ingestion Workflows
The engine differentiates cleanly between file processing and manual input:

1. **Workflow 1: File Ingestion (Bulk or Single Asset):**
   $$\text{File on Disk} \longrightarrow \text{Identifier} \longrightarrow \text{Decompressor} \longrightarrow \text{Sanitizer} \longrightarrow \text{Tokenizer} \longrightarrow \text{Embedder}$$
   Requires full extraction because semantic data is trapped inside compressed containers.

2. **Workflow 2: Raw Manual Text Input (CLI Console):**
   $$\text{Raw String} \longrightarrow [\text{Identifier Bypassed}] \longrightarrow [\text{Decompressor Bypassed}] \longrightarrow \text{Sanitizer} \longrightarrow \text{Tokenizer} \longrightarrow \text{Embedder}$$
   When a user types text directly into the CLI, the data is already uncompressed UTF-8. The engine bypasses Stages 1 and 2, passing the string straight to the sanitizer and tokenizer.

---

## 6. The Output Codec & Detokenizer (The Compressor)

Generative AI models do not output formatted `.mp4` or `.pdf` files directly; they output a sequence of discrete token probabilities. To render output for humans, Axiom defines the inverse pipeline via **Output Codecs**:

* **Text Detokenizer:** Inspects predicted token IDs in $[4, 64511]$, performs reverse lookup in `reverse_main_dictionary`, and streams UTF-8 strings to the console.
* **Visual Output Codec:** Takes discrete visual tokens $[64512, 64767]$, converts them back into RGB pixel grids, and packages them into standard image or video formats.
* **Acoustic Output Codec:** Takes discrete audio tokens $[64768, 65023]$, converts them to PCM amplitudes, and wraps them in RIFF/WAV headers.

This symmetry ensures that the core neural matrix remains modality-agnostic: the GPU processes pure geometric vectors, while the modular edges handle decompression upon entry and compression upon exit.
