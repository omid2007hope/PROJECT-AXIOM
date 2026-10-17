# 07. Multimodal Tokenizer and Inference Engine

In traditional artificial intelligence architectures, different modalities (text, vision, audio) are processed by separate, disjoint tokenizers and encoder stacks (e.g., BPE for text, ViT patches for vision, Mel-spectrogram convolutions for audio). These pipelines introduce severe software complexity and impedance mismatches when modalities interact.

Project Axiom implements an **Early-Fusion Omni-Modal Architecture**—the exact foundational design pattern leveraged by frontier models like GPT-4o and Gemini. All modalities are ingested through a unified four-stage pipeline, quantized into a dense, bounded codebook ($V = 65,536$), and embedded into a learnable 32-dimensional geometric space in GPU VRAM.

---

## 1. The Four-Stage Multimodal Ingestion Pipeline

Reading raw compressed bytes directly into a neural tokenizer is an architectural trap: the network wastes its parameter capacity trying to reverse-engineer zlib/FlateDecode compression tables or MP4 box framing rather than learning physics, language, or visual geometry.

Axiom enforces a strict, four-stage ingestion pipeline:

```
+-----------------------------------------------------------------------------------+
|                        THE FOUR-STAGE INGESTION PIPELINE                          |
|                                                                                   |
|  [Stage 1: THE IDENTIFIER] (The Switchboard)                                      |
|  • Inspects file extensions & magic bytes (e.g. 0xFF 0xD8 for JPEG, %PDF for PDF) |
|  • Routes raw file streams to the appropriate specialized decompressor            |
|                                      │                                            |
|                                      ▼                                            |
|  [Stage 2: THE DECOMPRESSOR] (The Extractor)                                      |
|  • Strips compression/container wrappers to expose the raw semantic payload:     |
|    - .pdf / .docx  ──► Clean UTF-8 text & equations (zlib stripped via decom_doc) |
|    - .mp4 / .mov   ──► Raw RGB pixel arrays (h.264 stripped via decom_video)      |
|    - .wav / .ogg   ──► Raw PCM audio amplitudes (RIFF stripped via decom_audio)   |
|    - .png / .jpg   ──► Uncompressed RGB bitmaps (decom_image)                     |
|                                      │                                            |
|                                      ▼                                            |
|  [Stage 3: THE SANITIZER & TOKENIZER] (The Quantizer)                             |
|  • Lowercases characters, strips punctuation, and assigns dense integer IDs       |
|  • Dense Unified Codebook: V = 65,536 (Words, numbers, pixels, and waveforms)     |
|                                      │                                            |
|                                      ▼                                            |
|  [Stage 4: THE EMBEDDER] (The VRAM Matrix)                                        |
|  • Uploads 128 integer token IDs (512 bytes) across PCIe to GPU                   |
|  • Fast CUDA gather kernel pulls 32D geometric vectors from d_embedding_table     |
+-----------------------------------------------------------------------------------+
```

### The Dual Ingestion Workflows
The engine differentiates cleanly between file processing and manual input:

1. **Workflow 1: File Ingestion (Bulk or Single Asset):**
   $$\text{File on Disk} \longrightarrow \text{Identifier} \longrightarrow \text{Decompressor} \longrightarrow \text{Sanitizer} \longrightarrow \text{Tokenizer} \longrightarrow \text{Embedder}$$
   Requires full extraction because semantic data is locked inside compressed containers.

2. **Workflow 2: Raw Manual Text Input (CLI Console):**
   $$\text{Raw String} \longrightarrow [\text{Identifier Bypassed}] \longrightarrow [\text{Decompressor Bypassed}] \longrightarrow \text{Sanitizer} \longrightarrow \text{Tokenizer} \longrightarrow \text{Embedder}$$
   When a user types text directly into the CLI, the data is already uncompressed UTF-8. The engine bypasses Stages 1 and 2, passing the string straight to the sanitizer and tokenizer.

---

## 2. Dense Early-Fusion Codebook ($V = 65,536$)

Legacy sparse architectures utilized multi-billion offsets (e.g. 1B for numbers, 2B for images, 3B for video). Indexing tables across billions of slots creates catastrophic VRAM memory walls and allocation failures.

Axiom consolidates all modalities into a **compact, contiguous 16-bit addressable range $[0, 65535]$**:

| Slot Range | Capacity | Modality | Content & Generation Method |
| :--- | :--- | :--- | :--- |
| **`[0 .. 3]`** | 4 tokens | **Special Reserved** | `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3`. |
| **`[4 .. 64,511]`** | 64,508 tokens | **Text & Numbers** | Unified vocabulary (`main_dictionary`). Preserves natural co-occurrence between words and quantities (e.g. `"gravity"` and `"9.8"`). |
| **`[64,512 .. 64,767]`** | 256 tokens | **Discrete Vision** | Discrete byte-level pixel values (`0x00 .. 0xFF`) and spatial codebook indices. |
| **`[64,768 .. 65,023]`** | 256 tokens | **Discrete Audio** | Discrete acoustic amplitude quantization slots (`0x00 .. 0xFF`). |
| **`[65,024 .. 65,535]`** | 512 tokens | **Reserved** | Expansion slots for 3D spatial latents and future modality codebooks. |

```
0   3 4                                                64511 64512 64767 64768 65023 65024    65535
+---+-------------------------------------------------------+-----+-----+-----+--------+
|SP |       UNIFIED TEXT & NUMERICAL DICTIONARY             | VIS | AUD | EXP | FUTURE |
+---+-------------------------------------------------------+-----+-----+-----+--------+
```

### Unifying Text and Numbers (`main_dictionary`)
Numbers and words share the same linguistic reality: a physics model reading `"acceleration 9.8 m/s^2"` must relate the quantity `"9.8"` to `"acceleration"`. By unifying text and numbers into `main_dictionary`, numbers are ingested directly as tokens, allowing semantic clustering in the vector space.

---

## 3. Learnable VRAM Embedding Matrix ($65,536 \times 32$)

Static trigonometric formulas (e.g. `sin(id * (d+1))`) are synthetically blind: words like `"bulk"` and `"training"` receive arbitrary unrelated coordinates simply because they were ingested at different times.

Axiom replaces synthetic formulas with a **Learnable Embedding Table** in GPU VRAM:

* **Table Dimensions:** $V \times D = 65,536 \times 32$ elements.
* **VRAM Footprint:**
  $$\text{Device Table (BF16)}: 65,536 \times 32 \times 2\text{ bytes} = 4,194,304\text{ bytes} \approx 4.19\text{ MB}$$
  $$\text{Master Table (FP32)}: 65,536 \times 32 \times 4\text{ bytes} = 8,388,608\text{ bytes} \approx 8.39\text{ MB}$$
* **Chaos Initialization:** Weights are initialized uniformly in $[-1/\sqrt{D}, +1/\sqrt{D}] \approx [-0.1768, +0.1768]$.
* **Semantic Emergence:** The backward scatter kernel with `atomicAdd` adjusts the master coordinates based on context, physically clustering related concepts together over time.

---

## 4. The Interactive Inference Pipeline (`--use`)

When launched with `app.exe --use`, the engine executes real-time local inference:

```
[User Input Prompt]
         │
         ▼
[Workflow 2: Sanitizer]    ===> Lowercase & punctuation stripping
         │
         ▼
[Tokenizer Lookup]         ===> Maps words to uint32_t IDs [4 .. 64511]
         │
         ▼
[Context Padding / Slice]  ===> Aligned to 128 tokens
         │
         ▼
[PCIe DMA Upload]          ===> 512 bytes sent to GPU pinned buffer
         │
         ▼
[CUDA Gather Kernel]       ===> Expands 128 IDs into 128x32 matrix in VRAM
         │
         ▼
[Tensor Core GEMM]         ===> CUTLASS EVT linear forward pass on RTX 4080
         │
         ▼
[Output Projection]        ===> Continuous prediction vector y in R^32
         │
         ▼
[Top-1 Dot Product / L2]   ===> Finds best matching token in VRAM
         │
         ▼
[Output Codec / Detoken]   ===> Maps token ID back to text or media
         │
         ▼
[Terminal Stream Output]
```

---

## 5. The Output Codecs (The Compressor)

Frontier generative architectures do not output finished `.mp4` or `.pdf` files directly; they output abstract mathematical token distributions. To render results for humans, Axiom defines the inverse pipeline via **Output Codecs**:

1. **Text Detokenizer:**
   - Evaluates token IDs in the linguistic range $[4, 64511]$.
   - Looks up the token string in `reverse_main_dictionary`.
   - Streams UTF-8 characters directly to the console output.
2. **Visual Output Codec:**
   - Evaluates discrete visual tokens $[64512, 64767]$.
   - Maps tokens back to RGB pixel values and spatial patches.
   - Encodes raw frames into standard container formats (e.g. BMP/PNG or h.264 MP4).
3. **Acoustic Output Codec:**
   - Evaluates discrete acoustic tokens $[64768, 65023]$.
   - Maps tokens back to PCM audio waveforms and writes standard RIFF/WAV headers.

This architectural symmetry ensures that while data ingestion strips away container entropy, output generation restores familiar human media formats.
