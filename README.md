# AI Core: Native C++ Artificial Intelligence Engine

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Standard: C++17](https://img.shields.io/badge/Standard-C%2B%2B17-green.svg)](#)
[![Architecture: Early--Fusion%20Multimodal](https://img.shields.io/badge/Architecture-Early--Fusion%20Multimodal-purple.svg)](#)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)](#)

Welcome to the **AI Core** technical documentation. AI Core is an open-source, zero-dependency artificial intelligence runtime engineered from fundamental principles in native C++17. The project bridges the gap between atomic neural computational units (the single Perceptron) and state-of-the-art foundation models (such as Meta's LLaMA, Chameleon, and Gemini-style early-fusion architectures).

---

## 1. Project Philosophy & Vision

Modern deep learning frameworks frequently hide fundamental computational mechanics behind heavyweight runtimes, dynamic Python bindings, and opaque memory managers. **AI Core** takes the opposite approach: complete architectural transparency, cache-conscious data structures, and deterministic systems programming in C++17.

```
========================================================================================
                               AI CORE EVOLUTIONARY ROADMAP
========================================================================================

  Stage 1: Atomic Perceptron Foundation
  --------------------------------------------------------------------------------------
  * Single artificial neuron: inputs, weights, bias, dot-product accumulation
  * Activation functions and scalar loss metrics
  * Verification of forward computation accuracy

                                          |
                                          v

  Stage 2: Universal Multimodal Early-Fusion Perception [CURRENT MILESTONE]
  --------------------------------------------------------------------------------------
  * Discrete tokenization across 5 sensory modalities (Text, Number, Image, Video, Audio)
  * Non-overlapping 10-million slot ID namespace allocation
  * Streaming binary ingestion directly from storage
  * Verifiable scale: 8,289,605 tokens generated across media assets

                                          |
                                          v

  Stage 3: Dense Latent Manifolds & Embedding Matrix [CURRENT MILESTONE]
  --------------------------------------------------------------------------------------
  * Discrete Token ID -> Continuous Dense Vector projection
  * Dynamic, deterministic harmonic embedding table
  * Verifiable scale: 786 unique vocabulary embeddings -> 33,158,420 dense floats

                                          |
                                          v

  Stage 4: The GPU Pivot & CUDA Infrastructure [ACTIVE DEVELOPMENT]
  --------------------------------------------------------------------------------------
  * Precision downgrade: 64-bit `double` to 32-bit `float` for Tensor Core alignment
  * CPU (Host) logistics to GPU (Device) muscle transfer via PCIe bus
  * `nvcc` integration and `cudaMalloc` / `cudaMemcpy` orchestration
  * Conductor Producer-Consumer pipeline synchronization

                                          |
                                          v

  Stage 5: LLaMA Foundation & Attention Primitives [UPCOMING]
  --------------------------------------------------------------------------------------
  * Scaled Dot-Product Attention & Multi-Head Self-Attention (MHA)
  * Rotary Positional Embeddings (RoPE)
  * RMSNorm pre-normalization & SwiGLU gated activations
  * Autoregressive Key-Value (KV) Cache mechanics
========================================================================================
```

---

## 2. Technical Documentation Index

All architectural specifications, memory models, algorithms, and engineering guidelines are maintained in modular documents:

| Specification Document | Focus Area |
| :--- | :--- |
| **[Multimodal Architecture Specification](architecture.md)** | Early-fusion discrete tokenization, universal sequence design, 10M ID offset spacing, and subsystem isolation. |
| **[Tokenization & Embeddings](tokenization_and_embeddings.md)** | Binary streaming, byte-level quantization, exact token census (8.28M tokens), and dense vector projection. |
| **[Memory Hierarchy & Hardware](memory_hierarchy_and_hardware.md)** | 3-tier memory model (NVMe -> Host RAM -> GPU VRAM), streaming batch loaders, footprint analysis, and quantization. |
| **[C++ Design Patterns & Data Structures](data_structures_and_cpp_design.md)** | C++17 systems programming, `add_assign_increase<T>`, type safety (`uint8_t`/`uint32_t`), and memory layout. |
| **[Build & Toolchain Guide](build_and_toolchain.md)** | Makefile targets, compiler flags, test harnesses, cross-platform path resolution, and debugging. |

---

## 3. High-Level Data Pipeline

```
+------------------------------------------------------------------------------------+
|                         RAW MULTIMODAL WORLD SENSORY INPUTS                        |
|  - Text Tokens:   {"The", "quick", "brown", "fox", "jumps", "over", ...}          |
|  - Number Data:   {5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2}    |
|  - Photo Asset:   "Photo.jpeg" (130,513 binary bytes)                              |
|  - Video Asset:   "Video.mp4"  (8,096,817 binary bytes)                            |
|  - Audio Asset:   "Voice.ogg"  (62,254 binary bytes)                               |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                       PARSER SUBSYSTEM (DISCRETE TOKENIZER)                        |
|  - Modality ID Allocator (10,000,000 slots per modality)                          |
|    * Text IDs:   [1 .. 9,999,999]                                                  |
|    * Number IDs: [10,000,000 .. 19,999,999]                                        |
|    * Image IDs:  [20,000,000 .. 29,999,999]                                        |
|    * Video IDs:  [30,000,000 .. 39,999,999]                                        |
|    * Voice IDs:  [40,000,000 .. 49,999,999]                                        |
|  - Generic Insertion Engine: Parser::add_assign_increase<T>()                     |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                    UNIFIED DISCRETE TOKEN SEQUENCE (tokenized_data)                |
|  - Type: std::vector<uint32_t>                                                     |
|  - Length: 8,289,605 discrete tokens                                               |
|  - Unique Vocabulary Entries: 786 distinct tokens                                  |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                   CONTINUOUS LATENT EMBEDDING PROJECTION (embedding)               |
|  - Lookup Table: std::unordered_map<uint32_t, std::vector<float>>                  |
|  - Latent Dimension (d_model): 4 floats per token                                 |
|  - Unified Output Buffer: std::vector<float> embedded_data                         |
|  - Total Output Floats: 33,158,420 continuous 32-bit values                        |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                        CPU-TO-GPU PCI-e LOGISTICS TRANSFER                         |
|  - Host (CPU): Conductor orchestrates token logistics and chunking.                |
|  - Bus: `cudaMemcpy` transfers the 512-float active context window.                |
|  - Device (GPU VRAM): Receives tensors for matrix multiplication.                  |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|               NEURAL CORE EXECUTION (GPU TENSOR CORES / PERCEPTRON)                |
|  - Forward Pass: Massive parallel dot product accumulation (y = W * x + b)         |
|  - Activation and Error Evaluation                                                 |
+------------------------------------------------------------------------------------+
```

---

## 4. Empirical System Benchmark

The complete verification harness executes natively against active media samples in `asset_sample/`:

| Modality | Source Entity | Raw Size | Generated Tokens | Unique Vocab IDs | Active ID Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Text** | In-Memory Words | 9 entries | 9 | 9 | `[1 .. 9]` |
| **Number** | In-Memory Doubles | 12 entries | 12 | 9 | `[10000000 .. 10000008]` |
| **Image** | `Photo.jpeg` | 130,513 B | 130,513 | 256 | `[20000000 .. 20000255]` |
| **Video** | `Video.mp4` | 8,096,817 B | 8,096,817 | 256 | `[30000000 .. 30000255]` |
| **Voice** | `Voice.ogg` | 62,254 B | 62,254 | 256 | `[40000000 .. 40000255]` |
| **TOTAL** | — | **8.289 MB** | **8,289,605** | **786** | **Disjoint 10M Blocks** |

---

## 5. Repository Directory Layout

```
learning/deepLearning/
├── Makefile                          # Unified build rules (all, test, clean)
├── asset_sample/                     # Real multimodal binary media test files
│   ├── Photo.jpeg                    # Visual sample asset (130 KB)
│   ├── Video.mp4                     # Temporal video sample asset (8.1 MB)
│   └── Voice.ogg                     # Acoustic audio sample asset (62 KB)
├── document/                         # Technical specifications & design docs
│   ├── README.md                     # Master project entrypoint (this file)
│   ├── architecture.md               # Multimodal early fusion architectural spec
│   ├── tokenization_and_embeddings.md# Tokenizer mechanics & embedding vectors
│   ├── memory_hierarchy_and_hardware.md # 3-tier memory model & VRAM budget
│   ├── data_structures_and_cpp_design.md# C++17 design patterns & containers
│   └── build_and_toolchain.md        # Build instructions, flags & test harness
├── include/                          # Header definitions
│   ├── parser.hpp                    # Input, Test, and Parser declarations
│   └── perceptron.hpp                # Perceptron & Key_Concepts definitions
├── ref/                              # Visual architectural reference assets
│   ├── Ai.jpg                        # Early neural network references
│   └── Deep_Learning.jpg             # Multimodal latent space diagrams
├── src/                              # Core application implementation
│   ├── main.cpp                      # Test runner entry point
│   └── perceptron.cpp                # Perceptron atomic forward pass
├── test/                             # Modular test verification suites
│   ├── run_test.ps1                  # Standalone verification script
│   ├── test_embeddings.cpp           # Embedding unit test harness
│   ├── test_header.hpp               # Header path inspection helper
│   ├── test_ifstream.cpp             # Binary I/O test suite
│   ├── test_path.cpp                 # Path resolution test suite
│   └── test_tokenization.cpp         # Tokenization & embedding verification
└── util/                             # Systems utilities & data parsers
    └── parser.cpp                    # Multimodal tokenizer & embedding logic
```

---

## 6. Quickstart & Build Instructions

### Prerequisites
- Any standard C++17 compliant compiler (`g++` >= 9.0, `clang++` >= 10.0, or MSVC 2019+).
- GNU Make or MinGW32-Make (`mingw32-make` on Windows).

### Build Targets

```bash
# Build the test harness and application
mingw32-make all

# Run the complete test suite
mingw32-make test

# Clean build artifacts
mingw32-make clean
```

### Manual Compilation
```bash
g++ -Wall -std=c++17 -Ilearning/deepLearning/include \
    learning/deepLearning/src/main.cpp \
    learning/deepLearning/src/perceptron.cpp \
    learning/deepLearning/test/test_tokenization.cpp \
    learning/deepLearning/test/test_path.cpp \
    learning/deepLearning/test/test_ifstream.cpp \
    learning/deepLearning/util/parser.cpp \
    -o learning/deepLearning/test.exe
```

---

## 7. License

This project is licensed under the **GNU General Public License v3 (GPL-3.0)**. You are free to copy, modify, distribute, and run this software, provided all modifications and derivative works remain open-source under the identical GPL-3.0 license.
