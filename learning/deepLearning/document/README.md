# AI Core: Native C++ Artificial Intelligence Engine (Version 1.0)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Standard: C++17](https://img.shields.io/badge/Standard-C%2B%2B17-green.svg)](#)
[![Status: Open Source](https://img.shields.io/badge/Status-Open%20Source-orange.svg)](#)

Welcome to the **AI Core** technical documentation. This project is an open-source, zero-dependency **Artificial Intelligence (AI)** engine written from scratch in native C++17, licensed under the **GNU General Public License v3 (GNU GPL v3)**.

---

## 1. Project Philosophy & Vision: From Perceptron to LLaMA

This repository is **not** a generic deep learning exercise—it is a full-stack **AI Architecture Project**. 

True machine intelligence begins at first principles:
1. **The Minimal Atomic Unit of AI:** We begin with the foundational mathematical nucleus of artificial intelligence: the single **Perceptron** (weights, biases, forward pass, error evaluation).
2. **Expansion to Frontiers:** From this simplest unit, the system iteratively evolves into complex cognitive systems: multimodal early-fusion perception, discrete tokenization, attention mechanisms, and reverse-engineering foundational Large Language Models (such as Meta's **LLaMA**).
3. **The Empirical Open-Source Pipeline:** The engine develops through iterative experimentation—conducting mathematical trials, producing high-performance modular components, stress-testing hardware limits, and polishing them into production-ready open-source AI building blocks.

```
+-------------------------------------------------------------------------------+
|                             AI Core Evolution Path                            |
+-------------------------------------------------------------------------------+
| Stage 1: The Perceptron (Minimal Atomic AI Nucleus)                           |
|          -> Linear decision boundaries, weights, biases, error metrics        |
+-------------------------------------------------------------------------------+
                                       |
                                       v
Stage 2: Multimodal Cognitive Perception (Current Milestone)
         -> Discrete early fusion, global tokenization across text, visual,
            auditory, and temporal sensory inputs
+-------------------------------------------------------------------------------+
                                       |
                                       v
Stage 3: Dense Representation & Continuous Latent Manifolds
         -> Embedding projection, semantic geometry, vector arithmetic
+-------------------------------------------------------------------------------+
                                       |
                                       v
Stage 4: Reverse-Engineering LLaMA & Transformer Primitives
         -> Multi-Head Attention, RoPE (Rotary Positional Embeddings),
            RMSNorm, SwiGLU activations, and KV Cache mechanics
+-------------------------------------------------------------------------------+
                                       |
                                       v
Stage 5: Local Hardware Execution & Quantization
         -> INT4/INT8 GGUF-style weights, RTX 4080 Tensor Core acceleration,
            and hybrid CPU/VRAM layer offloading
+-------------------------------------------------------------------------------+
```

---

## 2. Technical Documentation Index

Detailed architectural specifications and engineering standards are organized into modular documents:

| Specification Document | Topic Coverage |
| :--- | :--- |
| **[AI Multimodal Architecture Specification](file:///c:/Users/Omid/X/Code/Brain/learning/deepLearning/document/architecture.md)** | Early fusion discrete tokenization, unified sequences, modality ID offset spaces, collision prevention, and subsystem decoupling. |
| **[Tokenization & Embeddings Specification](file:///c:/Users/Omid/X/Code/Brain/learning/deepLearning/document/tokenization_and_embeddings.md)** | Evolution of text and sensory encoding in AI (ASCII $\to$ Polynomial Hash $\to$ BPE Subwords $\to$ Dense Vectors), VQ-VAE codebooks, and harmonic projection. |
| **[AI Memory Hierarchy & Hardware Acceleration](file:///c:/Users/Omid/X/Code/Brain/learning/deepLearning/document/memory_hierarchy_and_hardware.md)** | 3-tier memory model (NVMe Disk $\to$ RAM $\to$ GPU VRAM), immutable vocabularies, batch streaming, RTX 4080 (16GB) constraints, Tensor Cores, GGUF/AWQ quantization, and llama.cpp offloading. |
| **[C++ Design Patterns & Data Structures for AI](file:///c:/Users/Omid/X/Code/Brain/learning/deepLearning/document/data_structures_and_cpp_design.md)** | Standard library internals (`std::unordered_map`, `.find() == .end()`, `operator[]`), the DRY template engine (`add_assign_increase<T>`), memory safety, and verification harness. |

---

## 3. High-Level AI System Pipeline

```
+-----------------------------------------------------------------------------------+
|                            Raw Sensory Input (World State)                        |
|    - Text (Language symbols)                                                      |
|    - Numbers (Tabular feature vectors)                                            |
|    - Image (Visual pixel matrices)                                                |
|    - Video (Temporal motion deltas)                                               |
|    - Audio (Acoustic PCM waveforms)                                               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       AI Parser Subsystem (Discrete Perception)                   |
|    - Text Vocab:    IDs 1       - 9,999                                           |
|    - Number Vocab:  IDs 10,000  - 19,999                                          |
|    - Image Vocab:   IDs 20,000  - 29,999                                          |
|    - Video Vocab:   IDs 30,000  - 39,999                                          |
|    - Voice Vocab:   IDs 40,000  - 49,999                                          |
|    - DRY Template:  Parser::add_assign_increase<T>()                              |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      Unified Token Vector (tokenized_data)                        |
|                     [ 63 total discrete multimodal tokens ]                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         Continuous Embedding Projection                           |
|                      [ 189-dimensional dense latent vector ]                      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              AI Neural Engine (Perceptron)                        |
|                     Forward Pass Dot-Product, Activation, Inference               |
+-----------------------------------------------------------------------------------+
```

---

## 4. Directory Structure

```
learning/deepLearning/
├── document/                               # Open-Source Technical Documentation
│   ├── README.md                           # Master AI documentation index (this file)
│   ├── architecture.md                     # AI Multimodal system architecture & design
│   ├── tokenization_and_embeddings.md      # Tokenizer algorithms & vector embeddings in AI
│   ├── memory_hierarchy_and_hardware.md    # Local AI hardware scaling, VRAM, & batching
│   └── data_structures_and_cpp_design.md   # C++17 design patterns, templates & containers
├── include/                                # Header specifications
│   ├── parser.hpp                          # Parser class, Test, and Input structs
│   └── perceptron.hpp                      # Perceptron class & Key_Concepts struct
├── src/                                    # AI core neural implementation
│   ├── main.cpp                            # Main application entry point
│   └── perceptron.cpp                      # Perceptron implementation
├── util/                                   # Data processing & tokenization
│   └── parser.cpp                          # Multimodal parser & embedding routines
└── test/                                   # Validation test harnesses
    └── test_parser.cpp                     # Tokenizer & embedding verification tests
```

---

## 5. Build & Test Instructions

The codebase is built strictly to standard C++17 and requires zero third-party dependencies.

### Compiling & Running the Verification Suite
Using GCC (`g++`):
```bash
# Compile the test runner
g++ -std=c++17 -Wall -Wextra -pedantic test/test_parser.cpp -o test/test_parser.exe

# Execute the test suite
./test/test_parser.exe
```

### Syntax Verification on Core Modules
```bash
# Verify parser header & source syntax
g++ -fsyntax-only -Wall -Wextra -pedantic -std=c++17 util/parser.cpp

# Verify perceptron implementation syntax
g++ -fsyntax-only -Wall -Wextra -pedantic -std=c++17 src/perceptron.cpp
```

---

## 6. Project Roadmap

- [x] **Discrete Multimodal Tokenizer**: Segregated dictionary ranges for text, numbers, image, video, voice.
- [x] **DRY Generic Template Engine**: Unified `add_assign_increase<T>` routine.
- [x] **Dense Embedding Simulation**: Projection from discrete IDs to continuous latent vectors.
- [x] **Technical Documentation**: Open-source specification files covering architecture, memory hierarchy, and C++ design.
- [ ] **Forward Pass Dot Product**: Implement `input * weight + bias` matrix operations in `Perceptron`.
- [ ] **Non-Linear Activation**: Integrate activation functions (ReLU, Sigmoid, GELU).
- [ ] **Backpropagation & Gradient Descent**: Parameter optimization using computed loss derivatives.
- [ ] **Multi-Layer Architecture (MLP)**: Chaining perceptrons into hidden layers.
- [ ] **LLaMA Primitive Exploration**: Self-attention, RMSNorm, and Rotary Positional Embeddings (RoPE).

---

## 7. License

This project is licensed under the **GNU General Public License v3.0 (GNU GPL v3)**.  
You are free to copy, distribute, study, and modify this software provided that all modifications and derivative works remain open source under the same license terms.
