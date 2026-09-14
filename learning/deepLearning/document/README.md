# DeepLearning Core: Native C++ Engine Documentation

Welcome to the **DeepLearning Core** technical documentation. This project is a custom-built, zero-dependency deep learning engine and multimodal tokenization pipeline written from scratch in standard C++17.

It models the architectural foundations of modern foundation models (e.g., LLaMA, Gemini, Chameleon), focusing on systems-level clarity, memory hierarchy awareness, and mathematical transparency.

---

## Technical Documentation Index

Detailed specifications and architectural deep dives are modularized into the following documents:

| Specification Document | Topic Coverage |
| :--- | :--- |
| **[Multimodal Architecture Specification](file:///c:/Users/Omid/X/Code/Brain/learning/deepLearning/document/architecture.md)** | Early fusion discrete tokenization, unified sequences, modality ID offset ranges, collision prevention, and subsystem decoupling. |
| **[Tokenization & Embeddings Specification](file:///c:/Users/Omid/X/Code/Brain/learning/deepLearning/document/tokenization_and_embeddings.md)** | Text-to-number evolution (ASCII $\to$ Polynomial Rolling Hash $\to$ BPE/Tokens $\to$ Dense Vectors), VQ-VAE quantization concepts, semantic geometry, and harmonic embedding projection. |
| **[Memory Hierarchy & Hardware Acceleration](file:///c:/Users/Omid/X/Code/Brain/learning/deepLearning/document/memory_hierarchy_and_hardware.md)** | The 3-tier memory model (NVMe Disk $\to$ RAM $\to$ GPU VRAM), immutable vocabularies, batch streaming pipelines, RTX 4080 (16GB) constraints, Tensor Cores, GGUF/AWQ quantization, and llama.cpp layer offloading. |
| **[C++ Design Patterns & Data Structures](file:///c:/Users/Omid/X/Code/Brain/learning/deepLearning/document/data_structures_and_cpp_design.md)** | Standard library internals (`std::unordered_map`, `.find() == .end()`, `operator[]`), the DRY template refactor (`add_assign_increase<T>`), memory safety, and verification harness. |

---

## Architectural Summary

```
+-----------------------------------------------------------------------------------+
|                                Raw Multimodal Data                                |
|    - Text (Words)                                                                 |
|    - Tabular Numbers (Iris features)                                              |
|    - Image (Pixel arrays)                                                         |
|    - Video (Frame deltas)                                                         |
|    - Audio (16kHz PCM waveforms)                                                  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     Parser Subsystem (Discrete Early Fusion)                      |
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
|                             Perceptron Neural Engine                              |
|                       Inference, Forward Pass, Optimization                       |
+-----------------------------------------------------------------------------------+
```

---

## Directory Structure

```
learning/deepLearning/
├── document/                               # Open-Source Technical Documentation
│   ├── README.md                           # Master documentation index (this file)
│   ├── architecture.md                     # Multimodal system architecture & design
│   ├── tokenization_and_embeddings.md      # Tokenizer algorithms & vector embeddings
│   ├── memory_hierarchy_and_hardware.md    # RAM, VRAM, NVMe, batching & GPU scaling
│   └── data_structures_and_cpp_design.md   # C++17 design patterns, templates & containers
├── include/                                # Header specifications
│   ├── parser.hpp                          # Parser class, Test, and Input structs
│   └── perceptron.hpp                      # Perceptron class & Key_Concepts struct
├── src/                                    # Neural core implementation
│   ├── main.cpp                            # Main application entry point
│   └── perceptron.cpp                      # Perceptron implementation
├── util/                                   # Data processing & tokenization
│   └── parser.cpp                          # Multimodal parser & embedding routines
└── test/                                   # Validation test harnesses
    └── test_parser.cpp                     # Tokenizer & embedding verification tests
```

---

## Build & Test Instructions

The codebase adheres strictly to standard C++17 and requires zero external libraries.

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

## Project Roadmap

- [x] **Discrete Multimodal Tokenizer**: Segregated dictionary ranges for text, numbers, image, video, voice.
- [x] **DRY Generic Template Engine**: Unified `add_assign_increase<T>` routine.
- [x] **Dense Embedding Simulation**: Projection from discrete IDs to continuous latent vectors.
- [x] **Technical Documentation**: Open-source specification files covering architecture, memory hierarchy, and C++ design.
- [ ] **Forward Pass Dot Product**: Implement `input * weight + bias` matrix operations in `Perceptron`.
- [ ] **Non-Linear Activation**: Integrate activation functions (ReLU, Sigmoid, GELU).
- [ ] **Backpropagation & Gradient Descent**: Parameter optimization using computed loss derivatives.
