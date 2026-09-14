# Multimodal Architecture Specification

## 1. Architectural Overview

The DeepLearning Core engine implements an **Early Fusion Discrete Multimodal Architecture**, conceptually aligned with modern foundation models such as Google Gemini and Meta Chameleon. 

Rather than maintaining separate, disparate neural pathways for each input medium (images, text, audio, video, tabular numbers), the system maps all modalities into a **single, unified, discrete token space** before projecting them into a shared continuous embedding space.

```
+-----------------------------------------------------------------------------------+
|                                Raw Multimodal Data                                |
|  [Text: "The quick..."]  [Numbers: 5.1...]  [Images: 0..255]  [Video]  [Audio PCM]|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Discrete Tokenizer (Parser Subsystem)                       |
|   - Modality Dictionaries (std::unordered_map)                                   |
|   - Non-overlapping Global ID Ranges                                              |
|   - Template-driven dynamic ingestion (add_assign_increase<T>)                    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      Unified Token Sequence (tokenized_data)                      |
|                  [ 1, 2, 3 ... 10001 ... 20001 ... 30001 ... 40001 ]              |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            Shared Embedding Matrix                                |
|                       Vector Projection (d_model dimensions)                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             Neural Engine (Perceptron)                            |
|                            Forward Pass & Optimization                            |
+-----------------------------------------------------------------------------------+
```

---

## 2. Early Fusion vs. Late Fusion

In modern deep learning architectures, multimodal processing generally follows one of two paradigms:

### Option 1: Continuous Encoders / Late Fusion (GPT-4 Vision / CLIP Style)
- Text is converted into discrete token IDs.
- Vision and audio are routed through dedicated heavy encoders (e.g., Vision Transformer / CNN) that directly output dense vectors ($\mathbb{R}^d$).
- The vectors are concatenated or cross-attended inside the transformer layers.
- **Trade-off:** High model complexity; separate pre-trained feature extractors required for each non-text modality.

### Option 2: Discrete Quantization / Early Fusion (Gemini / Chameleon Style) — *Adopted*
- **Every** modality is converted into discrete integer tokens ($z \in \mathbb{Z}^+$) at the input boundary.
- Continuous visual/auditory signals are quantized into discrete codebook indices (Vector Quantization / VQ-VAE).
- All tokens enter a single, unified sequence vector (`input.tokenized_data`).
- A single embedding table maps all tokens—regardless of origin modality—into the latent dimension.
- **Advantage:** Maximum structural simplicity, uniform attention mechanisms, and complete modality agnosticism inside the neural processor.

---

## 3. Global Vocabulary Partitioning & Collision Prevention

When multiple modalities are flattened into one unified sequence, a critical ambiguity arises if token IDs overlap:

> **The Cross-Modal Collision Problem:**  
> If the text tokenizer assigns ID `1` to the word `"The"`, and the image quantizer assigns ID `1` to a `"Black Pixel (0.0)"`, the unified sequence receives `[1, 1]`.  
> The downstream Embedding Matrix receives only the integer array without modality metadata. Consequently, the matrix would look up row `1` for both tokens, assigning identical mathematical semantics to an English article and a black pixel.

### The ID Segregation Solution
To preserve complete semantic segregation while maintaining a unified sequence, the vocabulary space is partitioned into deterministic, disjoint numerical ranges:

| Modality | Data Type | Vocabulary Range | Storage Container | Key Type |
| :--- | :--- | :--- | :--- | :--- |
| **Text** | `std::string` | `1` – `9,999` | `input.text_vocab` | `std::string` |
| **Numbers** | `double` | `10,000` – `19,999` | `input.number_vocab` | `double` |
| **Image** | `double` (pixel) | `20,000` – `29,999` | `input.image_vocab` | `double` |
| **Video** | `double` (delta) | `30,000` – `39,999` | `input.video_vocab` | `double` |
| **Voice** | `double` (PCM) | `40,000` – `49,999` | `input.voice_vocab` | `double` |

```
0            10,000          20,000          30,000          40,000          50,000
+---------------+---------------+---------------+---------------+---------------+
|  Text Tokens  | Number Tokens |  Image Tokens |  Video Tokens |  Voice Tokens |
|  [1 - 9,999]  |[10,000-19,999]|[20,000-29,999]|[30,000-39,999]|[40,000-49,999]|
+---------------+---------------+---------------+---------------+---------------+
```

Because each ID range is strictly non-overlapping:
1. `Parser` never generates ambiguous tokens across modalities.
2. The `Embedding` layer can represent each modality in its own mathematical subspace without collision.
3. The neural network can correlate cross-modal patterns (e.g., matching the concept `"dog"` with visual pixel patterns of a dog) via cross-token attention.

---

## 4. Subsystem Decoupling

Following the Single Responsibility Principle, data parsing is strictly isolated from neural computation:

- **`Parser` (`util/parser.cpp`, `include/parser.hpp`)**: Handles tokenization, dictionary state tracking, vocabulary assignment, and embedding synthesis. It has zero knowledge of neural weights, biases, or forward pass logic.
- **`Perceptron` (`src/perceptron.cpp`, `include/perceptron.hpp`)**: Operates strictly on numerical input vectors (`std::vector<double>`). It has zero knowledge of strings, file formats, or tokenization dictionaries.
