# AI Multimodal Cognitive Architecture Specification

## 1. Architectural Overview

The **AI Core** engine implements an **Early Fusion Discrete Multimodal Architecture**, designed as the universal perceptual frontend for an advancing Artificial Intelligence system.

Rather than maintaining isolated, ad-hoc pipelines for different sensory formats (vision, natural language, audio, telemetry, temporal video), the AI maps all environmental inputs into a **single, unified, discrete token space** before projecting them into a shared continuous embedding space for the neural core.

```
+-----------------------------------------------------------------------------------+
|                        Raw Environmental Inputs (Sensory World)                   |
|  [Text: "The quick..."]  [Numbers: 5.1...]  [Images: 0..255]  [Video]  [Audio PCM]|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      AI Discrete Tokenizer (Parser Subsystem)                     |
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
|                            AI Neural Processing Core                              |
|           Perceptron (Atomic) -> MLP -> Transformer / LLaMA Primitives            |
+-----------------------------------------------------------------------------------+
```

---

## 2. Early Fusion vs. Late Fusion in Artificial Intelligence

Modern multimodal AI architectures process disparate media via two competing paradigms:

### Option 1: Continuous Encoders / Late Fusion (e.g., GPT-4 Vision / CLIP)
- Language is tokenized into integer IDs.
- Vision and audio bypass tokenization and are fed through heavy external encoders (e.g., Vision Transformer / ResNet / Whisper) that output continuous feature vectors ($\mathbb{R}^d$).
- These continuous vectors are fused via cross-attention layers.
- **Limitation:** High architectural complexity; requires maintaining separate large pre-trained models for each modality.

### Option 2: Discrete Quantization / Early Fusion (e.g., Gemini / Chameleon) — *Adopted*
- **All modalities** are converted into discrete integer tokens ($z \in \mathbb{Z}^+$) at the earliest perceptual stage.
- Continuous visual/auditory signals are quantized into discrete codebook indices (Vector Quantization / VQ-VAE).
- All tokens enter a single, unified sequence vector (`input.tokenized_data`).
- A unified embedding table projects all tokens into a common latent representation.
- **Advantage:** Maximum architectural elegance, unified representation, and complete modality agnosticism inside the AI neural core.

---

## 3. Global Vocabulary Partitioning & Collision Prevention

When multiple modalities are flattened into one unified sequence, an AI can suffer from severe semantic cross-contamination if token IDs overlap:

> **The Cross-Modal Collision Problem:**  
> If the text tokenizer assigns ID `1` to the word `"The"`, and the image quantizer assigns ID `1` to a `"Black Pixel (0.0)"`, the unified sequence receives `[1, 1]`.  
> The downstream Embedding Matrix receives only the raw integer vector without modality metadata. Consequently, the matrix would look up row `1` for both tokens, assigning identical mathematical semantics to an English article and a black pixel.

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
1. `Parser` never emits ambiguous tokens across modalities.
2. The `Embedding` layer projects each modality into its own distinct geometric subspace.
3. The AI neural engine can correlate cross-modal concepts (e.g., associating the word `"dog"` with the visual pixels of a dog) via associative weights.

---

## 4. Subsystem Decoupling: Perception vs. Cognition

Following the Single Responsibility Principle, sensory parsing is strictly isolated from neural computation:

- **`Parser` (`util/parser.cpp`, `include/parser.hpp`)**: Handles tokenization, dictionary state tracking, vocabulary assignment, and embedding synthesis. It has zero knowledge of neural weights, biases, or forward pass logic.
- **`Perceptron` (`src/perceptron.cpp`, `include/perceptron.hpp`)**: Operates strictly on numerical input tensors (`std::vector<double>`). It has zero knowledge of raw strings, image formats, or tokenization dictionaries.

---

## 5. Architectural Trajectory: From Perceptron to LLaMA

This architecture is deliberately designed as a progressive stepping stone:

1. **Atomic Foundation (Perceptron):** The simplest possible AI computational unit:
   $$y = \sigma\left(\sum_{i=1}^n w_i x_i + b\right)$$
   Where $x$ is the unified embedding vector emitted by `Parser::embedding`.
2. **Hidden Layer Synthesis (MLP):** Stacking perceptrons to create non-linear multi-layer representations.
3. **Sequence Modeling & Attention:** Introducing Query, Key, Value transformations over the unified token sequence:
   $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
4. **LLaMA Reverse-Engineering:** Integrating modern open-source LLM innovations:
   - **RoPE (Rotary Positional Embeddings):** Encoding positional awareness directly into Q and K vectors.
   - **RMSNorm:** Eliminating mean computation for faster layer normalization.
   - **SwiGLU Activation:** Gated linear units replacing standard ReLU/GELU.
   - **KV Caching:** Storing previous key-value states for fast autoregressive inference.
