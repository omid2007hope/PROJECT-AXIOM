# 07. Multimodal Tokenizer and Inference Engine

In traditional artificial intelligence architectures, different modalities (text, vision, audio) are processed by separate, disjoint tokenizers and encoder stacks (e.g., BPE for text, ViT patches for vision, Mel-spectrogram convolutions for audio). These pipelines introduce severe software complexity and impedance mismatches when modalities interact.

Project Axiom implements a unified, discrete **VQ-VAE inspired coordinate space** that ingests all modalities into a single mathematical continuum, paired with an ultra-low-latency **Euclidean $L_2$ Nearest-Neighbor decoding engine** for real-time inference.

---

## 1. Orthogonal Modality Namespaces (10,000,000 Offsets)

To prevent categorical token collisions across different data types, [`Parser`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/parser.h) establishes wide, fixed numerical offsets of **10,000,000 slots** per modality:

| Modality | Offset Base | Valid Token ID Range | Token Generation Method |
| :--- | :--- | :--- | :--- |
| **Special / Padding** | `0` | `0` | Fixed padding token ID. |
| **Text** | `1` | `1` – `9,999,999` | Character ASCII values (`c + 1`) and subword hash buckets. |
| **Numbers** | `10,000,000` | `10,000,000` – `19,999,999` | Integer value hash mappings: `10000000 + (val % 10000000)`. |
| **Images** | `20,000,000` | `20,000,000` – `29,999,999` | VQ-VAE discrete 2D spatial codebook indices. |
| **Video** | `30,000,000` | `30,000,000` – `39,999,999` | Spatiotemporal 3D patch codebook indices. |
| **Voice / Audio** | `40,000,000` | `40,000,000` – `49,999,999` | Acoustic vector quantization latent codes. |

```
0            10,000,000     20,000,000     30,000,000     40,000,000     50,000,000
+---------------------------------------------------------------------------------+
|   TEXT     |   NUMBERS    |    IMAGES    |    VIDEO     |    VOICE     |   ...    |
+---------------------------------------------------------------------------------+
```

### Why Orthogonal Offsets Matter
Under this architecture, the character `'A'` (ASCII 65 $\rightarrow$ Token ID 66), the integer `66` (Token ID 10,000,066), and an image pixel codebook index (Token ID 20,000,066) exist in mathematically distinct regions of the categorical spectrum. They will never collide or produce ambiguous embeddings during training or inference.

---

## 2. Deterministic 4-Dimensional Embedding Coordinates

A categorical token ID (such as `uint32_t token = 72`) cannot be passed directly into a linear matrix multiplication layer without implying arbitrary numeric magnitudes.

In [`AI::embed_token()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp), each discrete token ID is projected into a continuous 4-dimensional embedding vector $\mathbf{e} \in \mathbb{R}^4$:

$$\mathbf{e}(t) = \begin{bmatrix} e_0 \\ e_1 \\ e_2 \\ e_3 \end{bmatrix} = \begin{bmatrix} \sin(t \cdot 0.01) \\ \cos(t \cdot 0.01) \\ \sin(t \cdot 0.0001) \\ \cos(t \cdot 0.0001) \end{bmatrix}$$

```cpp
std::vector<float> AI::embed_token(uint32_t token_id) const {
    if (token_id == 0) return {0.0f, 0.0f, 0.0f, 0.0f}; // Padding
    float t = static_cast<float>(token_id);
    return {
        std::sin(t * 0.01f),
        std::cos(t * 0.01f),
        std::sin(t * 0.0001f),
        std::cos(t * 0.0001f)
    };
}
```

This deterministic trigonometric encoding maps every integer token across the 50-million slot multimodal spectrum onto a bounded, smooth 4D hypersphere with unit norm properties, providing a continuous geometric representation for Tensor Core matrix multiplication.

---

## 3. The Interactive Inference Pipeline (`--use`)

When launched with the `--use` flag ([`app.exe --use`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/main.cpp)), the engine enters the interactive, zero-overhead user inference session:

```
[User Input Prompt]
         │
         ▼
[Parser::text_to_tokens()]  ===> Tokenized to std::vector<uint32_t>
         │
         ▼
[Window Padding / Slicing]  ===> Aligned to 128 tokens (512 floats)
         │
         ▼
[Tensor Core Forward Pass]  ===> y = ReLU(W*x + b) computed on RTX 4080
         │
         ▼
[Predicted Vector y in R^4] ===> Continuous 4D output
         │
         ▼
[Euclidean L2 Decode]       ===> Nearest neighbor match to discrete token
         │
         ▼
[Parser::tokens_to_text()]  ===> Converted back to character string
         │
         ▼
[Terminal Stream Output]
```

### Autoregressive Context Window Padding
If the user provides an input prompt with fewer than 128 tokens (e.g., `"Hello world"` = 11 tokens), [`AI::predict_interactive()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp) pads the beginning of the context window with zeros:

```cpp
std::vector<uint32_t> context(128, 0); // 128 zero-padded slots
size_t copy_count = std::min(tokens.size(), (size_t)128);
std::copy(tokens.end() - copy_count, tokens.end(), context.end() - copy_count);
```

The tokens are embedded into the 512-float buffer and dispatched to the GPU for the forward pass.

---

## 4. Euclidean $L_2$ Nearest-Neighbor Vocabulary Decoding

The neural network's forward pass outputs a continuous 4-dimensional coordinate:
$$\mathbf{\hat{y}} = \begin{bmatrix} \hat{y}_0 & \hat{y}_1 & \hat{y}_2 & \hat{y}_3 \end{bmatrix}^T \in \mathbb{R}^4$$

To convert this continuous vector back into a discrete human-readable token or character, the engine executes an **$L_2$ Euclidean Nearest-Neighbor Search** across candidate vocabulary embeddings:

$$t^* = \arg\min_{k \in \mathcal{V}} \|\mathbf{\hat{y}} - \mathbf{e}(k)\|_2 = \arg\min_{k \in \mathcal{V}} \sqrt{\sum_{i=0}^3 (\hat{y}_i - e_i(k))^2}$$

In [`AI::decode_vector_to_token()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp):

```cpp
uint32_t AI::decode_vector_to_token(const std::vector<float>& vec) const {
    uint32_t best_token = 0;
    float min_dist_sq = std::numeric_limits<float>::max();

    // Search active candidate vocabulary (ASCII printable range)
    for (uint32_t candidate = 32; candidate <= 126; ++candidate) {
        uint32_t token_id = candidate + 1; // Text offset
        std::vector<float> emb = embed_token(token_id);
        
        float dist_sq = 0.0f;
        for (size_t i = 0; i < 4; ++i) {
            float diff = vec[i] - emb[i];
            dist_sq += diff * diff;
        }

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_token = token_id;
        }
    }
    return best_token;
}
```

### Autoregressive Streaming Generation
Once candidate token $t^*$ is selected:
1. $t^*$ is mapped back to its ASCII character via [`Parser::tokens_to_text()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/parser.cpp).
2. The character is printed immediately to `std::cout` without newline buffering.
3. $t^*$ is appended to the context window, shifting the 128-token window forward by one slot.
4. The forward pass repeats, generating a continuous autoregressive stream of output text.
