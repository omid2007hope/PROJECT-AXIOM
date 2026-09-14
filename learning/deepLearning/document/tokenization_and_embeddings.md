# Tokenization & Embeddings Specification

## 1. Evolution of Text Representation in Machine Learning

Machine learning models are strictly numerical systems; they cannot directly ingest strings or raw sensory data. The methodology for converting text into numerical tensors has evolved through three distinct paradigms:

### Phase 1: Direct ASCII / Character Encoding
- Each character is represented by its raw 8-bit integer ASCII value (e.g., `'a' = 97`, `'b' = 98`).
- **Limitation:** Fails to capture word-level semantics. Words are variable-length arrays of characters with high structural noise and zero conceptual meaning.

### Phase 2: Polynomial Rolling Hash
- Combines character values into a single scalar integer using a polynomial hash:
  $$\text{hash}(S) = \left( \sum_{i=0}^{n-1} S[i] \cdot p^i \right) \pmod m$$
  where $p$ is a prime base (e.g., 31) and $m$ is a large modulo (e.g., $10^9 + 9$).
- **Strengths:** Eliminates anagram collisions (e.g., `"cat"` vs. `"act"` yield completely distinct values). Dividing by $m$ normalizes the output into the range $[0.0, 1.0]$, matching normalized pixel inputs.
- **Critical Limitation:** Squashing an entire word into a single scalar destroys all relational geometry. Two semantically related words (e.g., `"doctor"` and `"nurse"`) produce arbitrary numerical scalar distances with no semantic correlation.

### Phase 3: Modern Tokenization & Dense Embeddings (Industry Standard)
Modern large language and multimodal models (LLaMA, GPT-4, Gemini) universally employ a **two-step pipeline**:
1. **Tokenization:** Text $\to$ Sequence of discrete integer IDs ($z \in \mathbb{Z}^+$).
2. **Embedding:** Discrete integer ID $\to$ Continuous dense vector ($\mathbf{v} \in \mathbb{R}^d$).

```
Raw Text: "The quick brown fox"
         |
         v [Step 1: Tokenizer / Dictionary Lookup]
Token IDs: [1, 2, 3, 4]
         |
         v [Step 2: Embedding Matrix Lookup / Projection]
Dense Vectors:
  Token 1 -> [ 0.12, -0.45,  0.88, ... d_model ]
  Token 2 -> [-0.03,  0.71, -0.19, ... d_model ]
  Token 3 -> [ 0.54, -0.22,  0.33, ... d_model ]
  Token 4 -> [-0.81,  0.09,  0.62, ... d_model ]
```

---

## 2. Tokenization Deep Dive

### Sub-word Tokenization (BPE & SentencePiece)
In production models, tokenization does not operate purely on whole words (which causes out-of-vocabulary issues and dictionary bloat) nor purely on characters (which causes excessive sequence lengths).

- **Byte-Pair Encoding (BPE):** Starts with base characters and iteratively merges the most frequent byte pairs into sub-word units.
  - `"unbelievable"` $\to$ `["un", "believ", "able"]` $\to$ `[42, 876, 230]`.
- **Vocabulary Size:** Typically fixed between $32,000$ and $128,000$ tokens.

### Discrete Tokenization Across Modalities
In the DeepLearning Core parser, tokenization is expanded across five distinct modalities:

1. **Text (`data_txt`)**: Extracted as words and mapped to integer IDs via `text_vocab`.
2. **Numbers (`data_num`)**: Categorized by numeric value into `number_vocab`.
3. **Images (`data_img`)**: In a production system, visual continuous patches are quantized using a pre-trained **Vector Quantized Variational AutoEncoder (VQ-VAE)** codebook. In this engine, pixel values are mapped into discrete visual tokens (`image_vocab`).
4. **Video (`data_video`)**: Frame-to-frame temporal deltas mapped to dynamic optical flow tokens (`video_vocab`).
5. **Voice (`data_voice`)**: Normalized PCM audio amplitudes mapped to acoustic tokens (`voice_vocab`).

---

## 3. Dense Vector Embeddings

### Why Dense Vectors?
While token IDs are discrete integers, deep learning operations rely on continuous gradient calculus (backpropagation). Embedding tables provide the differentiable interface between discrete symbols and continuous vector spaces.

Key properties of dense embedding spaces:
- **Latent Dimensionality ($d_{\text{model}}$):** Typically 768 to 4096 dimensions in modern models.
- **Semantic Distance:** Words with similar meanings cluster together under cosine similarity:
  $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
- **Vector Compositionality:** Geometric arithmetic reflects conceptual relationships:
  $$\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} + \mathbf{v}_{\text{Woman}} \approx \mathbf{v}_{\text{Queen}}$$

### Engine Implementation (`Parser::embedding`)
In the prototype parser, each discrete token is projected into an embedding vector. To simulate distinct semantic manifolds across segregated modalities without an external trained weight checkpoint, the engine utilizes orthogonal trigonometric harmonic functions:

```cpp
double embed_dim1 = std::sin(value * 1.5);
double embed_dim2 = std::cos(value * 2.1);
double embed_dim3 = std::sin(value * 3.3) * std::cos(value * 0.5);
```

Because token values are segregated by modality offsets ($1$, $10000$, $20000$, etc.), these periodic functions produce orthogonal feature clusters for text, images, audio, video, and tabular numbers.

---

## 4. The `add_assign_increase` Template Engine

To adhere to the **DRY (Don't Repeat Yourself)** principle while handling heterogeneous key types (`std::string` for text, `double` for numerical and sensor modalities), the parser implements a generic C++ template helper:

```cpp
template <typename T>
void add_assign_increase(const T &item, 
                         std::unordered_map<T, double> &vocab,
                         double &next_id, 
                         std::vector<double> &output_data) {
  if (vocab.find(item) == vocab.end()) {
    vocab[item] = next_id;
    next_id += 1.0;
  }
  output_data.push_back(vocab[item]);
}
```

### Execution Flow
1. **Query:** Checks dictionary membership in $O(1)$ average time using `vocab.find(item)`.
2. **Insert on Miss:** If the key does not exist (`== vocab.end()`), assigns the current `next_id` and increments the modality counter.
3. **Emit:** Appends the resolved token ID to `output_data` (`input.tokenized_data`).
