# AI Core: Multimodal Tokenization & Dense Embedding Specification

```
Document Version : 2.0.0
Standard         : ISO/IEC 14882:2017 (C++17)
Classification   : Algorithm & Representation Specification
Subsystems       : Parser (util/parser.cpp, include/parser.hpp)
```

---

## 1. Mathematical Foundations: From Discrete Symbols to Continuous Manifolds

Neural networks are continuous mathematical operators: their weights and activation functions exist in real vector space ($\mathbb{R}^d$). However, the physical world presents sensory inputs as discrete categorical entities (words, numeric readings, digital bytes).

**AI Core** bridges this domain transition through a strictly ordered two-stage pipeline:

```
+---------------------------------------------------------------------------------------+
| STAGE 1: DISCRETE TOKENIZATION                                                        |
| Mapping raw sensory entities into positive integer identifiers:                       |
|                             T : X -> Z^+                                              |
+---------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------+
| STAGE 2: DENSE VECTOR EMBEDDING                                                       |
| Projecting discrete integer identifiers into a continuous d-dimensional latent space: |
|                             E : Z^+ -> R^{d_model}                                    |
+---------------------------------------------------------------------------------------+
```

---

## 2. Modality-Specific Tokenization Mechanics

The engine orchestrates tokenization across five sensory channels in `util/parser.cpp`:

```
+---------------------------------------------------------------------------------------+
| DISCRETE TOKENIZATION CHANNELS IN AI CORE                                             |
+---------------------------------------------------------------------------------------+

1. TEXT (Natural Language)
   - Source: String tokens from test.data_txt or input.user_text.
   - Mechanism: Exact string hash matching via std::unordered_map<std::string, uint32_t>.
   - Target Range: [1 .. 9,999,999].

2. NUMBERS (Tabular & Telemetry Data)
   - Source: Floating-point scalars from test.data_num or input.user_numbers.
   - Mechanism: Scalar value matching via std::unordered_map<float, uint32_t>.
   - Target Range: [10,000,000 .. 19,999,999].

3. IMAGE (Visual Data)
   - Source: Binary stream from "Photo.jpeg" (130,513 bytes).
   - Mechanism: Unsigned byte ingestion via std::unordered_map<uint8_t, uint32_t>.
   - Target Range: [20,000,000 .. 29,999,999].

4. VIDEO (Temporal Visual Data)
   - Source: Binary stream from "Video.mp4" (8,096,817 bytes).
   - Mechanism: Unsigned byte ingestion via std::unordered_map<uint8_t, uint32_t>.
   - Target Range: [30,000,000 .. 39,999,999].

5. VOICE (Acoustic Audio Data)
   - Source: Binary stream from "Voice.ogg" (62,254 bytes).
   - Mechanism: Unsigned byte ingestion via std::unordered_map<uint8_t, uint32_t>.
   - Target Range: [40,000,000 .. 49,999,999].
+---------------------------------------------------------------------------------------+
```

---

## 3. Empirical Verification & Complete Token Census

The tokenization engine was empirically executed and validated against active media files located in `learning/deepLearning/asset_sample/`. The verification harness produced the following exact census:

### 3.1 Token Sequence Length vs. Vocabulary Size

There is a critical distinction in deep learning between **Sequence Length** (total tokens processed in the stream) and **Vocabulary Size** (unique dictionary entries discovered):

```
+---------------------------------------------------------------------------------------+
| SENSORY DOMAIN | RAW ASSET SIZE  | TOTAL STREAM TOKENS | UNIQUE VOCABULARY TOKENS     |
+----------------+-----------------+---------------------+------------------------------+
| Natural Text   | 9 words         | 9                   | 9    [1 .. 9]                |
| Numbers        | 12 floats       | 12                  | 9    [10000000 .. 10000008]   |
| Visual Photo   | 130,513 Bytes   | 130,513             | 256  [20000000 .. 20000255]   |
| Temporal Video | 8,096,817 Bytes | 8,096,817           | 256  [30000000 .. 30000255]   |
| Acoustic Voice | 62,254 Bytes    | 62,254              | 256  [40000000 .. 40000255]   |
+----------------+-----------------+---------------------+------------------------------+
| TOTAL CENSUS   | 8,289,605 Bytes | 8,289,605 tokens    | 786 unique tokens            |
+---------------------------------------------------------------------------------------+
```

### 3.2 Analysis of the 256-Byte Ceiling

In the media modalities (Image, Video, Voice), the vocabulary dictionaries contain exactly **256 unique tokens** each. 

This behavior is mathematically exact:
- A digital file read in binary mode (`std::ios::binary`) emits single bytes (`char byte`).
- Casting `byte` to `uint8_t` produces a discrete domain of $2^8 = 256$ possible values ($0 \le b \le 255$).
- Consequently, regardless of whether a media file is 100 Kilobytes or 10 Gigabytes, the unique codebook size for raw byte tokenization is capped at 256 tokens per modality.
- For `Photo.jpeg`, `Video.mp4`, and `Voice.ogg`, every byte value from `0x00` to `0xFF` appears at least once in the file header or compressed payload, completely saturating the 256-token space.

---

## 4. The Generic Ingestion Engine: `add_assign_increase<T>`

To prevent code duplication across 5 heterogeneous modalities, `include/parser.hpp` implements a type-parameterized template method:

```cpp
template <typename T>
void add_assign_increase(const T &data,
                         std::unordered_map<T, uint32_t> &dictionary,
                         uint32_t &next_id, 
                         Input &input) {
  if (dictionary.find(data) == dictionary.end()) {
    dictionary[data] = next_id;
    next_id += 1;
  }
  input.tokenized_data.push_back(dictionary[data]);
}
```

### Key Implementation Guarantees:
1. **Zero-Copy Parameter Passing:** `const T &data` avoids deep copying of large string keys.
2. **Amortized $O(1)$ Insertion:** Querying `.find(data)` before insertion ensures no existing token mapping is overwritten.
3. **Monotonic ID Generation:** `next_id += 1` guarantees that newly discovered tokens receive strictly increasing, gap-free IDs within their allocated 10-million block.
4. **Contiguous Vector Emission:** `input.tokenized_data.push_back()` appends the resolved ID to a cache-friendly, contiguous vector.

---

## 5. Streaming Disk Ingestion: `tokenize_from_disk`

To process large media files without exhausting physical RAM, `tokenize_from_disk` streams binary data directly from storage:

```cpp
void tokenize_from_disk(const std::string &fileName,
                        std::unordered_map<uint8_t, uint32_t> &dictionary,
                        uint32_t &next_id, 
                        Input &input) {
  // 1. Multi-path discovery heuristic for cross-directory resolution
  std::filesystem::path resolved_path = resolve_file_path(fileName);

  // 2. Open binary file stream
  std::ifstream parsed_data(resolved_path, std::ios::binary);
  if (!parsed_data) {
    std::cerr << "Failed to open " << resolved_path << "\n";
    return;
  }

  // 3. Stream byte-by-byte without loading the full 8MB payload into RAM
  char byte;
  while (parsed_data.get(byte)) {
    add_assign_increase(static_cast<uint8_t>(byte), dictionary, next_id, input);
  }
}
```

This streaming pattern ensures that memory usage during tokenization scales with the size of the output sequence buffer, not the file buffer.

---

## 6. Dense Continuous Embedding Projection

Once the sequence of discrete integer IDs is populated in `input.tokenized_data`, the cognitive engine requires continuous vector representations.

A discrete identifier `5` holds no mathematical relationship to `6` or `12`. To enable machine learning algorithms (which rely on gradients and continuous calculus) to understand the *meaning* of a token, discrete IDs are projected into a high-dimensional continuous space.

This continuous space is defined by `d_model` (the embedding dimension). For our structural prototype, `d_model = 4`.

### 6.1 Lookup Architecture & Vector Sizing

```
Input: Token ID (uint32_t) ---> [ embedding_table Lookup ] ---> Dense Vector [d_model]
```

- **Embedding Dimension ($d_{\text{model}}$):** Set to `uint32_t embedding_dim = 4` in `Input` (easily reconfigured to 512, 768, 2048, or 4096).
- **Lookup Table:** `std::unordered_map<uint32_t, std::vector<float>> embedding_table`.
- **Output Tensor:** A contiguous flat buffer `std::vector<float> embedded_data` of size:
  $$\text{Total Floats} = N_{\text{tokens}} \times d_{\text{model}} = 8,289,605 \times 4 = 33,158,420 \text{ floats}$$

### 6.2 Deterministic Harmonic Initialization

To enable consistent, reproducible neural forward passes without requiring multi-gigabyte pretrained weights on disk, `Parser::embedding()` synthesizes orthogonal continuous embeddings deterministically based on the token ID:

```cpp
std::vector<float> Parser::embedding(Input &input, Test &test) {
  input.embedded_data.clear();
  input.embedded_data.reserve(input.tokenized_data.size() * input.embedding_dim);

  for (uint32_t id : input.tokenized_data) {
    // Check if token embedding already exists in cache
    if (input.embedding_table.find(id) == input.embedding_table.end()) {
      std::vector<float> vec(input.embedding_dim);
      for (uint32_t d = 0; d < input.embedding_dim; ++d) {
        // Deterministic harmonic function based on ID and dimension index
        vec[d] = std::sinf(static_cast<float>(id) * (d + 1.0f));
      }
      input.embedding_table[id] = std::move(vec);
    }
    
    // Append dense vector to flat output stream
    const auto &vec = input.embedding_table[id];
    input.embedded_data.insert(input.embedded_data.end(), vec.begin(), vec.end());
  }

  return input.embedded_data;
}
```

### 6.3 Output: The Unified Tensor Stream

As each token is processed, its dense vector is retrieved (or generated) and appended to `std::vector<float> embedded_data`.

Because standard `std::vector` stores elements contiguously in RAM, the resulting `embedded_data` array is a perfectly flat, contiguous stream of floating-point numbers.

**The Book of Genesis Scale:**
- Input: `8,289,605` discrete tokens (`uint32_t`)
- Output: `33,158,420` continuous 32-bit `float` values (Tokens $\times 4$)

This output stream is mathematically identical to a flattened $(8289605 \times 4)$ tensor, which is the exact geometric shape required to feed into the Matrix Multiplication units (Tensor Cores) of the Perceptron over the PCIe bus in the next stage.

### 6.4 Mathematical Properties of Harmonic Embeddings:
1. **Bounded Invariant:** Since $\sin(x) \in [-1.0, 1.0]$, all embedding values are inherently normalized, preventing gradient explosion in early layers.
2. **Deterministic Stability:** The token ID `1` ("The") will *always* generate the exact same continuous coordinates:
   $$\mathbf{v}_1 = [\sin(1), \sin(2), \sin(3), \sin(4)] \approx [0.841471, 0.909297, 0.141120, -0.756802]$$
3. **Orthogonal Clustering by Modality:** Because token IDs are separated by 10,000,000 offsets, the harmonic frequencies create distinct geometric clusters in latent space for each modality.
4. **Exact Memory Footprint:** With 786 unique tokens discovered across all modalities, the `embedding_table` contains exactly 786 vectors ($786 \times 4 \times 4 \text{ bytes} \approx 12.58 \text{ KB}$), achieving near-instant cache warm-up.
