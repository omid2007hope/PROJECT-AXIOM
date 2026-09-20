# AI Core: Multimodal Tokenization & Dense Embedding Specification

```
Document Version : 3.0.0
Standard         : ISO/IEC 14882:2023 (C++23)
Classification   : Algorithm & Representation Specification
Subsystems       : Parser (util/parser.cpp, include/parser.hpp)
Data Types       : uint32_t (Tokens), std::bfloat16_t (Continuous Latent Vectors)
```

---

## 1. Mathematical Foundations: From Discrete Symbols to Continuous Manifolds

Neural networks are continuous mathematical operators: their weights and activation functions exist in real vector space ($\mathbb{R}^d$). However, the physical world presents sensory inputs as discrete categorical entities (words, numeric readings, digital bytes).

**AI Core** bridges this domain transition through a strictly ordered two-stage pipeline:

```
+---------------------------------------------------------------------------------------+
| STAGE 1: DISCRETE TOKENIZATION                                                        |
| Mapping raw sensory entities into positive integer identifiers:                       |
|                             T : X -> Z^+  (uint32_t)                                  |
+---------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------+
| STAGE 2: DENSE VECTOR EMBEDDING                                                       |
| Projecting discrete integer identifiers into a continuous d-dimensional latent space: |
|                             E : Z^+ -> BF16^{d_model}  (std::bfloat16_t)              |
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
   - Mechanism: Scalar value matching via std::unordered_map<double, uint32_t>.
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

```
+---------------------------------------------------------------------------------------+
| SENSORY DOMAIN | RAW ASSET SIZE  | TOTAL STREAM TOKENS | UNIQUE VOCABULARY TOKENS     |
+----------------+-----------------+---------------------+------------------------------+
| Natural Text   | 9 words         | 9                   | 9    [1 .. 9]                |
| Numbers        | 12 values       | 12                  | 9    [10000000 .. 10000008]   |
| Visual Photo   | 130,513 Bytes   | 130,513             | 256  [20000000 .. 20000255]   |
| Temporal Video | 8,096,817 Bytes | 8,096,817           | 256  [30000000 .. 30000255]   |
| Acoustic Voice | 62,254 Bytes    | 62,254              | 256  [40000000 .. 40000255]   |
+----------------+-----------------+---------------------+------------------------------+
| TOTAL CENSUS   | 8,289,605 Bytes | 8,289,605 tokens    | 786 unique tokens            |
+---------------------------------------------------------------------------------------+
```

### 3.1 The 256-Byte Ceiling
In the media modalities (Image, Video, Voice), the vocabulary dictionaries contain exactly **256 unique tokens** each. 
- A digital file read in binary mode (`std::ios::binary`) emits single bytes (`char byte`).
- Casting `byte` to `uint8_t` produces a discrete domain of $2^8 = 256$ possible values ($0 \le b \le 255$).
- Consequently, regardless of whether a media file is 100 Kilobytes or 10 Gigabytes, the unique codebook size for raw byte tokenization is capped at 256 tokens per modality.

---

## 4. Dense Continuous Embedding Projection in `std::bfloat16_t`

Once the sequence of discrete integer IDs is populated in `input.tokenized_data`, the cognitive engine requires continuous vector representations.

In **AI Core v3.0**, dense vectors are represented natively in 16-bit Brain Floating Point (`std::bfloat16_t`).

```
Input: Token ID (uint32_t) ---> [ embedding_table Lookup ] ---> Dense BF16 Vector [d_model = 4]
```

- **Embedding Dimension ($d_{\text{model}}$):** Set to `size_t embedding_dim = 4` in `Input`.
- **Lookup Table:** `std::unordered_map<uint32_t, std::vector<std::bfloat16_t>> embedding_table`.
- **Output Tensor:** A contiguous flat buffer `std::vector<std::bfloat16_t> embedded_data` of size:
  $$\text{Total BF16 Elements} = N_{\text{tokens}} \times d_{\text{model}} = 8,289,605 \times 4 = 33,158,420 \text{ elements}$$
- **Storage Footprint:**
  $$\text{Disk File Size} = 33,158,420 \times 2 \text{ bytes} = 66,316,840 \text{ bytes} \approx 66.32 \text{ MB}$$

### 4.1 Deterministic Harmonic Initialization

To enable consistent, reproducible neural forward passes without requiring multi-gigabyte pretrained weights on disk, `Parser::embedding()` synthesizes orthogonal continuous embeddings deterministically:

```cpp
std::vector<std::bfloat16_t> Parser::embedding(Input &input, Test &test) {
  input.embedded_data.clear();
  input.embedded_data.reserve(input.tokenized_data.size() * input.embedding_dim);

  for (uint32_t id : input.tokenized_data) {
    if (input.embedding_table.find(id) == input.embedding_table.end()) {
      std::vector<std::bfloat16_t> vec(input.embedding_dim);
      for (uint32_t d = 0; d < input.embedding_dim; ++d) {
        vec[d] = static_cast<std::bfloat16_t>(
            std::sin(static_cast<double>(id) * (d + 1.0)));
      }
      input.embedding_table[id] = std::move(vec);
    }

    const auto &vec = input.embedding_table[id];
    input.embedded_data.insert(input.embedded_data.end(), vec.begin(), vec.end());
  }

  save_data(input.embedded_data, "embedded_data.bin");
  return input.embedded_data;
}
```

### 4.2 Mathematical Properties of Harmonic BF16 Embeddings:
1. **Bounded Invariant:** Since $\sin(x) \in [-1.0, 1.0]$, all embedding values are inherently normalized, preventing gradient explosion in early layers.
2. **Deterministic Stability:** The token ID `1` ("The") will *always* generate identical continuous coordinates:
   $$\mathbf{v}_1 \approx [0.839844, 0.910156, 0.141602, -0.757812]$$
3. **Orthogonal Clustering by Modality:** Because token IDs are separated by 10,000,000 offsets, harmonic frequencies create distinct geometric clusters in latent space for each modality.
4. **Direct PCIe Alignment:** Output buffer `embedded_data` and file `database/embedded_data.bin` are bit-exact copies ready to stream into RAM and VRAM for consumption by the AI Tensor Cores.
