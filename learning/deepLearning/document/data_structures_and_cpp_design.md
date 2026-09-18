# AI Core: C++17 Design Patterns & Data Structures Specification

```
Document Version : 2.0.0
Standard         : ISO/IEC 14882:2017 (C++17)
Classification   : Systems Programming & Engineering Standards
Subsystems       : parser.hpp, parser.cpp, perceptron.hpp, perceptron.cpp
```

---

## 1. Core Engineering Philosophy

The **AI Core** codebase adheres to foundational systems-level programming principles inspired by high-performance Linux kernel development and foundational AI runtimes (e.g., `llama.cpp`):

```
+---------------------------------------------------------------------------------------+
| PILLARS OF AI CORE SYSTEMS ENGINEERING                                                |
+---------------------------------------------------------------------------------------+
1. ZERO-DEPENDENCY MANDATE:
   Built strictly upon the ISO C++17 Standard Library. No external dependencies, no
   third-party framework overhead, and zero opaque abstractions.

2. MECHANICAL SYMPATHY & CACHE LOCALITY:
   Prefer contiguous flat memory buffers (std::vector) over linked structures. Optimize
   for L1/L2 cache hit rates during high-throughput forward passes.

3. STRICT SEPARATION OF CONCERNS (DEC-OPS):
   Input parsing, vocabulary dictionaries, and sensory streaming (Parser) are decoupled
   from downstream mathematical tensor execution (Perceptron).

4. TYPE-EXPLICIT SAFETY:
   Categorical IDs use exact-width unsigned integers (uint32_t), byte inputs use uint8_t,
   and neural activations use double (or float). No floating-point types for identifiers.
+---------------------------------------------------------------------------------------+
```

---

## 2. Type System Refactoring: The "Double" Anti-Pattern

In early prototypes, floating-point types (`double` - 64-bit) were used indiscriminately for continuous mathematical values, discrete integer identifiers, and media byte keys. **AI Core v2.0** enforces strict type boundaries to align perfectly with Tensor Core hardware logic:

```cpp
// =========================================================================
// ANTI-PATTERN: Using double for discrete IDs, binary byte keys, and neural arrays
// =========================================================================
std::unordered_map<double, double> image_dictionary; // KEY: double, VALUE: double
double next_image_id = 20000.0;
std::vector<double> tokenized_data;
std::vector<double> embedded_data; // BAD: 64-bit floats tank Tensor Core performance

// =========================================================================
// PRODUCTION PATTERN: Exact-width categorical types & GPU-aligned 32-bit floats
// =========================================================================
std::unordered_map<uint8_t, uint32_t> image_dictionary; // KEY: byte, VALUE: ID
uint32_t next_image_id = 20000000;
std::vector<uint32_t> tokenized_data;
std::vector<float> embedded_data; // GOOD: 32-bit float ready for CUDA & Tensor Cores
```

### 2.1 Rationale for `uint32_t` Token Identifiers
1. **Integer Precision Guarantee:** A `double` (IEEE 754 double precision) uses a 53-bit mantissa. While exact for smaller integers, performing equality checks (`id_a == id_b`) or using floating-point types as dictionary keys incurs CPU hashing penalties and risks rounding edge cases.
2. **Memory Compaction:** Storing 8,289,605 tokens as `uint32_t` requires **33.16 MB** of contiguous memory. Storing them as `double` requires **66.32 MB**—doubling the memory bandwidth pressure over the system memory bus.
3. **Indexing Semantics:** Token IDs serve as direct row indices into embedding matrices ($E[\text{ID}]$). Standard array and tensor indices must be discrete unsigned integers (`uint32_t` or `size_t`).

### 2.2 Rationale for `float` (32-bit) Neural Tensors
A Tensor Core is a physical silicon chip hardwired to perform massive parallel multiplication on 32-bit (`float`), 16-bit (FP16), or Brain-Floats (BF16). If a 64-bit `double` is pushed to a Tensor Core, the GPU will silently fall back to legacy graphics cores, causing training to run exponentially slower. Transitioning to 32-bit `float` halves the RAM footprint and seamlessly prepares the buffers for `cudaMemcpy` to the GPU.

### 2.3 Rationale for `uint8_t` Media Dictionary Keys
Binary files emit raw octets (`char` with values `0..255`). Storing them as `double` keys in `unordered_map<double, uint32_t>` caused unnecessary float conversions and 8-byte key overhead. Using `uint8_t` restricts the key domain strictly to $2^8 = 256$ entries, matching the physical reality of byte-level tokenization.

---

## 3. Generic Ingestion Engine: `add_assign_increase<T>`

To avoid writing five duplicated insertion loops for text, numbers, images, video, and audio, `Parser` implements a generic template method:

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

### 3.1 Detailed Linage Analysis:

| Parameter | Type / Semantic | Rationale & Engineering Justification |
| :--- | :--- | :--- |
| `const T &data` | Input key (const ref) | Avoids deep copying of potentially expensive `std::string` text tokens. |
| `std::unordered_map<T, uint32_t> &dict` | Modality dictionary (ref) | Permits direct in-place mutation of the specific modality's vocabulary table. |
| `uint32_t &next_id` | Monotonic ID counter (ref) | Guarantees that token discovery monotonically increments the global namespace. |
| `Input &input` | State container (ref) | Directly appends the emitted token ID into the unified sequence buffer. |

### 3.2 Dictionary Mechanics: `.find()` vs. `operator[]`
In standard C++, using `dictionary[data]` unconditionally causes an **implicit default insertion** if the key is missing. By explicitly checking `if (dictionary.find(data) == dictionary.end())`:
- We unambiguously identify novel tokens.
- We assign the next sequential ID without accidental default construction.
- We increment the specific modality counter `next_id` only when a genuinely new token is cataloged.

---

## 4. Cross-Platform Filesystem Resolution: `std::filesystem`

Local execution environments present varying directory structures depending on how the application is launched (e.g., from the repository root, from the subfolder, or through test runners).

`tokenize_from_disk` in `include/parser.hpp` implements a robust multi-path discovery heuristic using C++17 `std::filesystem`:

```cpp
std::filesystem::path c_standard_fileName(fileName);

if (!c_standard_fileName.is_absolute()) {
  std::filesystem::path absolute_path(__FILE__);
  std::vector<std::filesystem::path> possible_paths = {
      absolute_path.parent_path().parent_path() / "asset_sample" / fileName,
      std::filesystem::path("Brain/learning/deepLearning/asset_sample") / fileName,
      std::filesystem::path("learning/deepLearning/asset_sample") / fileName,
      std::filesystem::path("deepLearning/asset_sample") / fileName,
      std::filesystem::path("asset_sample") / fileName
  };

  for (const auto &each_path : possible_paths) {
    if (std::filesystem::exists(each_path)) {
      c_standard_fileName = each_path;
      std::cout << "Found in\n" << c_standard_fileName << "\n";
      break;
    }
  }
}
```

### Architectural Benefits:
1. **Compile-Time Location Anchoring:** `std::filesystem::path absolute_path(__FILE__)` resolves relative to the source header itself, allowing test executables to run from any working directory.
2. **Graceful Fallback:** Checks alternative working directory paths in sequence before reporting failure.
3. **Cross-Platform Path Construction:** Uses the `/` operator of `std::filesystem::path`, which automatically maps to forward slashes on POSIX and backslashes on Windows.

---

## 5. Memory Safety & Move Semantics

In the embedding layer (`Parser::embedding` in `util/parser.cpp`), continuous vectors are generated and stored into the lookup table:

```cpp
if (input.embedding_table.find(id) == input.embedding_table.end()) {
  std::vector<float> vec(input.embedding_dim);
  for (uint32_t d = 0; d < input.embedding_dim; ++d) {
    vec[d] = std::sinf(static_cast<float>(id) * (d + 1.0f));
  }
  // Move semantics transfer vector ownership without heap reallocation
  input.embedding_table[id] = std::move(vec);
}
```

### Performance Impact:
- `std::move(vec)` casts the locally constructed vector to an rvalue reference, transferring its internal pointer, size, and capacity to the hash map bucket.
- This eliminates heap allocation copies for the 786 unique vocabulary vectors during initialization.
- Pre-allocating the sequence buffer via `input.embedded_data.reserve(...)` prevents repeated reallocation and memory fragmentation as 33.1 million floats are written.
