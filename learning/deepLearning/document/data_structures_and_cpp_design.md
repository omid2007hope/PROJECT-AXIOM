# C++ Design Patterns & Data Structures

## 1. Core Engineering Principles

The DeepLearning Core is designed around strict systems-level principles:
- **Zero-Dependency Mandate:** Built entirely with standard modern C++ (C++17 standard library) without reliance on external frameworks (e.g., PyTorch, ONNX, Eigen).
- **Simplicity Over Complexity:** Minimalistic, transparent data structures that expose raw computational mechanics rather than obfuscating them behind deep abstraction hierarchies.
- **Strict Decoupling:** Complete separation between the input parsing pipeline (`Parser`) and the neural execution pipeline (`Perceptron`).

---

## 2. Associative Containers & Dictionary Mechanics

The tokenization subsystem relies heavily on hash tables (`std::unordered_map`) to maintain associative vocabularies for each modality:

```cpp
std::unordered_map<std::string, double> text_vocab;
std::unordered_map<double, double> number_vocab;
std::unordered_map<double, double> image_vocab;
std::unordered_map<double, double> video_vocab;
std::unordered_map<double, double> voice_vocab;
```

### Hash Table Mechanics & Complexity
- **Lookup:** Average $O(1)$, Worst-case $O(N)$ (in the event of hash bucket collision storms).
- **Insertion:** Average $O(1)$ amortized.

### Resolving Common C++ Dictionary Misconceptions

#### A. The Meaning of `.find(item) == vocab.end()`
In C++, calling `find()` on an associative container traverses the bucket array:
- If the target key exists anywhere in the dictionary, `find()` returns an iterator pointing directly to the key-value pair.
- If the target key does **not** exist, `find()` returns the special sentinel iterator `vocab.end()`.
- **Critical Clarification:** `vocab.end()` does **not** point to the "last element" in the dictionary. It points to a one-past-the-end sentinel marker. In software engineering terms, `find() == end()` is the exact standard library equivalent of an **HTTP 404: Not Found** signal.

#### B. The Semantics of `vocab[item] = next_id`
Using `operator[]` on a `std::unordered_map` performs an **upsert** (insert if absent, assign if present):
- If the key is not found, `operator[]` allocates a new bucket entry, default-initializes the value, and binds the assignment.
- When preceded by a `find() == end()` check, this guarantees the insertion of a fresh token mapping without overwriting established IDs.

---

## 3. The DRY Refactoring: Template Metaprogramming

### The Code Duplication Issue
Prior to refactoring, each modality loop in `util/parser.cpp` executed duplicate insertion and ID increment logic:

```cpp
// Redundant across 5 modalities:
if (input.text_vocab.find(word) == input.text_vocab.end()) {
  input.text_vocab[word] = input.next_text_id;
  input.next_text_id += 1.0;
}
input.tokenized_data.push_back(input.text_vocab[word]);
```

Because text relies on `std::string` keys while sensory inputs rely on `double` keys, a standard function cannot accept both without code duplication or type erasure.

### The Template Solution
By introducing a C++ template method in the `Parser` class (`include/parser.hpp`), a single parameterized routine handles arbitrary key types seamlessly:

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

### Parameter Reference & Performance Lineage
- `const T &item`: Passed by constant reference to avoid expensive heap allocations or copying (particularly important for `std::string`).
- `std::unordered_map<T, double> &vocab`: Passed by mutable reference, allowing direct in-place mutation of the specific modality's dictionary.
- `double &next_id`: Passed by mutable reference, ensuring the persistent state counter for that modality is monotonically incremented.
- `std::vector<double> &output_data`: Passed by mutable reference to append resolved token IDs to the contiguous `tokenized_data` buffer.

---

## 4. Verification & Testing Strategy

The test harness in `test/test_parser.cpp` validates the end-to-end data pipeline:
1. **Instantiation:** Instantiates the raw sample tensors (`Test`) and tokenizer state (`Input`).
2. **Execution:** Calls `Parser::tokenization()` and `Parser::embedding()`.
3. **Verification Points:**
   - Unified token count matches total input elements ($9 + 12 + 15 + 12 + 15 = 63$).
   - Text tokens occupy $[1, 9999]$.
   - Number tokens occupy $[10000, 19999]$.
   - Image tokens occupy $[20000, 29999]$.
   - Video tokens occupy $[30000, 39999]$.
   - Voice tokens occupy $[40000, 49999]$.
   - Duplicate image pixels (e.g., repeated `0.0`, `128.0`, `255.0`) resolve to identical Token IDs without leaking new IDs.
   - Embeddings produce the correct 3-dimensional projection per token ($63 \times 3 = 189$ elements).
