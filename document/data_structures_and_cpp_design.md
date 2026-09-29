# AI Core: C++23 Design Patterns & Data Structures Specification

```
Document Version : 3.0.0
Standard         : ISO/IEC 14882:2023 (C++23)
Classification   : Systems Programming & Engineering Standards
Subsystems       : parser.hpp, parser.cpp, ai.hpp, ai.cpp, optimizer.hpp, optimizer.cpp, transfer.cuh, transfer.cu
```

---

## 1. Core Engineering Philosophy

The **AI Core** codebase adheres to foundational systems-level programming principles inspired by high-performance Linux kernel development and foundational AI runtimes (e.g., `llama.cpp`):

```
+---------------------------------------------------------------------------------------+
| PILLARS OF AI CORE SYSTEMS ENGINEERING                                                |
+---------------------------------------------------------------------------------------+
1. ZERO-DEPENDENCY MANDATE:
   Host CPU code is built strictly upon the ISO C++23 Standard Library. No external
   third-party framework overhead, no bloated runtimes, and zero opaque abstractions.

2. MECHANICAL SYMPATHY & CACHE LOCALITY:
   Prefer contiguous flat memory buffers (std::vector) over linked structures. Optimize
   for L1/L2 cache hit rates and PCIe DMA burst transfers.

3. STRICT SEPARATION OF CONCERNS (DEC-OPS):
   Input parsing, vocabulary dictionaries, and sensory streaming (Parser) are decoupled
   from downstream mathematical tensor execution (AI).

4. TYPE-EXPLICIT HARDWARE ALIGNMENT:
   Categorical IDs use exact-width unsigned integers (uint32_t), byte inputs use uint8_t,
   and continuous latent manifolds utilize native 16-bit Brain Floats (std::bfloat16_t).
+---------------------------------------------------------------------------------------+
```

---

## 2. Type System Evolution: Double $\rightarrow$ Float $\rightarrow$ Brain Float 16

In early prototypes, floating-point types (`double` - 64-bit) were used indiscriminately for continuous mathematical values, discrete integer identifiers, and media byte keys. **AI Core v3.0** enforces strict type boundaries that align directly with NVIDIA Tensor Core hardware logic:

```cpp
// =========================================================================
// ANTI-PATTERN: Using double for discrete IDs, binary byte keys, and neural arrays
// =========================================================================
std::unordered_map<double, double> image_dictionary; // KEY: double, VALUE: double
double next_image_id = 20000.0;
std::vector<double> tokenized_data;
std::vector<double> embedded_data; // BAD: 64-bit floats waste 75% memory bandwidth

// =========================================================================
// PRODUCTION PATTERN: Exact categorical types & Tensor Core native Brain Floats
// =========================================================================
std::unordered_map<uint8_t, uint32_t> image_dictionary; // KEY: byte, VALUE: ID
uint32_t next_image_id = 20000000;
std::vector<uint32_t> tokenized_data;
std::vector<std::bfloat16_t> embedded_data; // OPTIMAL: 16-bit native BF16 for Tensor Cores
```

### 2.1 Rationale for `uint32_t` Token Identifiers
1. **Integer Precision Guarantee:** A `double` uses a 53-bit mantissa. While exact for smaller integers, performing equality checks or using floating-point types as dictionary keys incurs CPU hashing penalties and risks rounding edge cases.
2. **Memory Compaction:** Storing 8,289,605 tokens as `uint32_t` requires **33.16 MB** of contiguous memory. Storing them as `double` requires **66.32 MB**—doubling memory bandwidth pressure over the system memory bus.
3. **Indexing Semantics:** Token IDs serve as direct row indices into embedding matrices ($E[\text{ID}]$). Array indices must be discrete unsigned integers (`uint32_t` or `size_t`).

### 2.2 Rationale for `std::bfloat16_t` (C++23 `<stdfloat>`)
Modern deep learning architectures (LLaMA, Gemini, GPT-4) standardize on **Brain Floating Point (BF16)**:
- **Bit Allocation:** 1 sign bit, 8 exponent bits, 7 mantissa (fraction) bits.
- **Dynamic Range Equivalence:** Because it possesses 8 exponent bits (identical to 32-bit `float`), `std::bfloat16_t` supports the exact same dynamic range ($\sim 10^{-38}$ to $10^{38}$) without causing underflow or gradient explosion.
- **50% Memory Reduction:** Each value occupies exactly **2 bytes** (`sizeof(std::bfloat16_t) == 2`). The entire 33,158,420 float manifold occupies only **66.32 MB** in RAM and VRAM (down from 132.63 MB in FP32).
- **Trivial Copyability:** Standard verification proves `std::is_trivially_copyable_v<std::bfloat16_t> == true`, ensuring raw pointer compatibility for direct DMA transfers over PCIe (`cudaMemcpyHostToDevice`).

### 2.3 Rationale for `uint8_t` Media Dictionary Keys
Binary files emit raw octets (`char` with values `0..255`). Storing them as `uint8_t` restricts the key domain strictly to $2^8 = 256$ entries, matching the physical reality of byte-level tokenization.

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

### Linage Analysis:

| Parameter | Type / Semantic | Rationale & Engineering Justification |
| :--- | :--- | :--- |
| `const T &data` | Input key (const ref) | Avoids deep copying of potentially expensive `std::string` text tokens. |
| `std::unordered_map<T, uint32_t> &dict` | Modality dictionary (ref) | Permits direct in-place mutation of the specific modality's vocabulary table. |
| `uint32_t &next_id` | Monotonic ID counter (ref) | Guarantees that token discovery monotonically increments the global namespace. |
| `Input &input` | State container (ref) | Directly appends the emitted token ID into the unified sequence buffer. |

---

## 4. Cross-Platform Filesystem Resolution: `std::filesystem`

`tokenize_from_disk` in `include/parser.hpp` implements a robust multi-path discovery heuristic using C++ `std::filesystem`:

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

---

## 5. Hardware Bridge Isolation: C ABI Dynamic Library Design

To bridge MinGW-w64 `g++` and NVIDIA `nvcc` (MSVC `cl.exe`), `src/CUDA/transfer.cuh` encapsulates GPU memory using opaque `void*` pointers and pure C linkage:

```cpp
#ifdef __cplusplus
extern "C" {
#endif

struct GpuPortion {
  void *d_context_window = nullptr; // Device pointer: Context Window X
  void *d_target         = nullptr; // Device pointer: Target Y
  size_t context_elements = 0;       // Element count in X
  size_t target_elements  = 0;       // Element count in Y
};

#ifdef BUILDING_CUDA_DLL
  #define CUDA_API __declspec(dllexport)
#else
  #define CUDA_API __declspec(dllimport)
#endif

CUDA_API bool upload_to_gpu(const void *h_context, size_t context_elements,
                            const void *h_target,  size_t target_elements,
                            GpuPortion *gpu);

CUDA_API bool download_from_gpu(void *h_context, size_t context_elements,
                                void *h_target,  size_t target_elements,
                                const GpuPortion *gpu);

CUDA_API void free_gpu(GpuPortion *gpu);

#ifdef __cplusplus
}
#endif
```

### Architectural Benefits:
1. **Host-Compiler Decoupling:** The host C++23 translation units never include NVIDIA-specific headers (`cuda_runtime.h` or `cuda_bf16.h`), eliminating MSVC-specific macro collisions.
2. **Zero-Reallocation Caching:** `upload_to_gpu()` inspects `gpu->context_elements`. If the window size matches the previous allocation, it skips `cudaFree` and `cudaMalloc`, reusing the active VRAM buffer.
3. **Double-Free Safety:** `free_gpu()` resets pointers to `nullptr` and element counts to 0, permitting idempotent calls without undefined behavior.
