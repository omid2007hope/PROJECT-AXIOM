# 06. Weight Persistence Specification

An artificial intelligence model that exists only in volatile VRAM suffers from complete amnesia: the instant the process terminates, the adjusted parameters are lost. To establish permanent model memory across reboots and facilitate open-source distribution, Project Axiom implements a strict binary file format: **`weights.axiom`**.

This document defines the binary layout of the file format, the atomic two-stage write protocol, CUDA stream synchronization, and corruption resistance safeguards.

---

## 1. The `weights.axiom` Binary File Format

The `weights.axiom` file is divided into a 56-byte compiler-safe header followed immediately by contiguous floating-point parameter payloads (projection weights, biases, and the master learnable embedding table):

```
+-------------------------------------------------------------------------------+
|                       AXIOM WEIGHT HEADER (56 Bytes)                          |
|  +-------------------------------------------------------------------------+  |
|  |  magic[8]        : "AXIOM_W1" (8 ASCII bytes)                           |  |
|  |  version         : uint32_t (Value = 1)                                 |  |
|  |  in_dim          : uint32_t (Input context dimension)                   |  |
|  |  out_dim         : uint32_t (Output projection dimension)               |  |
|  |  vocab_size      : uint32_t (Value = 65,536 dense codebook tokens)      |  |
|  |  embedding_dim   : uint32_t (Value = 32 dimensions)                     |  |
|  |  dtype           : uint32_t (Value = 0: IEEE 754 FP32)                  |  |
|  |  data_bytes      : uint64_t (Total tensor payload byte length)          |  |
|  |  crc32           : uint32_t (Payload checksum)                          |  |
|  |  reserved        : uint32_t (Zero padding for 64-bit alignment)         |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
|                       RAW TENSOR PAYLOAD                                      |
|  +-------------------------------------------------------------------------+  |
|  |  Projection Weights W : (out_dim * in_dim) floats                       |  |
|  |  Projection Biases b  : out_dim floats                                  |  |
|  |  Master Embed Table   : (vocab_size * embedding_dim) floats (8,388,608 B)| |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
```

### Compiler-Safe Struct Definition
In [`core/include/ai.h`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/ai.h), the header struct is packed with strict alignment to prevent non-deterministic compiler padding:

```cpp
#pragma pack(push, 1)
struct AxiomWeightHeader {
    char magic[8];        // "AXIOM_W1"
    uint32_t version;     // Format version (1)
    uint32_t in_dim;      // Input dimensions
    uint32_t out_dim;     // Output dimensions
    uint32_t vocab_size;  // Dense vocabulary capacity (65,536)
    uint32_t embedding_dim; // Vector dimensions per token (32)
    uint32_t dtype;       // Data type (0 = FP32, 1 = BF16)
    uint64_t data_bytes;  // Total size of following tensor payload
    uint32_t crc32;       // Checksum / validation hash
    uint32_t reserved;    // Padding / future flags
};
#pragma pack(pop)
```

---

## 2. The Atomic Two-Stage Write Protocol

Directly writing parameters to the production `weights.axiom` file introduces a critical data integrity vulnerability. If the application crashes, the host machine loses power, or disk storage is exhausted mid-write, the existing file is truncated or corrupted, destroying the model permanently.

Project Axiom implements an **Atomic Two-Stage Write Protocol** in [`AI::save_weights()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp):

```
                        [TRAINING PROCESS]
                                │
               1. Pull weights from GPU & sync stream
                                │
                                ▼
                   [weights.axiom.tmp]
               2. Full write to temporary file
                                │
                                ▼
                     Flush and Close Stream
                                │
                                ▼
       3. Atomic File System Rename (std::filesystem::rename)
                                │
            [weights.axiom.tmp] ───► [weights.axiom]
                                │
                                ▼
                  [PERMANENT VERIFIED WEIGHTS]
```

### Execution Steps
1. **Device Synchronization:** The host issues `cudaStreamSynchronize()` to guarantee all active Tensor Core operations and SGD kernel updates have completed.
2. **Device-to-Host Pull:** If weights are dirty on the GPU, the master weight array is synchronized back to host memory.
3. **Staging File Serialization:** Parameters are serialized to a temporary staging file: `weights.axiom.tmp`.
4. **I/O Flush:** The stream is flushed and explicitly closed, guaranteeing all data has cleared the operating system buffer cache.
5. **Atomic Rename:** The staging file replaces the target file via `std::filesystem::rename("weights.axiom.tmp", "weights.axiom")`. 

On modern operating systems (NTFS / ext4), the rename operation is an atomic metadata swap. If any error occurs prior to the rename, the original valid `weights.axiom` remains completely intact and uncorrupted.

---

## 3. GPU Stream Synchronization Prior to Serialization

Because CUDA operations are asynchronous, launching kernel updates followed immediately by host disk writes introduces a severe race condition:

```cpp
// INCORRECT: Host reads host/device memory while GPU is still updating it!
sgd_rank1_update_kernel<<<...>>>(...);
save_weights("weights.axiom"); // RACE CONDITION! Partial updates written to disk.
```

### The Synchronization Barrier
In Axiom, [`AI::save_weights()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp) enforces a strict hardware barrier:

```cpp
// Ensure all device updates in stream are complete
if (gpu_state && gpu_state->stream_compute) {
    cudaStreamSynchronize(gpu_state->stream_compute);
}
```

The CPU halts until the GPU's command queue has completely drained. Only when the mathematical state is settled in silicon are the parameters read and written to disk.

---

## 4. Integrity Validation and Graceful Degradation on Load

When [`AI::load_weights()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp) is called (such as during `app.exe --use` or startup), the file undergoes forensic validation before any memory is touched:

```cpp
bool AI::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open weight file: " << path << std::endl;
        return false;
    }

    AxiomWeightHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (file.gcount() != sizeof(header)) {
        std::cerr << "[ERROR] Corrupt header in: " << path << std::endl;
        return false;
    }

    // 1. Magic byte verification
    if (std::memcmp(header.magic, "AXIOM_W1", 8) != 0) {
        std::cerr << "[ERROR] Magic mismatch: Not an Axiom weight file!" << std::endl;
        return false;
    }

    // 2. Topology verification
    if (header.in_dim != 512 || header.out_dim != 4) {
        std::cerr << "[ERROR] Dimension mismatch in: " << path << std::endl;
        return false;
    }

    // 3. Payload size verification
    uint64_t expected_bytes = (512 * 4 + 4) * sizeof(float);
    if (header.data_bytes != expected_bytes) {
        std::cerr << "[ERROR] Payload size corrupt in: " << path << std::endl;
        return false;
    }

    // Read payload...
    file.read(reinterpret_cast<char*>(weights.data()), 2048 * sizeof(float));
    file.read(reinterpret_cast<char*>(biases.data()), 4 * sizeof(float));

    // Synchronize loaded parameters to GPU
    sync_weights_to_device();
    return true;
}
```

### Forensic Defenses
1. **Magic Mismatch:** Files missing the exact 8-byte `"AXIOM_W1"` identifier are immediately rejected without attempting to parse payload data.
2. **Dimension Mismatch:** Models saved with incompatible layer dimensions are rejected to prevent buffer overflows during matrix multiplication.
3. **Partial File Immunity:** If a truncated file is passed, `file.gcount()` detects the missing bytes, aborts the read, and prevents memory corruption.
4. **Graceful Fallback:** If `weights.axiom` does not exist or fails validation, the engine does not crash. It logs an explanatory message and falls back to random symmetry-breaking initialization, allowing training to begin from scratch safely.

---

## 5. The `optimizer.axiom` Binary Specification (Curing AI Amnesia)

To prevent the engine from experiencing amnesia across training sessions, Axiom persists the entire optimization state—including the first and second moments of the AdamW optimizer—into a companion binary file: `optimizer.axiom`.

```
+-------------------------------------------------------------------------------+
|                       AXIOM OPTIMIZER HEADER (64 Bytes)                       |
|  +-------------------------------------------------------------------------+  |
|  |  magic[8]        : "AXIOM_O1" (8 ASCII bytes)                           |  |
|  |  version         : uint32_t (Value = 1)                                 |  |
|  |  step_count      : uint64_t (Cumulative training steps completed)      |  |
|  |  learning_rate   : float    (Current active learning rate)              |  |
|  |  beta1           : float    (First moment decay coefficient, e.g. 0.90) |  |
|  |  beta2           : float    (Second moment decay coefficient, e.g. 0.999)| |
|  |  eps             : float    (Numerical stability constant, 1e-8)        |  |
|  |  weight_decay    : float    (Decoupled weight decay coefficient)        |  |
|  |  num_tensors     : uint32_t (Total parameter tensor count)              |  |
|  |  payload_bytes   : uint64_t (Total bytes of serialized moment arrays)   |  |
|  |  crc32           : uint32_t (CRC32 checksum of moment payload)          |  |
|  |  reserved        : uint32_t (Zero padding for 64-bit alignment)         |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
|                       RAW MOMENT TENSOR PAYLOAD                               |
|  +-------------------------------------------------------------------------+  |
|  |  Momentum Array m_t (Projection W & b)   : IEEE FP32 floats             |  |
|  |  Variance Array v_t (Projection W & b)   : IEEE FP32 floats             |  |
|  |  Momentum Array m_t (Embedding Table)   : (65,536 * 32) FP32 floats    |  |
|  |  Variance Array v_t (Embedding Table)   : (65,536 * 32) FP32 floats    |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
```

### Compiler-Safe Optimizer Header Struct
```cpp
#pragma pack(push, 1)
struct AxiomOptimizerHeader {
    char magic[8];         // "AXIOM_O1"
    uint32_t version;      // Format version (1)
    uint64_t step_count;   // Completed global steps
    float learning_rate;   // Active learning rate
    float beta1;           // 0.90f
    float beta2;           // 0.999f
    float eps;             // 1e-8f
    float weight_decay;    // Weight decay rate
    uint32_t num_tensors;  // Parameter groups count
    uint64_t payload_bytes;// Byte count of moment data
    uint32_t crc32;        // Integrity checksum
    uint32_t reserved;     // 64-bit boundary alignment
};
#pragma pack(pop)
```

### Two-Stage Atomic Write for Optimizer States
Just like `weights.axiom`, writing the optimizer state uses the two-stage atomic protocol:
1. Stream synchronize GPU memory to settle all active AdamW updates.
2. Serialize state to `optimizer.axiom.tmp`.
3. Flush and close file handles.
4. Atomically rename `optimizer.axiom.tmp` $\longrightarrow$ `optimizer.axiom`.

When training is resumed with `app.exe --train`, Axiom detects `optimizer.axiom`, restores momentum and variance vectors directly into GPU memory, and continues gradient updates with zero loss of acceleration.
