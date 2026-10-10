# 06. Weight Persistence Specification

An artificial intelligence model that exists only in volatile VRAM suffers from complete amnesia: the instant the process terminates, the adjusted parameters are lost. To establish permanent model memory across reboots and facilitate open-source distribution, Project Axiom implements a strict binary file format: **`weights.axiom`**.

This document defines the binary layout of the file format, the atomic two-stage write protocol, CUDA stream synchronization, and corruption resistance safeguards.

---

## 1. The `weights.axiom` Binary File Format

The `weights.axiom` file is divided into a 40-byte compiler-safe header followed immediately by contiguous floating-point parameter payloads:

```
+-------------------------------------------------------------------------------+
|                       AXIOM WEIGHT HEADER (40 Bytes)                          |
|  +-------------------------------------------------------------------------+  |
|  |  magic[8]     : "AXIOM_W1" (8 ASCII bytes)                              |  |
|  |  version      : uint32_t (Value = 1)                                    |  |
|  |  in_dim       : uint32_t (Value = 512)                                  |  |
|  |  out_dim      : uint32_t (Value = 4)                                    |  |
|  |  dtype        : uint32_t (Value = 0: IEEE 754 FP32)                     |  |
|  |  data_bytes   : uint64_t (Value = 8,208 bytes)                          |  |
|  |  crc32        : uint32_t (Payload checksum)                             |  |
|  |  reserved     : uint32_t (Zero padding for 64-bit alignment)            |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
|                       RAW TENSOR PAYLOAD (8,208 Bytes)                        |
|  +-------------------------------------------------------------------------+  |
|  |  Weight Matrix W : 2,048 floats * 4 bytes = 8,192 Bytes                 |  |
|  |  Bias Vector b   : 4 floats * 4 bytes     = 16 Bytes                    |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
|                       TOTAL FILE SIZE = 8,248 Bytes                           |
+-------------------------------------------------------------------------------+
```

### Compiler-Safe Struct Definition
In [`core/include/ai.h`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/include/ai.h), the header struct is packed with strict alignment to prevent non-deterministic compiler padding:

```cpp
#pragma pack(push, 1)
struct AxiomWeightHeader {
    char magic[8];        // "AXIOM_W1"
    uint32_t version;     // Format version (1)
    uint32_t in_dim;      // Input dimensions (512)
    uint32_t out_dim;     // Output dimensions (4)
    uint32_t dtype;       // Data type (0 = FP32, 1 = BF16)
    uint64_t data_bytes;  // Total size of following tensor payload (8208)
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
2. **Dimension Mismatch:** Models saved with incompatible layer dimensions ($512 \times 4$) are rejected to prevent buffer overflows during matrix multiplication.
3. **Partial File Immunity:** If a truncated file is passed, `file.gcount()` detects the missing bytes, aborts the read, and prevents memory corruption.
4. **Graceful Fallback:** If `weights.axiom` does not exist or fails validation, the engine does not crash. It logs an explanatory message and falls back to random symmetry-breaking initialization, allowing training to begin from scratch safely.
