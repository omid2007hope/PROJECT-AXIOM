# 08. CLI Manual and Operator Guide

In accordance with Project Axiom's Version 1.0 design doctrine, the engine operates as a zero-overhead, bare-metal command-line binary ([`core/app.exe`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/app.exe)). There are no electron runtimes, no browser windows, and no Python interpretative shims.

This document serves as the complete operator manual for executing inference, orchestrating training runs, utilizing runtime keyboard hotkeys, and building the binaries from source.

---

## 1. Command-Line Arguments and Options

The executable accepts standard UNIX-style command-line flags:

```bash
.\app.exe [OPTION] [ARGUMENT]
```

| Flag | Argument | Description |
| :--- | :--- | :--- |
| **`--use`** | _None_ | Launches the interactive local inference console. Automatically loads [`weights.axiom`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/weights.axiom) for zero-latency prompt evaluation with live HUD telemetry. |
| **`--train`** | `[STEPS]` (Optional) | Executes the continuous training loop on sample prompt data. Takes an optional step count integer (defaults to continuous if omitted). |
| **`--bulk`** | `[STEPS]` (Optional) | Automatically scans and trains on all assets inside `asset_sample` (Text, PDF, Images, Video, Audio, 3D models). Takes an optional step count limit. |
| **`--benchmark`** | _None_ | Executes the bare-metal hardware profiler using `cudaEvent_t` driver timers, printing the 4-variable Hardware Saturation HUD. |
| **`--test`** | _None_ | Runs the internal self-test validation routine. For the complete automated test suite, run [`test.exe --test`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/test.exe). |
| **`--weights`** | `<path>` | Designates custom neural weights checkpoint path (supports `--weights <path>` or `--weights=<path>`). |
| **`--vocab`** | `<path>` | Designates custom vocabulary dictionary path (supports `--vocab <path>` or `--vocab=<path>`). |
| **`--lr`** | `<rate>` | Sets optimizer learning rate override (e.g., `--lr 0.002`). |
| **`--data`** | `<dir>` | Sets custom bulk training directory override (e.g., `--data my_dataset`). |
| **`--archive`** | _None_ | Automatically archives processed files into `archive/` after bulk training completion. |
| **`--purge`** | _None_ | Automatically purges processed files from asset directory after bulk training completion. |
| **`--help`** | _None_ | Prints command-line argument documentation and version info to the console. |

### Interactive Main Menu (Default Execution)
If [`app.exe`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/app.exe) is launched without flags, it presents an interactive terminal dashboard:

```
======================================
           Welcome to AXIOM           
======================================
[1] Use it (Inference)
[2] Train it (Training)
[3] Run Tests
[4] Benchmark Hardware (Telemetry HUD)
[5] Bulk Training (asset_sample)
[6] Exit
> 
```

---

## 2. Interactive Local Inference Session (`--use`)

Launching the inference engine:
```powershell
.\app.exe --use
```

```
[AXIOM] Loading weights from: weights.axiom ...
[AXIOM] Weights loaded successfully (2,048 weights, 4 biases).
[AXIOM] Entering Interactive Inference Session (Type 'exit' to quit).

Prompt > Hello Axiom, what is the meaning of intelligence?
Prediction >  information structured over time.

Prompt >
```

- **Zero-Cloud Architecture:** The inference session executes entirely within local host RAM and GPU VRAM. Zero network packets are sent.
- **Latency Profile:** Token predictions are generated in microsecond timeframes via direct Tensor Core kernel launches.
- **Pairing Custom Weights and Vocabulary (`--weights` & `--vocab`):**
  Axiom enables pairing custom weight checkpoints with custom vocabulary dictionaries simultaneously from the CLI. This allows evaluating domain-specific models trained on specialized corpora without clobbering root baseline files:
  ```powershell
  # Forward inference pairing custom checkpoint and vocabulary:
  .\app.exe --use --weights experimental_run.axiom --vocab experimental_vocab.axiom

  # Equal-sign syntax supported across all commands:
  .\app.exe --use --weights=checkpoints/v01_model.axiom --vocab=dicts/v01_dict.axiom

  # Training with designated weights checkpoint, vocabulary output, and custom learning rate:
  .\app.exe --bulk 50000 --weights run_01.axiom --vocab vocab_01.axiom --lr 0.002 --archive
  ```

---

## 3. Training Loop Operations & Modality Scenarios

Axiom trains directly across multiple orthogonal modalities. Training can be initiated interactively through the dashboard (`.\app.exe` -> Option `[2]`), or directly via the command line:

```powershell
# Continuous training loop:
.\app.exe --train

# Fixed-step batch training (e.g. 50,000 steps with automatic save):
.\app.exe --train 50000
```

### Detailed Modality Training Scenarios

#### 1. Unified Text & Numerical Training
* **Interactive Path:** Select `[2] Train it` $\to$ Enter text, sentences, or numerical data.
* **Input:** Natural language sentences, technical corpora, or space-separated numerical values (e.g. `"The acceleration due to gravity is 9.8 m/s^2"`).
* **Tokenization:** Ingested into the unified `main_dictionary` within the dense codebook range `[4, 64,511]`. Words and numbers share context, allowing the engine to learn semantic co-occurrence between concepts and mathematical quantities.
* **Auto-Tiling:** Sequences shorter than 129 tokens are tiled seamlessly to fill the 128-token sliding context window.
* **Direct Workflow 2 Bypass:** Raw manual CLI text inputs completely bypass file extraction, feeding directly into character sanitization and tokenization.

#### 2. Image Decompression & Visual Token Training
* **Interactive Path:** Select `[2] Train it` $\to$ Select Image asset or folder.
* **Input:** Path to an image file (e.g., `asset_sample/Photo.jpeg`, `"C:\Data\Image.png"`).
* **Decompression Pipeline:** Routed to `decom_image` to strip container encodings (JPEG, PNG headers) and extract raw uncompressed RGB pixel arrays.
* **Tokenization:** Discrete visual tokens are mapped into the visual codebook block `[64,512, 64,767]`.

#### 3. Video Decompression & Frame Token Training
* **Interactive Path:** Select `[2] Train it` $\to$ Select Video asset.
* **Input:** Video file path (e.g., `asset_sample/Video.mp4`).
* **Decompression Pipeline:** Routed to `decom_video` to strip h.264/container bitstreams, exposing raw RGB pixel frames rather than compressed zlib/h.264 entropy.
* **Tokenization:** Frames are quantized into spatial codebook tokens in `[64,512, 64,767]`.

#### 4. Audio Waveform Decompression Training
* **Interactive Path:** Select `[2] Train it` $\to$ Select Voice/Audio asset.
* **Input:** Audio file path (e.g., `asset_sample/Voice.ogg`, `Voice.wav`).
* **Decompression Pipeline:** Routed to `decom_audio` to strip RIFF/OGG container metadata and extract clean PCM audio waveform amplitudes.
* **Tokenization:** Acoustic amplitudes are quantized into acoustic codebook tokens in `[64,768, 65,023]`.

#### 5. Cross-Modal Early Fusion (Text + Extracted Media)
* **Interactive Path:** Ingest text alongside media assets.
* **Mechanism:** Text tokens and decompressed media tokens are concatenated into a single continuous sequence. The 128-token sliding context window traverses the cross-modal seam, training the neural network on cross-modal associations without modality collisions.

#### 6. Bulk Directory Training from `asset_sample` (Mass Four-Stage Ingestion)
* **Interactive Path:** Select `[5] Bulk Training (asset_sample)` from the main menu, or execute `.\app.exe --bulk` (with optional step limit: `.\app.exe --bulk 50000`).
* **Four-Stage Ingestion Mechanics:** Axiom scans [`core/asset_sample`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/asset_sample) and routes each asset through the Identifier and specialized Decompressor:
  * **Text & Code (`.txt`, `.md`, `.csv`, `.json`, `.py`, `.cpp`):** Sanitized and tokenized into `main_dictionary` `[4, 64,511]`.
  * **Documents (`.pdf`, `.docx`):** Decompressed via `decom_doc` into clean UTF-8 text and equations (eliminating zlib compression entropy).
  * **Images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`):** Decompressed via `decom_image` into raw RGB pixel tensors.
  * **Video (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`):** Decoded via `decom_video` into raw RGB frame matrices.
  * **Audio (`.ogg`, `.wav`, `.mp3`, `.flac`):** Stripped via `decom_audio` into raw PCM waveforms.
* **Large File Progress Percentage Bar:**
  For files $\ge 5\text{ MB}$, Axiom displays a live progress percentage bar throttled at 2 MB intervals:
  ```text
  [..] Tokenizing Video Video.mp4 [======>                 ] 25.9% (2.0 MB / 7.7 MB)
  [..] Tokenizing Video Video.mp4 [========================] 100.0% (7.7 MB / 7.7 MB)
  ```
* **Continuous Fine-Tuning & Asset Management Workflow:**
  1. Fill [`core/asset_sample`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/asset_sample) with training files.
  2. Launch `.\app.exe` $\to$ Option `[5] Bulk Training (asset_sample)`.
  3. The model loads existing weights from `weights.axiom` and momentum from `optimizer.axiom`, continuing training with zero amnesia.
  4. Upon training completion or manual exit (`[Q]`), new weights and optimizer momentum are saved atomically.
  5. **Interactive Asset Management Prompt:**
     ```text
     Processed Asset Management (N files):
       [1] Keep files in asset_sample/ (Default)
       [2] Archive processed files to asset_sample/archive/
       [3] Purge (Delete) processed files from asset_sample/
     > 
     ```
     * **`[1] Keep` (Default):** Retains all files in `asset_sample/`.
     * **`[2] Archive`:** Moves processed files into `core/asset_sample/archive/`. Axiom automatically ignores `archive/` in subsequent directory scans, allowing immediate drop-in of new batches without re-ingesting old files.
     * **`[3] Purge`:** Prompts for safety confirmation (`Are you sure you want to proceed? (y/N)`). Entering `y` permanently deletes the processed files.

---

### Runtime Training Hotkeys (Non-Blocking Keyboard Control)
During the training loop, the engine continuously monitors non-blocking keyboard input via `_kbhit()` and `_getch()`. The developer can control execution dynamically without interrupting the compute pipeline:

| Key Press | Action | Technical Execution |
| :---: | :--- | :--- |
| **`[S]`** or **`[s]`** | **Atomic Checkpoint Save** | Flushes GPU streams, pulls weights & AdamW momentum/variance states, writes to `.tmp` files, and atomically renames them to `weights.axiom` and `optimizer.axiom`. Training resumes seamlessly with zero amnesia. |
| **`[R]`** or **`[r]`** | **Hard Weight Reset** | Re-randomizes all weights on CPU via Mersenne Twister (`std::mt19937`), pushes new weights to GPU VRAM, and resets step counters and momentum states in $<1\text{ ms}$. |
| **`[Q]`** or **`[q]`** | **Safe Exit** | Flushes final weights and optimizer state to disk atomically, neutralizes GPU pointers, releases pinned memory, and cleanly exits the application. |

---

## 4. Bare-Metal Hardware Benchmark (`--benchmark`)

Executing the zero-overhead hardware benchmark and kernel saturation sweep:
```powershell
.\app.exe --benchmark
```

The benchmark employs driver-level `cudaEvent_t` timers across active CUDA streams to execute a two-stage hardware characterization:
1. **Interactive Micro-Perceptron Inference (`512 -> 4`):** Measures bare-metal driver dispatch latency, prefill time-to-first-token (TTFT), and decode throughput on the active model architecture.
2. **CUTLASS EVT Kernel Saturation Sweep (`M=128, N=4096, K=4096`):** Allocates transformer-scale matrix tiles to saturate all Streaming Multiprocessors, verifying peak compute (TFLOPs), GDDR6 memory bus throughput, and Model FLOPs Utilization (MFU).

```text
========================================================
       AXIOM HARDWARE BENCHMARK & KERNEL SATURATION      
========================================================
Hardware Device : NVIDIA GeForce RTX 4080 Laptop GPU
Architecture    : sm_89 (Ada Lovelace)
Memory Bus      : 192-bit | Peak Bandwidth: 432.0 GB/s
Compute Engine  : 67.7 Peak Dense BF16 TFLOPs
VRAM Footprint  : 11061 MB Free / 12281 MB Total
--------------------------------------------------------
[BENCHMARK 1/2] Interactive Micro-Perceptron Inference (512 -> 4)
[..] Warming up Tensor Cores (10 forward passes)...
[..] Executing 100 Hardware-Timed Tensor Core Inferences...

┌────────────────────────────────────────────────────────────────────────┐
│  PROJECT AXIOM v1.0 • INFERENCE DISPATCH LATENCY [sm_89]               │
├────────────────────────────────────────────────────────────────────────┤
│  1. PREFILL LATENCY (TTFT)  :     0.1 ms                                │
│  2. DECODE THROUGHPUT       :  9302.3 tok/s  ( 0.11 ms/tok)             │
│  3. MEMORY BUS SATURATION   :   0.04 GB/s   (<0.1% of Peak)           │
│  4. COMPUTE EFFICIENCY (MFU):   0.0% MFU   (Tensor Core L1-Resident)  │
└────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------
[BENCHMARK 2/2] CUTLASS EVT Kernel Saturation Sweep (M=128, N=4096, K=4096)
[..] Allocating transformer-scale tensor buffers (34 MB VRAM footprint)...
[..] Warming up all Ada Lovelace Streaming Multiprocessors...
[..] Executing 50 Hardware-Timed Fused EVT Tensor Core Sweeps...

┌────────────────────────────────────────────────────────────────────────┐
│  PROJECT AXIOM v1.0 • TRANSFORMER KERNEL SATURATION [sm_89]            │
├────────────────────────────────────────────────────────────────────────┤
│  1. FUSED EVT SWEEP TIME    :    0.13 ms                                │
│  2. SUSTAINED COMPUTE POWER :    33.6 TFLOPs (BF16 Dense)                 │
│  3. MEMORY BUS SATURATION   :  278.9 GB/s   (64.5% of Peak)          │
│  4. COMPUTE EFFICIENCY (MFU):  49.6% MFU   (Tensor Core Saturation)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Compilation and Build Automation

Project Axiom employs a dual-compiler toolchain combining **MinGW-w64 GCC** (for ISO C++23 host code) and **NVIDIA NVCC** (for CUDA device code).

### Prerequisites
1. **Operating System:** Windows 10 or Windows 11 (64-bit).
2. **C++ Compiler:** MinGW-w64 `g++` 13.x or 14.x supporting `-std=c++23`.
3. **CUDA Toolkit:** NVIDIA CUDA Toolkit 12.x (`nvcc`) targeting Ada Lovelace (`sm_89`).
4. **Make Utility:** `mingw32-make` or GNU Make.

### Build Targets in [`core/Makefile`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/Makefile)

```powershell
# Navigate to the core source directory
cd c:\Users\Omid\X\Code\PROJECT-AXIOM\core

# 1. Compile all binaries (CUDA DLL, app.exe, test.exe)
mingw32-make all

# 2. Run the automated 200-test suite
mingw32-make test

# 3. Clean all build products, link libraries, and scratch artifacts
mingw32-make clean
```

### Clean Target Artifact Purge
Executing `mingw32-make clean` performs a deep cleanup of the build workspace without touching persistent checkpoints (`weights.axiom`):
* **Binaries & Libraries:** Purges `app.exe`, `test.exe`, `cuda_transfer.dll`, `cuda_transfer.lib`, and `cuda_transfer.exp`.
* **Object Files & Compiler Outputs:** Purges all intermediate `*.o`, `*.obj`, `*.exp`, `*.lib`, and `*.tmp` files.
* **Transient Test Artifacts:** Cleans up `test_*.exe`, `test_*.axiom`, and `corrupt_*.axiom` created during test runs.
* **Database Caches:** Flushes generated `database/*.bin` token datasets so subsequent training sessions can re-ingest fresh data.

### Build Pipeline Breakdown
1. **`cuda_transfer.dll` Compilation (NVIDIA NVCC):**
   ```powershell
   nvcc -arch=sm_89 --std=c++17 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" -I"$(CUDA_DIR)/include" -Ithird_party/cutlass/include --expt-relaxed-constexpr -Xcompiler "/Zc:preprocessor,/Zc:__cplusplus" -DBUILDING_CUDA_DLL -shared src/CUDA/gpu/*.cu src/CUDA/kernel/*.cu src/CUDA/util/*.cu -o cuda_transfer.dll
   ```
2. **`app.exe` Host Compilation (MinGW-w64 GCC):**
   ```powershell
   g++ -Wall -std=c++23 -Iinclude src/*.cpp test/*.cpp util/*.cpp cuda_transfer.dll -o app.exe -lpsapi
   ```
3. **`test.exe` Test Harness Compilation (MinGW-w64 GCC):**
   ```powershell
   g++ -Wall -std=c++23 -Iinclude src/*.cpp test/*.cpp util/*.cpp cuda_transfer.dll -o test.exe -lpsapi
   ```
