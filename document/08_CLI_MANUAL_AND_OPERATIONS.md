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
| **`--use`** | _None_ | Launches the interactive local inference console. Automatically loads [`weights.axiom`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/weights.axiom) for zero-latency prompt evaluation. |
| **`--train`** | `[STEPS]` (Optional) | Executes the continuous training loop on `training_data.txt`. Takes an optional step count integer (defaults to `10000` if omitted). |
| **`--test`** | _None_ | Runs the internal self-test validation routine. For the complete 200-test suite, run [`test.exe --test`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/test.exe). |
| **`--help`** | _None_ | Prints command-line argument documentation and version info to the console. |

### Interactive Main Menu (Default Execution)
If [`app.exe`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/app.exe) is launched without flags, it presents an interactive terminal dashboard:

```
======================================================
           PROJECT AXIOM: SILICON-NATIVE AI           
======================================================
 [1] Use Model (Direct Offline Inference)
 [2] Train Model (Continuous SGD Loop)
 [3] Run Automated Verification Suite
 [4] Exit Engine
======================================================
 Select option [1-4]: 
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

---

## 3. Training Loop Operations (`--train [N]`)

Launching a training session:
```powershell
.\app.exe --train 50000
```

The engine begins streaming tokens through the 4-tier pipeline, reporting progress, step counts, Mean Squared Error (MSE) loss, and throughput:

```
[AXIOM] Initializing Training Loop (50,000 steps)...
[AXIOM] Streaming data from: training_data.txt
[TRAIN] Step: 1000/50000 | Loss: 0.08412 | LR: 0.01000 | Stream: ACTIVE
[TRAIN] Step: 2000/50000 | Loss: 0.06120 | LR: 0.01000 | Stream: ACTIVE
[TRAIN] Step: 3000/50000 | Loss: 0.04289 | LR: 0.01000 | Stream: ACTIVE
...
```

### Runtime Training Hotkeys (Non-Blocking Keyboard Control)
During the training loop, the engine continuously monitors non-blocking keyboard input via `_kbhit()` and `_getch()`. The developer can control execution dynamically without interrupting the compute pipeline:

| Key Press | Action | Technical Execution |
| :---: | :--- | :--- |
| **`[S]`** or **`[s]`** | **Atomic Weight Save** | Flushes GPU streams, pulls weights, writes to `weights.axiom.tmp`, and atomically renames to `weights.axiom`. Training resumes seamlessly. |
| **`[R]`** or **`[r]`** | **Hard Weight Reset** | Re-randomizes all 2,048 weights on CPU via Mersenne Twister (`std::mt19937`), pushes new weights to GPU VRAM, and resets step counters in $<1\text{ ms}$. |
| **`[Q]`** or **`[q]`** | **Safe Exit** | Flushes final weights to disk atomically, neutralizes GPU pointers, releases pinned memory, and cleanly exits the application. |

---

## 4. Compilation and Build Automation

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

# 1. Compile all binaries (DLL, app.exe, test.exe)
mingw32-make all

# 2. Run the automated test suite
mingw32-make test

# 3. Clean all build artifacts (.exe, .dll, .obj, .o)
mingw32-make clean
```

### Build Pipeline Breakdown
1. **`cuda_transfer.dll` Compilation:**
   ```powershell
   nvcc -O3 -shared -Xcompiler "/O2 /MD" -arch=sm_89 -Iinclude -Ithird_party/cutlass/include src/tensor_cores.cu -o cuda_transfer.dll
   ```
2. **`app.exe` Host Compilation:**
   ```powershell
   g++ -std=c++23 -O3 -Iinclude src/main.cpp src/ai.cpp src/loader.cpp src/parser.cpp -L. -lcuda_transfer -o app.exe
   ```
3. **`test.exe` Test Harness Compilation:**
   ```powershell
   g++ -std=c++23 -O3 -Iinclude tests/test_main.cpp src/ai.cpp src/loader.cpp src/parser.cpp -L. -lcuda_transfer -o test.exe
   ```
