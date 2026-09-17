# AI Core: Build System, Toolchain & Testing Specification

```
Document Version : 2.0.0
Standard         : ISO/IEC 14882:2017 (C++17)
Classification   : Systems Toolchain & Continuous Verification
Build Tool       : GNU Make / MinGW32-Make
Compiler Targets : GCC >= 9.0, Clang >= 10.0, MSVC 2019+
```

---

## 1. Build System Architecture

The build process for **AI Core** is governed by a unified GNU Makefile located at the project root (`learning/deepLearning/Makefile` and root `Makefile`).

### 1.1 Makefile Structure & Target Breakdown

```makefile
CXX = g++
CXXFLAGS = -Wall -std=c++17 -Ilearning/deepLearning/include

# Directories
SRC_DIR = learning/deepLearning/src
TEST_DIR = learning/deepLearning/test
UTIL_DIR = learning/deepLearning/util

# Source Files
UTIL_SRCS = $(wildcard $(UTIL_DIR)/*.cpp)
ALL_SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(TEST_DIR)/*.cpp) $(UTIL_SRCS)
APP_SRCS = $(ALL_SRCS)
TEST_SRCS = $(ALL_SRCS)

# Output Targets
APP_TARGET = learning/deepLearning/app.exe
TEST_TARGET = learning/deepLearning/test.exe
```

### 1.2 Target Definitions

| Target | Command | Purpose |
| :--- | :--- | :--- |
| `all` | `mingw32-make all` | Compiles both `app.exe` and `test.exe` binaries. |
| `test` | `mingw32-make test` | Builds `test.exe` (if out-of-date) and immediately executes the test suite. |
| `clean` | `mingw32-make clean` | Removes compiled binaries (`app.exe`, `test.exe`). |

---

## 2. Compilation Flags & Systems Rationale

```bash
g++ -Wall -std=c++17 -Ilearning/deepLearning/include ...
```

- `-std=c++17`: Activates ISO C++17 language features, specifically:
  - `std::filesystem::path` and path decomposition operators.
  - Template type deduction for constructors.
  - Structured binding declarations (`for (const auto& [key, id] : dict)`).
- `-Wall`: Activates all standard compiler diagnostic warnings, ensuring zero silent type mismatches or uninitialized variables.
- `-Ilearning/deepLearning/include`: Adds the header directory to the compiler's include search path, enabling clean `#include "parser.hpp"` and `#include "perceptron.hpp"` directives.

---

## 3. Test Harness Topology

The test executable (`learning/deepLearning/test.exe`) is invoked from `src/main.cpp` and coordinates four modular test suites:

```
+---------------------------------------------------------------------------------------+
|                               TEST HARNESS TOPOLOGY                                   |
+---------------------------------------------------------------------------------------+

                                [ src/main.cpp ]
                                        |
      +-----------------+---------------+---------------+-----------------+
      |                 |                               |                 |
      v                 v                               v                 v
[ test_path() ]  [ test_ifstream() ]          [ test_tokenization() ]  [ test_embeddings ]
(test_path.cpp) (test_ifstream.cpp)           (test_tokenization.cpp) (test_embeddings.cpp)
      |                 |                               |
Verifies binary   Verifies direct binary       Performs full multimodal
path resolution   file stream opening on       tokenization & dense
via __FILE__      sample Photo.jpeg            embedding projection
```

### 3.1 Test Suite Breakdown

1. **`test_path()` (`test/test_path.cpp`, `test/test_header.hpp`)**:
   - Inspects compile-time `__FILE__` macros.
   - Verifies relative-to-absolute path traversal to `asset_sample/Photo.jpeg`.
2. **`test_ifstream()` (`test/test_ifstream.cpp`)**:
   - Opens `Photo.jpeg` via `std::ifstream` in `std::ios::binary` mode.
   - Validates low-level stream state flags.
3. **`test_tokenization()` (`test/test_tokenization.cpp`)**:
   - Instantiates `Input`, `Test`, and `Parser`.
   - Executes `Parser::tokenization(input, test)`.
   - Emits first 1,000 generated tokens to console for visual inspection.
   - Computes and prints the **Vocabulary Summary** (counts and min/max ID intervals for each modality).
   - Executes `Parser::embedding(input, test)`.
   - Emits the **Embedding Summary** (total floats, vector dimension, cache entry count, and sample coordinates).

---

## 4. Empirical Verification Execution Log

The following execution output is captured directly from the standard build and test run:

```
$ mingw32-make test
g++ -Wall -std=c++17 -Ilearning/deepLearning/include ... -o learning/deepLearning/test.exe
learning/deepLearning/test.exe

--- Running test_path ---
Header __FILE__ is: learning/deepLearning/test/test_header.hpp
Resolved path is: "learning/deepLearning\asset_sample\Photo.jpeg"
Absolute path is: "C:\Users\Omid\X\Code\Brain\learning\deepLearning\asset_sample\Photo.jpeg"

--- Running test_ifstream ---
Path: "learning/deepLearning/util/..\asset_sample\Photo.jpeg"
Success!

--- Running test_tokenization ---
Found in "learning/deepLearning/util/..\asset_sample\Photo.jpeg"
Found in "learning/deepLearning/util/..\asset_sample\Video.mp4"
Found in "learning/deepLearning/util/..\asset_sample\Voice.ogg"
Successfully generated 8289605 tokens!
All tokens: 1 2 3 4 5 6 7 8 9 10000000 10000001 10000002 10000003 ...

--- Vocabulary Summary ---
  Text  : 9 unique tokens  [1 .. 9]
  Number: 9 unique tokens  [10000000 .. 10000008]
  Image : 256 unique tokens  [20000000 .. 20000255]
  Video : 256 unique tokens  [30000000 .. 30000255]
  Voice : 256 unique tokens  [40000000 .. 40000255]
--------------------------

--- Embedding Summary ---
  Tokens       : 8289605
  Dim per token: 4
  Total floats : 33158420
  Unique embeds: 786
  First token [1] -> [0.841471, 0.909297, 0.14112, -0.756802]
-------------------------

Press Enter to exit...
```

---

## 5. Cross-Platform Portability & Toolchain Guide

### 5.1 Windows (MinGW-w64 / MSYS2 UCRT64)
- When executing in PowerShell or Command Prompt, use `mingw32-make test`.
- When linking executables, ensure any previously running instances of `test.exe` are terminated to prevent Windows `cannot open output file ... Permission denied` errors.

### 5.2 Linux (Ubuntu / Debian / Fedora / Arch)
- Toolchain: Standard `build-essential` or `gcc-c++`.
- Command:
  ```bash
  make clean
  make test
  ```
- File paths are normalized via `std::filesystem::path` and seamlessly resolve across POSIX directory hierarchies.

### 5.3 macOS (Apple Silicon / Clang)
- Toolchain: Xcode Command Line Tools (`clang++`).
- Command:
  ```bash
  make CXX=clang++ test
  ```
