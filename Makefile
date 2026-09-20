CXX      = g++
CXXFLAGS = -Wall -std=c++23 -Ilearning/deepLearning/include

# ─── CUDA ────────────────────────────────────────────────────────────────────
CUDA_DIR  = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3
NVCC      = "$(CUDA_DIR)/bin/nvcc.exe"
# sm_89 = RTX 4080 Laptop (Ada Lovelace, compute capability 8.9)
# -ccbin g++ : use MinGW g++ as the host compiler instead of cl.exe (MSVC)
NVCCFLAGS = -arch=sm_89 --std=c++17 -ccbin g++ \
            -I"$(CUDA_DIR)/include"

CUDA_SRC_DIR = learning/deepLearning/src/CUDA
CUDA_SRCS    = $(wildcard $(CUDA_SRC_DIR)/*.cu)
CUDA_OBJ     = learning/deepLearning/cuda_transfer.o

# CUDA runtime library (links cudaMalloc, cudaMemcpy, etc.)
CUDA_LIBS = -L"$(CUDA_DIR)/lib/x64" -lcudart

# ─── CPU sources ─────────────────────────────────────────────────────────────
SRC_DIR  = learning/deepLearning/src
TEST_DIR = learning/deepLearning/test
UTIL_DIR = learning/deepLearning/util

UTIL_SRCS = $(wildcard $(UTIL_DIR)/*.cpp)
ALL_SRCS  = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(TEST_DIR)/*.cpp) $(UTIL_SRCS)
APP_SRCS  = $(ALL_SRCS)
TEST_SRCS = $(ALL_SRCS)

# ─── Output targets ──────────────────────────────────────────────────────────
APP_TARGET  = learning/deepLearning/app.exe
TEST_TARGET = learning/deepLearning/test.exe

# ─── Default: compile everything ─────────────────────────────────────────────
all: $(APP_TARGET) $(TEST_TARGET)

# ─── Compile CUDA sources into a single relocatable object ───────────────────
# If nvcc fails (e.g., MSVC cl.exe not yet installed), skip and build CPU only.
$(CUDA_OBJ): $(CUDA_SRCS)
	-$(NVCC) $(NVCCFLAGS) -c $(CUDA_SRCS) -o $(CUDA_OBJ)

# Helper: return the CUDA obj only if it was successfully built
CUDA_LINK = $(if $(wildcard $(CUDA_OBJ)),$(CUDA_OBJ),)
CUDA_LIB  = $(if $(wildcard $(CUDA_OBJ)),$(CUDA_LIBS),)

# ─── Link CPU + GPU into the app ─────────────────────────────────────────────
$(APP_TARGET): $(APP_SRCS) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) $(APP_SRCS) $(CUDA_LINK) $(CUDA_LIB) -o $(APP_TARGET)

# ─── Link CPU + GPU into the test binary ─────────────────────────────────────
$(TEST_TARGET): $(TEST_SRCS) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) $(CUDA_LINK) $(CUDA_LIB) -o $(TEST_TARGET)

# ─── Clean ───────────────────────────────────────────────────────────────────
clean:
	-rm -f $(APP_TARGET) $(TEST_TARGET) $(CUDA_OBJ)

# ─── Run tests ───────────────────────────────────────────────────────────────
test: $(TEST_TARGET)
	$(TEST_TARGET)
