CXX = g++
CXXFLAGS = -Wall -std=c++17 -Ilearning/deepLearning/include

# Directories
SRC_DIR = learning/deepLearning/src
TEST_DIR = learning/deepLearning/test
UTIL_DIR = learning/deepLearning/util

# Source Files (using wildcards!)
UTIL_SRCS = $(wildcard $(UTIL_DIR)/*.cpp)
APP_SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(UTIL_SRCS)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp) $(UTIL_SRCS)

# Output Targets
APP_TARGET = learning/deepLearning/app.exe
TEST_TARGET = learning/deepLearning/test.exe

# Default: compile both!
all: $(APP_TARGET) $(TEST_TARGET)

# Compile the main application
$(APP_TARGET): $(APP_SRCS)
	$(CXX) $(CXXFLAGS) $(APP_SRCS) -o $(APP_TARGET)

# Compile the test application
$(TEST_TARGET): $(TEST_SRCS)
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) -o $(TEST_TARGET)

# Clean both executables
clean:
	cmd.exe /C del /Q /F learning\deepLearning\*.exe

# Run the test
test: $(TEST_TARGET)
	cmd.exe /C ".\learning\deepLearning\test.exe"
