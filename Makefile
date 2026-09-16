CXX = g++
CXXFLAGS = -Wall -std=c++17

TARGET = learning/deepLearning/test/test_tokenization.exe
SRC = learning/deepLearning/test/test_tokenization.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	cmd.exe /C del /Q /F learning\deepLearning\test\*.exe

test: $(TARGET)
	cmd.exe /C "cd learning\deepLearning\test && test_tokenization.exe"
