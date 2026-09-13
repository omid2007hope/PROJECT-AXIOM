#include "../include/parser.hpp"
#include <iostream>
#include <vector>

// Add a mock definition so it compiles if parser.cpp is linked
#include "../util/parser.cpp"

void conduct_test() {
  Test_Sample test_data;
  Input input_data;
  Polynomial polynomial;
  Parser parser;

  std::cout << "--- Step 1: Tokenization (Mock BPE) ---\n";
  std::vector<double> tokens = parser.tokenization(test_data, input_data, polynomial);
  
  for (size_t i = 0; i < 9; i++) { // First 9 are our text words
    std::cout << "Word: '" << test_data.data_txt[i] << "' -> Token ID: " << tokens[i] << "\n";
  }
  
  std::cout << "\n--- Step 2: Embeddings ---\n";
  std::vector<double> embeddings = parser.embedding(test_data, input_data, polynomial);
  
  for (size_t i = 0; i < 3; i++) { // Show embeddings for first 3 tokens
    std::cout << "Token ID: " << tokens[i] << " -> Embedding Vector: [" 
              << embeddings[i*3] << ", " 
              << embeddings[i*3+1] << ", " 
              << embeddings[i*3+2] << "]\n";
  }
}

int main() {
  conduct_test();
  return 0;
}
