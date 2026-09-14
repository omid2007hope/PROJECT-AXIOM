#include "../include/parser.hpp"
#include <iostream>
#include <vector>

// Add a mock definition so it compiles if parser.cpp is linked
#include "../util/parser.cpp"

void conduct_test() {
  Test test_data;
  Input input_data;
  Parser parser;

  std::cout << "=== Step 1: Discrete Tokenization ===\n";
  std::vector<double> tokens = parser.tokenization(input_data, test_data);
  
  std::cout << "Total Unified Tokens Generated: " << tokens.size() << "\n\n";

  std::cout << "--- Text Tokens (Expected IDs 1+) ---\n";
  for (size_t i = 0; i < test_data.data_txt.size(); i++) { 
    std::cout << "Text: '" << test_data.data_txt[i] << "' -> Token ID: " << tokens[i] << "\n";
  }

  size_t offset = test_data.data_txt.size();
  std::cout << "\n--- Number Tokens (Expected IDs 10000+) ---\n";
  for (size_t i = 0; i < test_data.data_num.size(); i++) { 
    std::cout << "Number: " << test_data.data_num[i] << " -> Token ID: " << tokens[offset + i] << "\n";
  }

  offset += test_data.data_num.size();
  std::cout << "\n--- Image Tokens (Expected IDs 20000+) ---\n";
  for (size_t i = 0; i < test_data.data_img.size(); i++) { 
    std::cout << "Pixel: " << test_data.data_img[i] << " -> Token ID: " << tokens[offset + i] << "\n";
  }

  offset += test_data.data_img.size();
  std::cout << "\n--- Video Tokens (Expected IDs 30000+) ---\n";
  for (size_t i = 0; i < test_data.data_video.size(); i++) { 
    std::cout << "Delta: " << test_data.data_video[i] << " -> Token ID: " << tokens[offset + i] << "\n";
  }

  offset += test_data.data_video.size();
  std::cout << "\n--- Voice Tokens (Expected IDs 40000+) ---\n";
  for (size_t i = 0; i < test_data.data_voice.size(); i++) { 
    std::cout << "Sample: " << test_data.data_voice[i] << " -> Token ID: " << tokens[offset + i] << "\n";
  }
  
  std::cout << "\n=== Step 2: Embeddings ===\n";
  std::vector<double> embeddings = parser.embedding(input_data, test_data);
  
  std::cout << "Total Embedding Values Generated: " << embeddings.size() << "\n\n";
  std::cout << "Previewing Embeddings for Image Tokens:\n";
  
  offset = test_data.data_txt.size() + test_data.data_num.size();
  for (size_t i = 0; i < 3; i++) { // Show embeddings for first 3 Image tokens
    std::cout << "Image Token ID: " << tokens[offset + i] << " -> Embedding Vector: [" 
              << embeddings[(offset + i)*3] << ", " 
              << embeddings[(offset + i)*3+1] << ", " 
              << embeddings[(offset + i)*3+2] << "]\n";
  }
}

int main() {
  conduct_test();
  return 0;
}
