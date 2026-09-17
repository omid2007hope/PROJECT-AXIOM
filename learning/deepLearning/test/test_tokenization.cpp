#include "../include/parser.hpp"

void test_tokenization() {
  Input input;
  Test test;
  Parser parser;

  std::vector<uint32_t> tokenized = parser.tokenization(input, test);

  std::cout << "Successfully generated " << tokenized.size() << " tokens!\n";
  std::cout << "All tokens: ";

  // Print all tokens
  for (size_t i = 0; i < tokenized.size() && i < 1000; ++i) {
    std::cout << tokenized[i] << " ";
  }
  std::cout << "\n";
}