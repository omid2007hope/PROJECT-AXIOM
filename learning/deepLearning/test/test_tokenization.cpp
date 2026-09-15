#include "../include/parser.hpp"
#include "../util/parser.cpp"

void test_tokenization() {
  Input input;
  Test test;
  Parser parser;

  std::vector<double> tokenized = parser.tokenization(input, test);

  std::cout << "Successfully generated " << tokenized.size() << " tokens!\n";
  std::cout << "First 15 tokens: ";
  
  // Print first 15 tokens (otherwise the video tokens will flood the screen!)
  for (size_t i = 0; i < std::min(tokenized.size(), (size_t)15); ++i) {
      std::cout << tokenized[i] << " ";
  }
  std::cout << "...\n";
}

int main() {
  test_tokenization();
  return 0;
}