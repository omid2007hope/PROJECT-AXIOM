#include "../include/parser.hpp"
#include "../util/parser.cpp"

void test_tokenization() {
  Input input;
  Test test;
  Parser parser;

  std::vector<double> tokenized = parser.tokenization(input, test);

  std::cout << "Successfully generated " << tokenized.size() << " tokens!\n";
  std::cout << "All tokens: ";
  
  // Print all tokens
  for (size_t i = 0; i < tokenized.size(); ++i) {
      std::cout << tokenized[i] << " ";
  }
  std::cout << "\n";
}

int main() {
  test_tokenization();
  return 0;
}