#include "../include/parser.hpp"

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

int main(int argc, char *argv[]) {
  test_tokenization();

  // Wait for user input before closing, so the console doesn't close immediately
  std::cout << "Press Enter to exit..." << std::endl;
  std::cin.get();

  return 0;
}