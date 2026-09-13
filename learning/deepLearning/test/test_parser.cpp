#include "../include/perceptron.hpp"
#include "../src/perceptron.cpp"
#include <iostream>

void conduct_test() {
  Test_Sample test_data;
  Input input_data;

  // Execute the standalone data parser without instantiating Perceptron
  std::vector<double> result = tokenization(test_data, input_data);

  std::cout << "--- Parser Unification Test ---\n";
  for (size_t i = 0; i < result.size(); i++) {
    std::cout << "Data Point " << i << ": " << result[i] << "\n";
  }
}

int main() {
  conduct_test();
  return 0;
}
