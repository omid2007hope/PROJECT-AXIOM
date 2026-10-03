#include <iostream>
#include <string>

#include "../include/ai.hpp"
#include "../include/optimizer.hpp"

void test_path();
void test_ifstream();
void test_embeddings();
void test_tokenization();
void test_optimizer();
void test_cuda_transfer();

auto fetch_argument(int count, char *argument[]) {
  for (int i = 1; i < count; i++) {
    return std::string(argument[i]);
  }
}

int main(int argc, char *argv[]) {
  auto arg = fetch_argument(argc, argv);

  if (arg == "--train") {
    std::cout << "--- Training AI ---\n";

    // Instantiate the required objects for the AI to train
    Portion portion;
    Optimizer optimizer;

    AI ai(portion.all_elements, portion.target_elements);

    ai.train(optimizer, portion);

  } else if (arg == "--test") {
    std::cout << "--- test_path AI ---\n";
    test_path();

    std::cout << "--- test_ifstream AI ---\n";
    test_ifstream();

    std::cout << "--- test_embeddings AI ---\n";
    test_embeddings();

    std::cout << "--- test_tokenization AI ---\n";
    test_tokenization();

    std::cout << "--- test_optimizer AI ---\n";
    test_optimizer();

    std::cout << "--- test_cuda_transfer AI ---\n";
    test_cuda_transfer();
  } else {
    std::cout << "Please provide --train or --test\n";
  }

  return 0;
}
