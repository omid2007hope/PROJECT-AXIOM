#include <iostream>
#include <string>

#include "../include/ai.hpp"
#include "../include/optimizer.hpp"

void AI::train(Optimizer &optimizer, Portion &portion);

void test_path();
void test_ifstream();
void test_embeddings();
void test_tokenization();
void test_optimizer();
void test_cuda_transfer();

std::string fetch_argument(int count, char *argument[]) {
  for (int i = 0; i <= count; i++) {
    return std::string(argument[i]);
  };
};

int main(int argc, char *argv[]) {

  if (fetch_argument(argc, argv) == "app.exe --train" || "mingw32-make") {
    std::cout << "--- Training AI ---\n";
    train();

  } else if (fetch_argument(argc, argv) == "app.exe --test" ||
             "mingw32-make test ") {

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
  };

  return 0;
};
