#include <iostream>

void test_path();
void test_ifstream();
void test_embeddings();
void test_tokenization();
void test_optimizer();
void test_cuda_transfer();

int main(int argc, char *argv[]) {
    std::cout << "--- Running test_path ---\n";
    test_path();

    std::cout << "\n--- Running test_ifstream ---\n";
    test_ifstream();

    std::cout << "\n--- Running test_embeddings ---\n";
    test_embeddings();

    std::cout << "\n--- Running test_tokenization ---\n";
    test_tokenization();

    std::cout << "\n--- Running test_optimizer ---\n";
    test_optimizer();

    std::cout << "\n--- Running test_cuda_transfer ---\n";
    test_cuda_transfer();

    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();
    return 0;
}