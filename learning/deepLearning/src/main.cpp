#include <iostream>

void test_tokenization();
void test_path();
void test_ifstream();
void test_optimizer();

int main(int argc, char *argv[]) {
    std::cout << "--- Running test_path ---\n";
    test_path();
    
    std::cout << "\n--- Running test_ifstream ---\n";
    test_ifstream();
    
    std::cout << "\n--- Running test_tokenization ---\n";
    test_tokenization();

    std::cout << "\n--- Running test_optimizer ---\n";
    test_optimizer();

    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();
    return 0; 
}