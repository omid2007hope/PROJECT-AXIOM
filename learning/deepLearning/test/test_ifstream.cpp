#include <iostream>
#include <fstream>
#include <filesystem>

int main() {
    std::filesystem::path p("learning/deepLearning/util/..\\asset_sample\\Photo.jpeg");
    std::cout << "Path: " << p << "\n";
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        std::cout << "Failed to open.\n";
    } else {
        std::cout << "Success!\n";
    }
    return 0;
}
