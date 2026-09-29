#include <iostream>
#include <fstream>
#include <filesystem>

void test_ifstream() {
    std::filesystem::path p("util/..\\asset_sample\\Photo.jpeg");
    std::cout << "Path: " << p << "\n";
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        std::cout << "Failed to open.\n";
    } else {
        std::cout << "Success!\n";
    }
}
