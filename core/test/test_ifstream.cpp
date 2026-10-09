#include <filesystem>
#include <fstream>
#include <iostream>

void test_ifstream() {
	std::filesystem::path absolute_path(__FILE__);
	std::filesystem::path p =
		absolute_path.parent_path().parent_path() / "asset_sample" / "Photo.jpeg";
	std::cout << "Path: " << p << "\n";
	std::ifstream f(p, std::ios::binary);
	if (!f) {
		std::cout << "Failed to open.\n";
	} else {
		std::cout << "Success!\n";
	}
}
