#pragma once
#include <iostream>
#include <filesystem>

inline void print_file_path() {
    std::cout << "Header __FILE__ is: " << __FILE__ << "\n";
    std::filesystem::path p(__FILE__);
    std::filesystem::path resolved = p.parent_path().parent_path() / "asset_sample" / "Photo.jpeg";
    std::cout << "Resolved path is: " << resolved << "\n";
    std::cout << "Absolute path is: " << std::filesystem::absolute(resolved) << "\n";
}
