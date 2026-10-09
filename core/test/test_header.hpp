#pragma once
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "../include/parser.hpp"

inline void print_file_path() {
	std::cout << "Header __FILE__ is: " << __FILE__ << "\n";
	std::filesystem::path p(__FILE__);
	std::filesystem::path resolved =
		p.parent_path().parent_path() / "asset_sample" / "Photo.jpeg";
	std::cout << "Resolved path is: " << resolved << "\n";
	std::cout << "Absolute path is: " << std::filesystem::absolute(resolved) << "\n";
}

// ─── Test fixtures ──────────────────────────────────────────────────────────
// The utility layers decide nothing on their own, so each test hands them the
// same kind of configuration main() does.

// core/ folder (this header lives in core/test/)
inline std::filesystem::path test_core_dir() {
	return std::filesystem::path(__FILE__).parent_path().parent_path();
}

inline std::filesystem::path test_embedded_path() {
	return test_core_dir() / "database" / "embedded_data.bin";
}

inline std::vector<std::filesystem::path> test_asset_dirs() {
	return {test_core_dir() / "asset_sample"};
}

// Raw data samples
inline Test make_test_data() {
	Test test;
	test.data_txt = {"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"};
	test.data_num = {5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2};
	test.data_image = {"Photo.jpeg"};
	test.data_video = {"Video.mp4"};
	test.data_voice = {"Voice.ogg"};
	return test;
}
