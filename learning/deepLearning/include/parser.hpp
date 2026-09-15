#ifndef include_parser
#define include_parser

#include <cstddef>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

// Raw data samples
struct Test {
  // Text by user
  std::vector<std::string> data_txt = {"The",  "quick", "brown", "fox", "jumps",
                                       "over", "the",   "lazy",  "dog"};
  // Number by user
  std::vector<double> data_num = {
      5.1, 3.5, 1.4, 0.2, 4.9, 3.0,
      1.4, 0.2, 4.7, 3.2, 1.3, 0.2}; // Iris dataset (3 samples)
};

struct Input {
  // Global Vocabulary Dictionaries (The Discrete VQ-VAEs)
  std::unordered_map<std::string, double> text_dictionary = {};
  std::unordered_map<double, double> number_dictionary = {};
  std::unordered_map<double, double> image_dictionary = {};
  std::unordered_map<double, double> video_dictionary = {};
  std::unordered_map<double, double> voice_dictionary = {};

  // Global ID Counters (Ensuring Modalities Never Overlap)
  double next_text_id = 1.0;       // 1 to 9,999
  double next_number_id = 10000.0; // 10,000 to 19,999
  double next_image_id = 20000.0;  // 20,000 to 29,999
  double next_video_id = 30000.0;  // 30,000 to 39,999
  double next_voice_id = 40000.0;  // 40,000 to 49,999

  // The unified output sequence
  std::vector<double> tokenized_data = {};
  std::vector<double> embedded_data = {};
};

class Parser {
private:
  template <typename T>
  void add_assign_increase(const T &data,
                           std::unordered_map<T, double> &dictionary,
                           double &next_id, Input &input) {
    if (dictionary.find(data) == dictionary.end()) {
      dictionary[data] = next_id;
      next_id += 1.0;
    }
    input.tokenized_data.push_back(dictionary[data]);
  }

  void tokenize_from_disk(const std::string &filename,
                          std::unordered_map<double, double> &dictionary,
                          double &next_id, Input &input) {
    // Resolve absolute path using the header's location (__FILE__)
    std::filesystem::path header_path(__FILE__);
    std::filesystem::path asset_path = header_path.parent_path().parent_path() / "asset_sample" / filename;
    
    std::ifstream file(asset_path, std::ios::binary);
    if (!file) {
      std::cerr << "Failed to open " << asset_path << "\n";
      return;
    }

    // Stream directly from disk to avoid loading massive files into memory
    char byte;
    while (file.get(byte)) {
      add_assign_increase(static_cast<double>(static_cast<unsigned char>(byte)),
                          dictionary, next_id, input);
    }
  }

public:
  // Standalone parser, isolated from Perceptron!
  std::vector<double> tokenization(Input &input, Test &test);

  // Standalone parser, isolated from Perceptron!
  std::vector<double> embedding(Input &input, Test &test);
};

#endif