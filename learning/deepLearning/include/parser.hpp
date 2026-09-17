#ifndef include_parser
#define include_parser

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// Raw data samples
struct Test {
  // Text by user
  std::vector<std::string> data_txt = {"The",  "quick", "brown", "fox", "jumps",
                                       "over", "the",   "lazy",  "dog"};
  // Number by user
  std::vector<double> data_num = {5.1, 3.5, 1.4, 0.2, 4.9, 3.0,
                                  1.4, 0.2, 4.7, 3.2, 1.3, 0.2};
};

struct Input {
  // Global Vocabulary Dictionaries (The Discrete VQ-VAEs)
  std::unordered_map<std::string, uint32_t> text_dictionary = {};
  std::unordered_map<double, uint32_t> number_dictionary = {};
  std::unordered_map<uint8_t, uint32_t> image_dictionary = {};
  std::unordered_map<uint8_t, uint32_t> video_dictionary = {};
  std::unordered_map<uint8_t, uint32_t> voice_dictionary = {};

  // Production Mode User Inputs (When developer_mode = false)
  std::vector<std::string> user_text = {};
  std::vector<double> user_numbers = {};
  std::vector<std::string> user_image_paths = {};
  std::vector<std::string> user_video_paths = {};
  std::vector<std::string> user_voice_paths = {};

  // Global ID Counters (Ensuring Modalities Never Overlap)
  uint32_t next_text_id = 1;             // 1 to 9,999,999
  uint32_t next_number_id = 10000000;    // 10,000,000 to 19,999,999
  uint32_t next_image_id = 20000000;     // 20,000,000 to 29,999,999
  uint32_t next_video_id = 30000000;     // 30,000,000 to 39,999,999
  uint32_t next_voice_id = 40000000;     // 40,000,000 to 49,999,999

  // The unified output sequence
  std::vector<uint32_t> tokenized_data = {};
  std::vector<double>   embedded_data  = {};

  // Embedding lookup table: token_id -> dense vector of size embedding_dim
  uint32_t embedding_dim = 4;
  std::unordered_map<uint32_t, std::vector<double>> embedding_table = {};
};

class Parser {
private:
  // add to dictionaries
  template <typename T>
  void add_assign_increase(const T &data,
                           std::unordered_map<T, uint32_t> &dictionary,
                           uint32_t &next_id, Input &input) {
    if (dictionary.find(data) == dictionary.end()) {
      dictionary[data] = next_id;
      next_id += 1;
    }
    input.tokenized_data.push_back(dictionary[data]);
  }

  // Read from disk
  void tokenize_from_disk(const std::string &fileName,
                          std::unordered_map<uint8_t, uint32_t> &dictionary,
                          uint32_t &next_id, Input &input) {

    std::filesystem::path c_standard_fileName(fileName);

    if (!c_standard_fileName.is_absolute()) {
      std::filesystem::path absolute_path(__FILE__);
      std::vector<std::filesystem::path> possible_paths = {
          absolute_path.parent_path().parent_path() / "asset_sample" / fileName,
          std::filesystem::path("Brain/learning/deepLearning/asset_sample") / fileName,
          std::filesystem::path("learning/deepLearning/asset_sample") / fileName,
          std::filesystem::path("deepLearning/asset_sample") / fileName,
          std::filesystem::path("asset_sample") / fileName
      };

      for (const auto &each_path : possible_paths) {
        if (std::filesystem::exists(each_path)) {
          c_standard_fileName = each_path;
          std::cout << "Found in\n" << c_standard_fileName << "\n";
          break;
        }
      }
    }

    std::ifstream parsed_data(c_standard_fileName, std::ios::binary);

    if (!parsed_data) {
      std::cerr << "Failed to open " << c_standard_fileName << "\n";
      return;
    }

    // Stream directly from disk to avoid loading massive files into memory
    char byte;
    while (parsed_data.get(byte)) {
      add_assign_increase(static_cast<uint8_t>(byte),
                          dictionary, next_id, input);
    }
  }

public:
  // Mode
  bool developer_mode = true;

  // Standalone parser, isolated from Perceptron!
  std::vector<uint32_t> tokenization(Input &input, Test &test);

  // Standalone parser, isolated from Perceptron!
  std::vector<double> embedding(Input &input, Test &test);
};

#endif