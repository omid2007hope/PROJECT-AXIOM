#ifndef include_parser
#define include_parser

#include <cstddef>
#include <ifstream>
#include <ios>
#include <string>
#include <unordered_map>
#include <vector>

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

  void from_disk(const T &path) {
    std::ifstream file(path, std::ios::binary) if (file) {
      std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file),
                                        {})
    }
    else {
      std::cerr << "Failed to open", path, "/n";
    }
  };

  void add_assign_increase(const T &data,
                           std::unordered_map<T, double> &dictionary,
                           double &next_id, Input &input) {
    if (dictionary.find(data) == dictionary.end()) {
      dictionary[data] = next_id;
      next_id += 1.0;
    }
    input.tokenized_data.push_back(dictionary[data]);
  }

public:
  // Standalone parser, isolated from Perceptron!
  std::vector<double> tokenization(Input &input, Test &test);

  // Standalone parser, isolated from Perceptron!
  std::vector<double> embedding(Input &input, Test &test);
};

#endif