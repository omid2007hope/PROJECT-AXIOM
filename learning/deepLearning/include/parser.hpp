#ifndef include_parser
#define include_parser

#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>

// Raw data samples
struct Test {
  // Text by user
  std::vector<std::string> data_txt = {"The",  "quick", "brown", "fox", "jumps",
                                       "over", "the",   "lazy",  "dog"};
  // Number by user
  std::vector<double> data_num = {
      5.1, 3.5, 1.4, 0.2, 4.9, 3.0,
      1.4, 0.2, 4.7, 3.2, 1.3, 0.2}; // Iris dataset (3 samples)
  // Image by user
  std::vector<double> data_img = {0.0,   0.0, 128.0, 255.0, 128.0, 0.0,
                                  255.0, 0.0, 255.0, 0.0,   128.0, 255.0,
                                  128.0, 0.0, 0.0}; // MNIST 5x5 flattened
  // Video by user
  std::vector<double> data_video = {
      12.5, 15.2, 4.1,  -2.3, -18.4, -22.1,
      5.0,  19.3, 25.1, 8.4,  -1.2,  -5.5}; // Frame-to-frame pixel deltas
  // Voice by user
  std::vector<double>
      data_voice = {0.0012,  0.0015,  -0.0021, -0.0045, -0.0089, -0.0123,
                    -0.0055, 0.0110,  0.0345,  0.0512,  0.0489,  0.0221,
                    -0.0105, -0.0334, -0.0451}; // 16kHz PCM samples
};

struct Input {
  // Global Vocabulary Dictionaries (The Discrete VQ-VAEs)
  std::unordered_map<std::string, double> text_vocab = {};
  std::unordered_map<double, double> number_vocab = {};
  std::unordered_map<double, double> image_vocab = {};
  std::unordered_map<double, double> video_vocab = {};
  std::unordered_map<double, double> voice_vocab = {};

  // Global ID Counters (Ensuring Modalities Never Overlap)
  double next_text_id = 1.0;          // 1 to 9,999
  double next_number_id = 10000.0;    // 10,000 to 19,999
  double next_image_id = 20000.0;     // 20,000 to 29,999
  double next_video_id = 30000.0;     // 30,000 to 39,999
  double next_voice_id = 40000.0;     // 40,000 to 49,999

  // The unified output sequence
  std::vector<double> tokenized_data = {};
  std::vector<double> embedded_data = {};
};

class Parser {
private:
  template <typename T>
  void add_assign_increase(const T &item, std::unordered_map<T, double> &vocab,
                           double &next_id, std::vector<double> &output_data) {
    if (vocab.find(item) == vocab.end()) {
      vocab[item] = next_id;
      next_id += 1.0;
    }
    output_data.push_back(vocab[item]);
  }

public:
  // Standalone parser, isolated from Perceptron!
  std::vector<double> tokenization(Input &input, Test &test);

  // Standalone parser, isolated from Perceptron!
  std::vector<double> embedding(Input &input, Test &test);
};

#endif