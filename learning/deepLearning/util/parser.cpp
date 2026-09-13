#include "../include/parser.hpp"

/*
Turn raw data samples into the parsed input vector;
Now isolated from the neural network entirely.
*/
std::vector<double> data_parser(Test_Sample &test, Input &input,
                                Polynomial &polynomial) {
  // 1. Parse Text: Polynomial rolling hash to prevent anagram collisions and
  // normalize magnitude
  for (const std::string &word : test.data_txt) {
    for (unsigned char c : word) {
      polynomial.hash_value =
          (polynomial.hash_value + c * polynomial.p_pow) % polynomial.m;
      polynomial.p_pow = (polynomial.p_pow * polynomial.p) % polynomial.m;
    }

    // Normalize to [0.0, 1.0] to mathematically align with other modalities
    // (like image pixels)
    double unified =
        static_cast<double>(polynomial.hash_value) / static_cast<double>(m);
    input.data.push_back(unified);
  }

  // 2. Parse Numbers: Min-Max normalize based on known dataset bounds (e.g.
  // Iris max ~10.0)
  for (double val : test.data_num) {
    double unified = val / 10.0;
    input.data.push_back(unified);
  }

  // 3. Parse Images: Normalize pixel values (0-255) to (0.0-1.0)
  for (double pixel : test.data_img) {
    double unified = pixel / 255.0;
    input.data.push_back(unified);
  }

  // 4. Parse Video: Normalize frame deltas (-255 to 255) to (-1.0 to 1.0)
  for (double delta : test.data_video) {
    double unified = delta / 255.0;
    input.data.push_back(unified);
  }

  // 5. Parse Voice: Push audio PCM samples directly
  for (double sample : test.data_voice) {
    double unified = sample;
    input.data.push_back(unified);
  }

  return input.data;
}
