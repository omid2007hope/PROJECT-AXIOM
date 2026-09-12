#include "../include/perceptron.hpp"

Perceptron::Perceptron(Key_Concepts &keys) {
  // Just a prototype now!
  weight = keys.weight;
  biases = keys.biases;
  forward_pass = keys.forward_pass;
  loss_function = keys.loss_function;

  // Test keys
  weight = 0.0;
  biases = 0.0;
  forward_pass = 0.0;
  loss_function = 0.0;
}

/*
Turn raw data samples into the parsed input vector;
Now isolated from the neural network entirely.
*/
std::vector<double> data_parser(Test_Sample &test, Input &input) {
  // 1. Parse Text: Polynomial rolling hash to prevent anagram collisions and normalize magnitude
  for (const std::string &word : test.data_txt) {
    long long hash_value = 0;
    long long p = 31;
    long long m = 1000000009; // Large prime modulus (1e9 + 9)
    long long p_pow = 1;

    for (unsigned char c : word) {
      hash_value = (hash_value + c * p_pow) % m;
      p_pow = (p_pow * p) % m;
    }
    
    // Normalize to [0.0, 1.0] to mathematically align with other modalities (like image pixels)
    double unified = static_cast<double>(hash_value) / static_cast<double>(m);
    input.data.push_back(unified);
  }

  // 2. Parse Numbers: Min-Max normalize based on known dataset bounds (e.g. Iris max ~10.0)
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

double Perceptron::mind() {
  Test_Sample test;
  Input input;
  std::vector<double> entry = data_parser(test, input);
  return 0.0;
}
