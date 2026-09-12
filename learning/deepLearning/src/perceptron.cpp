#include "../include/perceptron.hpp"

Perceptron::Perceptron(Key_Concepts &keys) {
  // Just a prototype now!
  double weight = keys.weight;
  double biases = keys.biases;
  double forward_pass = keys.forward_pass;
  double loss_function = keys.loss_function;

  // Test keys
  weight = 0.0;
  biases = 0.0;
  forward_pass = 0.0;
  loss_function = 0.0;
}

/*
Turn raw data samples into the parsed input vector;
*/
double Perceptron::data_parser(Test_Sample &test, Input &input) {
  // Parsed data

  // 1. Parse Text: Convert strings to doubles (e.g., sum of ASCII values)
  for (const std::string &word : test.data_txt) {
    double unified = 0.0;
    for (char c : word) {
      unified += static_cast<double>(c); // Simple embedding
    }
    input.data.push_back(unified);
  }

  // 2. Parse Numbers: Already doubles, push directly
  for (double val : test.data_num) {
    double unified = val;
    input.data.push_back(unified);
  }

  // 3. Parse Images: Normalize pixel values (0-255) to (0.0-1.0)
  for (double pixel : test.data_img) {
    double unified = pixel / 255.0;
    input.data.push_back(unified);
  }

  // 4. Parse Video: Push frame deltas directly
  for (double delta : test.data_video) {
    double unified = delta;
    input.data.push_back(unified);
  }

  // 5. Parse Voice: Push audio PCM samples directly
  for (double sample : test.data_voice) {
    double unified = sample;
    input.data.push_back(unified);
  }

  return input.data;
};
