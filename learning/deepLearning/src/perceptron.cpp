#include "../include/perceptron.hpp"

Perceptron::Perceptron() {
  // Just a prototype now!

  Key_Concepts keys;

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
void Perceptron::data_parser() {
  // Parsed data
  std::vector<double> input = {};

  double unified = 0.0;

  input.push_back(unified);
}
