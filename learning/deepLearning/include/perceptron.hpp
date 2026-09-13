#ifndef include_perceptron
#define include_perceptron

#include <cstddef>
#include <string>
#include <vector>

struct Key_Concepts {
  double weight = 0.0;        // Weight
  double biases = 0.0;        // Bias
  double forward_pass = 0.0;  // Guess
  double loss_function = 0.0; // Calculate how wrong the guess was
};

class Perceptron {
public:
  double weight;
  double biases;
  double forward_pass;
  double loss_function;

  Perceptron(Key_Concepts &keys);
};

#endif // include_perceptron
