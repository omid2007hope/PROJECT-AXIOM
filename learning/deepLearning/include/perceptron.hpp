#ifndef include_perceptron
#define include_perceptron

#include <cstddef>
#include <vector>

struct Key_Concepts {
  double weight;        // Weight
  double biases;        // Bias
  double forward_pass;  // Guess
  double loss_function; // Calculate how wrong the guess was
};

class Perceptron {
public:
  Perceptron();
};

#endif // include_perceptron
