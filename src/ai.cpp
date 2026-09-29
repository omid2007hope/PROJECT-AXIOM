#include "../include/ai.hpp"

AI::AI(Key_Concepts &keys) {
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
