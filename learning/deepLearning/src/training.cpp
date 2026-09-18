#include "../include/training.hpp"
#include "../include/optimizer.hpp"

// Right now the only thing that it does is make contextWindow in use True
void Training::training(Train &train) {
  Portion portion;
  Optimizer optimizer;

  train.training_in_use = true;

  optimizer.ram_to_contextWindow(portion, train.training_in_use);
}
