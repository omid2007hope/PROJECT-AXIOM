#include <iostream>
#include <string>

#include "../include/ai.hpp"
#include "../include/optimizer.hpp"

void test_path();
void test_ifstream();
void test_embeddings();
void test_tokenization();
void test_optimizer();
void test_cuda_transfer();

auto fetch_argument(int count, char *argument[]) {
  for (int i = 1; i < count; i++) {
    return std::string(argument[i]);
  }
  return std::string("");
}

int main(int argc, char *argv[]) {
  auto arg = fetch_argument(argc, argv);

  if (arg == "--train") {
    std::cout << "--- Training AI ---\n";

    // Instantiate the required objects for the AI to train
    Portion portion;
    Optimizer optimizer;

    /* LEARN: Omid Teimory 2026 All rights reserved
    AI is the type (your class).

    ai is just the name we decided to give this specific variable. We could have
    named it my_brain, model, or test_network.

    (...) tells C++: "Create this variable, and instantly pass these two numbers
    into the AI(size_t in_dim, size_t out_dim) constructor to set it up."
    */

    AI ai(portion.all_elements, portion.target_elements);

    /* LEARN: Omid Teimory 2026 All rights reserved
    Locks the Thread: It sets training_in_use to true so the CPU data-loader
    thread doesn't collide with the GPU thread.

    Preps the Data: It asks the optimizer to load the batch of data from system
    RAM into the context window.

    PCIe Transfer: It pushes that data across the motherboard into your RTX
    4080's VRAM using upload_to_gpu().

    Predicts (Forward Pass): It fires your custom CUDA forward_pass kernel to
    push the data through the neural network and generate a prediction.

    Learns (Backward Pass): It fires the CUDA backward_pass kernel to calculate
    the errors and adjust the network's weights.

    Swaps Buffers: It flips the buffer_index (e.g., from 0 to 1) so the CPU can
    start loading the next batch of data while the GPU finishes its current
    math.
    */

    ai.train(optimizer, portion);

  } else if (arg == "--test") {
    std::cout << "--- test_path AI ---\n";
    test_path();

    std::cout << "--- test_ifstream AI ---\n";
    test_ifstream();

    std::cout << "--- test_embeddings AI ---\n";
    test_embeddings();

    std::cout << "--- test_tokenization AI ---\n";
    test_tokenization();

    std::cout << "--- test_optimizer AI ---\n";
    test_optimizer();

    std::cout << "--- test_cuda_transfer AI ---\n";
    test_cuda_transfer();
  } else {
    std::cout << "Please provide --train or --test\n";
  }

  return 0;
}
