#include <iostream>
#include <stdfloat> // Required for std::bfloat16_t (C++23)
#include <string>
#include <vector>

#include "../include/ai.hpp"
#include "../include/optimizer.hpp"

void test_path();
void test_ifstream();
void test_embeddings();
void test_tokenization();
void test_optimizer();
void test_cuda_transfer();

// Validator for final stage v-1
template <typename T> T Validate(std::string &c_out) {

  std::string c_in;
  T data;

  while (true) {
    std::cout << c_out << "\n";

    if (!std::getline(std::cin, c_in)) {
      std::cin.clear();
      std::cerr << "\n[!] Input stream closed. Exiting.\n";
      std::exit(1);
    };

    char extra;
    std::istringstream iss(c_in);

    if ((iss >> data) && !(iss >> extra)) {
      if constexpr (std::is_arithmetic_v<T>) {
        // Numeric types: enforce positive-only or allow zero constraint
        if (data > 0 || (data == 0)) {
          return data;
        }
      } else {
        // Non-numeric types (e.g. std::string): accept any non-empty value
        return data;
      }
    }
  };
};

// fetch arguments
auto fetch_argument(int count, char *argument[]) {
  for (int i = 1; i < count; i++) {
    return std::string(argument[i]);
  }
  return std::string("");
}

// main function
int main(int argc, char *argv[]) {
  auto arg = fetch_argument(argc, argv);

  if (arg == "") {
    std::cout << "Welcome to AXIOM\n\n" 
              << "1: Use it\n" 
              << "2: Train it\n"
              << "> ";
    int option = 1;
    if (std::cin >> option) {
      // For later on
    }
  } else if (arg == "--train") {
    std::cout << "--- Training AI ---\n";

    // These are needed later on
    Parser parser;
    Input input;
    Test test;

      // Instantiate the required objects for the AI to train
      Portion portion;
      Optimizer optimizer;

      // First thing first parser has top be fully executed!
      // Read and tokenize all
      std::vector<uint32_t> tokenize = parser.tokenization(input, test);
      // Embedded and Save them all
      std::vector<std::bfloat16_t> embed = parser.embedding(input, test);

      // allocate space
      portion.allocate_pinned();

      // Fill the buffer from the saved data in
      // C:\Users\Omid\X\Code\PROJECT-AXIOM\core\database

      optimizer.disk_to_ram(portion);

      /* LEARN: Omid Teimory 2026 All rights reserved
      AI is the type (your class).

      ai is just the name we decided to give this specific variable. We could
      have named it my_brain, model, or test_network.

      (...) tells C++: "Create this variable, and instantly pass these two
      numbers into the AI(size_t in_dim, size_t out_dim) constructor to set it
      up."
      */

      AI ai(portion.all_elements, portion.target_elements);

      /* LEARN: Omid Teimory 2026 All rights reserved
      Locks the Thread: It sets training_in_use to true so the CPU data-loader
      thread doesn't collide with the GPU thread.

      Preps the Data: It asks the optimizer to load the batch of data from
      system RAM into the context window.

      PCIe Transfer: It pushes that data across the motherboard into your RTX
      4080's VRAM using upload_to_gpu().

      Predicts (Forward Pass): It fires your custom CUDA forward_pass kernel
      to push the data through the neural network and generate a prediction.

      Learns (Backward Pass): It fires the CUDA backward_pass kernel to
      calculate the errors and adjust the network's weights.

      Swaps Buffers: It flips the buffer_index (e.g., from 0 to 1) so the CPU
      can start loading the next batch of data while the GPU finishes its
      current math.
      */

      // send the data from cpu to gpu
      ai.upload_weights_to_gpu();

      // start training
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
  }

  return 0;
}
