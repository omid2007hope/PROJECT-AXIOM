#include <conio.h> // For _kbhit() and _getch()
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

// ! Clears input stream errors and flushes invalid buffer characters
void safeCin() {
  if (!std::cin) {
    std::cin.clear();
  }
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

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

  // These are needed later on
  Parser parser;
  Input input;
  Test test;

  // Instantiate the required objects for the AI to train
  Portion portion;
  Optimizer optimizer;

  if (arg == "") {
    std::cout << "==========================\n"
              << "     Welcome to AXIOM     \n"
              << "==========================\n"
              << "[1] Use it\n"
              << "[2] Train it\n"
              << "> ";
    int option = 1;
    std::cin >> option;
    safeCin();

    if (option == 1) {
      // For later on
    } else if (option == 2) {
      std::cout << "\n--- Training Mode: ON ---\n\n";

      int type = 1;
      std::cout << "Select Data Type:\n"
                << "[1] Text\n"
                << "[2] Number\n"
                << "[3] Image\n"
                << "[4] Video\n"
                << "[5] Voice\n"
                << "> ";
      std::cin >> type;
      safeCin();

      if (type == 1) {
        input.user_text = Validate<std::vector<std::string>>("Enter Text: ");

        int media = 1;
        std::cout << "\nAdd media?\n[1] Yes\n[2] No\n> ";
        std::cin >> media;
        safeCin();

        if (media == 1) {
          // For later
        } else if (media == 2) {
          // First thing first parser has top be fully executed!
          // Read and tokenize all
          std::vector<uint32_t> tokenize = parser.tokenization(input, test);

          // Embedded and Save them all
          std::vector<std::bfloat16_t> embed = parser.embedding(input, test);
        }
      } else if (type == 2) {
        input.user_numbers = Validate<std::vector<double>>("Enter Number: ");

        int media = 1;
        std::cout << "\nAdd media?\n[1] Yes\n[2] No\n> ";
        std::cin >> media;
        safeCin();

        if (media == 1) {
          // For later
        } else if (media == 2) {
          // First thing first parser has top be fully executed!
          // Read and tokenize all
          std::vector<uint32_t> tokenize = parser.tokenization(input, test);

          // Embedded and Save them all
          std::vector<std::bfloat16_t> embed = parser.embedding(input, test);
        }
      } else if (type == 3) {
        input.user_image_paths = Validate<std::vector<std::string>>("Enter Image Path: ");
      } else if (type == 4) {
        input.user_video_paths = Validate<std::vector<std::string>>("Enter Video Path: ");
      } else if (type == 5) {
        input.user_voice_paths = Validate<std::vector<std::string>>("Enter Voice Path: ");
      }
    }
  } else if (arg == "--train") {
    std::cout << "--- Training AI ---\n";

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
    std::cout << "Beginning continuous training loop...\n";
    std::cout << "Press 'R' at any time to reset AI weights and start fresh.\n";

    while (true) {
      // Check for interactive user commands without blocking
      if (_kbhit()) {
        char ch = _getch();
        if (ch == 'r' || ch == 'R') {
          std::cout << "\n[!] Manual Reset Triggered! Wiping Brain...\n";
          ai.reset_weights();
          ai.upload_weights_to_gpu(); // push new chaotic weights to VRAM
          std::cout << "[!] Brain wipe complete. Resuming training.\n\n";
        }
      }

      // Feed forward & backward pass
      ai.train(optimizer, portion);

      // When the RAM buffer is exhausted, the offset resets to 0.
      // We must pull the next massive chunk from disk into RAM.
      if (portion.offset == 0) {
        optimizer.disk_to_ram(portion);
      }
    }

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
