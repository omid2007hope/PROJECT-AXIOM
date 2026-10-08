// =============================================================================
// AXIOM — main.cpp
//
// main() is the orchestrator. Every path, hyperparameter and sample value is
// decided in this file and handed down to the Parser, Optimizer and AI. The
// utility layers never decide anything on their own.
//
//   app.exe            interactive menu
//   app.exe --train    train on the built-in sample data
//   app.exe --test     run the test suites
// =============================================================================

#define NOMINMAX // keep <windows.h> from defining min()/max() macros

#include <conio.h> // For _kbhit() and _getch()
#include <psapi.h>
#include <windows.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdfloat> // Required for std::bfloat16_t (C++23)
#include <string>
#include <type_traits>
#include <vector>

#include "../include/ai.hpp"
#include "../include/optimizer.hpp"
#include "../src/CUDA/util/telemetry.cuh"

void test_path();
void test_ifstream();
void test_embeddings();
void test_tokenization();
void test_optimizer();
void test_cuda_transfer();

// *****************************************************************************
// Configuration: the single source of truth for everything the backend uses
// *****************************************************************************
namespace config {

// Hyperparameters
constexpr size_t ram_buffer_size = 100000;
constexpr size_t token_dim = 4;
constexpr size_t context_window_size = 128;
constexpr float learning_rate = 0.001f;

// Initial weights are drawn uniformly from [weight_min, weight_max]
constexpr float weight_min = -0.1f;
constexpr float weight_max = 0.1f;

// Folder layout, relative to the core/ folder
constexpr const char *database_folder = "database";
constexpr const char *embedded_file = "embedded_data.bin";
constexpr const char *asset_folder = "asset_sample";

} // namespace config

// core/ folder, located from this source file (core/src/main.cpp)
std::filesystem::path locate_core_dir() {
  return std::filesystem::absolute(__FILE__).parent_path().parent_path();
}

// Raw data samples used by developer mode (--train)
Test make_sample_data() {
  Test test;
  test.data_txt = {"The",  "quick", "brown", "fox", "jumps",
                   "over", "the",   "lazy",  "dog"};
  test.data_num = {5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2};
  test.data_image = {"Photo.jpeg"};
  test.data_video = {"Video.mp4"};
  test.data_voice = {"Voice.ogg"};
  return test;
}

// *****************************************************************************
// Console input
// *****************************************************************************

// ! Clears input stream errors and flushes invalid buffer characters
void safeCin() {
  if (!std::cin) {
    std::cin.clear();
  }
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

// Reads a menu choice; returns 0 when the input is not a number
int read_choice() {
  int choice = 0;
  std::cin >> choice;
  safeCin();
  return choice;
}

// Validates a single whitespace-free token
template <typename E> bool parse_token(const std::string &token, E &value) {
  char extra;
  std::istringstream iss(token);

  if (!(iss >> value) || (iss >> extra)) {
    return false;
  }

  if constexpr (std::is_arithmetic_v<E>) {
    // Numeric types: zero or positive only
    return value >= 0;
  } else {
    // Non-numeric types (e.g. std::string): any non-empty token is accepted
    return true;
  }
}

// Validator for final stage v-1
// Asks until the whole line is a non-empty, space-separated list of valid
// values. T is a vector of the wanted element type.
template <typename T> T Validate(const std::string &c_out) {
  using Element = typename T::value_type;

  std::string c_in;

  while (true) {
    std::cout << c_out << "\n";

    if (!std::getline(std::cin, c_in)) {
      std::cin.clear();
      std::cerr << "\n[!] Input stream closed. Exiting.\n";
      std::exit(1);
    }

    T data;
    Element value{};
    std::string token;
    std::istringstream iss(c_in);
    bool valid = true;

    while (iss >> token) {
      if (!parse_token(token, value)) {
        valid = false;
        break;
      }
      data.push_back(value);
    }

    if (valid && !data.empty()) {
      return data;
    }

    std::cout << "[!] Invalid input. Please try again.\n";
  }
}

// fetch arguments
std::string fetch_argument(int count, char *argument[]) {
  return count > 1 ? std::string(argument[1]) : std::string();
}

// Interactive menu. Fills `input` and returns true when it is ready to be
// parsed, or false when there is nothing to train on.
bool collect_user_input(Input &input) {
  std::cout << "==========================\n"
            << "     Welcome to AXIOM     \n"
            << "==========================\n"
            << "[1] Use it\n"
            << "[2] Train it\n"
            << "> ";

  const int option = read_choice();

  if (option == 1) {
    // For later on
    std::cout << "\n[!] 'Use it' is not available yet.\n";
    return false;
  }

  if (option != 2) {
    std::cout << "\n[!] Invalid option.\n";
    return false;
  }

  std::cout << "\n--- Training Mode: ON ---\n\n";

  std::cout << "Select Data Type:\n"
            << "[1] Text\n"
            << "[2] Number\n"
            << "[3] Image\n"
            << "[4] Video\n"
            << "[5] Voice\n"
            << "> ";

  const int type = read_choice();

  if (type == 1) {
    input.user_text = Validate<std::vector<std::string>>("Enter Text: ");
  } else if (type == 2) {
    input.user_numbers = Validate<std::vector<double>>("Enter Number: ");
  } else if (type == 3) {
    input.user_image_paths =
        Validate<std::vector<std::string>>("Enter Image Path: ");
  } else if (type == 4) {
    input.user_video_paths =
        Validate<std::vector<std::string>>("Enter Video Path: ");
  } else if (type == 5) {
    input.user_voice_paths =
        Validate<std::vector<std::string>>("Enter Voice Path: ");
  } else {
    std::cout << "\n[!] Invalid data type.\n";
    return false;
  }

  // Text and numbers can be paired with media
  if (type == 1 || type == 2) {
    std::cout << "\nAdd media?\n[1] Yes\n[2] No\n> ";

    const int media = read_choice();

    if (media == 1) {
      // For later
      std::cout << "\n[!] Adding media is not available yet.\n";
      return false;
    }

    if (media != 2) {
      std::cout << "\n[!] Invalid option.\n";
      return false;
    }
  }

  return true;
}

// *****************************************************************************
// Live hardware telemetry
// *****************************************************************************
void print_hardware_telemetry(const Portion &portion) {
  PROCESS_MEMORY_COUNTERS pmc{};
  GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));

  MEMORYSTATUSEX memInfo{};
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  GlobalMemoryStatusEx(&memInfo);

  size_t gpu_free = 0, gpu_total = 0;
  get_gpu_telemetry(&gpu_free, &gpu_total);

  std::error_code ec;
  const std::filesystem::space_info disk_space =
      std::filesystem::space(std::filesystem::current_path(ec), ec);
  const double disk_free_gb =
      ec ? 0.0 : disk_space.available / 1024.0 / 1024.0 / 1024.0;

  std::cout << "\n========================================================\n"
            << "                LIVE HARDWARE TELEMETRY                 \n"
            << "========================================================\n"
            << std::fixed << std::setprecision(2)
            << "[CPU] L1/L2/L3 Cache: BYPASSED (Pinned Write-Combining)\n"
            << "[RAM] Engine Stream : "
            << (portion.ram_buffer_size * sizeof(std::bfloat16_t) / 1024.0 /
                1024.0)
            << " MB\n"
            << "[RAM] Process Usage : "
            << (pmc.WorkingSetSize / 1024.0 / 1024.0) << " MB\n"
            << "[RAM] System Avail  : "
            << (memInfo.ullAvailPhys / 1024.0 / 1024.0 / 1024.0) << " GB / "
            << (memInfo.ullTotalPhys / 1024.0 / 1024.0 / 1024.0) << " GB\n"
            << "[GPU] Context Window: "
            << (portion.all_elements * sizeof(std::bfloat16_t) / 1024.0)
            << " KB\n"
            << "[GPU] VRAM Available: " << (gpu_free / 1024.0 / 1024.0)
            << " MB / " << (gpu_total / 1024.0 / 1024.0) << " MB\n"
            << "[DSK] Disk Free     : " << disk_free_gb << " GB\n"
            << "========================================================\n\n";
}

// *****************************************************************************
// Training loop (runs until the process is stopped)
// *****************************************************************************
[[noreturn]] void run_training_loop(AI &ai, Optimizer &optimizer,
                                    Portion &portion) {
  std::cout << "Beginning continuous training loop...\n";
  std::cout << "Press 'R' at any time to reset AI weights and start fresh.\n";

  while (true) {
    // Check for interactive user commands without blocking
    if (_kbhit()) {
      char ch = _getch();
      if (ch == 'r' || ch == 'R') {
        std::cout << "\n[!] Manual Reset Triggered! Wiping Brain...\n";
        ai.reset_weights();
        if (ai.upload_weights_to_gpu()) { // push new chaotic weights to VRAM
          std::cout << "[!] Brain wipe complete. Resuming training.\n\n";
        } else {
          std::cerr << "[!] Brain wipe failed: could not upload weights.\n\n";
        }
      }
    }

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

    // Feed forward & backward pass
    ai.train(optimizer, portion);

    // When the RAM buffer is exhausted, the offset resets to 0.
    // We must pull the next massive chunk from disk into RAM.
    if (portion.offset == 0) {
      optimizer.disk_to_ram(portion);
    }
  }
}

void run_tests() {
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

// main function
int main(int argc, char *argv[]) {
  const std::string arg = fetch_argument(argc, argv);

  if (arg == "--test") {
    run_tests();
    return 0;
  }

  if (!arg.empty() && arg != "--train") {
    std::cerr << "Usage: app.exe [--train | --test]\n"
              << "  (no argument)  interactive menu\n"
              << "  --train        train on the built-in sample data\n"
              << "  --test         run the test suites\n";
    return 1;
  }

  // Developer mode tokenizes the built-in samples; the menu uses real input
  const bool developer_mode = (arg == "--train");

  // Everything below is configured here and handed down to the backend
  const std::filesystem::path core_dir = locate_core_dir();
  const std::filesystem::path embedded_path =
      core_dir / config::database_folder / config::embedded_file;
  const std::vector<std::filesystem::path> asset_dirs = {core_dir /
                                                         config::asset_folder};

  // These are needed later on
  Parser parser(developer_mode, embedded_path, asset_dirs);
  Input input;
  input.embedding_dim = config::token_dim;
  Test test = make_sample_data();

  // Instantiate the required objects for the AI to train
  Portion portion(config::ram_buffer_size, config::token_dim,
                  config::context_window_size);
  Optimizer optimizer(embedded_path, config::learning_rate);

  if (developer_mode) {
    std::cout << "--- Training AI ---\n";
  } else if (!collect_user_input(input)) {
    return 0;
  }

  // *********************************************************
  // Phase 1: Parsing and Storage
  // *********************************************************

  // First thing first parser has top be fully executed!
  // Read and tokenize all
  parser.tokenization(input, test);

  // Embedded and Save them all
  parser.embedding(input, test);

  std::cout << "\n[OK] Data successfully generated at: " << embedded_path
            << "\n";

  // *********************************************************
  // Phase 2: Pinned Memory Allocation and Disk-to-RAM Stream
  // *********************************************************

  std::cout << "[..] Optimizing data and loading to RAM...\n";

  // Allocate pinned host memory for ultra-fast DMA transfers
  if (!portion.allocate_pinned()) {
    std::cerr << "[!] Failed to allocate pinned host memory.\n";
    return 1;
  }

  // Stream the initial chunk of parsed data from disk into the RAM buffer
  optimizer.disk_to_ram(portion);

  // The AI needs at least one full window plus its target to train on
  if (portion.ram_buffer_tokens.size() <
      portion.all_elements + portion.target_elements) {
    std::cerr << "[!] Not enough data in " << embedded_path
              << " to fill one context window.\n";
    return 1;
  }

  print_hardware_telemetry(portion);

  // *********************************************************
  // Phase 3: AI Instantiation and GPU PCIe Transfer
  // *********************************************************

  std::cout << "[..] Initializing AI model and uploading to VRAM...\n";

  /* LEARN: Omid Teimory 2026 All rights reserved
  AI is the type (your class).

  ai is just the name we decided to give this specific variable. We could
  have named it my_brain, model, or test_network.

  (...) tells C++: "Create this variable, and instantly pass these numbers
  into the AI(size_t in_dim, size_t out_dim, float weight_min, float
  weight_max) constructor to set it up."
  */

  AI ai(portion.all_elements, portion.target_elements, config::weight_min,
        config::weight_max);

  // Transfer the chaotic starting weights from CPU to GPU VRAM
  if (!ai.upload_weights_to_gpu()) {
    std::cerr << "[!] Failed to upload the AI weights to VRAM.\n";
    return 1;
  }

  run_training_loop(ai, optimizer, portion);
}
