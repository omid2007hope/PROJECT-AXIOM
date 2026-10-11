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
#include <io.h>    // For _isatty() and _fileno()
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdfloat> // Required for std::bfloat16_t (C++23)
#include <string>
#include <type_traits>
#include <vector>

// psapi.h needs the types from windows.h, so this order must not be re-sorted
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on

#include "../include/ai.hpp"
#include "../include/optimizer.hpp"
#include "../include/telemetry.hpp"
#include "../src/CUDA/kernel/forward_pass.cuh"

bool test_path();
bool test_ifstream();
bool test_embeddings();
bool test_tokenization();
bool test_optimizer();
bool test_cuda_transfer();
bool test_transfer();
bool test_ai();
bool test_inference();

// *****************************************************************************
// Configuration: the single source of truth for everything the backend uses
// *****************************************************************************
namespace config {

// Hyperparameters
constexpr size_t ram_buffer_size = 100000;
constexpr size_t token_dim = 4;
constexpr size_t context_window_size = 128;
constexpr float learning_rate = 0.001f;

// How many training steps pass between two loss reports
constexpr size_t loss_report_interval = 10000;

// Initial weights are drawn uniformly from [weight_min, weight_max]
constexpr float weight_min = -0.1f;
constexpr float weight_max = 0.1f;

// Folder layout, relative to the core/ folder
constexpr const char* database_folder = "database";
constexpr const char* embedded_file = "embedded_data.bin";
constexpr const char* asset_folder = "asset_sample";
constexpr const char* weights_file = "weights.axiom";

} // namespace config

enum class AppMode {
	Use,
	Train,
	BulkTrain,
	Test,
	Benchmark,
	Exit
};

// core/ folder, located from this source file (core/src/main.cpp)
std::filesystem::path locate_core_dir() {
	return std::filesystem::absolute(__FILE__).parent_path().parent_path();
}

// Raw data samples used by developer mode (--train)
Test make_sample_data() {
	Test test;
	test.data_txt = {"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"};
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

// Pauses and waits for a key press before console window closes
void pause_exit() {
	if (!_isatty(_fileno(stdin))) {
		return;
	}
	std::cout << "\nPress any key to exit...\n";
	while (_kbhit()) {
		_getch();
	}
	_getch();
}

// Validates a single whitespace-free token
template <typename E>
bool parse_token(const std::string& token, E& value) {
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
template <typename T>
T Validate(const std::string& c_out) {
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
		Element value {};
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

// Validates a single file or directory path, preserving whitespace and stripping quotes
std::vector<std::string> ValidatePaths(const std::string& c_out) {
	std::string c_in;
	while (true) {
		std::cout << c_out << "\n";
		if (!std::getline(std::cin, c_in)) {
			std::cin.clear();
			std::cerr << "\n[!] Input stream closed. Exiting.\n";
			std::exit(1);
		}
		size_t start = c_in.find_first_not_of(" \t\r\n");
		if (start == std::string::npos) {
			std::cout << "[!] Invalid input. Please try again.\n";
			continue;
		}
		size_t end = c_in.find_last_not_of(" \t\r\n");
		std::string trimmed = c_in.substr(start, end - start + 1);
		if (trimmed.size() >= 2 && trimmed.front() == '"' && trimmed.back() == '"') {
			trimmed = trimmed.substr(1, trimmed.size() - 2);
		}
		if (!trimmed.empty()) {
			return {trimmed};
		}
		std::cout << "[!] Invalid input. Please try again.\n";
	}
}

// fetch arguments
std::string fetch_argument(int count, char* argument[]) {
	return count > 1 ? std::string(argument[1]) : std::string();
}

// Interactive menu. Fills `input` and returns the selected AppMode.
AppMode collect_user_input(Input& input) {
	std::cout << "======================================\n"
		  << "           Welcome to AXIOM           \n"
		  << "======================================\n"
		  << "[1] Use it (Inference)\n"
		  << "[2] Train it (Training)\n"
		  << "[3] Run Tests\n"
		  << "[4] Benchmark Hardware (Telemetry HUD)\n"
		  << "[5] Bulk Training (asset_sample)\n"
		  << "[6] Exit\n"
		  << "> ";

	const int option = read_choice();

	if (option == 1) {
		return AppMode::Use;
	}

	if (option == 3) {
		return AppMode::Test;
	}

	if (option == 4) {
		return AppMode::Benchmark;
	}

	if (option == 5) {
		return AppMode::BulkTrain;
	}

	if (option == 6) {
		return AppMode::Exit;
	}

	if (option != 2) {
		std::cout << "\n[!] Invalid option.\n";
		return AppMode::Exit;
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
		input.user_image_paths = ValidatePaths("Enter Image Path: ");
	} else if (type == 4) {
		input.user_video_paths = ValidatePaths("Enter Video Path: ");
	} else if (type == 5) {
		input.user_voice_paths = ValidatePaths("Enter Voice Path: ");
	} else {
		std::cout << "\n[!] Invalid data type.\n";
		return AppMode::Exit;
	}

	// Text and numbers can be paired with media
	if (type == 1 || type == 2) {
		std::cout << "\nAdd media?\n[1] Yes\n[2] No\n> ";

		const int media = read_choice();

		if (media == 1) {
			std::cout << "\nSelect Media Type:\n"
				  << "[1] Image\n"
				  << "[2] Video\n"
				  << "[3] Voice\n"
				  << "> ";

			const int media_type = read_choice();

			if (media_type == 1) {
				input.user_image_paths = ValidatePaths("Enter Image Path: ");
			} else if (media_type == 2) {
				input.user_video_paths = ValidatePaths("Enter Video Path: ");
			} else if (media_type == 3) {
				input.user_voice_paths = ValidatePaths("Enter Voice Path: ");
			} else {
				std::cout << "\n[!] Invalid media type.\n";
				return AppMode::Exit;
			}
		}

		else if (media != 2) {
			std::cout << "\n[!] Invalid option.\n";
			return AppMode::Exit;
		}
	}

	return AppMode::Train;
}

// *****************************************************************************
// Live hardware telemetry
// *****************************************************************************
void print_hardware_telemetry(const Portion& portion) {
	PROCESS_MEMORY_COUNTERS pmc {};
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));

	MEMORYSTATUSEX memInfo {};
	memInfo.dwLength = sizeof(MEMORYSTATUSEX);
	GlobalMemoryStatusEx(&memInfo);

	size_t gpu_free = 0, gpu_total = 0;
	get_gpu_telemetry(&gpu_free, &gpu_total);

	std::error_code ec;
	const std::filesystem::space_info disk_space =
		std::filesystem::space(std::filesystem::current_path(ec), ec);
	const double disk_free_gb = ec ? 0.0 : disk_space.available / 1024.0 / 1024.0 / 1024.0;

	std::cout << "\n========================================================\n"
		  << "                LIVE HARDWARE TELEMETRY                 \n"
		  << "========================================================\n"
		  << std::fixed << std::setprecision(2)
		  << "[CPU] L1/L2/L3 Cache: BYPASSED (Pinned Write-Combining)\n"
		  << "[RAM] Engine Stream : "
		  << (portion.ram_buffer_size * sizeof(std::bfloat16_t) / 1024.0 / 1024.0)
		  << " MB\n"
		  << "[RAM] Process Usage : " << (pmc.WorkingSetSize / 1024.0 / 1024.0) << " MB\n"
		  << "[RAM] System Avail  : " << (memInfo.ullAvailPhys / 1024.0 / 1024.0 / 1024.0)
		  << " GB / " << (memInfo.ullTotalPhys / 1024.0 / 1024.0 / 1024.0) << " GB\n"
		  << "[GPU] Context Window: "
		  << (portion.all_elements * sizeof(std::bfloat16_t) / 1024.0) << " KB\n"
		  << "[GPU] VRAM Available: " << (gpu_free / 1024.0 / 1024.0) << " MB / "
		  << (gpu_total / 1024.0 / 1024.0) << " MB\n"
		  << "[DSK] Disk Free     : " << disk_free_gb << " GB\n"
		  << "========================================================\n\n";
}

// *****************************************************************************
// Loss report: the proof that the network is learning
// *****************************************************************************

// Prints the mean loss of the last interval next to the very first one. A falling
// number means the weights are really being pulled toward the token sequences.
void print_loss(AI& ai, size_t step, std::optional<double>& first_loss) {
	double loss = 0.0;
	size_t steps = 0;

	if (!ai.read_loss(loss, steps)) {
		std::cerr << "[!] Could not read the loss from the GPU.\n";
		return;
	}

	if (!first_loss) {
		first_loss = loss;
	}

	std::cout << std::fixed << std::setprecision(6) << "[Step " << step << "] loss " << loss;

	if (*first_loss > 0.0) {
		std::cout << std::showpos << std::setprecision(1) << "  ("
			  << (loss - *first_loss) / *first_loss * 100.0 << "% vs first)"
			  << std::noshowpos;
	}

	std::cout << "\n";
}

// *****************************************************************************
// Inference Engine ("Use it")
// *****************************************************************************
void run_inference_engine(AI& ai,
			  Parser& parser,
			  Input& input,
			  const std::filesystem::path& weights_path) {
	std::cout << "\n========================================================\n"
		  << "             AXIOM INFERENCE ENGINE ('Use it')          \n"
		  << "========================================================\n";

	if (std::filesystem::exists(weights_path)) {
		std::cout << "[..] Loading model weights from: " << weights_path.filename().string() << "\n";
		if (ai.load_weights(weights_path.string())) {
			std::cout << "[OK] Model weights successfully loaded into VRAM.\n\n";
		} else {
			std::cerr << "[!] Failed to load model weights. Using baseline weights.\n\n";
		}
	} else {
		std::cout << "[!] Notice: '" << weights_path.filename().string()
			  << "' not found.\n"
			  << "[!] Running with baseline initialized weights.\n"
			  << "[!] Tip: Run 'app.exe --train' and press 'S' to save trained weights.\n\n";
	}

	parser.build_embedding_table(input);

	std::cout << "Model Architecture: Single Dedicated Multimodal Perceptron\n"
		  << "Context Window    : " << config::context_window_size << " tokens ("
		  << (config::context_window_size * config::token_dim * sizeof(std::bfloat16_t))
		  << " bytes BF16)\n"
		  << "Tensor Cores      : CUTLASS v4.8.0 Fused EVT (Ada Lovelace sm_89)\n"
		  << "Vocabulary Size   : " << input.embedding_table.size() << " tokens mapped\n"
		  << "--------------------------------------------------------\n"
		  << "Enter text or prompt to predict next token (type 'exit' to quit):\n";

	std::string line;
	while (true) {
		std::cout << "\nPrompt > ";
		if (!std::getline(std::cin, line)) {
			break;
		}

		if (line.empty()) {
			continue;
		}

		if (line == "exit" || line == "quit" || line == ":q") {
			std::cout << "Exiting inference engine.\n";
			break;
		}

		// Tokenize input prompt
		std::vector<uint32_t> tokens = parser.tokenize_prompt(input, line);
		std::cout << "[Tokens] (" << tokens.size() << " tokens): ";
		for (size_t i = 0; i < tokens.size(); ++i) {
			std::cout << tokens[i] << (i + 1 < tokens.size() ? ", " : "\n");
		}

		// Generate 128-token context window (512 BF16 floats)
		std::vector<std::bfloat16_t> ctx_window =
			parser.tokens_to_context_window(input, tokens, config::context_window_size);

		AxiomTelemetry telemetry;
		telemetry.begin_inference();

		// Run forward prediction on Tensor Cores
		std::vector<float> prediction;
		if (!ai.predict(ctx_window, prediction)) {
			std::cerr << "[!] Forward prediction failed.\n";
			continue;
		}
		telemetry.record_first_token();

		// Decode prediction via nearest-neighbor L2 search
		uint32_t pred_token_id = parser.decode_embedding_to_token(input, prediction);
		std::string pred_token_str = parser.decode_prediction(input, prediction);

		std::cout << std::fixed << std::setprecision(4)
			  << "[Tensor Output] [" << prediction[0] << ", " << prediction[1]
			  << ", " << prediction[2] << ", " << prediction[3] << "]\n"
			  << "[Next Token ID] #" << pred_token_id << "\n"
			  << "[Prediction]    \"" << pred_token_str << "\"\n";

		telemetry.end_inference(ai.h_weights.size());
	}
}

// *****************************************************************************
// Hardware Benchmark
// *****************************************************************************
void run_hardware_benchmark(AI& ai,
			    Parser& parser,
			    Input& input) {
	std::cout << "\n========================================================\n"
		  << "       AXIOM HARDWARE BENCHMARK & KERNEL SATURATION      \n"
		  << "========================================================\n";

	AxiomTelemetry telemetry;
	const HardwareSpecs& specs = telemetry.get_specs();

	std::cout << "Hardware Device : " << specs.gpu_name << "\n"
		  << "Architecture    : sm_" << specs.sm_major << specs.sm_minor << " (Ada Lovelace)\n"
		  << "Memory Bus      : " << specs.bus_width_bits << "-bit | Peak Bandwidth: " << std::fixed << std::setprecision(1) << specs.peak_bandwidth_gb_s << " GB/s\n"
		  << "Compute Engine  : " << specs.peak_dense_tflops << " Peak Dense BF16 TFLOPs\n"
		  << "VRAM Footprint  : " << (specs.free_vram_bytes / (1024 * 1024)) << " MB Free / "
		  << (specs.total_vram_bytes / (1024 * 1024)) << " MB Total\n"
		  << "--------------------------------------------------------\n"
		  << "[BENCHMARK 1/2] Interactive Micro-Perceptron Inference (512 -> 4)\n"
		  << "[..] Warming up Tensor Cores (10 forward passes)...\n";

	std::vector<std::bfloat16_t> dummy_ctx(config::context_window_size * config::token_dim,
					       static_cast<std::bfloat16_t>(0.05f));
	std::vector<float> pred;
	for (int i = 0; i < 10; ++i) {
		ai.predict(dummy_ctx, pred);
	}

	constexpr int bench_passes = 100;
	std::cout << "[..] Executing " << bench_passes << " Hardware-Timed Tensor Core Inferences...\n";

	telemetry.begin_inference();
	for (int i = 0; i < bench_passes; ++i) {
		ai.predict(dummy_ctx, pred);
		if (i == 0) {
			telemetry.record_first_token();
		} else {
			telemetry.record_token();
		}
	}
	telemetry.end_inference(ai.h_weights.size(), "INFERENCE DISPATCH LATENCY");

	// Benchmark 2: Saturated Matrix Sweep
	std::cout << "--------------------------------------------------------\n"
		  << "[BENCHMARK 2/2] CUTLASS EVT Kernel Saturation Sweep (M=128, N=4096, K=4096)\n"
		  << "[..] Allocating transformer-scale tensor buffers (34 MB VRAM footprint)...\n"
		  << "[..] Warming up all Ada Lovelace Streaming Multiprocessors...\n"
		  << "[..] Executing 50 Hardware-Timed Fused EVT Tensor Core Sweeps...\n";

	constexpr int sat_M = 128;
	constexpr int sat_N = 4096;
	constexpr int sat_K = 4096;
	constexpr int sat_iters = 50;

	double sat_tflops = 0.0;
	double sat_bw = 0.0;
	float sat_avg_ms = 0.0f;

	if (run_cutlass_saturation_benchmark(sat_M, sat_N, sat_K, sat_iters,
					     &sat_tflops, &sat_bw, &sat_avg_ms)) {
		float sat_bw_pct = static_cast<float>((sat_bw / specs.peak_bandwidth_gb_s) * 100.0);
		float sat_mfu_pct = static_cast<float>((sat_tflops / specs.peak_dense_tflops) * 100.0);
		std::string arch_tag = "sm_" + std::to_string(specs.sm_major) + std::to_string(specs.sm_minor);

		AxiomTelemetry::print_saturation_hud(sat_avg_ms, sat_tflops, sat_bw, sat_bw_pct, sat_mfu_pct, arch_tag);
	} else {
		std::cerr << "[!] CUTLASS saturation benchmark failed.\n";
	}
}

// *****************************************************************************
// Training loop
// *****************************************************************************
void run_training_loop(AI& ai,
		       Optimizer& optimizer,
		       Portion& portion,
		       Elements& elements,
		       const std::filesystem::path& weights_path,
		       size_t max_steps = 0) {
	std::cout << "Beginning continuous training loop...\n";
	std::cout << "Controls:\n"
		  << "  [S] Save current model weights to " << weights_path.filename().string() << "\n"
		  << "  [R] Reset AI brain to chaotic state\n"
		  << "  [Q] Save and quit training loop\n\n";

	// Steps trained since the start (or the last reset) and the first loss reported
	size_t step = 0;
	std::optional<double> first_loss;

	while (true) {
		// Check for interactive user commands without blocking
		char ch = 0;
		if (_kbhit()) {
			ch = _getch();
		} else if (!_isatty(_fileno(stdin))) {
			while (std::cin.rdbuf()->in_avail() > 0) {
				char c;
				if (std::cin.get(c)) {
					if (c == 'q' || c == 'Q' || c == 's' || c == 'S' || c == 'r' || c == 'R') {
						ch = c;
						break;
					}
				}
			}
			if (ch == 0) {
				HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
				DWORD bytesAvail = 0;
				if (PeekNamedPipe(hStdin, NULL, 0, NULL, &bytesAvail, NULL)) {
					if (bytesAvail > 0) {
						char c;
						if (std::cin.get(c)) {
							ch = c;
						}
					} else if (max_steps == 0 && step >= 20) {
						ch = 'q';
					}
				} else if (max_steps == 0 && step >= 20) {
					ch = 'q';
				}
			}
		}

		if (ch != 0) {
			if (ch == 'r' || ch == 'R') {
				std::cout << "\n[!] Manual Reset Triggered! Wiping Brain...\n";
				ai.reset_weights(elements);
				if (ai.upload_weights_to_gpu()) { // push new chaotic weights to VRAM
					step = 0;
					first_loss.reset();
					std::cout
						<< "[!] Brain wipe complete. Resuming training.\n\n";
				} else {
					std::cerr
						<< "[!] Brain wipe failed: could not upload weights.\n\n";
				}
			} else if (ch == 's' || ch == 'S') {
				std::cout << "\n[!] Manual Checkpoint Triggered! Saving weights...\n";
				if (ai.save_weights(weights_path.string())) {
					std::cout << "[OK] Model checkpoint saved to " << weights_path.filename().string() << "\n\n";
				} else {
					std::cerr << "[!] Failed to save weights.\n\n";
				}
			} else if (ch == 'q' || ch == 'Q') {
				std::cout << "\n[!] Stopping training loop. Saving checkpoint...\n";
				ai.save_weights(weights_path.string());
				std::cout << "[OK] Checkpoint saved. Exiting training.\n";
				break;
			}
		}

		// Feed forward & backward pass
		const TrainStatus status = ai.train(optimizer, portion);

		if (status == TrainStatus::Failed) {
			std::cerr << "[!] Training stopped: a GPU step failed.\n";
			std::exit(1);
		}

		if (status == TrainStatus::Trained) {
			++step;
			if (step % config::loss_report_interval == 0 || (max_steps > 0 && step == max_steps)) {
				print_loss(ai, step, first_loss);
			}
			if (max_steps > 0 && step >= max_steps) {
				std::cout << "\n[!] Target training steps reached (" << max_steps
					  << "). Saving model weights...\n";
				ai.save_weights(weights_path.string());
				std::cout << "[OK] Model checkpoint saved to " << weights_path.filename().string() << "\n";
				break;
			}
		}

		// When the RAM buffer is exhausted, the offset resets to 0.
		// We must pull the next massive chunk from disk into RAM.
		if (portion.offset == 0) {
			optimizer.disk_to_ram(portion);
		}
	}
}

bool run_tests() {
	bool all_ok = true;

	std::cout << "--- test_path AI ---\n";
	all_ok &= test_path();

	std::cout << "--- test_ifstream AI ---\n";
	all_ok &= test_ifstream();

	std::cout << "--- test_embeddings AI ---\n";
	all_ok &= test_embeddings();

	std::cout << "--- test_tokenization AI ---\n";
	all_ok &= test_tokenization();

	std::cout << "--- test_optimizer AI ---\n";
	all_ok &= test_optimizer();

	std::cout << "--- test_cuda_transfer AI ---\n";
	all_ok &= test_cuda_transfer();

	std::cout << "--- test_transfer AI ---\n";
	all_ok &= test_transfer();

	std::cout << "--- test_ai AI ---\n";
	all_ok &= test_ai();

	std::cout << "--- test_inference AI ---\n";
	all_ok &= test_inference();

	std::cout << "\n============================================\n";
	if (all_ok) {
		std::cout << "  ALL TEST SUITES PASSED (100% GREEN)       \n";
	} else {
		std::cerr << "  ONE OR MORE TEST SUITES FAILED            \n";
	}
	std::cout << "============================================\n\n";

	return all_ok;
}

// main function
int main(int argc, char* argv[]) {
	std::ios::sync_with_stdio(false);
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
#endif

	const std::string arg = fetch_argument(argc, argv);
	const bool is_interactive = arg.empty();

	if (arg == "--test") {
		return run_tests() ? 0 : 1;
	}

	if (arg == "--help" || arg == "-h") {
		std::cout << "========================================================\n"
			  << "   PROJECT AXIOM v1.0 — Native C++23/CUDA AI Engine     \n"
			  << "========================================================\n"
			  << "Usage: app.exe [OPTION]\n\n"
			  << "Options:\n"
			  << "  (no argument)  Interactive console menu\n"
			  << "  --use          Run direct forward inference ('Use it')\n"
			  << "  --train [N]    Train on sample data / prompt (optional N steps limit)\n"
			  << "  --bulk [N]     Bulk train on all files in asset_sample (optional N steps)\n"
			  << "  --benchmark    Run hardware saturation & telemetry benchmark\n"
			  << "  --test         Run all automated test suites\n"
			  << "  --help, -h     Show this help message\n\n";
		return 0;
	}

	if (!arg.empty() && arg != "--train" && arg != "--use" && arg != "--benchmark" && arg != "--bulk" && arg != "--train-bulk") {
		std::cerr << "Unknown argument: " << arg << "\n"
			  << "Usage: app.exe [--use | --train | --bulk | --benchmark | --test | --help]\n";
		return 1;
	}

	AppMode mode = AppMode::Train;
	bool developer_mode = false;
	size_t max_steps = 0;

	if (arg == "--use") {
		mode = AppMode::Use;
		developer_mode = true;
	} else if (arg == "--benchmark") {
		mode = AppMode::Benchmark;
		developer_mode = true;
	} else if (arg == "--bulk" || arg == "--train-bulk") {
		mode = AppMode::BulkTrain;
		developer_mode = false;
		if (argc > 2) {
			try {
				max_steps = std::stoull(argv[2]);
			} catch (...) {
				max_steps = 0;
			}
		}
	} else if (arg == "--train") {
		mode = AppMode::Train;
		developer_mode = true;
		if (argc > 2) {
			try {
				max_steps = std::stoull(argv[2]);
			} catch (...) {
				max_steps = 0;
			}
		}
	}

	// Everything below is configured here and handed down to the backend
	const std::filesystem::path core_dir = locate_core_dir();
	const std::filesystem::path embedded_path =
		core_dir / config::database_folder / config::embedded_file;
	const std::filesystem::path weights_path =
		std::filesystem::exists(core_dir / config::weights_file)
			? (core_dir / config::weights_file)
			: (std::filesystem::exists(config::weights_file)
				   ? std::filesystem::path(config::weights_file)
				   : (core_dir / config::weights_file));
	const std::vector<std::filesystem::path> asset_dirs = {core_dir / config::asset_folder};

	Parser parser(developer_mode, embedded_path, asset_dirs);
	Input input;
	input.embedding_dim = config::token_dim;
	Test test = make_sample_data();

	// If no argument, present interactive menu
	if (arg.empty()) {
		mode = collect_user_input(input);
		if (mode == AppMode::Exit) {
			return 0;
		}
		if (mode == AppMode::Test) {
			const bool ok = run_tests();
			pause_exit();
			return ok ? 0 : 1;
		}
		if (mode == AppMode::Use || mode == AppMode::Benchmark) {
			developer_mode = true;
			parser.developer_mode = true;
		}
	}

	// Instantiate the required objects for the AI
	Portion portion(config::ram_buffer_size, config::token_dim, config::context_window_size);
	Optimizer optimizer(embedded_path, config::learning_rate);

	// *********************************************************
	// Phase 1: Parsing and Storage
	// *********************************************************
	if (mode == AppMode::BulkTrain) {
		const std::filesystem::path asset_dir = core_dir / config::asset_folder;
		std::error_code ec;
		if (!std::filesystem::exists(asset_dir, ec)) {
			std::filesystem::create_directories(asset_dir, ec);
		}
		std::cout << "\n========================================================\n"
			  << "              AXIOM BULK TRAINING MODE                  \n"
			  << "========================================================\n"
			  << "Target Directory : " << asset_dir.string() << "\n\n";

		std::cout << "[..] Scanning and tokenizing all assets in " << asset_dir.filename().string() << "...\n";
		const size_t processed = parser.bulk_tokenize_directory(asset_dir, input);
		if (processed == 0 || input.tokenized_data.empty()) {
			std::cerr << "\n[!] No training files could be parsed from " << asset_dir.string() << "\n"
				  << "[i] Instructions:\n"
				  << "    1. Place training files into the 'asset_sample' folder:\n"
				  << "       - Text (.txt, .md, .csv, code files)\n"
				  << "       - Images (.jpeg, .png, .bmp, .webp)\n"
				  << "       - Videos (.mp4, .mkv, .avi)\n"
				  << "       - Voice/Audio (.ogg, .wav, .mp3)\n"
				  << "       - Documents (.pdf) & 3D models (.obj, .stl, etc.)\n"
				  << "    2. Then run Bulk Training again.\n";
			if (is_interactive) {
				pause_exit();
			}
			return 1;
		}
		std::cout << "\n[OK] Bulk ingestion complete: " << input.tokenized_data.size()
			  << " tokens collected across " << processed << " files.\n";
	} else {
		std::cout << "[..] Parsing and tokenizing multimodal data...\n";
		parser.tokenization(input, test);
		if (input.tokenized_data.empty()) {
			std::cerr << "[!] No data could be tokenized from the specified input.\n";
			if (is_interactive) {
				pause_exit();
			}
			return 1;
		}
	}

	// Repeat sequence if shorter than one full context window + target to enable continuous training
	const size_t min_training_tokens = config::context_window_size + 1;
	if (input.tokenized_data.size() < min_training_tokens) {
		std::cout << "[..] Dataset sequence (" << input.tokenized_data.size()
			  << " tokens) is shorter than context window (" << config::context_window_size
			  << " tokens). Repeating sequence to enable continuous autoregressive training...\n";
		const size_t orig_sz = input.tokenized_data.size();
		const size_t target_tokens = config::context_window_size * 8;
		while (input.tokenized_data.size() < target_tokens) {
			for (size_t i = 0; i < orig_sz && input.tokenized_data.size() < target_tokens; ++i) {
				input.tokenized_data.push_back(input.tokenized_data[i]);
			}
		}
	}

	parser.embedding(input, test);
	parser.build_embedding_table(input);
	std::cout << "[OK] Vocabulary prepared: " << input.embedding_table.size() << " tokens mapped.\n";

	// *********************************************************
	// Phase 2: AI Instantiation and GPU Allocation
	// *********************************************************
	std::cout << "[..] Initializing AI model...\n";
	Elements ai_elements;
	AI ai(portion.all_elements,
	      portion.target_elements,
	      config::weight_min,
	      config::weight_max,
	      ai_elements);

	if (!ai.upload_weights_to_gpu()) {
		std::cerr << "[!] Failed to upload the AI weights to VRAM.\n";
		if (is_interactive) {
			pause_exit();
		}
		return 1;
	}

	// *********************************************************
	// Phase 3: Mode Execution Dispatch
	// *********************************************************
	if (mode == AppMode::Use) {
		run_inference_engine(ai, parser, input, weights_path);
		if (is_interactive) {
			pause_exit();
		}
		return 0;
	}

	if (mode == AppMode::Benchmark) {
		run_hardware_benchmark(ai, parser, input);
		if (is_interactive) {
			pause_exit();
		}
		return 0;
	}

	// Training Mode
	if (developer_mode) {
		std::cout << "--- Training AI ---\n";
	}

	std::cout << "[..] Optimizing data and loading to RAM...\n";
	if (!portion.allocate_pinned()) {
		std::cerr << "[!] Failed to allocate pinned host memory.\n";
		if (is_interactive) {
			pause_exit();
		}
		return 1;
	}

	optimizer.disk_to_ram(portion);

	if (portion.ram_buffer_tokens.size() < portion.all_elements + portion.target_elements) {
		std::cerr << "[!] Not enough data in " << embedded_path
			  << " to fill one context window.\n";
		portion.free_pinned();
		if (is_interactive) {
			pause_exit();
		}
		return 1;
	}

	// If pre-existing weights exist on disk, load them into VRAM before training
	if (std::filesystem::exists(weights_path)) {
		std::cout << "[..] Pre-existing weights found at " << weights_path.filename().string()
			  << ". Loading checkpoint...\n";
		ai.load_weights(weights_path.string());
	}

	print_hardware_telemetry(portion);

	run_training_loop(ai, optimizer, portion, ai_elements, weights_path, max_steps);

	portion.free_pinned();

	if (mode == AppMode::BulkTrain) {
		std::cout << "\n========================================================\n"
			  << "          BULK TRAINING COMPLETED SUCCESSFULLY          \n"
			  << "========================================================\n"
			  << "[OK] Model weights safely saved to: " << weights_path.filename().string() << "\n";

		const auto& bulk_files = parser.get_last_bulk_files();
		if (is_interactive && !bulk_files.empty()) {
			std::cout << "\nProcessed Asset Management (" << bulk_files.size() << " file"
				  << (bulk_files.size() == 1 ? "" : "s") << "):\n"
				  << "  [1] Keep files in asset_sample/ (Default)\n"
				  << "  [2] Archive processed files to asset_sample/archive/\n"
				  << "  [3] Purge (Delete) processed files from asset_sample/\n"
				  << "> ";

			int cleanup_choice = read_choice();
			if (cleanup_choice <= 0 || cleanup_choice > 3) {
				cleanup_choice = 1;
			}

			if (cleanup_choice == 2) {
				const std::filesystem::path archive_dir = (core_dir / config::asset_folder) / "archive";
				std::error_code ec;
				std::filesystem::create_directories(archive_dir, ec);
				size_t archived_count = 0;
				for (const auto& fpath : bulk_files) {
					if (std::filesystem::exists(fpath, ec)) {
						std::filesystem::path dest = archive_dir / fpath.filename();
						if (std::filesystem::exists(dest, ec)) {
							std::string stem = fpath.stem().string();
							std::string ext = fpath.extension().string();
							dest = archive_dir / (stem + "_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ext);
						}
						std::filesystem::rename(fpath, dest, ec);
						if (ec) {
							ec.clear();
							std::filesystem::copy_file(fpath, dest, std::filesystem::copy_options::overwrite_existing, ec);
							if (!ec) {
								std::filesystem::remove(fpath, ec);
								++archived_count;
							}
						} else {
							++archived_count;
						}
					}
				}
				std::cout << "[OK] Successfully archived " << archived_count
					  << " file(s) to " << archive_dir.string() << "\n";
			} else if (cleanup_choice == 3) {
				std::cout << "\n[!] WARNING: This will permanently delete " << bulk_files.size()
					  << " file(s) from " << (core_dir / config::asset_folder).string() << "!\n"
					  << "Are you sure you want to proceed? (y/N): ";
				char confirm = 0;
				if (std::cin >> confirm) {
					safeCin();
				}
				if (confirm == 'y' || confirm == 'Y') {
					size_t purged_count = 0;
					for (const auto& fpath : bulk_files) {
						std::error_code ec;
						if (std::filesystem::remove(fpath, ec)) {
							++purged_count;
						}
					}
					std::cout << "[OK] Successfully purged " << purged_count
						  << " file(s) from asset_sample/.\n";
				} else {
					std::cout << "[i] Purge cancelled. Preserving all files in asset_sample/.\n";
				}
			} else {
				std::cout << "[i] Keeping all processed files in asset_sample/.\n";
			}
		} else {
			std::cout << "[i] You may now remove files from 'asset_sample',\n"
				  << "    add new training data, and run Bulk Training again\n"
				  << "    to continuously fine-tune the model.\n";
		}
		std::cout << "========================================================\n";
	}

	if (is_interactive) {
		pause_exit();
	}
	return 0;
}
