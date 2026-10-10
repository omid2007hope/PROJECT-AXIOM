#ifndef include_parser
#define include_parser

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdfloat>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Raw data samples (filled in by the caller, e.g. main)
struct Test {
	// Text by user
	std::vector<std::string> data_txt = {};
	// Number by user
	std::vector<double> data_num = {};
	// Media file names, resolved against the parser's asset directories
	std::vector<std::string> data_image = {};
	std::vector<std::string> data_video = {};
	std::vector<std::string> data_voice = {};
};

struct Input {
	// Global Vocabulary Dictionaries (The Discrete VQ-VAEs)
	std::unordered_map<std::string, uint32_t> text_dictionary = {};
	std::unordered_map<double, uint32_t> number_dictionary = {};
	std::unordered_map<uint8_t, uint32_t> image_dictionary = {};
	std::unordered_map<uint8_t, uint32_t> video_dictionary = {};
	std::unordered_map<uint8_t, uint32_t> voice_dictionary = {};

	// Production Mode User Inputs (When developer_mode = false)
	std::vector<std::string> user_text = {};
	std::vector<double> user_numbers = {};
	std::vector<std::string> user_image_paths = {};
	std::vector<std::string> user_video_paths = {};
	std::vector<std::string> user_voice_paths = {};

	// Global ID Counters (Ensuring Modalities Never Overlap)
	uint32_t next_text_id = 1;	    // 1 to 9,999,999
	uint32_t next_number_id = 10000000; // 10,000,000 to 19,999,999
	uint32_t next_image_id = 20000000;  // 20,000,000 to 29,999,999
	uint32_t next_video_id = 30000000;  // 30,000,000 to 39,999,999
	uint32_t next_voice_id = 40000000;  // 40,000,000 to 49,999,999

	// The unified output sequence
	std::vector<uint32_t> tokenized_data = {};

	// bfloat16_t
	std::vector<std::bfloat16_t> embedded_data = {};

	// Embedding lookup table: token_id -> dense vector of size embedding_dim
	// bfloat16_t
	size_t embedding_dim = 4;

	// bfloat16_t
	std::unordered_map<uint32_t, std::vector<std::bfloat16_t>> embedding_table = {};
};

class Parser {
private:
	// Binary file written by embedding(); decided by the caller.
	std::filesystem::path embedded_path;

	// Folders searched for relative media file names; decided by the caller.
	std::vector<std::filesystem::path> asset_dirs;

	// ********************
	// add to dictionaries
	// ********************
	template <typename T>
	void add_assign_increase(const T& data,
				 std::unordered_map<T, uint32_t>& dictionary,
				 uint32_t& next_id,
				 Input& input) {
		if (dictionary.find(data) == dictionary.end()) {
			dictionary[data] = next_id;
			next_id += 1;
		}
		input.tokenized_data.push_back(dictionary[data]);
	}

	// ********************
	// Generic data saver for any vector type (floats, ints, etc.)
	// ********************
	template <typename T>
	void save_data(const std::vector<T>& data, const std::filesystem::path& full_path) {
		// Create the output folder (and any subfolders) if they don't exist
		std::filesystem::create_directories(full_path.parent_path());

		// Use full_path for the file stream
		std::ofstream out_file(full_path, std::ios::binary | std::ios::trunc);

		if (!out_file) {
			std::cerr << "Warning: Failed to open " << full_path << " for writing.\n";
			return;
		}
		// Write the entire vector to disk in one lightning-fast operation
		out_file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
		out_file.close();
		std::cout << "Saved " << data.size() << " items to " << full_path << "\n";
	}

	// ********************
	// Read from disk
	// ********************
	void tokenize_from_disk(const std::string& fileName,
				std::unordered_map<uint8_t, uint32_t>& dictionary,
				uint32_t& next_id,
				Input& input) {

		std::filesystem::path c_standard_fileName(fileName);

		if (!c_standard_fileName.is_absolute()) {
			for (const auto& dir : asset_dirs) {
				const std::filesystem::path each_path = dir / fileName;
				if (std::filesystem::exists(each_path)) {
					c_standard_fileName = each_path;
					std::cout << "Found in\n" << c_standard_fileName << "\n";
					break;
				}
			}
		}

		std::ifstream parsed_data(c_standard_fileName, std::ios::binary);

		if (!parsed_data) {
			std::cerr << "Failed to open " << c_standard_fileName << "\n";
			return;
		}

		// Stream directly from disk to avoid loading massive files into memory
		char byte;
		while (parsed_data.get(byte)) {
			add_assign_increase(static_cast<uint8_t>(byte), dictionary, next_id, input);
		}
	}

public:
	// Mode (true: tokenize the Test samples, false: tokenize the user's Input)
	bool developer_mode;

	// Every setting is handed in by the caller; the parser decides nothing.
	Parser(bool developer_mode,
	       std::filesystem::path embedded_path,
	       std::vector<std::filesystem::path> asset_dirs)
	    : embedded_path(std::move(embedded_path)),
	      asset_dirs(std::move(asset_dirs)),
	      developer_mode(developer_mode) {}

	// Standalone parser, isolated from AI!
	std::vector<uint32_t> tokenization(Input& input, Test& test);

	// Standalone parser, isolated from AI!
	std::vector<std::bfloat16_t> embedding(Input& input, Test& test);

	// Decoder: finds the nearest token ID in the embedding space for a continuous prediction vector
	uint32_t decode_embedding_to_token(const Input& input, const std::vector<float>& embedding_vector);

	// Formats a discrete token ID back into its human-readable representation
	std::string token_to_string(const Input& input, uint32_t token_id);

	// Decodes continuous embedding prediction directly to human-readable string
	std::string decode_prediction(const Input& input, const std::vector<float>& embedding_vector);

	// Ensures the embedding lookup table is populated for all known tokens
	void build_embedding_table(Input& input);

	// Tokenizes a text prompt string into discrete token IDs, registering any unseen words
	std::vector<uint32_t> tokenize_prompt(Input& input, const std::string& prompt);

	// Converts a sequence of token IDs into a 128-token context window vector (size = context_tokens * embedding_dim)
	std::vector<std::bfloat16_t> tokens_to_context_window(const Input& input,
							       const std::vector<uint32_t>& tokens,
							       size_t context_tokens = 128);
};

#endif