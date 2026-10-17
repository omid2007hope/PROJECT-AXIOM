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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>


struct Test {
	std::vector<std::string> test_input {};

	std::vector<std::string> data_image = {};
	std::vector<std::string> data_video = {};
	std::vector<std::string> data_voice = {};
};

struct Input {
	std::vector<uint32_t> tokenized_data = {};
	std::vector<std::bfloat16_t> embedded_data = {};

	mutable std::vector<uint32_t> decode_ids = {};
	mutable std::vector<float> decode_vectors = {};

	std::unordered_map<uint32_t, std::vector<std::bfloat16_t>> embedding_table = {};
	size_t embedding_dim = 4;

	// IDs
	uint32_t last_token = std::numeric_limits<uint32_t>::max();

	// Global Vocabulary Dictionaries (The Discrete VQ-VAEs)
	// text, number, symbols
	std::unordered_map<std::string, uint32_t> main_dictionary = {};
	std::unordered_map<uint32_t, std::string> reverse_main_dictionary = {};

	std::unordered_map<uint8_t, uint32_t> image_dictionary = {};
	std::unordered_map<uint8_t, uint32_t> video_dictionary = {};
	std::unordered_map<uint8_t, uint32_t> voice_dictionary = {};

	// Production Mode User Inputs (When developer_mode = false)
	std::vector<std::string> user_input = {};

	std::vector<std::string> user_image_paths = {};
	std::vector<std::string> user_video_paths = {};
	std::vector<std::string> user_voice_paths = {};

	// Global ID Counters (Ensuring Modalities Never Overlap)
	uint32_t next_text_id = 1;	      // 1 to 999,999,999
	uint32_t next_number_id = 1000000000; // 1,000,000,000 to 1,999,999,999
	uint32_t next_image_id = 2000000000;  // 2,000,000,000 to 2,999,999,999
	uint32_t next_video_id = 3000000000;  // 3,000,000,000 to 3,999,999,999
	uint32_t next_voice_id = 4000000000;  // 4,000,000,000 to 4,200,000,000

	// Optimization for test
	size_t CHUNK_TOKENS = 65536;
	size_t chunk_pos = 0;
	size_t tokens_written = 0;
	size_t last_pct = 0;

	// Demonstrate progress with bar
	size_t minimum_limit = 500000;
	size_t ram_buffer_size = 100000;
};

// ***********
// File size formatter
// ***********
inline std::string format_file_size(uintmax_t bytes) {
	std::ostringstream oss;

	if (bytes < 1024) {
		oss << bytes << " B";
	} else if (bytes < 1024 * 1024) {
		oss << std::fixed << std::setprecision(1) << (bytes / 1024.0) << " KB";
	} else if (bytes < 1024ULL * 1024ULL * 1024ULL) {
		oss << std::fixed << std::setprecision(1) << (bytes / (1024.0 * 1024.0)) << " MB";
	} else {
		oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0 * 1024.0))
		    << " GB";
	}

	return oss.str();
}

class Parser {
private:
	// Binary file written by embedding(); decided by the caller.
	std::filesystem::path embedded_path;

	// Folders searched for relative media file names; decided by the caller.
	std::vector<std::filesystem::path> asset_dirs;

	// Files processed in the last bulk tokenization
	std::vector<std::filesystem::path> last_bulk_files = {};

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
			if constexpr (std::is_same_v<T, std::string>) {
				input.reverse_text_dictionary[next_id] = data;
			}
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
		if (!full_path.parent_path().empty()) {
			std::error_code ec;
			std::filesystem::create_directories(full_path.parent_path(), ec);
		}

		// Use full_path for the file stream
		std::ofstream out_file(full_path, std::ios::binary | std::ios::trunc);

		if (!out_file) {
			std::cerr << "Warning: Failed to open " << full_path << " for writing.\n";
			return;
		}
		// Write the entire vector to disk in one lightning-fast operation
		out_file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
		out_file.close();
		std::cout << "Saved " << data.size() << " items to " << full_path << "\n"
			  << std::flush;
	}

	// ********************
	// Read from disk
	// ********************
	void tokenize_from_disk(const std::string& fileName,
				std::unordered_map<uint8_t, uint32_t>& dictionary,
				uint32_t& next_id,
				Input& input) {

		std::string cleaned_name = fileName;
		if (cleaned_name.size() >= 2 && cleaned_name.front() == '"' &&
		    cleaned_name.back() == '"') {
			cleaned_name = cleaned_name.substr(1, cleaned_name.size() - 2);
		}

		std::filesystem::path c_standard_fileName(cleaned_name);

		if (!c_standard_fileName.is_absolute()) {
			for (const auto& dir : asset_dirs) {
				const std::filesystem::path each_path = dir / cleaned_name;
				if (std::filesystem::exists(each_path)) {
					c_standard_fileName = each_path;
					std::cout << "Found in\n" << c_standard_fileName << "\n";
					break;
				}
			}
		}

		auto tokenize_single_file = [&](const std::filesystem::path& path) {
			std::error_code ec;
			uintmax_t fsize = std::filesystem::file_size(path, ec);
			if (ec)
				fsize = 0;
			std::ifstream parsed_data(path, std::ios::binary);
			if (!parsed_data) {
				std::cerr << "Failed to open " << path << "\n";
				return false;
			}
			char buf[65536];
			uintmax_t bytes_read = 0;
			uintmax_t last_progress_bytes = 0;
			const bool show_progress = (fsize >= 5 * 1024 * 1024);
			const size_t initial_tokens = input.tokenized_data.size();

			while (parsed_data.read(buf, sizeof(buf)) || parsed_data.gcount() > 0) {
				std::streamsize n = parsed_data.gcount();
				for (std::streamsize i = 0; i < n; ++i) {
					add_assign_increase(static_cast<uint8_t>(buf[i]),
							    dictionary,
							    next_id,
							    input);
				}
				bytes_read += static_cast<uintmax_t>(n);
				if (show_progress &&
				    (bytes_read - last_progress_bytes >= 2 * 1024 * 1024 ||
				     bytes_read >= fsize)) {
					last_progress_bytes = bytes_read;
					const double pct =
						(fsize > 0) ? (static_cast<double>(bytes_read) /
							       static_cast<double>(fsize)) *
								      100.0
							    : 100.0;
					constexpr int bar_w = 24;
					const int filled = static_cast<int>((pct / 100.0) * bar_w);
					std::string bar = "[";
					for (int b = 0; b < bar_w; ++b) {
						if (b < filled) {
							bar += '=';
						} else if (b == filled && filled < bar_w) {
							bar += '>';
						} else {
							bar += ' ';
						}
					}
					bar += "]";
					std::cout << "\r  [..] Ingesting "
						  << path.filename().string() << " " << bar << " "
						  << std::fixed << std::setprecision(1) << pct
						  << "%" << std::flush;
				}
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			if (show_progress) {
				std::cout
					<< "\r                                                                                \r";
			}
			std::cout << "  [OK] Ingested " << path.filename().string() << " ("
				  << format_file_size(fsize) << ") -> " << added
				  << " tokens mapped\n"
				  << std::flush;
			return true;
		};

		std::error_code ec;
		if (std::filesystem::is_directory(c_standard_fileName, ec)) {
			bool found_any = false;
			for (const auto& entry :
			     std::filesystem::directory_iterator(c_standard_fileName, ec)) {
				if (entry.is_regular_file(ec)) {
					if (tokenize_single_file(entry.path())) {
						found_any = true;
					}
				}
			}
			if (!found_any) {
				std::cerr << "Failed to open or find files in "
					  << c_standard_fileName << "\n";
			}
			return;
		}

		if (!std::filesystem::exists(c_standard_fileName, ec)) {
			std::cerr << "Failed to open " << c_standard_fileName << "\n";
			return;
		}

		tokenize_single_file(c_standard_fileName);
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

	// Bulk directory parser: scans and tokenizes all files in a folder (Text, PDF, Images, Video, Audio, 3D Models)
	size_t bulk_tokenize_directory(const std::filesystem::path& dir_path,
				       Input& input,
				       size_t max_files = 0);

	// Returns the list of files processed in the most recent bulk tokenization run
	const std::vector<std::filesystem::path>& get_last_bulk_files() const {
		return last_bulk_files;
	}

	// Standalone parser, isolated from AI!
	std::vector<std::bfloat16_t> embedding(Input& input, Test& test);

	// Decoder: finds the nearest token ID in the embedding space for a continuous prediction vector
	uint32_t decode_embedding_to_token(const Input& input,
					   const std::vector<float>& embedding_vector);

	// Formats a discrete token ID back into its human-readable representation
	std::string token_to_string(const Input& input, uint32_t token_id);

	// Decodes continuous embedding prediction directly to human-readable string
	std::string decode_prediction(const Input& input,
				      const std::vector<float>& embedding_vector);

	// Ensures the embedding lookup table is populated for all known tokens
	void build_embedding_table(Input& input);

	// Tokenizes a text prompt string into discrete token IDs, registering any unseen words
	std::vector<uint32_t> tokenize_prompt(Input& input, const std::string& prompt);

	// Converts a sequence of token IDs into a 128-token context window vector (size = context_tokens * embedding_dim)
	std::vector<std::bfloat16_t> tokens_to_context_window(const Input& input,
							      const std::vector<uint32_t>& tokens,
							      size_t context_tokens = 128);

	// Vocabulary serialization & persistence (Flat binary AXVK format)
	bool save_vocabulary(const std::filesystem::path& filepath, const Input& input);
	bool load_vocabulary(const std::filesystem::path& filepath, Input& input);
};

#endif