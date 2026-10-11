#include "../include/parser.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>

// Data Streaming (or Batching).

/*
Turn raw data samples into the parsed input vector;
Now isolated from the neural network entirely.
*/

// MOCK DISCRETE TOKENIZER (Gemini / Chameleon Early Fusion style)
std::vector<uint32_t> Parser::tokenization(Input& input, Test& test) {

	if (developer_mode) {
		// 1. Text Tokenization (IDs 1 - 9999)
		for (const std::string& word : test.data_txt) {
			add_assign_increase(word, input.text_dictionary, input.next_text_id, input);
		}

		// 2. Number Tokenization (IDs 10000 - 19999)
		for (double val : test.data_num) {
			add_assign_increase(
				val, input.number_dictionary, input.next_number_id, input);
		}

		// 3. Image Tokenization (IDs 20000 - 29999)
		for (const std::string& file : test.data_image) {
			tokenize_from_disk(
				file, input.image_dictionary, input.next_image_id, input);
		}

		// 4. Video Tokenization (IDs 30000 - 39999)
		for (const std::string& file : test.data_video) {
			tokenize_from_disk(
				file, input.video_dictionary, input.next_video_id, input);
		}

		// 5. Voice/Audio Tokenization (IDs 40000 - 49999)
		for (const std::string& file : test.data_voice) {
			tokenize_from_disk(
				file, input.voice_dictionary, input.next_voice_id, input);
		}

	} else {
		// Production Mode: Dynamically Tokenize User's UI Data
		for (const std::string& word : input.user_text) {
			add_assign_increase(word, input.text_dictionary, input.next_text_id, input);
		}
		for (double val : input.user_numbers) {
			add_assign_increase(
				val, input.number_dictionary, input.next_number_id, input);
		}
		for (const std::string& path : input.user_image_paths) {
			tokenize_from_disk(
				path, input.image_dictionary, input.next_image_id, input);
		}
		for (const std::string& path : input.user_video_paths) {
			tokenize_from_disk(
				path, input.video_dictionary, input.next_video_id, input);
		}
		for (const std::string& path : input.user_voice_paths) {
			tokenize_from_disk(
				path, input.voice_dictionary, input.next_voice_id, input);
		}
	}

	return input.tokenized_data;
}

// EMBEDDINGS: Convert global token IDs to a flat dense vector
// Each unique token ID gets a fixed embedding vector (seeded by ID for
// stability). Layout: [emb(token_0)..., emb(token_1)..., ...] size =
// tokenized_data.size() * embedding_dim
std::vector<std::bfloat16_t> Parser::embedding(Input& input, Test& test) {
	input.embedded_data.clear();
	input.embedded_data.reserve(input.tokenized_data.size() * input.embedding_dim);

	for (uint32_t id : input.tokenized_data) {
		// Create entry on first encounter, seeded by the token ID for determinism
		if (input.embedding_table.find(id) == input.embedding_table.end()) {
			std::vector<std::bfloat16_t> vec(input.embedding_dim);
			// Simple deterministic init: sin/cos of (id * dim_index) — no random
			// needed
			for (uint32_t d = 0; d < input.embedding_dim; ++d) {
				vec[d] = static_cast<std::bfloat16_t>(
					std::sin(static_cast<double>(id) * (d + 1.0)));
			}
			input.embedding_table[id] = std::move(vec);
		}

		// Append the dense vector for this token to the RAM output
		const auto& vec = input.embedding_table[id];
		input.embedded_data.insert(input.embedded_data.end(), vec.begin(), vec.end());
	}

	// Use our new generalized save method
	save_data(input.embedded_data, embedded_path);

	return input.embedded_data;
}

void Parser::build_embedding_table(Input& input) {
	auto ensure_embedding = [&](uint32_t id) {
		if (input.embedding_table.find(id) == input.embedding_table.end()) {
			std::vector<std::bfloat16_t> vec(input.embedding_dim);
			for (uint32_t d = 0; d < input.embedding_dim; ++d) {
				vec[d] = static_cast<std::bfloat16_t>(
					std::sin(static_cast<double>(id) * (d + 1.0)));
			}
			input.embedding_table[id] = std::move(vec);
		}
	};

	for (const auto& [_, id] : input.text_dictionary) ensure_embedding(id);
	for (const auto& [_, id] : input.number_dictionary) ensure_embedding(id);
	for (const auto& [_, id] : input.image_dictionary) ensure_embedding(id);
	for (const auto& [_, id] : input.video_dictionary) ensure_embedding(id);
	for (const auto& [_, id] : input.voice_dictionary) ensure_embedding(id);
}

uint32_t Parser::decode_embedding_to_token(const Input& input,
					   const std::vector<float>& embedding_vector) {
	if (input.embedding_table.empty()) {
		return 0;
	}

	uint32_t best_id = 0;
	float min_dist_sq = std::numeric_limits<float>::max();

	for (const auto& [id, vec] : input.embedding_table) {
		float dist_sq = 0.0f;
		for (size_t d = 0; d < input.embedding_dim && d < embedding_vector.size() && d < vec.size(); ++d) {
			float diff = embedding_vector[d] - static_cast<float>(vec[d]);
			dist_sq += diff * diff;
		}
		if (dist_sq < min_dist_sq) {
			min_dist_sq = dist_sq;
			best_id = id;
		}
	}

	return best_id;
}

std::string Parser::token_to_string(const Input& input, uint32_t token_id) {
	// 1. Text Modality (1 to 9,999,999)
	if (token_id < 10000000) {
		for (const auto& [word, id] : input.text_dictionary) {
			if (id == token_id) return word;
		}
		return "[Text #" + std::to_string(token_id) + "]";
	}

	// 2. Number Modality (10,000,000 to 19,999,999)
	if (token_id < 20000000) {
		for (const auto& [num, id] : input.number_dictionary) {
			if (id == token_id) {
				std::ostringstream oss;
				oss << num;
				return oss.str();
			}
		}
		return "[Number #" + std::to_string(token_id) + "]";
	}

	// 3. Image Modality (20,000,000 to 29,999,999)
	if (token_id < 30000000) {
		for (const auto& [byte_val, id] : input.image_dictionary) {
			if (id == token_id) {
				return "[Image byte: " + std::to_string(static_cast<int>(byte_val)) + "]";
			}
		}
		return "[Image #" + std::to_string(token_id) + "]";
	}

	// 4. Video Modality (30,000,000 to 39,999,999)
	if (token_id < 40000000) {
		for (const auto& [byte_val, id] : input.video_dictionary) {
			if (id == token_id) {
				return "[Video byte: " + std::to_string(static_cast<int>(byte_val)) + "]";
			}
		}
		return "[Video #" + std::to_string(token_id) + "]";
	}

	// 5. Voice Modality (40,000,000 to 49,999,999)
	if (token_id < 50000000) {
		for (const auto& [byte_val, id] : input.voice_dictionary) {
			if (id == token_id) {
				return "[Voice byte: " + std::to_string(static_cast<int>(byte_val)) + "]";
			}
		}
		return "[Voice #" + std::to_string(token_id) + "]";
	}

	return "[Unknown #" + std::to_string(token_id) + "]";
}

std::string Parser::decode_prediction(const Input& input,
				      const std::vector<float>& embedding_vector) {
	uint32_t token_id = decode_embedding_to_token(input, embedding_vector);
	if (token_id == 0) return "[Empty Vocabulary]";
	return token_to_string(input, token_id);
}

std::vector<uint32_t> Parser::tokenize_prompt(Input& input, const std::string& prompt) {
	std::vector<uint32_t> tokens;
	std::istringstream iss(prompt);
	std::string word;
	while (iss >> word) {
		if (input.text_dictionary.find(word) == input.text_dictionary.end()) {
			input.text_dictionary[word] = input.next_text_id++;
		}
		tokens.push_back(input.text_dictionary[word]);
	}
	build_embedding_table(input);
	return tokens;
}

std::vector<std::bfloat16_t> Parser::tokens_to_context_window(const Input& input,
							       const std::vector<uint32_t>& tokens,
							       size_t context_tokens) {
	std::vector<std::bfloat16_t> window(context_tokens * input.embedding_dim,
					    static_cast<std::bfloat16_t>(0.0f));
	if (tokens.empty()) return window;

	size_t count = std::min(tokens.size(), context_tokens);
	size_t start_token_idx = context_tokens - count;

	for (size_t i = 0; i < count; ++i) {
		uint32_t tok_id = tokens[tokens.size() - count + i];
		auto it = input.embedding_table.find(tok_id);
		if (it != input.embedding_table.end()) {
			for (size_t d = 0; d < input.embedding_dim; ++d) {
				window[(start_token_idx + i) * input.embedding_dim + d] = it->second[d];
			}
		}
	}
	return window;
}

static std::string format_file_size(uintmax_t bytes) {
	std::ostringstream oss;
	if (bytes < 1024) {
		oss << bytes << " B";
	} else if (bytes < 1024 * 1024) {
		oss << std::fixed << std::setprecision(1) << (bytes / 1024.0) << " KB";
	} else {
		oss << std::fixed << std::setprecision(1) << (bytes / 1024.0 / 1024.0) << " MB";
	}
	return oss.str();
}

size_t Parser::bulk_tokenize_directory(const std::filesystem::path& dir_path, Input& input) {
	std::error_code ec;
	if (!std::filesystem::exists(dir_path, ec) || !std::filesystem::is_directory(dir_path, ec)) {
		std::cerr << "[!] Invalid or non-existent asset directory: " << dir_path << "\n";
		return 0;
	}

	std::vector<std::filesystem::path> files;
	for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path, ec)) {
		if (entry.is_regular_file(ec)) {
			std::string fname = entry.path().filename().string();
			if (!fname.empty() && fname.front() != '.') {
				files.push_back(entry.path());
			}
		}
	}

	if (files.empty()) {
		std::cout << "[!] No files found in " << dir_path << "\n";
		return 0;
	}

	std::sort(files.begin(), files.end());
	std::cout << "\n[+] Discovered " << files.size() << " asset file(s) in " << dir_path.string() << ":\n";

	size_t files_processed = 0;
	for (const auto& fpath : files) {
		std::string ext = fpath.extension().string();
		for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

		uintmax_t fsize = std::filesystem::file_size(fpath, ec);
		if (ec) fsize = 0;

		const size_t initial_tokens = input.tokenized_data.size();

		bool is_text_ext = (ext == ".txt" || ext == ".text" || ext == ".md" ||
				    ext == ".markdown" || ext == ".json" || ext == ".csv" ||
				    ext == ".log" || ext == ".c" || ext == ".cpp" ||
				    ext == ".h" || ext == ".hpp" || ext == ".cu" ||
				    ext == ".cuh" || ext == ".py" || ext == ".html" ||
				    ext == ".htm" || ext == ".xml" || ext == ".yaml" ||
				    ext == ".yml" || ext == ".ini" || ext == ".cfg" ||
				    ext == ".conf" || ext == ".sql" || ext == ".rs" ||
				    ext == ".go" || ext == ".java" || ext == ".ts" ||
				    ext == ".js" || ext == ".sh" || ext == ".bat" ||
				    ext == ".ps1");

		bool is_image_ext = (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
				     ext == ".bmp" || ext == ".webp" || ext == ".gif" ||
				     ext == ".tiff" || ext == ".tif" || ext == ".ico" ||
				     ext == ".ppm" || ext == ".pgm" || ext == ".svg");

		bool is_video_ext = (ext == ".mp4" || ext == ".mkv" || ext == ".avi" ||
				     ext == ".mov" || ext == ".webm" || ext == ".wmv" ||
				     ext == ".flv" || ext == ".m4v" || ext == ".mpg" ||
				     ext == ".mpeg" || ext == ".3gp");

		bool is_audio_ext = (ext == ".ogg" || ext == ".wav" || ext == ".mp3" ||
				     ext == ".flac" || ext == ".aac" || ext == ".m4a" ||
				     ext == ".wma" || ext == ".opus" || ext == ".aiff" ||
				     ext == ".mid" || ext == ".midi");

		bool is_3d_ext = (ext == ".obj" || ext == ".stl" || ext == ".gltf" ||
				  ext == ".glb" || ext == ".ply" || ext == ".3ds" ||
				  ext == ".fbx" || ext == ".dae" || ext == ".blend");

		bool is_binary = false;
		if (!is_text_ext && !is_image_ext && !is_video_ext && !is_audio_ext) {
			std::ifstream probe(fpath, std::ios::binary);
			char sample[1024];
			probe.read(sample, sizeof(sample));
			std::streamsize n_read = probe.gcount();
			for (std::streamsize i = 0; i < n_read; ++i) {
				if (sample[i] == '\0') {
					is_binary = true;
					break;
				}
			}
		}

		if (is_text_ext || (!is_image_ext && !is_video_ext && !is_audio_ext && !is_binary)) {
			std::ifstream in(fpath);
			if (!in) {
				std::cerr << "  [!] Failed to open " << fpath.filename().string() << "\n";
				continue;
			}
			std::string word;
			while (in >> word) {
				add_assign_increase(word, input.text_dictionary, input.next_text_id, input);
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			std::cout << "  [OK] " << fpath.filename().string()
				  << " (" << format_file_size(fsize) << ") -> Text: "
				  << added << " tokens mapped\n";
			++files_processed;
		} else if (is_image_ext) {
			std::ifstream in(fpath, std::ios::binary);
			if (!in) {
				std::cerr << "  [!] Failed to open " << fpath.filename().string() << "\n";
				continue;
			}
			char buf[65536];
			while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
				std::streamsize n = in.gcount();
				for (std::streamsize i = 0; i < n; ++i) {
					add_assign_increase(static_cast<uint8_t>(buf[i]), input.image_dictionary, input.next_image_id, input);
				}
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			std::cout << "  [OK] " << fpath.filename().string()
				  << " (" << format_file_size(fsize) << ") -> Image: "
				  << added << " tokens mapped\n";
			++files_processed;
		} else if (is_video_ext) {
			std::ifstream in(fpath, std::ios::binary);
			if (!in) {
				std::cerr << "  [!] Failed to open " << fpath.filename().string() << "\n";
				continue;
			}
			char buf[65536];
			while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
				std::streamsize n = in.gcount();
				for (std::streamsize i = 0; i < n; ++i) {
					add_assign_increase(static_cast<uint8_t>(buf[i]), input.video_dictionary, input.next_video_id, input);
				}
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			std::cout << "  [OK] " << fpath.filename().string()
				  << " (" << format_file_size(fsize) << ") -> Video: "
				  << added << " tokens mapped\n";
			++files_processed;
		} else if (is_audio_ext) {
			std::ifstream in(fpath, std::ios::binary);
			if (!in) {
				std::cerr << "  [!] Failed to open " << fpath.filename().string() << "\n";
				continue;
			}
			char buf[65536];
			while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
				std::streamsize n = in.gcount();
				for (std::streamsize i = 0; i < n; ++i) {
					add_assign_increase(static_cast<uint8_t>(buf[i]), input.voice_dictionary, input.next_voice_id, input);
				}
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			std::cout << "  [OK] " << fpath.filename().string()
				  << " (" << format_file_size(fsize) << ") -> Voice/Audio: "
				  << added << " tokens mapped\n";
			++files_processed;
		} else {
			// PDF, 3D model, or arbitrary binary document
			std::ifstream in(fpath, std::ios::binary);
			if (!in) {
				std::cerr << "  [!] Failed to open " << fpath.filename().string() << "\n";
				continue;
			}
			char buf[65536];
			while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
				std::streamsize n = in.gcount();
				for (std::streamsize i = 0; i < n; ++i) {
					add_assign_increase(static_cast<uint8_t>(buf[i]), input.image_dictionary, input.next_image_id, input);
				}
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			std::string tag = is_3d_ext ? "3D Model" : (ext == ".pdf" ? "PDF Document" : "Binary Asset");
			std::cout << "  [OK] " << fpath.filename().string()
				  << " (" << format_file_size(fsize) << ") -> " << tag << ": "
				  << added << " tokens mapped\n";
			++files_processed;
		}
	}

	return files_processed;
}
