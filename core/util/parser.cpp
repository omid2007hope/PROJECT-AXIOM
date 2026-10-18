#include "../include/parser.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../include/Identifier.hpp"
#include "../include/decompressor/decom_audio.hpp"
#include "../include/decompressor/decom_doc.hpp"
#include "../include/decompressor/decom_image.hpp"
#include "../include/decompressor/decom_video.hpp"


// Data Streaming (or Batching).

/*
Turn raw data samples into the parsed input vector;
Now isolated from the neural network entirely.
*/

// *************
// Tokenization phase
// *************

// Cleans raw token strings: lowercases ASCII and strips leading/trailing punctuation
static std::string sanitize_word(const std::string& raw) {
	if (raw.empty()) return "";
	if (raw.size() >= 3 && raw.front() == '<' && raw.back() == '>') {
		return raw;
	}
	size_t start = 0;
	while (start < raw.size() && (std::ispunct(static_cast<unsigned char>(raw[start])))) {
		++start;
	}
	size_t end = raw.size();
	while (end > start && (std::ispunct(static_cast<unsigned char>(raw[end - 1])))) {
		--end;
	}
	if (start >= end) return "";
	std::string clean = raw.substr(start, end - start);
	for (char& c : clean) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	return clean;
}

// MOCK DISCRETE TOKENIZER (Gemini / Chameleon Early Fusion style)
std::vector<uint32_t> Parser::tokenization(Input& input, Test& test) {

	// *************
	// Developer mode : on
	// Data comes from test structure
	// *************

	if (developer_mode == true) {
		// 1. Text Tokenization (IDs 1 - 9999)
		for (const std::string& word : test.test_input) {
			std::string clean = sanitize_word(word);
			if (!clean.empty()) {
				add_assign_increase(clean, input.main_dictionary, input.next_text_id, input);
			}
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

		// *************
		// Developer mode : off
		// Data comes from input structure
		// *************

	} else {
		// Production Mode: Dynamically Tokenize User's UI Data
		for (const std::string& word : input.user_input) {
			std::string clean = sanitize_word(word);
			if (!clean.empty()) {
				add_assign_increase(clean, input.main_dictionary, input.next_text_id, input);
			}
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

// *************
// Embedding phase
// *************

// EMBEDDINGS: Convert global token IDs to a flat dense vector
// Streams directly to disk in fixed-size chunks to eliminate RAM saturation
// and SSD page thrashing on massive datasets (hundreds of millions of tokens).
std::vector<std::bfloat16_t> Parser::embedding(Input& input, Test& test) {
	input.chunk_pos = 0;
	input.tokens_written = 0;
	input.last_pct = 0;

	build_embedding_table(input);

	if (!embedded_path.parent_path().empty()) {
		std::error_code ec;
		std::filesystem::create_directories(embedded_path.parent_path(), ec);
	}

	std::ofstream out(embedded_path, std::ios::binary | std::ios::trunc);

	if (!out) {
		std::cerr << "[!] Fatal: Failed to open " << embedded_path
			  << " for writing embeddings.\n";

		return {};
	}

	// If we are testing which means developer mode is on
	// if token size is less than equal to ram's buffer size
	const bool keep_in_ram =
		developer_mode == true || (input.tokenized_data.size() <= input.ram_buffer_size);

	if (keep_in_ram == true) {
		input.embedded_data.resize(input.tokenized_data.size() * input.embedding_dim);
	} else {
		input.embedded_data.clear();
		input.embedded_data.shrink_to_fit();
	}

	std::vector<std::bfloat16_t> chunk(input.CHUNK_TOKENS * input.embedding_dim);
	std::vector<std::bfloat16_t> current_vec(input.embedding_dim);

	const bool show_progress = (input.tokenized_data.size() >= input.minimum_limit);

	for (size_t i = 0; i < input.tokenized_data.size(); ++i) {

		// each single token
		const uint32_t id = input.tokenized_data[i];

		// Insure it creates a new unique ID each time
		if (id != input.last_token) {
			input.last_token = id;
			auto it = input.embedding_table.find(id);
			if (it != input.embedding_table.end()) {
				current_vec = it->second;
			} else {
				float bound = 1.0f / std::sqrt(static_cast<float>(input.embedding_dim > 0 ? input.embedding_dim : 4));
				for (size_t d = 0; d < input.embedding_dim; ++d) {
					uint32_t h = id * 0x45d9f3b + static_cast<uint32_t>(d + 1) * 0x119de1f3;
					h = ((h >> 16) ^ h) * 0x45d9f3b;
					h = ((h >> 16) ^ h);
					float norm = static_cast<float>(h) / static_cast<float>(0xFFFFFFFFU);
					current_vec[d] = static_cast<std::bfloat16_t>(-bound + 2.0f * bound * norm);
				}
			}
		}

		if (keep_in_ram == true) {
			for (size_t d = 0; d < input.embedding_dim; ++d) {

				input.embedded_data[i * input.embedding_dim + d] = current_vec[d];
			}
		}

		for (size_t d = 0; d < input.embedding_dim; ++d) {
			chunk[input.chunk_pos++] = current_vec[d];
		}

		if (input.chunk_pos == chunk.size()) {
			out.write(reinterpret_cast<const char*>(chunk.data()),
				  chunk.size() * sizeof(std::bfloat16_t));
			input.tokens_written += input.CHUNK_TOKENS;
			input.chunk_pos = 0;

			if (show_progress) {
				const size_t pct =
					(input.tokens_written * 100) / input.tokenized_data.size();
				if (pct != input.last_pct) {
					input.last_pct = pct;
					constexpr int bar_w = 24;
					const int filled = static_cast<int>((pct / 100.0) * bar_w);
					std::string bar = "[";
					for (int b = 0; b < bar_w; ++b) {
						if (b < filled)
							bar += '=';
						else if (b == filled)
							bar += '>';
						else
							bar += ' ';
					}
					bar += "]";
					std::cout << "\r  [..] Synthesizing continuous embeddings "
						  << bar << " " << pct << "% ("
						  << input.tokens_written << "/"
						  << input.tokenized_data.size() << " tokens)"
						  << std::flush;
				}
			}
		}
	}

	// std::ofstream out(embedded_path, std::ios::binary | std::ios::trunc)
	if (input.chunk_pos > 0) {
		out.write(reinterpret_cast<const char*>(chunk.data()),
			  input.chunk_pos * sizeof(std::bfloat16_t));
		input.tokens_written += (input.chunk_pos / input.embedding_dim);
	}
	out.close();

	if (show_progress) {
		const uintmax_t bytes =
			input.tokenized_data.size() * input.embedding_dim * sizeof(std::bfloat16_t);
		std::cout
			<< "\r                                                                                \r"
			<< "  [OK] Continuous embeddings synthesized: "
			<< input.tokenized_data.size() << " tokens (" << format_file_size(bytes)
			<< ") streamed to disk.\n"
			<< std::flush;
	}

	return input.embedded_data;
}

void Parser::build_embedding_table(Input& input) {
	const size_t total_unique = input.main_dictionary.size() +
				    input.image_dictionary.size() + input.video_dictionary.size() +
				    input.voice_dictionary.size();
	if (total_unique > 0) {
		input.embedding_table.reserve(total_unique);
	}

	float bound = 1.0f / std::sqrt(static_cast<float>(input.embedding_dim > 0 ? input.embedding_dim : 4));
	auto ensure_embedding = [&](uint32_t id) {
		if (input.embedding_table.find(id) == input.embedding_table.end()) {
			std::vector<std::bfloat16_t> vec(input.embedding_dim);
			for (uint32_t d = 0; d < input.embedding_dim; ++d) {
				uint32_t h = id * 0x45d9f3b + static_cast<uint32_t>(d + 1) * 0x119de1f3;
				h = ((h >> 16) ^ h) * 0x45d9f3b;
				h = ((h >> 16) ^ h);
				float norm = static_cast<float>(h) / static_cast<float>(0xFFFFFFFFU);
				vec[d] = static_cast<std::bfloat16_t>(-bound + 2.0f * bound * norm);
			}
			input.embedding_table[id] = std::move(vec);
		}
	};

	for (const auto& [_, id] : input.main_dictionary)
		ensure_embedding(id);
	for (const auto& [_, id] : input.image_dictionary)
		ensure_embedding(id);
	for (const auto& [_, id] : input.video_dictionary)
		ensure_embedding(id);
	for (const auto& [_, id] : input.voice_dictionary)
		ensure_embedding(id);
}

uint32_t Parser::decode_embedding_to_token(const Input& input,
					   const std::vector<float>& embedding_vector) {
	if (input.embedding_table.empty()) {
		return 0;
	}

	// Rebuild the contiguous mirror whenever the table has grown or shrunk
	if (input.decode_ids.size() != input.embedding_table.size()) {
		input.decode_ids.clear();
		input.decode_vectors.clear();
		input.decode_ids.reserve(input.embedding_table.size());
		input.decode_vectors.reserve(input.embedding_table.size() * input.embedding_dim);

		for (const auto& [id, vec] : input.embedding_table) {
			input.decode_ids.push_back(id);
			for (size_t d = 0; d < input.embedding_dim; ++d) {
				input.decode_vectors.push_back(
					d < vec.size() ? static_cast<float>(vec[d]) : 0.0f);
			}
		}
	}

	const size_t compared_dims = std::min(input.embedding_dim, embedding_vector.size());

	for (size_t d = 0; d < compared_dims; ++d) {
		if (!std::isfinite(embedding_vector[d])) {
			return 0;
		}
	}

	const float* row = input.decode_vectors.data();

	uint32_t best_id = 0;
	float min_dist_sq = std::numeric_limits<float>::max();

	for (size_t i = 0; i < input.decode_ids.size(); ++i, row += input.embedding_dim) {
		float dist_sq = 0.0f;
		for (size_t d = 0; d < compared_dims; ++d) {
			const float diff = embedding_vector[d] - row[d];
			dist_sq += diff * diff;
			if (dist_sq >= min_dist_sq) {
				break;
			}
		}
		if (dist_sq < min_dist_sq) {
			min_dist_sq = dist_sq;
			best_id = input.decode_ids[i];
		}
	}

	return best_id;
}

std::string Parser::token_to_string(const Input& input, uint32_t token_id) {
	if (token_id == vocab::PAD) {
		return "<PAD>";
	}
	if (token_id == vocab::UNK) {
		return "<UNK>";
	}
	if (token_id == vocab::BOS) {
		return "<BOS>";
	}
	if (token_id == vocab::EOS) {
		return "<EOS>";
	}

	// 1. Text & Number Modality (Main Dictionary: 4 to 64,511)
	if (token_id >= vocab::TEXT_START_ID && token_id <= vocab::TEXT_MAX_ID) {
		auto it = input.reverse_main_dictionary.find(token_id);
		if (it != input.reverse_main_dictionary.end()) {
			return it->second;
		}
		for (const auto& [word, id] : input.main_dictionary) {
			if (id == token_id)
				return word;
		}
		return "[Token #" + std::to_string(token_id) + "]";
	}

	// 2. Visual Modality (Images & Video frames: 64,512 to 64,767)
	if (token_id >= vocab::VISUAL_START_ID && token_id <= vocab::VISUAL_END_ID) {
		uint8_t byte_val = static_cast<uint8_t>(token_id - vocab::VISUAL_START_ID);
		for (const auto& [b, id] : input.video_dictionary) {
			if (id == token_id) {
				return "[Video byte: " +
				       std::to_string(static_cast<int>(byte_val)) + "]";
			}
		}
		return "[Image byte: " +
		       std::to_string(static_cast<int>(byte_val)) + "]";
	}

	// 3. Acoustic Modality (Voice & Audio waveforms: 64,768 to 65,023)
	if (token_id >= vocab::ACOUSTIC_START_ID && token_id <= vocab::ACOUSTIC_END_ID) {
		uint8_t byte_val = static_cast<uint8_t>(token_id - vocab::ACOUSTIC_START_ID);
		return "[Voice byte: " +
		       std::to_string(static_cast<int>(byte_val)) + "]";
	}

	// 4. Reserved Expansion Slots (65,024 to 65,535)
	if (token_id >= vocab::EXPANSION_START_ID && token_id <= vocab::EXPANSION_MAX_ID) {
		return "[Expansion #" + std::to_string(token_id) + "]";
	}

	return "[Unknown #" + std::to_string(token_id) + "]";
}

std::string Parser::decode_prediction(const Input& input,
				      const std::vector<float>& embedding_vector) {
	uint32_t token_id = decode_embedding_to_token(input, embedding_vector);
	if (token_id == 0)
		return "[Empty Vocabulary]";
	return token_to_string(input, token_id);
}

std::vector<uint32_t> Parser::tokenize_prompt(Input& input, const std::string& prompt) {
	std::vector<uint32_t> tokens;
	std::istringstream iss(prompt);
	std::string word;
	while (iss >> word) {
		std::string clean = sanitize_word(word);
		if (clean.empty()) continue;

		auto it = input.main_dictionary.find(clean);
		if (it == input.main_dictionary.end()) {
			it = input.main_dictionary.find(word);
		}

		if (it == input.main_dictionary.end()) {
			if (input.next_text_id <= vocab::TEXT_MAX_ID) {
				uint32_t new_id = input.next_text_id++;
				input.main_dictionary[clean] = new_id;
				input.reverse_main_dictionary[new_id] = clean;
				tokens.push_back(new_id);
			} else {
				input.main_dictionary[clean] = vocab::UNK;
				tokens.push_back(vocab::UNK);
			}
		} else {
			tokens.push_back(it->second);
		}
	}
	build_embedding_table(input);
	return tokens;
}

std::vector<std::bfloat16_t> Parser::tokens_to_context_window(const Input& input,
							      const std::vector<uint32_t>& tokens,
							      size_t context_tokens) {
	std::vector<std::bfloat16_t> window(context_tokens * input.embedding_dim,
					    static_cast<std::bfloat16_t>(0.0f));
	if (tokens.empty())
		return window;

	size_t count = std::min(tokens.size(), context_tokens);
	size_t start_token_idx = context_tokens - count;

	for (size_t i = 0; i < count; ++i) {
		uint32_t tok_id = tokens[tokens.size() - count + i];
		auto it = input.embedding_table.find(tok_id);
		if (it != input.embedding_table.end()) {
			for (size_t d = 0; d < input.embedding_dim; ++d) {
				window[(start_token_idx + i) * input.embedding_dim + d] =
					it->second[d];
			}
		}
	}
	return window;
}

static void print_progress_bar(const std::string& filename,
			       const std::string& action,
			       uintmax_t current_bytes,
			       uintmax_t total_bytes) {
	const double pct = (total_bytes > 0) ? std::min(100.0,
							(static_cast<double>(current_bytes) /
							 static_cast<double>(total_bytes)) *
								100.0)
					     : 100.0;
	constexpr int bar_width = 24;
	const int filled = static_cast<int>((pct / 100.0) * bar_width);

	std::string bar = "[";
	for (int i = 0; i < bar_width; ++i) {
		if (i < filled) {
			bar += '=';
		} else if (i == filled && filled < bar_width) {
			bar += '>';
		} else {
			bar += ' ';
		}
	}
	bar += "]";

	std::cout << "\r  [..] " << action << " " << filename << " " << bar << " " << std::fixed
		  << std::setprecision(1) << pct << "% (" << format_file_size(current_bytes)
		  << " / " << format_file_size(total_bytes) << ")" << std::flush;
}

size_t Parser::bulk_tokenize_directory(const std::filesystem::path& dir_path,
				       Input& input,
				       size_t max_files) {
	last_bulk_files.clear();

	std::error_code ec;
	if (!std::filesystem::exists(dir_path, ec) ||
	    !std::filesystem::is_directory(dir_path, ec)) {
		std::cerr << "[!] Invalid or non-existent asset directory: " << dir_path << "\n";
		return 0;
	}

	std::vector<std::filesystem::path> files;
	for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path, ec)) {
		if (entry.is_regular_file(ec)) {
			std::filesystem::path rel = entry.path().lexically_relative(dir_path);
			bool in_archive = false;
			for (const auto& part : rel) {
				if (part == "archive") {
					in_archive = true;
					break;
				}
			}
			if (in_archive) {
				continue;
			}
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
	std::cout << "\n[+] Discovered " << files.size() << " asset file(s) in "
		  << dir_path.string() << ":\n";

	size_t files_processed = 0;
	const size_t total_files =
		(max_files > 0 && max_files < files.size()) ? max_files : files.size();

	auto report_file_done = [&](const std::filesystem::path& fpath,
				    uintmax_t fsize,
				    const std::string& modality_tag,
				    size_t added_tokens) {
		++files_processed;
		last_bulk_files.push_back(fpath);

		if (developer_mode) {
			std::cout << "  [OK] " << fpath.filename().string() << " ("
				  << format_file_size(fsize) << ") -> " << modality_tag << ": "
				  << added_tokens << " tokens mapped\n";
		} else {
			const double pct =
				(total_files > 0) ? std::min(100.0,
							     (static_cast<double>(files_processed) *
							      100.0 / total_files))
						  : 100.0;
			constexpr int bar_w = 24;
			int filled = static_cast<int>((pct / 100.0) * bar_w);
			if (filled > bar_w)
				filled = bar_w;
			std::string bar = "[";
			for (int b = 0; b < bar_w; ++b) {
				if (b < filled)
					bar += '=';
				else if (b == filled && filled < bar_w)
					bar += '>';
				else
					bar += ' ';
			}
			bar += "]";

			std::string fname = fpath.filename().string();
			if (fname.size() > 20)
				fname = fname.substr(0, 17) + "...";

			std::cout << "\r  [..] Ingesting assets " << bar << " " << std::fixed
				  << std::setprecision(1) << pct << "% (" << files_processed << "/"
				  << total_files << " files | " << input.tokenized_data.size()
				  << " tokens | " << fname << ")   " << std::flush;
		}
	};

	for (const auto& fpath : files) {
		if (max_files > 0 && files_processed >= max_files) {
			break;
		}
		IdentifiedFile id_file = Identifier::identify(fpath);
		std::string ext = fpath.extension().string();
		for (char& c : ext)
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

		uintmax_t fsize = std::filesystem::file_size(fpath, ec);
		if (ec)
			fsize = 0;

		const size_t initial_tokens = input.tokenized_data.size();

		bool is_text_ext =
			(ext == ".txt" || ext == ".text" || ext == ".md" || ext == ".markdown" ||
			 ext == ".json" || ext == ".csv" || ext == ".log" || ext == ".c" ||
			 ext == ".cpp" || ext == ".h" || ext == ".hpp" || ext == ".cu" ||
			 ext == ".cuh" || ext == ".py" || ext == ".html" || ext == ".htm" ||
			 ext == ".xml" || ext == ".yaml" || ext == ".yml" || ext == ".ini" ||
			 ext == ".cfg" || ext == ".conf" || ext == ".sql" || ext == ".rs" ||
			 ext == ".go" || ext == ".java" || ext == ".ts" || ext == ".js" ||
			 ext == ".sh" || ext == ".bat" || ext == ".ps1");

		bool is_doc = (id_file.modality == ModalityCategory::DOCUMENT || is_text_ext ||
			       ext == ".pdf" || ext == ".docx");

		bool is_image = (id_file.modality == ModalityCategory::IMAGE ||
				 ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
				 ext == ".webp" || ext == ".gif" || ext == ".tiff" || ext == ".tif");

		bool is_video = (id_file.modality == ModalityCategory::VIDEO ||
				 ext == ".mp4" || ext == ".mkv" || ext == ".avi" || ext == ".mov" ||
				 ext == ".webm" || ext == ".wmv");

		bool is_audio = (id_file.modality == ModalityCategory::AUDIO ||
				 ext == ".ogg" || ext == ".wav" || ext == ".mp3" || ext == ".flac" ||
				 ext == ".aac" || ext == ".m4a" || ext == ".wma" || ext == ".opus");

		if (is_doc) {
			// Phase 1.2: Sanitize Document Ingestion & Eliminate PDF Byte Streaming
			// Extract clean text from PDF, DOCX, TXT, MD so linguistic terms enter main_dictionary
			std::string extracted_text = Decom_doc::extract_text(fpath);
			std::istringstream iss(extracted_text);
			std::string word;
			while (iss >> word) {
				std::string clean = sanitize_word(word);
				if (!clean.empty()) {
					add_assign_increase(clean, input.main_dictionary, input.next_text_id, input);
				}
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			std::string tag = (ext == ".pdf") ? "PDF Document" : ((ext == ".docx") ? "DOCX Document" : "Text");
			report_file_done(fpath, fsize, tag, added);
		} else if (is_image) {
			// Phase 1.3: Genuine visual media to discrete visual byte tokenizer
			std::ifstream in(fpath, std::ios::binary);
			if (!in) {
				std::cerr << "  [!] Failed to open " << fpath.filename().string()
					  << "\n";
				continue;
			}
			char buf[65536];
			uintmax_t bytes_read = 0;
			uintmax_t last_progress_bytes = 0;
			const bool show_progress = (fsize >= 5 * 1024 * 1024);

			while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
				std::streamsize n = in.gcount();
				for (std::streamsize i = 0; i < n; ++i) {
					add_assign_increase(static_cast<uint8_t>(buf[i]),
							    input.image_dictionary,
							    input.next_image_id,
							    input);
				}
				bytes_read += static_cast<uintmax_t>(n);
				if (show_progress &&
				    (bytes_read - last_progress_bytes >= 2 * 1024 * 1024 ||
				     bytes_read >= fsize)) {
					last_progress_bytes = bytes_read;
					print_progress_bar(fpath.filename().string(),
							   "Ingesting Image",
							   bytes_read,
							   fsize);
				}
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			if (show_progress) {
				std::cout
					<< "\r                                                                                \r";
			}
			report_file_done(fpath, fsize, "Image", added);
		} else if (is_video) {
			// Video visual frames to discrete visual byte tokenizer
			std::ifstream in(fpath, std::ios::binary);
			if (!in) {
				std::cerr << "  [!] Failed to open " << fpath.filename().string()
					  << "\n";
				continue;
			}
			char buf[65536];
			uintmax_t bytes_read = 0;
			uintmax_t last_progress_bytes = 0;
			const bool show_progress = (fsize >= 5 * 1024 * 1024);

			while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
				std::streamsize n = in.gcount();
				for (std::streamsize i = 0; i < n; ++i) {
					add_assign_increase(static_cast<uint8_t>(buf[i]),
							    input.video_dictionary,
							    input.next_video_id,
							    input);
				}
				bytes_read += static_cast<uintmax_t>(n);
				if (show_progress &&
				    (bytes_read - last_progress_bytes >= 2 * 1024 * 1024 ||
				     bytes_read >= fsize)) {
					last_progress_bytes = bytes_read;
					print_progress_bar(fpath.filename().string(),
							   "Tokenizing Video",
							   bytes_read,
							   fsize);
				}
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			if (show_progress) {
				std::cout
					<< "\r                                                                                \r";
			}
			report_file_done(fpath, fsize, "Video", added);
		} else if (is_audio) {
			// Genuine audio waveforms to discrete acoustic byte tokenizer
			std::ifstream in(fpath, std::ios::binary);
			if (!in) {
				std::cerr << "  [!] Failed to open " << fpath.filename().string()
					  << "\n";
				continue;
			}
			char buf[65536];
			uintmax_t bytes_read = 0;
			uintmax_t last_progress_bytes = 0;
			const bool show_progress = (fsize >= 5 * 1024 * 1024);

			while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
				std::streamsize n = in.gcount();
				for (std::streamsize i = 0; i < n; ++i) {
					add_assign_increase(static_cast<uint8_t>(buf[i]),
							    input.voice_dictionary,
							    input.next_voice_id,
							    input);
				}
				bytes_read += static_cast<uintmax_t>(n);
				if (show_progress &&
				    (bytes_read - last_progress_bytes >= 2 * 1024 * 1024 ||
				     bytes_read >= fsize)) {
					last_progress_bytes = bytes_read;
					print_progress_bar(fpath.filename().string(),
							   "Ingesting Audio",
							   bytes_read,
							   fsize);
				}
			}
			const size_t added = input.tokenized_data.size() - initial_tokens;
			if (show_progress) {
				std::cout
					<< "\r                                                                                \r";
			}
			report_file_done(fpath, fsize, "Voice/Audio", added);
		} else {
			// Phase 1.3: Non-media binary assets are NOT streamed into image_dictionary
			report_file_done(fpath, fsize, "Binary Asset (Skipped Raw Streaming)", 0);
		}
	}

	if (!developer_mode && files_processed > 0) {
		std::cout << "\r  [OK] Ingesting assets [========================] 100.0% ("
			  << files_processed << "/" << total_files << " files | "
			  << input.tokenized_data.size() << " tokens)            \n"
			  << std::flush;
	}

	return files_processed;
}

bool Parser::save_vocabulary(const std::filesystem::path& filepath, const Input& input) {
	if (input.main_dictionary.empty()) {
		if (std::filesystem::exists(filepath)) {
			std::cout << "[i] Preserving existing vocabulary at "
				  << filepath.filename().string()
				  << " (no text tokens in current dataset).\n";
			return true;
		}
		std::cout << "[i] Skipping vocabulary serialization: text dictionary is empty.\n";
		return true;
	}

	// 1. Write to .tmp to prevent corruption during an aborted save
	std::filesystem::path tmp_filepath = filepath.string() + ".tmp";
	if (!tmp_filepath.parent_path().empty()) {
		std::error_code ec;
		std::filesystem::create_directories(tmp_filepath.parent_path(), ec);
	}
	std::ofstream out(tmp_filepath, std::ios::binary);
	if (!out) {
		std::cerr << "[!] Failed to open temporary vocabulary file for writing: "
			  << tmp_filepath.string() << "\n";
		return false;
	}

	// 2. Write Magic Header (Axiom Vocab Key: 'A', 'X', 'V', 'K')
	const char magic[4] = {'A', 'X', 'V', 'K'};
	out.write(magic, 4);

	// 3. Write Dictionary Size
	uint64_t vocab_size = static_cast<uint64_t>(input.main_dictionary.size());
	out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(uint64_t));

	// 4. Stream Key-Value Pairs [Length] [Bytes] [ID]
	for (const auto& [word, id] : input.main_dictionary) {
		uint32_t word_len = static_cast<uint32_t>(word.size());
		out.write(reinterpret_cast<const char*>(&word_len), sizeof(uint32_t));
		out.write(word.data(), word_len);
		out.write(reinterpret_cast<const char*>(&id), sizeof(uint32_t));
	}

	out.close();
	if (!out) {
		std::cerr << "[!] Write failure during vocabulary serialization.\n";
		return false;
	}

	// 5. Atomic Rename (.tmp -> target)
	std::error_code ec;
	std::filesystem::rename(tmp_filepath, filepath, ec);
	if (ec) {
		ec.clear();
		std::filesystem::copy_file(tmp_filepath,
					   filepath,
					   std::filesystem::copy_options::overwrite_existing,
					   ec);
		std::filesystem::remove(tmp_filepath, ec);
	}

	std::cout << "[OK] Serialized " << vocab_size << " text tokens to "
		  << filepath.filename().string() << "\n"
		  << std::flush;
	return true;
}

bool Parser::load_vocabulary(const std::filesystem::path& filepath, Input& input) {
	std::ifstream in(filepath, std::ios::binary);
	if (!in) {
		std::cerr << "[!] Failed to open vocabulary file: " << filepath.string() << "\n";
		return false;
	}

	// 1. Validate Magic Header
	char magic[4] = {};
	in.read(magic, 4);
	if (in.gcount() != 4 || magic[0] != 'A' || magic[1] != 'X' || magic[2] != 'V' ||
	    magic[3] != 'K') {
		std::cerr << "[!] Fatal: Invalid magic bytes in vocab file: "
			  << filepath.filename().string() << " (expected 'AXVK')\n";
		return false;
	}

	// 2. Read Size and Allocate
	uint64_t vocab_size = 0;
	in.read(reinterpret_cast<char*>(&vocab_size), sizeof(uint64_t));
	if (!in) {
		std::cerr << "[!] Fatal: Corrupt vocabulary size header in "
			  << filepath.filename().string() << "\n";
		return false;
	}

	if (vocab_size == 0) {
		std::cout << "[i] Vocabulary file " << filepath.filename().string()
			  << " contains 0 text tokens; preserving initialized vocabulary.\n";
		in.close();
		return true;
	}

	input.main_dictionary.clear();
	input.reverse_main_dictionary.clear();
	input.main_dictionary.reserve(vocab_size);
	input.reverse_main_dictionary.reserve(vocab_size);

	// 3. Reconstruct Dictionaries
	for (uint64_t i = 0; i < vocab_size; ++i) {
		uint32_t word_len = 0;
		in.read(reinterpret_cast<char*>(&word_len), sizeof(uint32_t));
		if (!in) {
			std::cerr << "[!] Incomplete vocabulary entry length at index " << i
				  << "\n";
			return false;
		}

		std::string word(word_len, '\0');
		in.read(word.data(), word_len);
		if (!in) {
			std::cerr << "[!] Incomplete vocabulary word data at index " << i << "\n";
			return false;
		}

		uint32_t id = 0;
		in.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
		if (!in) {
			std::cerr << "[!] Incomplete vocabulary token ID at index " << i << "\n";
			return false;
		}

		input.main_dictionary[word] = id;
		input.reverse_main_dictionary[id] = word;
		if (id >= input.next_text_id) {
			input.next_text_id = id + 1;
		}
	}

	in.close();
	build_embedding_table(input);
	std::cout << "[OK] Loaded " << vocab_size << " vocabulary tokens from "
		  << filepath.filename().string() << "\n"
		  << std::flush;
	return true;
}
