#include "../include/parser.hpp"
#include "test_header.hpp"

bool test_tokenization() {
	std::cout << "\n=== Tokenization & Modality Test Suite ===\n";
	int s_pass = 0, s_fail = 0;

	Input input;
	Test test = make_test_data();
	Parser parser(true, test_embedded_path(), test_asset_dirs());

	std::vector<uint32_t> tokenized = parser.tokenization(input, test);
	test_check(!tokenized.empty(), "Tokenized stream is non-empty", s_pass, s_fail);
	test_check_eq(tokenized.size(), input.tokenized_data.size(), "Parser output matches input.tokenized_data", s_pass, s_fail);

	// Modality dictionaries must be populated
	test_check(!input.text_dictionary.empty(), "Text vocabulary populated", s_pass, s_fail);
	test_check(!input.number_dictionary.empty(), "Number vocabulary populated", s_pass, s_fail);
	test_check(!input.image_dictionary.empty(), "Image vocabulary populated", s_pass, s_fail);
	test_check(!input.video_dictionary.empty(), "Video vocabulary populated", s_pass, s_fail);
	test_check(!input.voice_dictionary.empty(), "Voice vocabulary populated", s_pass, s_fail);

	// Generate and verify embeddings
	std::vector<std::bfloat16_t> embedded = parser.embedding(input, test);
	test_check(!embedded.empty(), "Embedded vector is non-empty", s_pass, s_fail);
	test_check_eq(embedded.size(), tokenized.size() * input.embedding_dim, "Embedded floats == tokens * embedding_dim", s_pass, s_fail);
	test_check(!input.embedding_table.empty(), "Embedding lookup table is populated", s_pass, s_fail);

	// Bulk directory tokenization verification
	Input bulk_input;
	bulk_input.embedding_dim = input.embedding_dim;
	Parser bulk_parser(false, test_embedded_path(), test_asset_dirs());
	size_t files_processed = bulk_parser.bulk_tokenize_directory(test_asset_dirs()[0], bulk_input);
	test_check(files_processed > 0, "bulk_tokenize_directory processes files in asset_sample", s_pass, s_fail);
	test_check(!bulk_input.tokenized_data.empty(), "bulk_tokenize_directory produces non-empty token stream", s_pass, s_fail);
	test_check(!bulk_input.image_dictionary.empty(), "bulk_tokenize_directory populates image namespace", s_pass, s_fail);
	test_check(!bulk_input.video_dictionary.empty(), "bulk_tokenize_directory populates video namespace", s_pass, s_fail);
	test_check(!bulk_input.voice_dictionary.empty(), "bulk_tokenize_directory populates voice namespace", s_pass, s_fail);
	test_check_eq(bulk_parser.get_last_bulk_files().size(), files_processed, "bulk_parser tracks all processed bulk files", s_pass, s_fail);

	// Test archive exclusion: files placed in archive/ are ignored
	std::error_code ec;
	const std::filesystem::path test_archive_dir = test_asset_dirs()[0] / "archive";
	std::filesystem::create_directories(test_archive_dir, ec);
	const std::filesystem::path dummy_archive_file = test_archive_dir / "ignored_dummy.txt";
	{
		std::ofstream dummy_out(dummy_archive_file);
		dummy_out << "archived content should be ignored by bulk tokenizer";
	}
	Input archive_test_input;
	archive_test_input.embedding_dim = input.embedding_dim;
	size_t archive_test_processed = bulk_parser.bulk_tokenize_directory(test_asset_dirs()[0], archive_test_input);
	test_check_eq(archive_test_processed, files_processed, "bulk_tokenize_directory strictly ignores archive folder", s_pass, s_fail);
	std::filesystem::remove(dummy_archive_file, ec);
	std::filesystem::remove(test_archive_dir, ec);

	// Non-existent directory handling
	Input missing_input;
	size_t missing_processed = bulk_parser.bulk_tokenize_directory("non_existent_folder_xyz", missing_input);
	test_check_eq(missing_processed, static_cast<size_t>(0), "bulk_tokenize_directory safely returns 0 on missing dir", s_pass, s_fail);

	std::cout << "=== Tokenization & Modality Test Suite: " << s_pass << " passed, " << s_fail << " failed ===\n";
	return s_fail == 0;
}