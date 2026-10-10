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

	std::cout << "=== Tokenization & Modality Test Suite: " << s_pass << " passed, " << s_fail << " failed ===\n";
	return s_fail == 0;
}