#include "../include/parser.hpp"

void test_tokenization() {
  Input input;
  Test test;
  Parser parser;

  std::vector<uint32_t> tokenized = parser.tokenization(input, test);

  std::cout << "Successfully generated " << tokenized.size() << " tokens!\n";
  std::cout << "All tokens: ";

  // Print all tokens
  for (size_t i = 0; i < tokenized.size() && i < 1000; ++i) {
    std::cout << tokenized[i] << " ";
  }
  std::cout << "\n";

  // Vocabulary summary: count and ID range per modality
  auto summary = [](const char* name, auto& dict) {
    if (dict.empty()) { std::cout << "  " << name << ": 0 unique tokens\n"; return; }
    uint32_t lo = UINT32_MAX, hi = 0;
    for (const auto& [key, id] : dict) {
      if (id < lo) lo = id;
      if (id > hi) hi = id;
    }
    std::cout << "  " << name << ": " << dict.size()
              << " unique tokens  [" << lo << " .. " << hi << "]\n";
  };

  std::cout << "\n--- Vocabulary Summary ---\n";
  summary("Text  ", input.text_dictionary);
  summary("Number", input.number_dictionary);
  summary("Image ", input.image_dictionary);
  summary("Video ", input.video_dictionary);
  summary("Voice ", input.voice_dictionary);
  std::cout << "--------------------------\n";

  // Run embedding
  std::vector<double> embedded = parser.embedding(input, test);

  std::cout << "\n--- Embedding Summary ---\n";
  std::cout << "  Tokens       : " << tokenized.size() << "\n";
  std::cout << "  Dim per token: " << input.embedding_dim << "\n";
  std::cout << "  Total floats : " << embedded.size() << "\n";
  std::cout << "  Unique embeds: " << input.embedding_table.size() << "\n";
  std::cout << "  First token [" << tokenized[0] << "] -> [";
  for (uint32_t d = 0; d < input.embedding_dim; ++d) {
    std::cout << embedded[d];
    if (d + 1 < input.embedding_dim) std::cout << ", ";
  }
  std::cout << "]\n";
  std::cout << "-------------------------\n";
}