#include "../include/parser.hpp"

#include <cmath>
#include <unordered_map>
#include <vector>

/*
Turn raw data samples into the parsed input vector;
Now isolated from the neural network entirely.
*/

// Using Byte-Pair Encoding (BPE) or SentencePiece concept
std::vector<double> Parser::tokenization(Input &input) {

  // 1. MOCK TOKENIZATION: Map words to integer IDs
  std::unordered_map<std::string, double> text;
  double next_id = 1.0;

  // for each word
  for (const std::string &word : test.data_txt) {
    // If the word isn't in our dictionary, add it
    if (text.find(word) == text.end()) {
      text[word] = next_id;
      next_id += 1.0;
    }
    // Push the token ID (e.g. 1.0, 2.0) into the input
    input.data.push_back(text[word]);
  }

  return input.data;
}

// 2. MOCK EMBEDDINGS: Convert token IDs to Dense Vectors
std::vector<double> Parser::embedding(Input &input) {
  std::vector<double> embedded_data;

  // For every parsed value (token IDs, normalized pixels, etc.)
  for (double value : input.data) {
    // We mock an embedding matrix by generating 3 unique floats
    // for every discrete value using mathematical transformations.
    // In real AI, these values are weights that the AI learns over time!
    double embed_dim1 = std::sin(value * 1.5);
    double embed_dim2 = std::cos(value * 2.1);
    double embed_dim3 = std::sin(value * 3.3) * std::cos(value * 0.5);

    embedded_data.push_back(embed_dim1);
    embedded_data.push_back(embed_dim2);
    embedded_data.push_back(embed_dim3);
  }

  return embedded_data;
}
