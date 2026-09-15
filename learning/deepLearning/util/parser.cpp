#include "../include/parser.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

// Data Streaming (or Batching).

/*
Turn raw data samples into the parsed input vector;
Now isolated from the neural network entirely.
*/

// MOCK DISCRETE TOKENIZER (Gemini / Chameleon Early Fusion style)
std::vector<double> Parser::tokenization(Input &input, Test &test) {

  // 1. Text Tokenization (IDs 1 - 9999)
  for (const std::string &word : test.data_txt) {
    add_assign_increase(word, input.text_dictionary, input.next_text_id, input);
  }

  // 2. Number Tokenization (IDs 10000 - 19999)
  for (double val : test.data_num) {
    add_assign_increase(val, input.number_dictionary, input.next_number_id,
                        input);
  }

  // 3. Image Tokenization (IDs 20000 - 29999)
  // Note: We use exact matches for now. A real VQ-VAE would round/quantize
  // these.
  tokenize_from_disk("Photo.jpeg", input.image_dictionary, input.next_image_id,
                     input);

  // 4. Video Tokenization (IDs 30000 - 39999)
  tokenize_from_disk("Video.mp4", input.video_dictionary, input.next_video_id,
                     input);

  // 5. Voice/Audio Tokenization (IDs 40000 - 49999)
  tokenize_from_disk("Voice.ogg", input.voice_dictionary, input.next_voice_id,
                     input);

  return input.tokenized_data;
}

// MOCK EMBEDDINGS: Convert global token IDs to Dense Vectors
std::vector<double> Parser::embedding(Input &input, Test &test) {}
