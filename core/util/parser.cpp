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
std::vector<uint32_t> Parser::tokenization(Input &input, Test &test) {

  if (developer_mode) {
    // 1. Text Tokenization (IDs 1 - 9999)
    for (const std::string &word : test.data_txt) {
      add_assign_increase(word, input.text_dictionary, input.next_text_id,
                          input);
    }

    // 2. Number Tokenization (IDs 10000 - 19999)
    for (double val : test.data_num) {
      add_assign_increase(val, input.number_dictionary, input.next_number_id,
                          input);
    }

    // 3. Image Tokenization (IDs 20000 - 29999)
    tokenize_from_disk("Photo.jpeg", input.image_dictionary,
                       input.next_image_id, input);

    // 4. Video Tokenization (IDs 30000 - 39999)
    tokenize_from_disk("Video.mp4", input.video_dictionary, input.next_video_id,
                       input);

    // 5. Voice/Audio Tokenization (IDs 40000 - 49999)
    tokenize_from_disk("Voice.ogg", input.voice_dictionary, input.next_voice_id,
                       input);

  } else {
    // Production Mode: Dynamically Tokenize User's UI Data
    for (const std::string &word : input.user_text) {
      add_assign_increase(word, input.text_dictionary, input.next_text_id,
                          input);
    }
    for (double val : input.user_numbers) {
      add_assign_increase(val, input.number_dictionary, input.next_number_id,
                          input);
    }
    for (const std::string &path : input.user_image_paths) {
      tokenize_from_disk(path, input.image_dictionary, input.next_image_id,
                         input);
    }
    for (const std::string &path : input.user_video_paths) {
      tokenize_from_disk(path, input.video_dictionary, input.next_video_id,
                         input);
    }
    for (const std::string &path : input.user_voice_paths) {
      tokenize_from_disk(path, input.voice_dictionary, input.next_voice_id,
                         input);
    }
  }

  return input.tokenized_data;
}

// EMBEDDINGS: Convert global token IDs to a flat dense vector
// Each unique token ID gets a fixed embedding vector (seeded by ID for
// stability). Layout: [emb(token_0)..., emb(token_1)..., ...] size =
// tokenized_data.size() * embedding_dim
std::vector<std::bfloat16_t> Parser::embedding(Input &input, Test &test) {
  input.embedded_data.clear();
  input.embedded_data.reserve(input.tokenized_data.size() *
                              input.embedding_dim);

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
    const auto &vec = input.embedding_table[id];
    input.embedded_data.insert(input.embedded_data.end(), vec.begin(),
                               vec.end());
  }

  // Use our new generalized save method
  save_data(input.embedded_data, "embedded_data.bin");

  return input.embedded_data;
}
