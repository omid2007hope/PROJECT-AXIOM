#include "../include/parser.hpp"

#include <cmath>
#include <unordered_map>
#include <vector>

/*
Turn raw data samples into the parsed input vector;
Now isolated from the neural network entirely.
*/

// MOCK DISCRETE TOKENIZER (Gemini / Chameleon Early Fusion style)
std::vector<double> Parser::tokenization(Input &input, Test &test) {

  // 1. Text Tokenization (IDs 1 - 9999)
  for (const std::string &word : test.data_txt) {
    if (input.text_vocab.find(word) == input.text_vocab.end()) {
      input.text_vocab[word] = input.next_text_id;
      input.next_text_id += 1.0;
    }
    input.tokenized_data.push_back(input.text_vocab[word]);
  }

  // 2. Number Tokenization (IDs 10000 - 19999)
  for (double val : test.data_num) {
    if (input.number_vocab.find(val) == input.number_vocab.end()) {
      input.number_vocab[val] = input.next_number_id;
      input.next_number_id += 1.0;
    }
    input.tokenized_data.push_back(input.number_vocab[val]);
  }

  // 3. Image Tokenization (IDs 20000 - 29999)
  // Note: We use exact matches for now. A real VQ-VAE would round/quantize these.
  for (double pixel : test.data_img) {
    if (input.image_vocab.find(pixel) == input.image_vocab.end()) {
      input.image_vocab[pixel] = input.next_image_id;
      input.next_image_id += 1.0;
    }
    input.tokenized_data.push_back(input.image_vocab[pixel]);
  }

  // 4. Video Tokenization (IDs 30000 - 39999)
  for (double delta : test.data_video) {
    if (input.video_vocab.find(delta) == input.video_vocab.end()) {
      input.video_vocab[delta] = input.next_video_id;
      input.next_video_id += 1.0;
    }
    input.tokenized_data.push_back(input.video_vocab[delta]);
  }

  // 5. Voice/Audio Tokenization (IDs 40000 - 49999)
  for (double sample : test.data_voice) {
    if (input.voice_vocab.find(sample) == input.voice_vocab.end()) {
      input.voice_vocab[sample] = input.next_voice_id;
      input.next_voice_id += 1.0;
    }
    input.tokenized_data.push_back(input.voice_vocab[sample]);
  }

  return input.tokenized_data;
}

// MOCK EMBEDDINGS: Convert global token IDs to Dense Vectors
std::vector<double> Parser::embedding(Input &input, Test &test) {

  for (double value : input.tokenized_data) {
    // We mock an embedding matrix. Because the IDs are strictly segregated 
    // by modality (e.g. 1 vs 20000), the math naturally produces 
    // wildly different embedding vectors for different modalities!
    double embed_dim1 = std::sin(value * 1.5);
    double embed_dim2 = std::cos(value * 2.1);
    double embed_dim3 = std::sin(value * 3.3) * std::cos(value * 0.5);

    input.embedded_data.push_back(embed_dim1);
    input.embedded_data.push_back(embed_dim2);
    input.embedded_data.push_back(embed_dim3);
  }

  return input.embedded_data;
}
