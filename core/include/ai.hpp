#ifndef INCLUDE_AI_HPP
#define INCLUDE_AI_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdfloat>
#include <string>
#include <vector>

#include "../src/CUDA/transfer.cuh"
#include "optimizer.hpp"

#pragma pack(push, 1)

struct AxiomWeightHeader {
	char magic[8] = {'A', 'X', 'I', 'O', 'M', '_', 'V', '1'};
	uint64_t input_dimension = 0;
	uint64_t output_dimension = 0;
	uint64_t weight_count = 0;
	uint64_t bias_count = 0;
	uint64_t vocab_size = 0;
	uint64_t embedding_dim = 0;
};

#pragma pack(pop)

static_assert(sizeof(AxiomWeightHeader) == 56, "AxiomWeightHeader must be exactly 56 bytes");

struct Elements {
	size_t weight_elements = 0;
	size_t bias_elements = 0;
};

// What one AI::train() call did
enum class TrainStatus {
	Trained, // one full forward + backward step ran
	NoData,	 // the RAM buffer is exhausted: the caller must refill it
	Failed	 // a CUDA step failed (already reported on stderr)
};

struct SamplingConfig {
	float temperature = 0.7f;
	int top_k = 40;
	float top_p = 0.9f;
	uint64_t seed = 0;
};

class Parser;
struct Input;

class AI {
private:
	size_t input_dimension;
	size_t output_dimension;

	// Bounds of the uniform distribution used to initialize the weights
	float weight_min;
	float weight_max;

	// Training steps launched since the last read_loss()
	size_t pending_steps = 0;

	// Copies the last forward pass output and the GPU-argmax token to the host
	bool read_prediction(std::vector<float>& output_prediction);

public:
	AI(const AI&) = delete;

	AI& operator=(const AI&) = delete;

	size_t vocab_size = 65536;
	size_t embedding_dim = 4;
	size_t active_vocab_size = 0;

	// Milestone 3: Discrete Autoregressive Token Classification
	uint32_t last_predicted_token_id = 0;

	// CPU Memory (High precision float for initialization and updates)
	std::vector<float> host_weights;
	std::vector<float> host_bias;
	std::vector<float> host_embeddings;

	// GPU Memory State (The pointers for the VRAM)
	GpuWeights gpu_weights;
	GpuActivations gpu_activations;
	GpuPortion gpu_portion; // Reused across

	// iterations to prevent reallocation
	GpuEmbeddings gpu_embeddings;

	// Constructor: Takes dimensions and initializes with Chaos (Randomness)
	// inside [weight_min, weight_max]
	AI(size_t input_dimension,
	   size_t output_dimension,
	   float weight_min,
	   float weight_max,
	   Elements& elements);

	~AI();

	// Method to move memory to GPU
	bool upload_weights();

	// Method to copy the trained weights back into host_weights / host_bias
	bool download_weights();

	// Method to reset all parameters/weights back to chaotic state
	void reset_weights(Elements& elements);

	// The Training Loop Orchestrator
	TrainStatus train(Optimizer& optimizer, Portion& portion);

	// Mean training loss (0.5 * squared error per step, measured before each update)
	// over the steps since the previous call. `steps` receives how many that was.
	// Waits for the GPU to finish, so call it once per report interval, not per step.
	bool read_loss(double& mean_loss, size_t& steps);

	// Weight persistence methods
	bool save_weights(const std::string& path = "weights.axiom");

	bool load_weights(const std::string& path = "weights.axiom");

	// Forward Inference Engine (no gradient updates)
	bool predict(const std::vector<std::bfloat16_t>& context_window,
		     std::vector<float>& output_prediction);

	// Forward Inference from token IDs (right-aligned, PAD = 0): the GPU gathers the
	// trained embeddings, so inference sees the same inputs as training.
	bool predict_tokens(const std::vector<uint32_t>& context_ids,
			    std::vector<float>& output_prediction);

	// Autoregressive generation loop with real-time streaming
	bool generate(const std::string& prompt,
		      Parser& parser,
		      Input& input,
		      size_t max_new_tokens = 128,
		      const SamplingConfig& config = {},
		      std::function<void(const std::string&)> token_callback = nullptr);
};

#endif
