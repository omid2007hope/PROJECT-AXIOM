#ifndef include_transfer
#define include_transfer

#include <cstddef>
#include <vector>

#include "../src/CUDA/transfer.cuh"


class Transfer {
public:
	// Method to move memory to GPU
	static bool upload_weights_to_gpu(const std::vector<float>& h_weights,
					  const std::vector<float>& h_bias,
					  size_t& pending_steps,
					  GpuWeights& gpu_weights,
					  GpuActivations& gpu_activations);

	// Method to copy the trained weights back into h_weights / h_bias
	static bool download_weights_from_gpu(std::vector<float>& h_weights,
					      std::vector<float>& h_bias,
					      const GpuWeights& gpu_weights);

	// Method to move embeddings to GPU
	static bool upload_embeddings_to_gpu(const std::vector<float>& h_embeddings,
					      GpuEmbeddings& gpu_embeddings);

	// Method to copy trained embeddings back into h_embeddings
	static bool download_embeddings_from_gpu(std::vector<float>& h_embeddings,
						 const GpuEmbeddings& gpu_embeddings);
};

#endif