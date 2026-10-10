#ifndef include_transfer
#define include_transfer

#include <cstddef>
#include <vector>

#include "../src/CUDA/transfer.cuh"


class Transfer {
private:
public:
	// Method to move memory to GPU
	bool upload_weights_to_gpu(const std::vector<float>& h_weights,
				   const std::vector<float>& h_bias,
				   size_t& pending_steps,
				   GpuWeights& gpu_weights,
				   GpuActivations& gpu_activations,
				   [[maybe_unused]] GpuPortion& gpu_portion);

	// Method to copy the trained weights back into h_weights / h_bias
	bool download_weights_from_gpu(std::vector<float>& h_weights,
				       std::vector<float>& h_bias,
				       [[maybe_unused]] size_t pending_steps,
				       const GpuWeights& gpu_weights,
				       [[maybe_unused]] const GpuActivations& gpu_activations,
				       [[maybe_unused]] const GpuPortion& gpu_portion);
};

#endif