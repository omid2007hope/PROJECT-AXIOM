#ifndef CUDA_TRANSFER_CUH
#define CUDA_TRANSFER_CUH

// This file is now an umbrella header that includes all the modularized CUDA APIs.
// CPU-side code can still simply #include "transfer.cuh" and gain access to all structs and functions.

#include "util/gpu_types.cuh"
#include "util/memory_allocation.cuh"
#include "gpu/upload.cuh"
#include "gpu/download.cuh"
#include "kernel/forward_pass.cuh"
#include "kernel/backward_pass.cuh"

#endif // CUDA_TRANSFER_CUH
