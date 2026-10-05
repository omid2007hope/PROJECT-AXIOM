#include "telemetry.cuh"
#include <cuda_runtime.h>
#include <iostream>

extern "C" void get_gpu_telemetry(size_t *free_mem, size_t *total_mem) {
    if (free_mem && total_mem) {
        cudaError_t err = cudaMemGetInfo(free_mem, total_mem);
        if (err != cudaSuccess) {
            *free_mem = 0;
            *total_mem = 0;
        }
    }
}
