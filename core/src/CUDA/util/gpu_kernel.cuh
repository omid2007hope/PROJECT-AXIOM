#ifndef GPU_KERNEL
#define GPU_KERNEL

#include "../../../../include/ai.hpp"

struct Grid {
  int thread = 256;
  int block = static_cast<int>((output_dim + thread - 1) / thread);
  int index = 0;
 };

#endif // GPU_KERNEL
