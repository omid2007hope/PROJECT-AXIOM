Based on the roadmap and the fact that we just finished implementing the synchronized data pipeline and the native CUDA forward pass, here are the exact next steps we need to tackle to turn this into a true high-performance training engine:

### 1. Multi-Stream Double Buffering (Phase 3 of the Roadmap)

Right now, our pipeline is **synchronous**: it slides the window, waits for the PCIe upload to finish, and then runs the forward pass.

- **The Goal**: We need to overlap communication and computation. We'll create two non-blocking CUDA streams (`cudaStreamNonBlocking`).
- **How it works**: While Stream A is executing the neural network math (`forward_pass`) on Window 1, Stream B will be actively transferring Window 2 across the PCIe bus. This hides the memory transfer latency entirely, keeping the Tensor Cores fed at 100% capacity.

### 2. The Loss Function & Backward Pass (Backpropagation)

We have the forward pass $y = \text{ReLU}(Wx + b)$, but the network isn't actually _learning_ yet.

- **The Goal**: Implement the backward pass to calculate gradients and update the weights.
- **How it works**: We'll write a custom CUDA kernel to calculate the error between our prediction ($y$) and the actual `target_tokens`. Then we'll calculate the gradients (the derivatives) and apply a learning rate to update the $W$ matrix and $b$ vector in VRAM.

### 3. Stream-Ordered VRAM Management (Phase 5)

As we implement the backward pass, we will need temporary VRAM buffers to hold intermediate gradients and activations.

- **The Goal**: Prevent `cudaMalloc` and `cudaFree` from causing synchronization bottlenecks during the training loop.
- **How it works**: We will introduce `cudaMallocAsync` and `cudaMemPool_t` (or start building our BFC Caching Allocator) so that memory for the backward pass can be recycled instantly on the GPU without pausing the CPU orchestration loop.

**Which of these would you like to attack first?** I highly recommend starting with the **Backward Pass**, as completing the mathematical loop (Forward $\rightarrow$ Loss $\rightarrow$ Backward) will give us a fully functional (albeit synchronous) neural network before we optimize its latency with streams and custom memory allocators.
