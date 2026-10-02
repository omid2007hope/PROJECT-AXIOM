### Option A: Implement the Training Loop (The Backward Pass Integration)

**Roadmap Phase:** Phase 4 (Neural Nucleus) / _Step 2 in `2th step.md`_
Right now, your CUDA DLL exposes a `backward_pass` function (which calculates the error between the prediction and `target_tokens`, computes gradients, and updates the weights). However, your C++ engine hasn't wired this up to run in a continuous training loop.

- **Why do this next?** A neural network isn't a neural network until it can _learn_. Wiring up the full mathematical loop (Upload $\rightarrow$ Forward $\rightarrow$ Backward) will give you a fully functional, mathematically complete AI engine.
- **Mentor's Recommendation:** **Highly Recommended.** It is always best to make the system _correct_ (learning) before making it _blisteringly fast_ (double-buffering).

### Option B: C++ Asynchronous Double Buffering (Overlapping IO/Compute)

**Roadmap Phase:** Phase 3 / _Step 1 in `2th step.md`_
While the CUDA DLL accepts a `buffer_index` and allocates two `cudaStream_t`s, your CPU `Optimizer` thread is currently operating synchronously. It waits for data to move across the PCIe bus before asking the GPU to compute.

- **Why do this next?** If you want to see the pure hardware optimization of your pipeline, we can wire up the C++ threads to "ping-pong". While the GPU is crunching numbers on Buffer 0, the CPU will DMA transfer the next context window into Buffer 1. This completely hides the PCIe latency.
- **Mentor's Recommendation:** Do this _after_ Option A. It's much easier to debug a synchronous training loop than an asynchronous one.

### Option C: The CUTLASS Integration for Tensor Cores

**Roadmap Phase:** Phase 4
Right now, our `forward_pass_kernel` and `backward_pass_kernel` in `kernel/` are written using standard nested CUDA threads. While they use `bfloat16`, they will execute on the generic CUDA cores, not the specialized Tensor Cores of your RTX GPU.

- **Why do this next?** To saturate your GPU and hit max TeraFLOPS, we need to replace our custom matrix multiplication with NVIDIA's CUTLASS library (C++ Templates for Linear Algebra).
- **Mentor's Recommendation:** Hold off. This is a massive dependency injection and requires the mathematical loop to be fully functional first so we can verify CUTLASS doesn't break our math.
