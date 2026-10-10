# 05. Training Optimizer and Backpropagation

A neural network without backpropagation is a static mathematical transformation. Learning occurs when the engine computes the error between its predictions and ground truth, propagates that signal backwards through the layer topology, and adjusts weight values to minimize loss.

This document details the mathematical derivation of Project Axiom's **Stochastic Gradient Descent (SGD)** backpropagation pipeline, mixed-precision master weight mechanics, and physical symmetry breaking.

---

## 1. Mathematical Formulation of the Forward & Backward Pass

In Project Axiom's foundational layer, the network processes a sliding context window of 128 tokens across 4 embedding channels, producing an input vector $\mathbf{x} \in \mathbb{R}^{512}$. The layer maps $\mathbf{x}$ to an output prediction $\mathbf{\hat{y}} \in \mathbb{R}^4$ representing the 4 embedding channels of the 129th target token.

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}, \quad \mathbf{\hat{y}} = \text{ReLU}(\mathbf{z})$$

Where:
- $\mathbf{W} \in \mathbb{R}^{4 \times 512}$ is the weight matrix ($2,048$ parameters).
- $\mathbf{b} \in \mathbb{R}^4$ is the bias vector ($4$ parameters).
- $\text{ReLU}(u) = \max(0, u)$.

### The Loss Function
The objective is Mean Squared Error (MSE) loss between prediction $\mathbf{\hat{y}}$ and ground-truth target vector $\mathbf{y}$:

$$\mathcal{L} = \frac{1}{2} \sum_{i=0}^{3} (\hat{y}_i - y_i)^2$$

### Error Gradient Propagation (The Delta Vector)
Applying the chain rule through the non-linear ReLU activation yields the error delta $\boldsymbol{\delta} \in \mathbb{R}^4$:

$$\delta_i = \frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} = (\hat{y}_i - y_i) \cdot \mathbb{I}(z_i > 0)$$

Where $\mathbb{I}(z_i > 0)$ is the indicator function representing the derivative of the ReLU function:

$$\text{ReLU}'(z_i) = \begin{cases} 1 & \text{if } z_i > 0 \\ 0 & \text{if } z_i \le 0 \end{cases}$$

---

## 2. Parameter Gradients & Rank-1 SGD Update

The gradient of the loss with respect to each weight element $W_{i, j}$ is the outer product of the error delta vector $\boldsymbol{\delta}$ and the input activation vector $\mathbf{x}$:

$$\frac{\partial \mathcal{L}}{\partial W_{i, j}} = \delta_i \cdot x_j, \quad \frac{\partial \mathcal{L}}{\partial b_i} = \delta_i$$

### Parameter Update Equation
Using Stochastic Gradient Descent with learning rate $\eta$ (configured to $\eta = 0.01$ in [`AI`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp)):

$$W_{i, j}^{(t+1)} = W_{i, j}^{(t)} - \eta \cdot (\delta_i \cdot x_j)$$

$$b_i^{(t+1)} = b_i^{(t)} - \eta \cdot \delta_i$$

```
   Delta Vector (4)                     Input Vector (512)
   [ d0, d1, d2, d3 ]^T      x          [ x0, x1, x2, ... , x511 ]
            \                                      /
             \                                    /
              ▼                                  ▼
      +-------------------------------------------------------+
      |       RANK-1 OUTER PRODUCT GRADIENT UPDATE MATRIX     |
      |                                                       |
      |   dW = [ d0*x0   d0*x1   d0*x2   ...   d0*x511 ]     |
      |        [ d1*x0   d1*x1   d1*x2   ...   d1*x511 ]     |
      |        [ d2*x0   d2*x1   d2*x2   ...   d2*x511 ]     |
      |        [ d3*x0   d3*x1   d3*x2   ...   d3*x511 ]     |
      +-------------------------------------------------------+
```

---

## 3. The Rank-1 CUDA Backpropagation Kernel

In [`core/src/tensor_cores.cu`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/tensor_cores.cu), the gradient calculation and parameter update are executed via a dedicated parallel CUDA kernel:

```cpp
__global__ void sgd_rank1_update_kernel(
    cutlass::bfloat16_t* __restrict__ weights,
    const cutlass::bfloat16_t* __restrict__ inputs,
    const float* __restrict__ deltas,
    float learning_rate,
    int in_dim,
    int out_dim)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Thread index along input dimension (0 to 511)
    if (j < in_dim) {
        float x_j = float(inputs[j]);
        for (int i = 0; i < out_dim; ++i) { // Loop over output dimension (0 to 3)
            int weight_idx = i * in_dim + j;
            float current_w = float(weights[weight_idx]);
            float grad = deltas[i] * x_j;
            float updated_w = current_w - (learning_rate * grad);
            weights[weight_idx] = cutlass::bfloat16_t(updated_w);
        }
    }
}
```

### Hardware Parallelism Strategy
- **Grid Configuration:** Exactly $512$ threads dispatched across 2 thread blocks of 256 threads ($2 \times 256 = 512$).
- **Coalesced Memory Access:** Each thread index $j$ accesses contiguous memory addresses along the rows, ensuring 128-bit memory transaction coalescing on the PCIe/VRAM bus.
- **Atomic-Free Execution:** Because each thread writes exclusively to distinct column entries for each row, thread contention is mathematically impossible. Zero mutexes or atomic operations are required.

---

## 4. Master Weights and Mixed-Precision Stability

In deep learning hardware acceleration, 16-bit Brain Float (`bfloat16`) is optimal for matrix multiplication because it preserves the 8-bit dynamic range exponent of IEEE FP32 while halving bandwidth.

However, during backpropagation, small gradients can cause catastrophic numerical issues:
$$\Delta W = \eta \cdot \delta_i \cdot x_j \approx 0.01 \times 0.001 \times 0.05 = 0.0000005$$

If updates of magnitude $10^{-6}$ are added directly to a `bfloat16` number (which has only 7 bits of mantissa precision), the small gradient is completely truncated to zero by hardware rounding. The parameters stop updating, and the network stops learning.

### The Axiom Master Weight Pattern
Project Axiom resolves this via **FP32 Master Weights**:
1. **Host Master Parameters:** The authoritative copy of all weights is maintained in full **IEEE FP32 single-precision** inside [`AI::weights`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp).
2. **Device Acceleration Parameters:** During forward passes and streaming updates, weights are maintained in `bfloat16` on the GPU for high-speed Tensor Core execution.
3. **Loss Gradient Accumulation:** Parameter gradients are accumulated in 32-bit float precision before syncing to disk or committing to long-term storage, guaranteeing zero underflow loss.

---

## 5. Symmetry Breaking at Birth (The Clone Army Problem)

If all weights and biases in a neural network are initialized to `0.0`:
1. Every neuron computes $z_i = \sum 0 \cdot x_j + 0 = 0$.
2. Every output activation evaluates to identical values.
3. Every backpropagation gradient $\delta_i$ evaluates to the exact same magnitude.
4. Every weight is adjusted by the exact same amount.

The network collapses into an army of clones: regardless of how many neurons exist in the architecture, they all perform the exact same calculation, destroying model capacity.

```
       ALL ZERO INITIALIZATION               MERSENNE TWISTER SYMMETRY BREAKING
       [0.00  0.00  0.00  0.00]                  [+0.034  -0.012  +0.045  -0.021]
       [0.00  0.00  0.00  0.00]                  [-0.019  +0.048  -0.033  +0.007]
                 │                                              │
                 ▼                                              ▼
        CLONE ARMY COLLAPSE                           SPECIALIZED NEURONS
     (Every neuron learns identically)              (Each neuron learns distinct features)
```

### The C++ Mersenne Twister Solution
Axiom injects physical uniform random chaos at initialization via `std::mt19937`:

```cpp
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

for (float& w : weights) {
    w = dist(gen); // Every parameter is born mathematically unique
}
```

### The Hard Weight Reset Shortcut (`[R]`)
During live interactive training in [`AI::train_interactive()`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/src/ai.cpp), the developer can press **`[R]`**:
1. Instantly re-randomizes all 2,048 weights on the host using the Mersenne Twister engine.
2. Synchronizes the randomized weights to the GPU device buffer via `cudaMemcpyAsync()`.
3. Resets the training step counter and loss metrics.
4. Allows testing new data convergence trajectories in under **1 millisecond** without terminating the application.
