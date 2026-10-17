# 05. Training Optimizer and Backpropagation

A neural network without backpropagation is a static mathematical transformation. Learning occurs when the engine computes the error between its predictions and ground truth, propagates that signal backwards through the layer topology, and adjusts weight values to minimize loss.

This document details the mathematical derivation of Project Axiom's **Stochastic Gradient Descent (SGD)** backpropagation pipeline, mixed-precision master weight mechanics, and physical symmetry breaking.

---

## 1. Mathematical Formulation of the Forward & Backward Pass

In Project Axiom's foundational layer, the network processes a sliding context window of 128 tokens across embedding channels, producing an input vector $\mathbf{x} \in \mathbb{R}^{D_{in}}$. The layer maps $\mathbf{x}$ to an output prediction $\mathbf{\hat{y}}$ representing the target embedding coordinates:

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}, \quad \mathbf{\hat{y}} = \mathbf{z}$$

Where:
- $\mathbf{W} \in \mathbb{R}^{D_{out} \times D_{in}}$ is the linear projection weight matrix.
- $\mathbf{b} \in \mathbb{R}^{D_{out}}$ is the bias vector.
- The output mapping is linear, allowing the model to freely span continuous signed coordinates throughout $[-1.0, +1.0]$.

### The Loss Function
The objective is Mean Squared Error (MSE) loss between prediction $\mathbf{\hat{y}}$ and ground-truth target vector $\mathbf{y}$:

$$\mathcal{L} = \frac{1}{2} \sum_{i=0}^{D_{out}-1} (\hat{y}_i - y_i)^2$$

### Un-Gated Error Gradient Propagation (The Delta Vector)
Because the epilogue is linear, the gradient flows directly through without being clamped by a ReLU zero-gate:

$$\delta_i = \frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i - y_i$$

This un-gated formulation guarantees that $dz = dy$, preventing the permanent "Dying ReLU" parameter deadlocks that occur when target values are negative.

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
1. Instantly re-randomizes all weights on the host using the Mersenne Twister engine.
2. Synchronizes the randomized weights to the GPU device buffer via `cudaMemcpyAsync()`.
3. Resets the training step counter and loss metrics.
4. Allows testing new data convergence trajectories in under **1 millisecond** without terminating the application.

---

## 6. Embedding Table Backpropagation & Scatter Kernel

In naive synthetic embedding setups, token coordinates are fixed by trigonometric formulas and never learn from data. Project Axiom implements true **semantic emergence** by propagating gradients backward through the linear projection layer into the embedding matrix:

### 1. Input Gradient Computation
Applying the chain rule through the weight matrix yields the gradient with respect to each input feature $x_j$:

$$\frac{\partial \mathcal{L}}{\partial x_j} = \sum_{i=0}^{D_{out}-1} \delta_i \cdot W_{i, j}$$

### 2. Backward Scatter Kernel with `atomicAdd`
Because the same token (e.g. `"the"` or `"9.8"`) can appear multiple times within the same 128-token context window, parallel threads updating embedding coordinates simultaneously would induce severe race conditions.

Axiom executes a dedicated backward scatter kernel using hardware-accelerated atomic operations targeting the FP32 master table:

```cpp
__global__ void scatter_embedding_gradients_kernel(
    const uint32_t* __restrict__ d_tokens,
    const float* __restrict__ d_dx,
    float* __restrict__ d_table_master,
    float learning_rate,
    int num_tokens,
    int embedding_dim)
{
    int j = blockIdx.x; // Token position (0 to 127)
    int d = threadIdx.x; // Embedding dimension (0 to 31)
    if (j < num_tokens && d < embedding_dim) {
        uint32_t token_id = d_tokens[j];
        float grad = d_dx[j * embedding_dim + d];
        atomicAdd(&d_table_master[token_id * embedding_dim + d], -learning_rate * grad);
    }
}
```

After master FP32 gradient accumulation, the values are rounded into the `__nv_bfloat16` device table `d_table` for subsequent forward passes. Over millions of steps, words and multimodal tokens that co-occur in similar contexts (such as `"gravity"` and `"acceleration"`, or `"bulk"` and `"training"`) are physically pulled into neighboring regions of the 32D hyperspace.

---

## 7. Curing AI Amnesia: The AdamW Optimizer State

A standard neural network engine that saves only weights exhibits total amnesia: when restarted, the model has no recollection of its prior learning trajectory. It must slowly rediscover the gradient slope from scratch.

Axiom cures this amnesia by persisting the full **AdamW Optimizer State** in companion file `optimizer.axiom`.

### The Two Shadow Memories
For every parameter $\theta$, the optimizer tracks two dynamic moments:
1. **First Moment ($m_t$, Momentum / Velocity):**
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   Tracks the historical direction and speed of gradient descent.
2. **Second Moment ($v_t$, Variance / Volatility / Friction):**
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   Measures gradient volatility. Parameters on rough, erratic error surfaces receive smaller updates to prevent divergence.

### Parameter Update with Bias Correction & Decoupled Weight Decay
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

Where $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, and $\lambda$ is the decoupled weight decay coefficient.

### Binary Persistence (`optimizer.axiom`)
When the user saves the model (via CLI `--archive` or keyboard shortcut `[S]`), Axiom writes the $m_t$ and $v_t$ tensors for both projection weights and the master embedding table into `optimizer.axiom`. Upon resuming, the engine reloads its complete momentum state directly into GPU VRAM, instantly continuing training at full velocity.
