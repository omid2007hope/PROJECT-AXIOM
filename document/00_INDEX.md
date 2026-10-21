# Project Axiom: Master Technical Documentation Suite

Welcome to the definitive engineering documentation for **Project Axiom** 🌌 — a silicon-native, zero-overhead multimodal AI engine built from scratch in bare-metal **C++23** and **NVIDIA CUDA**.

---

## Documentation Directory Map

| Chapter | Document | Core Focus |
| :--- | :--- | :--- |
| **01** | [01_PROJECT_VISION_AND_PHILOSOPHY.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/01_PROJECT_VISION_AND_PHILOSOPHY.md) | The philosophy of bare-metal computing, Terry Davis simplicity doctrine, rejection of Python bloat, the raw-byte compression trap, the dual ingestion workflow, and curing AI amnesia. |
| **02** | [02_SYSTEM_ARCHITECTURE_AND_PIPELINE.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/02_SYSTEM_ARCHITECTURE_AND_PIPELINE.md) | The Four-Stage Multimodal Ingestion Pipeline (Identifier, Decompressor, Sanitizer/Tokenizer, Embedder), dense early-fusion codebook ($V=65,536$), the 4-tier memory logistics conveyor belt, and Output Codecs. |
| **03** | [03_MEMORY_HIERARCHY_AND_PCIE_DMA.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/03_MEMORY_HIERARCHY_AND_PCIE_DMA.md) | Hardware memory pyramid, `cudaHostAllocWriteCombined` pinned memory, token ID PCIe streaming (512 bytes/window), asynchronous PCIe DMA, and ping-pong dual non-blocking CUDA streams. |
| **04** | [04_CUDA_AND_TENSOR_CORES_COMPUTE.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/04_CUDA_AND_TENSOR_CORES_COMPUTE.md) | CUDA execution hierarchy, branchless VRAM gather kernel, CUTLASS v4.8.0 templates on Ada Lovelace (`sm_89`), linear EVT epilogue fusion, and Mixture of Experts (MoE) architecture. |
| **05** | [05_TRAINING_OPTIMIZER_AND_BACKPROPAGATION.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/05_TRAINING_OPTIMIZER_AND_BACKPROPAGATION.md) | Un-gated linear backpropagation ($dz = dy$), backward scatter kernel with `atomicAdd`, master FP32 embedding tables, and Curing AI Amnesia with AdamW Optimizer States ($m_t, v_t$). |
| **06** | [06_WEIGHT_PERSISTENCE_SPECIFICATION.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/06_WEIGHT_PERSISTENCE_SPECIFICATION.md) | Binary specifications for `weights.axiom` (parameters & master embedding table) and `optimizer.axiom` (AdamW momentum/variance tensors), two-stage atomic write protocol, and stream sync. |
| **07** | [07_MULTIMODAL_TOKENIZER_AND_INFERENCE.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/07_MULTIMODAL_TOKENIZER_AND_INFERENCE.md) | The Four-Stage Ingestion Pipeline, elimination of compressed byte streaming, dense early-fusion codebook ($V=65,536$), learnable 32D VRAM embeddings, and Output Codecs / Detokenizers. |
| **08** | [08_CLI_MANUAL_AND_OPERATIONS.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/08_CLI_MANUAL_AND_OPERATIONS.md) | Operator guide for command-line arguments, interactive terminal UI, unified text/number training, decompressor asset ingestion, runtime hotkeys (`[S]`, `[R]`, `[Q]`), and compilation instructions. |
| **09** | [09_VERIFICATION_AND_TEST_SUITE.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/09_VERIFICATION_AND_TEST_SUITE.md) | Forensic analysis of automated test harness verifying persistence, header rejection, dense codebook integrity, decompressor pipelines, zero VRAM leaks, and mathematical convergence. |
| **10** | [10_MATHEMATICAL_FOUNDATIONS_AND_PHYSICS.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/10_MATHEMATICAL_FOUNDATIONS_AND_PHYSICS.md) | Mathematical derivations of autoregression, Shannon entropy of compressed files vs semantic manifolds, shared text/number representations, AdamW momentum dynamics, and "The Rewind Test". |
| **11** | [11_ROADMAP_AND_SCALING_V2.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/11_ROADMAP_AND_SCALING_V2.md) | Technical blueprint for Version 2.0: Four-Stage Ingestion, Learnable VRAM Embeddings, AdamW Optimizer State, BFC Allocator, PagedAttention, CUDA Graphs, and Mixture of Experts (MoE). |
| **12** | [12_PUBLIC_LAUNCH_AND_ANNOUNCEMENT_KIT.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/12_PUBLIC_LAUNCH_AND_ANNOUNCEMENT_KIT.md) | Release kits for LinkedIn, X (Twitter), Reddit, and Hacker News outlining the project launch, four-stage multimodal pipeline, zero-amnesia optimizer state, and open-source contribution workflow. |
| **13** | [training_structure.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/training_structure.md) | Domain-specific expert training blueprint, 500 MB dataset curation, corpus hygiene rules, multi-epoch progression, and Attention prerequisite. |

---

## Visual Reference Asset Index

All architectural and hardware reference diagrams are organized in [`core/ref/`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/):

1. **Hardware Foundations & Limits:**
   - [`byte_and_bits_structure.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/byte_and_bits_structure.png): 1-byte / 8-bit bitmask layout.
   - [`powers_of_two_hardware_table.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/powers_of_two_hardware_table.png): Powers of 2 hardware alignment mapping ($2^1$ to $2^{10}$).
   - [`binary_combination_growth_curve.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/binary_combination_growth_curve.png): Exponential $2^n$ state capacity scaling.
   - [`cpu_architecture_32bit_vs_64bit_limits.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/cpu_architecture_32bit_vs_64bit_limits.png): Address space ceilings and 64-bit pointer architecture.
2. **CUDA Execution & Memory:**
   - [`cuda_software_to_hardware_mapping.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/cuda_software_to_hardware_mapping.png): Software Grid/Block/Warp/Thread to physical SM/Sub-partition/ALU.
   - [`cuda_execution_hierarchy_grids_blocks_threads.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/cuda_execution_hierarchy_grids_blocks_threads.png): 3D Thread mapping within Thread Blocks and Grids.
   - [`cuda_grid_block_thread_hierarchy.jpg`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/cuda_grid_block_thread_hierarchy.jpg): Multi-tier execution scheduling breakdown.
   - [`thread_blocks_to_sm_scheduling.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/thread_blocks_to_sm_scheduling.png): Hardware block allocation across Streaming Multiprocessors.
   - [`cuda_sm_occupancy_calculator.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/cuda_sm_occupancy_calculator.png): Warp scheduling and 100% occupancy saturation.
   - [`cuda_pointer_lifecycle_and_hazards.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/cuda_pointer_lifecycle_and_hazards.png): Null-guard, free, and neutralize safety state machine.
   - [`gpu_memory_hierarchy_pyramid.jpg`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/gpu_memory_hierarchy_pyramid.jpg): Latency vs capacity hierarchy (Registers $\rightarrow$ Shared $\rightarrow$ L2 $\rightarrow$ VRAM $\rightarrow$ PCIe).
   - [`memory_hierarchy_latency_tradeoffs.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/memory_hierarchy_latency_tradeoffs.png): Cycle counts and throughput across memory tiers.
   - [`gpu_graphics_rendering_pipeline.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/gpu_graphics_rendering_pipeline.png): Traditional GPU raster pipeline comparison.
3. **Neural Computing & Taxonomy:**
   - [`single_perceptron_binary_model.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/single_perceptron_binary_model.png): Classical threshold perceptron unit.
   - [`artificial_neuron_perceptron_model.jpg`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/artificial_neuron_perceptron_model.jpg): Modern continuous $y = \sigma(Wx + b)$ artificial neuron model.
   - [`neural_network_mlp_layers_diagram.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/neural_network_mlp_layers_diagram.png): Multilayer Perceptron connectivity.
   - [`neural_network_token_dimension_topology.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/neural_network_token_dimension_topology.png): Dimension and token channel topology.
   - [`hierarchical_feature_representation.jpg`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/hierarchical_feature_representation.jpg): Unsupervised to supervised hierarchical feature emergence.
   - [`traditional_vs_machine_learning_paradigm.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/traditional_vs_machine_learning_paradigm.png): Classical rule-based logic vs learned weight systems.
   - [`ai_ml_deep_learning_venn_diagram.jpg`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/ai_ml_deep_learning_venn_diagram.jpg): AI, Machine Learning, and Deep Learning subset relationships.
   - [`ai_ml_deep_learning_comparison.jpg`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/ai_ml_deep_learning_comparison.jpg): Operational breakdown of AI concepts.
   - [`ai_ml_nn_dl_taxonomy_diagram.png`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/ai_ml_nn_dl_taxonomy_diagram.png): Concentric taxonomy layers from Narrow AI to Deep Learning.
4. **Cognitive & Physics Research:**
   - [`cognitive_architecture_pipeline.jpg`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/cognitive_architecture_pipeline.jpg): Multi-path cognitive synthesis and context anchoring architecture.
   - [`the_rewind_test_determinism_vs_randomness.jpg`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/ref/the_rewind_test_determinism_vs_randomness.jpg): The Rewind Test: deterministic origin convergence vs source-less physical entropy.

---

## Key System Binaries and Artifacts

- **Release Executable:** [`core/app.exe`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/app.exe)
- **Automated Verification Harness:** [`core/test.exe`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/test.exe)
- **CUDA Acceleration Runtime:** [`core/cuda_transfer.dll`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/cuda_transfer.dll)
- **Persistent Binary Weights:** [`core/weights.axiom`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/weights.axiom) (8,248 bytes)
- **Build Automation:** [`core/Makefile`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/core/Makefile)
