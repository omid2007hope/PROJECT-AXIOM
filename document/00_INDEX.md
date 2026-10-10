# Project Axiom: Master Technical Documentation Suite

Welcome to the definitive engineering documentation for **Project Axiom** 🌌 — a silicon-native, zero-overhead multimodal AI engine built from scratch in bare-metal **C++23** and **NVIDIA CUDA**.

---

## Documentation Directory Map

| Chapter | Document | Core Focus |
| :--- | :--- | :--- |
| **01** | [01_PROJECT_VISION_AND_PHILOSOPHY.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/01_PROJECT_VISION_AND_PHILOSOPHY.md) | The philosophy of bare-metal computing, Terry Davis simplicity doctrine, rejection of Python bloat, and the single dedicated model architecture. |
| **02** | [02_SYSTEM_ARCHITECTURE_AND_PIPELINE.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/02_SYSTEM_ARCHITECTURE_AND_PIPELINE.md) | The 4-Tier logistics conveyor belt (NVMe $\rightarrow$ RAM $\rightarrow$ Pinned DMA $\rightarrow$ VRAM), "The Seam" carry-forward window, and the Library/Book/Page operational model. |
| **03** | [03_MEMORY_HIERARCHY_AND_PCIE_DMA.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/03_MEMORY_HIERARCHY_AND_PCIE_DMA.md) | Hardware memory pyramid, `cudaHostAllocWriteCombined` pinned memory, CPU cache bypassing, asynchronous PCIe DMA, and ping-pong dual non-blocking CUDA streams. |
| **04** | [04_CUDA_AND_TENSOR_CORES_COMPUTE.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/04_CUDA_AND_TENSOR_CORES_COMPUTE.md) | CUDA execution hierarchy (Grid/Block/Warp/Thread), NVIDIA CUTLASS v4.8.0 templates on Ada Lovelace (`sm_89`), `mma.sync` instructions, and Epilogue Visitor Tree (EVT) kernel fusion. |
| **05** | [05_TRAINING_OPTIMIZER_AND_BACKPROPAGATION.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/05_TRAINING_OPTIMIZER_AND_BACKPROPAGATION.md) | Stochastic Gradient Descent (SGD) backprop kernel, FP32 master weights vs BF16 device weights, mixed-precision stability, and Mersenne Twister symmetry breaking. |
| **06** | [06_WEIGHT_PERSISTENCE_SPECIFICATION.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/06_WEIGHT_PERSISTENCE_SPECIFICATION.md) | The `weights.axiom` binary specification, 40-byte `AxiomWeightHeader` layout, two-stage atomic write via `.tmp` swap, CUDA stream synchronization, and corruption resistance. |
| **07** | [07_MULTIMODAL_TOKENIZER_AND_INFERENCE.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/07_MULTIMODAL_TOKENIZER_AND_INFERENCE.md) | Discrete VQ-VAE inspired orthogonal coordinate spaces (Text, Numbers, Images, Video, Audio), deterministic coordinate embedding, autoregressive prediction, and Euclidean $L_2$ decoding. |
| **08** | [08_CLI_MANUAL_AND_OPERATIONS.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/08_CLI_MANUAL_AND_OPERATIONS.md) | Operator guide for command-line arguments (`--use`, `--train [N]`, `--test`, `--help`), interactive terminal UI, runtime keyboard shortcuts (`[S]`, `[R]`, `[Q]`), and compilation instructions. |
| **09** | [09_VERIFICATION_AND_TEST_SUITE.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/09_VERIFICATION_AND_TEST_SUITE.md) | Forensic analysis of all 200 automated tests across 9 test suites in `test.exe` verifying persistence, corrupt header rejection, zero VRAM leaks, and mathematical roundtrip identity. |
| **10** | [10_MATHEMATICAL_FOUNDATIONS_AND_PHYSICS.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/10_MATHEMATICAL_FOUNDATIONS_AND_PHYSICS.md) | Mathematical derivations of autoregressive loss, spectral norm bounds, hardware entropy injection (`std::random_device`), and "The Rewind Test" on physical determinism. |
| **11** | [11_ROADMAP_AND_SCALING_V2.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/11_ROADMAP_AND_SCALING_V2.md) | Technical blueprint for Version 2.0: Best-Fit Coalescing (BFC) Caching Allocator, PagedAttention KV-Cache with Copy-On-Write prefix caching, CUDA Graphs DAGs, and ATen Dispatcher. |
| **12** | [12_PUBLIC_LAUNCH_AND_ANNOUNCEMENT_KIT.md](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/12_PUBLIC_LAUNCH_AND_ANNOUNCEMENT_KIT.md) | Release kits for LinkedIn, X (Twitter), Reddit, and Hacker News outlining the project launch, feature overview, and open-source contribution workflow. |

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
