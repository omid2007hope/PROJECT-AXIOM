# Domain-Specific Expert Training Architecture & Execution Blueprint

This document specifies the operational blueprint for curating high-signal training datasets and executing multi-epoch expert model training on a single **NVIDIA GeForce RTX 4080** (16 GB VRAM ceiling) within **Project Axiom**.

---

## 1. Executive Summary & Core Philosophy

An artificial intelligence model is mathematically a mirror of its training corpus. Feeding noisy, contradictory, unformatted, or low-density text causes parameter optimization to converge on high-entropy noise. 

To train an expert model under desktop hardware constraints, engineering effort must shift from runtime orchestration to **strict dataset curation, corpus hygiene, and multi-epoch progression**.

```
+-----------------------------------------------------------------------------------+
|                        PHASE 1: EXPERT BOUNDARY SELECTION                         |
|   Narrow domain focus (C++/CUDA Systems, Kinetic Physics, or Stylistic Fiction)  |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                        PHASE 2: GATHERING & DISTILLATION                          |
|         100 MB to 500 MB pure text (~125M to 150M categorical tokens)             |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                        PHASE 3: STRICT CORPUS HYGIENE                             |
|    UTF-8 validation, whitespace normalization, and frequency filtering (< 3)     |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                        PHASE 4: MULTI-EPOCH TRAINING LOOP                         |
|   Epoch 1 (Clustering) -> Epochs 2-3 (Syntax) -> Epochs 4-5 (Contextual Logic)    |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                  ARCHITECTURAL PREREQUISITE: HORIZON 2 ATTENTION                  |
|     Multi-Head Self-Attention (MHA) to replace raw single-layer context averaging |
+-----------------------------------------------------------------------------------+
```

---

## 2. Phase 1: Define the Domain (The Expert Boundary)

Given the 16 GB physical VRAM ceiling of the RTX 4080, training an unconstrained generalist model is mathematically unfeasible. The model must target **one** narrow, highly constrained domain to maximize representational capacity:

| Domain Option | Focus Area | Source Material | Target Mastery |
| :--- | :--- | :--- | :--- |
| **Option A: The Systems Engineer** | C++23 & NVIDIA CUDA Kernels | Production repositories (CUTLASS, PyTorch ATen, Godot) | Low-level memory management, RAII, warp intrinsics, pointer arithmetic |
| **Option B: The Ballistics Physicist** | Terminal ballistics & material impact | Research papers, textbook chapters, MOP terminal simulation assets | Continuum mechanics, shock waves, penetration depth formulas, equation of state |
| **Option C: The Stylistic Author** | Specialized narrative fiction | Curated literary series, single-author anthologies | Coherent character voice, dialogue mechanics, narrative pacing |

> **Architectural Law:** The narrower the target domain, the faster the loss manifold converges on deterministic logic rather than memorizing disjointed surface statistics.

---

## 3. Phase 2: Data Gathering & Distillation

The target dataset footprint is **100 MB to 500 MB** of uncompressed, plain-text data.

* **Token Density:** In technical and standard English text, 1 byte yields approximately $0.25$ to $0.30$ tokens.
* **Volume:** 500 MB of clean text delivers **125 to 150 million tokens**, matching the throughput envelope of an RTX 4080 weekend training session.

### Domain A: C++ / CUDA Source Ingestion
1. Clone 20 to 30 reference repositories maintaining pristine formatting and comments (e.g., CUTLASS, PyTorch ATen backend, Godot Engine core).
2. Purge all binary assets, build artifacts (`.obj`, `.dll`), `.git` directories, documentation images, and configuration scripts.
3. Isolate strictly source and header extensions: `.cpp`, `.hpp`, `.cu`, `.cuh`.

### Domain B: Kinetic Physics & Ballistics Ingestion
1. Collect 20 to 40 academic textbooks, peer-reviewed journals, and declassified impact mechanics papers.
2. Feed PDF and compressed documents through the Project Axiom [`Decompressor`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/02_SYSTEM_ARCHITECTURE_AND_PIPELINE.md#the-four-stage-multimodal-ingestion-pipeline) pipeline to extract continuous text streams, discarding raster graphics and layout artifacts.

---

## 4. Phase 3: Corpus Hygiene & Formatting

Raw textual data frequently contains non-semantic byte sequences (corrupted PDF symbols, Base64 strings, unescaped HTML, or hex dumps) that degrade Tensor Core efficiency.

Before raw text enters the [`Identifier`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/02_SYSTEM_ARCHITECTURE_AND_PIPELINE.md) stage, the following hygiene rules must be applied:

1. **UTF-8 Sanitization:**
   * Strip invalid UTF-8 byte sequences, control characters (excluding newline `\n`), and non-semantic formatting anomalies.
2. **Whitespace Normalization:**
   * Convert all tab characters (`\t`) to 4 spaces.
   * Collapse consecutive blank lines (`\n\n\n+`) into a single newline boundary (`\n`).
3. **Frequency Filtering ($N < 3$ Threshold):**
   * Run a global token frequency pass across the entire corpus.
   * If any token or character sequence appears fewer than 3 times across the 500 MB corpus, replace it with `<UNK>` (Token ID `1`) or excise it.
   * **Rationale:** Protects the 65,536-entry [`main_dictionary`](file:///c:/Users/Omid/X/Code/PROJECT-AXIOM/document/07_MULTIMODAL_TOKENIZER_AND_INFERENCE.md) from being flooded by typos, GUIDs, and random hexadecimal strings.

---

## 5. Phase 4: Multi-Epoch Training Loop Strategy

A single linear pass over the dataset is insufficient for functional synthesis. The dataset must be trained over **multiple shuffled epochs**:

```
[Dataset: 500 MB / 150M Tokens]
         │
         ├──> Epoch 1: Geometric Representation (Token Clustering)
         ├──> Epoch 2: Syntactic Co-occurrence (Grammar & Operators)
         ├──> Epoch 3: Structural Refinement (Control Flow & Typing)
         ├──> Epoch 4: Long-Range Invariants (Memory Pairs: malloc -> free)
         └──> Epoch 5: Deep Contextual Inference (Algorithmic Logic)
```

### Cognitive Epoch Progression

1. **Epoch 1 (Geometric Clustering):**
   * Embeddings align into fundamental semantic clusters (e.g., numerical datatypes `int`, `float`, `size_t` group into proximate vector subspaces).
2. **Epochs 2–3 (Syntactic Structure):**
   * The model internalizes local syntax (e.g., `float` is routinely followed by an identifier, assignment operator `=`, and a numerical literal).
3. **Epochs 4–5 (Contextual Logic):**
   * The model masters paired programmatic invariants across the sliding window (e.g., invocation of `cudaMalloc` mandates an eventual `cudaFree` within the function scope).

### Compute Budget & Throughput Math (RTX 4080)

* **Dataset Size:** 125,000,000 tokens.
* **Micro-Dispatch Speed:** $\sim 300\text{ tokens/second}$ (unbatched PCIe streaming).
  $$\text{Execution Time (Unbatched)} = \frac{125,000,000}{300 \times 3600} \approx 115.7\text{ hours } (\sim 4.8\text{ days per epoch})$$
* **Batched Tensor Core Dispatch (Target):** With CUDA batching ($M \ge 16$ or $M = 32$ parallel sliding windows per CUTLASS sweep), throughput scales to $3,000 - 5,000\text{ tok/s}$:
  $$\text{Execution Time (Batched)} = \frac{125,000,000}{3,500 \times 3600} \approx 9.9\text{ hours per epoch}$$

---

## 6. Architectural Prerequisite: Multi-Head Self-Attention (Horizon 2)

Training 500 MB of high-density domain data on a single-layer perceptron projection risks creating a high-capacity memorization engine ("drunk historian") that predicts the next token based only on the immediate static average of the context window.

Before committing to massive multi-epoch runs, the engine architecture must incorporate **Horizon 2: Multi-Head Self-Attention (MHA)**:

1. **The Representation Limitation:**
   * A linear perceptron computes $Y = X \cdot W + B$, flattening 256 tokens into a fixed static sum without pairwise relational gating.
2. **The Attention Mechanism:**
   * Query-Key-Value ($Q, K, V$) projections compute dynamic relational weights:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   * This mechanism enables Token #255 to attend directly back to Token #12 (e.g., resolving a pointer declaration to its point of destruction), forming structural comprehension across the sequence.

---

## 7. Immediate Action Checklist Prior to Training

- [ ] Select domain focus (**Option A: Systems/CUDA C++** or **Option B: Kinetic Physics**).
- [ ] Collate raw `.cpp` / `.cu` files or extract uncompressed research text to `asset_sample/`.
- [ ] Execute UTF-8 sanitization and whitespace normalization script.
- [ ] Enforce frequency threshold ($\ge 3$ occurrences) to preserve dictionary vocabulary slots.
- [ ] Implement batched sliding window gathering in CUDA to maximize Tensor Core MFU.
- [ ] Integrate Horizon 2 Self-Attention layer prior to final weight convergence pass.
