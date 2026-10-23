# High-Density Curated Training Plan: Control-Token Architecture

## 1. Core Philosophy

This training pipeline discards the brute-force method of bulk data scraping. Instead, it utilizes High-Density Curated Data combined with Explicit Learning Instructions (Control Tokens). By manually engineering the text and injecting structural markers, we mathematically force the attention mechanism to focus on logical rules, drastically steepening the gradient and reducing training time from 48 hours to less than 5 hours.

---

## 2. Tokenizer Engineering (The Learning Instructions)

Before writing a single line of training data, you must define the markers that will guide the engine. These markers are added to the tokenizer's vocabulary as `special_tokens` so they are never split into smaller byte-chunks.

### Recommended Control Tokens for Assembly
- `<|concept_start|>` and `<|concept_end|>`: Bounds a specific logical rule.
- `<|rule_register|>`: Anchors attention to register manipulation.
- `<|rule_memory|>`: Anchors attention to memory addressing and pointers.
- `<|rule_control_flow|>`: Anchors attention to jumps and loops.
- `<|commentary|>`: Separates the code from your plain-English explanations.

### Implementation Note
When initializing your tokenizer (e.g., in HuggingFace/PyTorch), add these explicitly:

```python
tokenizer.add_special_tokens({'additional_special_tokens': ['<|concept_start|>', '<|rule_register|>', ...]})
```

> [!IMPORTANT]
> **Critical:** You must resize the model's embedding matrix afterward:
> ```python
> model.resize_token_embeddings(len(tokenizer))
> ```

---

## 3. Dataset Architecture (The Handpicked Data)

Your `.md` or `.txt` training files must be strictly formatted. Every example should follow a predictable sequence: Introduction -> Code -> Instruction Marker -> Explanation.

### Example Training Block
```text
<|concept_start|>
Moving immediate values into a 32-bit register.
<|code|>
MOV EAX, 42
<|rule_register|>
<|commentary|>
The MOV instruction copies the decimal value 42 directly into the EAX register. 
The destination is always on the left, the source is on the right.
<|concept_end|>
```

By placing `<|rule_register|>` right after the code, the engine's attention query ($Q$) matrix will form a massive weight correlation between the token `MOV`, the register `EAX`, and your English explanation.

---

## 4. Curriculum Design (The Training Sequence)

To prevent the loss curve from plateauing, you must feed the data to the engine sequentially, from lowest complexity to highest complexity. Do not shuffle the dataset across the entire corpus during the first epoch.

### Epoch 1: Progressive Staging
- **Phase 1: Foundational Syntax**
  - **Focus:** Basic `MOV`, `ADD`, `SUB`, and register architecture.
  - **Data:** ~1,000 highly curated blocks.
- **Phase 2: Memory & Pointers**
  - **Focus:** `PUSH`, `POP`, `LEA`, stack manipulation, and offsets (e.g., `[EBP-4]`).
  - **Data:** ~1,500 curated blocks with heavy use of `<|rule_memory|>`.
- **Phase 3: Control Flow & Logic**
  - **Focus:** `JMP`, `JZ`, `JNZ`, `CMP`, loops, and function prologues/epilogues.
  - **Data:** ~2,000 curated blocks.

### Epochs 2–3: Reinforcement
Now, you can shuffle the entire dataset. The engine has learned the foundational rules and will now solidify the matrix weights.

---

## 5. Hyperparameter Configuration

Because this data is mathematically dense and contains zero noise, standard bulk-training hyperparameters will destroy the model (causing catastrophic overfitting).

- **Learning Rate (LR):** Use a significantly lower learning rate than bulk training (e.g., $1 \times 10^{-5}$ instead of $5 \times 10^{-4}$). The data is so clean that you don't need large gradient steps to escape noise.
- **Batch Size:** Keep it relatively small (e.g., 4 to 8). Large batch sizes average out the gradients, but you want the engine to deeply feel the impact of each specific `<|concept_start|>` block.
- **Epochs:** 2 to 4 maximum. High-density data gets memorized very quickly. If you train for 10 epochs, the model will just memorize your exact `.md` files and fail to write novel assembly code.
- **Weight Decay:** Set this slightly higher (e.g., 0.1) to aggressively punish the network for memorizing and force it to generalize the underlying rules.

---

## 6. Execution & Monitoring

- **Monitor the Loss Curve:** You should see a rapid, near-vertical drop in loss within the first 30 minutes, followed by a smooth, low-slope decline.
- **Evaluate on Unseen Prompts:** After Epoch 2, pause the training and ask the model to generate a block of assembly it has never seen before (e.g., *"Write a loop that counts to 10 in ECX"*).
- **Check the Control Tokens:** In your testing, prompt the model using your markers (e.g., `User: Write a memory offset <|rule_memory|>`). If the training worked, the model's output will instantly snap to the correct syntax.
