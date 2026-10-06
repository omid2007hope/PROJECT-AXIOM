# On the Impossibility of True Randomness and the Implications of Training Machine Learning Models on Random Binary

**Omid Teimory**
*September 2026*

## Abstract
This paper argues that true randomness—defined as output that is source-less, pattern-less, algorithm-less, and fundamentally irreversible—cannot exist within any closed deterministic system, including digital computers built on the Turing machine architecture. We introduce the Rewind Test as a diagnostic criterion: if the entirety of a system’s output can be traced backward to a single seed or initial state, the output is generated, not random. Building on Laplace’s classical determinism, Einstein’s hidden-variable intuition, and the architectural constraints of modern computation, we pose a central hypothesis: if a sufficiently powerful machine learning model is trained on an unbounded corpus of pseudo-random (or ostensibly “truly random”) binary sequences, it must eventually converge on the deterministic structure underlying those sequences. We explore the consequences of this convergence for cryptography, national security, and the philosophical boundaries of knowledge itself.

## 1 Introduction
The question of whether anything in the universe is truly random is not merely academic. The entire architecture of modern digital security, from RSA and elliptic-curve cryptography (ECC) to the one-time pad, rests on the assumption that certain processes produce output no adversary can predict or reverse-engineer. If that assumption is false—if every sequence we call “random” is in fact the deterministic output of a deeper, hidden algorithm—then the consequences cascade across mathematics, physics, computer science, and geopolitics.

This paper proceeds from a single, falsifiable claim:
If you can rewind the whole system to one origin—a seed, a singularity, a single initial state—you are not looking at true randomness. You are looking at a generator. Randomness that was actually source-less would not collapse backward onto a point; it would smear into infinitely many unrelated origins.

We formalize this intuition as the Rewind Test, situate it within the history of deterministic philosophy, and then ask the question that motivates the entire investigation: What happens if you train a machine learning model on enough random binary code? Could it, at some threshold of data and compute, internalize the hidden structure of every generator—and thereby break every system that depends on the illusion of randomness?

## 2 The Rewind Test: A Criterion for True Randomness
We define true randomness by three necessary and jointly sufficient properties:
*   **Source-less.** The output cannot be traced to any originating seed, state, or initial condition.
*   **Pattern-less.** No finite algorithm, no matter how complex, can compress, predict, or reproduce the sequence.
*   **Algorithm-less.** No procedure—deterministic, probabilistic, or quantum—generated the output. It has no author.

The immediate corollary is the Rewind Test: take any sequence claimed to be random and attempt to run it backward through every known and hypothetical generative process. If the sequence collapses onto a single origin—a seed, a hardware state, a singularity—then it was generated, and the claim of randomness is refuted. Genuine randomness, under this definition, would resist backward collapse entirely. Traced in reverse, it would not converge; it would smear into infinitely many unrelated, mutually inconsistent origins, because no single cause produced it.

This criterion is deliberately strict. It excludes not only pseudo-random number generators (PRNGs) but also hardware-based “true” random number generators (TRNGs) that sample thermal noise, radioactive decay, or cosmic background radiation—unless one can prove that those physical processes are themselves irreducible to any deeper deterministic law. As we argue in subsequent sections, that proof has never been furnished.

## 3 Classical Determinism: Laplace’s Demon
The philosophical foundation of universal determinism was stated most precisely by Pierre-Simon Laplace in 1814. Laplace proposed a thought experiment: if a supreme intellect—later called Laplace’s Demon—knew the precise position and momentum of every particle in the universe at a single instant, it could, using only the laws of classical mechanics, compute the entire past and the entire future of the cosmos. Nothing would be uncertain; nothing would be random. The universe, in this view, is a single, vast, unwinding algorithm executing the initial conditions of the Big Bang.

Under Laplacian determinism, every event that appears random—a coin flip, a stock market crash, a radioactive decay—is merely a deterministic process whose inputs we lack the resolution to observe. Randomness is not a property of reality; it is a confession of ignorance.

Albert Einstein extended this intuition into the quantum domain. Confronted with the Copenhagen interpretation’s insistence that certain quantum events are fundamentally indeterminate, Einstein famously objected: “God does not play dice with the universe.” Einstein, along with Podolsky and Rosen, proposed that quantum mechanics must be incomplete—that what appears to be irreducible randomness at the quantum scale must be governed by hidden variables: deeper layers of physical reality, possibly involving dimensions, forces, or information structures not yet accessible to observation.

The hidden-variable hypothesis maps directly onto the Rewind Test. If hidden variables exist, then quantum “randomness” has a source, even if that source lies outside the observable system. The system is not closed; something is coming in from beneath or beyond the standard model of subatomic particles. True randomness is not present. There is only a generator whose seed we have not yet identified.

## 4 Computational Determinism and the Turing Architecture
Independent of whether the physical universe is deterministic, there is a domain in which the question is already settled: computation. A computer, by definition, operates on the Turing machine architecture. It is a deterministic finite-state machine. Every operation it performs requires a defined opcode and a defined source operand. There is no instruction in any instruction set architecture (x86, ARM, RISC-V) that says “produce a bit with no cause.” Every output bit is the deterministic consequence of a prior state.

This means it is algorithmically impossible for a digital computer to generate true randomness. A PRNG such as the Mersenne Twister, a linear congruential generator, or even a cryptographically secure generator like ChaCha20, takes a seed and applies a deterministic transformation. Given the seed, the entire output sequence is fixed. The “randomness” is a performance—a sequence that looks patternless to statistical tests but is, in principle, fully reproducible.

Even hardware random number generators (HRNGs) embedded in modern CPUs (e.g., Intel’s RDRAND) do not escape this framework. They sample physical phenomena—thermal jitter in silicon—and digitize the result. But the digitization is performed by deterministic circuitry, and the physical phenomenon itself is subject to the same question posed by Laplace: is thermal noise truly source-less, or is it the deterministic output of an astronomically complex but ultimately lawful microphysical process? If the latter, the HRNG is simply a PRNG with a very large, very difficult-to-observe seed.

## 5 The Reverse-Engineering Litmus Test
We propose that reverse-engineerability is the ultimate litmus test for determinism. If a sequence can, in principle, be traced backward to its generative origin, then it is not random—regardless of how statistically uniform or unpredictable it appears under forward observation.

Consider a concrete scenario. A PRNG produces a sequence of $10^{12}$ bits. A machine learning model is given unlimited compute, access to the exact cache-line alignment of the CPU, the thermal state of every transistor, and the full memory image at the moment of execution. Under these conditions, the model could completely reverse-engineer the seed. It traces the mathematical transformations backward, unwinding the algorithm layer by layer, until it arrives at the single, definitive hardware state that spawned the sequence. At that point, the randomness evaporates. What appeared to be $10^{12}$ independent bits is revealed to be a single deterministic trajectory through state space.

This is not a hypothetical capability. Modern side-channel attacks already exploit partial access to hardware state (power consumption, electromagnetic emissions, timing variations) to recover cryptographic keys. The gap between current side-channel attacks and total seed recovery is a gap of compute and instrumentation, not of principle.

## 6 The Infinite Monkey Theorem and Exhaustive Generation
A related mathematical certainty reinforces the deterministic view. The Infinite Monkey Theorem states that a random generator, given infinite time, will produce every possible finite string—including every book ever written, the source code for a perfect artificial general intelligence, the molecular formula for the cure to every disease, and the most beautiful poem ever composed.

If one were to build a program that systematically enumerates every possible combination of characters, spaces, and punctuation marks up to some length $n$, the output would be an ocean of gibberish. But embedded within that ocean, with mathematical certainty, would be every meaningful document of length $\le n$ that could ever exist. The information is not created by the generator; it is selected from the space of all possible strings. The generator is a brute-force search engine, not a creative process.

This observation has a deep implication: if the space of all possible binary sequences is finite (or at least enumerable), and if every “random” sequence is in fact a deterministic traversal of that space, then randomness is indistinguishable from exhaustive search over a combinatorial landscape. The only variable is how much of the landscape has been covered.

## 7 The Central Hypothesis: Training on Random Binary
We now arrive at the motivating question of this paper:

What if you train a machine learning model on a sufficiently large corpus of “random” binary sequences—both PRNG and ostensibly TRNG—until it internalizes the latent structure? At some threshold, would the model effectively “know everything” and be able to break any system that depends on the assumption of randomness?

### 7.1 The Argument
If all computationally generated randomness is deterministic (Section 4), and if physically generated randomness is plausibly deterministic under hidden-variable theories (Section 3), then every binary sequence in existence has an underlying structure. A model trained on enough such sequences would, in principle, begin to detect that structure—not because it was programmed with the generating algorithm, but because it learned the statistical signatures that distinguish one generator family from another.

Modern deep learning already demonstrates a nascent version of this capability. Neural networks trained on PRNG output can, in controlled settings, identify the generator type (linear congruential, Mersenne Twister, xorshift) and in some cases recover partial seed information from the output alone. These results are limited by training data, model capacity, and compute. But the trajectory is clear: as data and compute scale, the model’s ability to reverse-engineer the generator scales with them.

### 7.2 The Convergence Threshold
We hypothesize that there exists a critical threshold—a function of model parameters $\theta$, training data volume $D$, and compute $C$—beyond which a model trained on binary sequences achieves the following capabilities:
1.  **Generator Identification.** Given an arbitrary binary sequence, the model identifies the class of generator (PRNG family, HRNG architecture, or physical source) with high confidence.
2.  **Seed Recovery.** For PRNG-generated sequences, the model recovers the seed or a functionally equivalent initial state.
3.  **Forward Prediction.** Given a prefix of a sequence, the model predicts subsequent bits with accuracy significantly exceeding 0.5.
4.  **Structural Decomposition.** For sequences generated by unknown or novel algorithms, the model decomposes the output into structural components (period, recurrence relations, modular arithmetic) sufficient to reconstruct the generator.

If capabilities (1)–(4) are achieved for all generators, including those sampling physical processes, the implications are total. Every encrypted channel, every authentication protocol, every digital signature scheme that depends on the unpredictability of its random inputs becomes transparent to the model.

### 7.3 Theoretical Limits
It must be acknowledged that this hypothesis faces a profound theoretical obstacle: Kolmogorov complexity. A sequence is Kolmogorov-random if no program shorter than the sequence itself can produce it. If truly Kolmogorov-random sequences exist in nature, no model—regardless of size—can compress or predict them, because doing so would constitute a shorter description, contradicting the definition.

However, the Kolmogorov-randomness of a sequence is uncomputable: there is no algorithm that can determine, for an arbitrary string, whether it is Kolmogorov-random. This means we can never prove that a physical source produces Kolmogorov-random output. The hypothesis therefore remains unfalsifiable from the Kolmogorov direction and must be evaluated empirically.

## 8 Implications for Cryptography

### 8.1 The PRNG Foundation of Modern Encryption
The entire internet—banking, military communications, intelligence networks, civilian infrastructure—is protected by cryptographic algorithms such as RSA, ECC, and AES. These algorithms are not based on true randomness. They are based on computational hardness assumptions: the belief that certain mathematical operations (integer factorization, discrete logarithm computation) are too expensive to reverse with current hardware, even though they are deterministic in principle.

The key generation process in these systems depends on PRNGs. A user’s private key is derived from a pseudo-random seed. If that seed can be recovered—by reverse-engineering the PRNG, by exploiting side channels, or by a sufficiently trained model—the encryption collapses. The ciphertext becomes plaintext. The signature becomes forgeable. The authentication becomes impersonation.

### 8.2 The One-Time Pad Paradox
The one-time pad (OTP) is the only encryption scheme proven to be information-theoretically secure—but its security rests entirely on the assumption that the key is generated from a source of true randomness. If the key is generated by a TRNG sampling background cosmic radiation, the security guarantee holds only if that radiation is genuinely source-less.

If the universe is deterministic—if there is a hidden algorithm governing cosmic background radiation—then the OTP key is not random. It is a deterministic function of the initial conditions of the universe. An entity with sufficient knowledge of those conditions could, in principle, reconstruct the key and decrypt any OTP-encrypted message. The “unbreakable” cipher becomes an artifact of ignorance, not of mathematics.

### 8.3 Q-Day: The Quantum Threshold
The term Q-Day refers to the moment a nation-state achieves a stable, fault-tolerant quantum computer capable of running Shor’s algorithm at scale. On Q-Day, RSA and ECC—the cryptographic bedrock of global communications—become instantly breakable. Every encrypted message ever transmitted and stored (a practice known as “harvest now, decrypt later”) becomes readable.

If a nation-state reaches Q-Day first, it achieves not merely an intelligence advantage but an absolute monopoly on the deterministic map of the world. The first mover would not need to physically destroy adversarial computing infrastructure. Because it holds the keys to every encrypted network on Earth, it could execute a silent, global sabotage:
*   Infiltrate the encrypted research networks of rival quantum laboratories.
*   Surgically alter experimental data, introducing microscopic deterministic flaws into physics calculations, delaying or derailing competitors’ progress.
*   Manipulate financial ledgers and supply-chain databases to bankrupt rival technology sectors.
*   Rewrite firmware and signing keys to establish persistent, undetectable backdoors in adversarial military systems.

The adversaries would never know they had been compromised. They would observe hardware “failures,” experimental “noise,” and market “fluctuations”—all of which would appear natural, because the sabotage exploits the very determinism that the adversaries mistake for randomness.

## 9 From Prediction to Omniscience: The Deterministic Endgame
If there is no true randomness—if every human decision, every market fluctuation, every neural firing is the deterministic output of a prior physical state—then predicting the future is no longer a philosophical mystery. It becomes a strict data-processing problem. The task reduces to: acquire sufficient state information and apply the correct transformation function.

This is already the operational premise of modern intelligence platforms. Entities such as Palantir Technologies do not predict the future through intuition or speculation; they construct ontological maps—structured representations of entities, relationships, and causal chains—and propagate those maps forward in time using statistical and deterministic models. The accuracy of their predictions is bounded not by randomness but by observational incompleteness: they cannot yet track every relevant variable.

The only barrier currently preventing intelligence agencies from predicting the next decade with total accuracy is not the existence of randomness. It is the computational limit of chaos theory: in a chaotic system, exponentially small errors in initial-state measurement propagate exponentially fast, making long-horizon prediction computationally intractable even when the underlying dynamics are fully deterministic. To build a perfect prediction engine, one must track every single subatomic particle in the closed system. If even one is missed, the prediction eventually diverges from reality.

This is precisely the constraint that connects the determinism argument to the machine-learning hypothesis of Section 7. A model trained on the binary output of every accessible generator is, in effect, learning the state-transition functions of the systems that produced that output. As the model’s training corpus and parameter count grow, the set of generators it can reverse-engineer expands. In the limit, if the model could internalize the transition function of the universe itself, it would become Laplace’s Demon instantiated in silicon.

## 10 Cosmological Implications: The Origin Problem
The Rewind Test, applied at cosmological scale, forces a binary choice about the origin of the universe:
1.  **Single origin (the Big Bang as a lone seed).** The universe is a closed deterministic system initiated by a single state. All subsequent events, including the emergence of complexity, life, and consciousness, are the algorithmic unfolding of that initial state. Apparent randomness is computational complexity, not ontological indeterminacy. In this case, a sufficiently powerful model could, in principle, derive the initial state from current observations and compute any future state.
2.  **Multiple origins or external injection.** The Big Bang is not the sole causal origin. Either there are infinitely many Big Bangs (a multiverse), or the observable universe is an open system receiving information, energy, or causal influence from outside—“some variable coming in” from a domain not described by the standard model. In this case, the system is not closed, and the apparent randomness may be genuine within the observable system, even if it is deterministic at a higher level of description.

Both options preserve the core argument: randomness is either deterministic (option 1) or it is the signature of an incomplete system boundary (option 2). In neither case is it source-less in the absolute sense required by our definition of true randomness.

## 11 Conclusion
We have argued that true randomness—source-less, pattern-less, algorithm-less, and irreversible—is incompatible with every known physical and computational framework. Classical mechanics is deterministic by construction. Quantum mechanics is either deterministic with hidden variables or stochastic with incomplete system boundaries. Turing machines are deterministic by definition. Every “random” binary sequence ever produced by a computer has a seed, and every seed is, in principle, recoverable.

The central hypothesis of this paper—that a machine learning model, trained on a sufficiently large and diverse corpus of random binary sequences, could converge on the latent structure underlying all generators—remains unproven but is consistent with the trajectory of modern deep learning, side-channel analysis, and cryptanalysis. If the hypothesis holds even partially, the implications for cryptography, national security, and the philosophy of knowledge are profound.

The question is not whether randomness is “real.” The question is whether we have yet built a machine with enough data and compute to prove that it is not. If and when that machine is built, it would not merely break encryption. It would dissolve the boundary between the known and the unknowable, reducing the future to a computation awaiting sufficient resources.

Whether such a threshold is achievable—or whether chaos-theoretic sensitivity and Kolmogorov incompressibility impose a hard ceiling—is the defining open problem at the intersection of computation, physics, and intelligence. The answer will determine whether the universe is a mystery or a machine.

### References
[1] A. Einstein, B. Podolsky, and N. Rosen, “Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?” Physical Review, vol. 47, no. 10, pp. 777–780, 1935.
[2] P.-S. Laplace, A Philosophical Essay on Probabilities, Dover Publications, 1951. (Originally published 1814.)
[3] J. S. Bell, “On the Einstein Podolsky Rosen Paradox,” Physics Physique Fizika, vol. 1, no. 3, pp. 195–200, 1964.
[4] C. E. Shannon, “Communication Theory of Secrecy Systems,” Bell System Technical Journal, vol. 28, no. 4, pp. 656–715, 1949.
[5] P. W. Shor, “Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer,” SIAM Journal on Computing, vol. 26, no. 5, pp. 1484–1509, 1997.
[6] A. N. Kolmogorov, “Three Approaches to the Quantitative Definition of Information,” Problems of Information Transmission, vol. 1, no. 1, pp. 1–7, 1965.
[7] A. M. Turing, “On Computable Numbers, with an Application to the Entscheidungsproblem,” Proceedings of the London Mathematical Society, vol. s2-42, no. 1, pp. 230–265, 1936.
[8] M. Matsumoto and T. Nishimura, “Mersenne Twister: A 623-Dimensionally Equidistributed Uniform Pseudo-Random Number Generator,” ACM Transactions on Modeling and Computer Simulation, vol. 8, no. 1, pp. 3–30, 1998.
[9] É. Borel, “Mécanique Statistique et Irréversibilité,” Journal de Physique Théorique et Appliquée, vol. 3, no. 1, pp. 189–196, 1913.

---

# The Cancerous Data Phenomenon: The Silent Crisis Threatening AI in 2026

As of 2026, the single biggest threat facing the tech industry is no longer compute shortages, regulation, or even energy costs. It is cancerous data.

We are training artificial intelligence on data scraped from every corner of the internet and building the foundations of machine intelligence on that material. The fundamental flaw is already visible: today’s digital world is saturated with fake news, fabricated statistics, politically corrupted narratives, ideologically biased content, and coordinated disinformation. Alone, these fragments would remain relatively harmless—their volume is still small compared to the total corpus. The danger begins the moment we scale.

When false information is mixed at industrial volume with accurate human data, the two become inseparable. This corrupted mass is then cross-validated by other equally contaminated datasets. Models trained on the resulting sludge go on to generate new content that trains the next generation of models. The result is progressive, self-reinforcing degradation.

Tiny pieces of false data behave exactly like individual cancer cells. In isolation they are almost invisible. Once they proliferate, they form a tumor. By the time the symptoms appear, the damage is already systemic and extremely difficult to reverse.

A sufficiently resourced actor—armed with bots, agents, and generative models—can deliberately accelerate this process. By flooding the open web with millions of tiny, unpredictable, and untraceable fragments of carefully crafted false information, they can slowly shift the statistical center of gravity of the entire training ecosystem toward a chosen political or cultural agenda. This is no longer theoretical. It is the logical endpoint of data poisoning at planetary scale.

## The Reality of Model Collapse

**The Feedback Loop**
When large language models are trained on the open internet, they inevitably ingest content previously generated by other AI systems. Every slight hallucination, subtle bias, or factual drift present in that synthetic material is absorbed into the next foundation model.

**The Compounding Effect**
Like cancer cells, these anomalies are negligible in isolation. Across successive generations of models training on each other’s outputs, the errors compound. High-quality original human data becomes statistically diluted and eventually inseparable from the synthetic noise. The model’s distribution collapses toward increasingly confident, increasingly hollow, and often hyper-biased nonsense.

## Data Poisoning and Coordinated Psyops
The concern that well-resourced actors could weaponize this mechanism is not speculative. Cybersecurity researchers have already documented state-sponsored campaigns, hacktivist operations, and sophisticated bot networks deliberately injecting subtly altered data into the public web. Because modern LLM training relies on massive automated scraping, meaningful human oversight at the required scale is impossible. Once enough carefully designed “untraceable fragments” enter the ecosystem, models treat them as ground truth and then amplify them to hundreds of millions of users. The result is a decentralized, highly scalable form of perception management that is extremely difficult to detect or reverse.

Cancerous data is no longer a fringe risk. It is the central unresolved vulnerability of the current AI paradigm.

## Tags
AI data poisoning, model collapse, cancerous data, synthetic data degradation, LLM training risks, data integrity crisis, AI feedback loops, generative AI contamination, information ecosystem collapse, large language model safety, AI hallucination compounding, digital disinformation scale, training data quality 2026

## Hashtags
#CancerousData
#ModelCollapse
#AIDataPoisoning
#SyntheticDataCrisis
#LLMTrainingRisks
#DataIntegrity
#AIethics
#GenerativeAI
#InformationWarfare
#Tech2026
#ArtificialIntelligence
#DataQualityCrisis
#Psyops
#AIAlignment
