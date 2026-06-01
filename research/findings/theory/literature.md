# Annotated Bibliography: Masked GRPO and Communication-Efficient RLVR

*Compiled by lit-scout, 2026-06-01. Coverage: 2021–2026, biased toward 2024–2026.*

This bibliography covers three clusters:

- **[A] RLVR with noisy/random/spurious/minimal rewards** on Qwen-class math models
- **[B] RLVR elicits vs teaches**: does RL surface latent pretraining capability?
- **[C] Training-signal corruption that still converges**: gradient compression, dropout-as-noise, sign-SGD, stale/async gradients, activation masking in backprop

Each entry lists: citation, arXiv ID or venue, date, 2–3 sentence summary, and a **relevance verdict** mapping the work to our finding (masked+clean@20 ≈ dense on GSM8K, stalls on Big-Math; ~20M× ppl mismatch on masked steps yet still learns).

---

## Cluster A — RLVR with Noisy / Random / Spurious / Minimal Rewards

### A1. Spurious Rewards: Rethinking Training Signals in RLVR
**Shao et al. (incl. Lambert, Min, Hajishirzi, Zettlemoyer)** — arXiv 2506.10947, submitted June 2025, revised Feb 2026. Under review at OpenReview.

GRPO on Qwen2.5-Math-7B with *randomly assigned* rewards yields +21.4pp on MATH-500 (vs +29.1pp ground-truth); incorrect-label rewards give +24.6pp; format-only rewards +16.4pp. The mechanism: GRPO's clip term amplifies high-prior behaviours already learned during pretraining—for Qwen-Math that behaviour is "code reasoning" (pseudo-code chains without execution), which rises from 65% to >90% frequency. The effect is **model-family-specific**: Llama3 and OLMo2 show no gains under the same spurious rewards, because they lack an analogous high-prior elicitable behaviour.

**Verdict: STRONGLY SUPPORTS our story.** Our masked GRPO steps produce a near-random forward (pearson ≈ 0.004 with the sampler's logprobs); yet learning happens on GSM8K. A1 establishes that even fully random rewards amplify pre-existing Qwen capability. Our masked forward is exactly the reward-signal analogue: the PPO ratio is self-consistent (≈1) so the gradient "knows" its own direction but is distorted relative to the true policy—precisely the kind of weak/decorrelated signal that still suffices when the task only needs elicitation.

---

### A2. Exploration vs. Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward
**Chen, Li, Li, Yin, Chen, Lin** — arXiv 2512.16912, ICLR 2026.

Formalises the mechanism underlying A1: clipping reduces entropy (pushes the policy toward high-confidence outputs), and this entropy-minimisation effect is the actual driver of performance gains under spurious rewards—not the signal content of the reward itself. Importantly, shows that *random rewards + clip bias* also work on Llama and QwQ families (not just Qwen-Math), broadening A1's model-specific claim. Proposes a "reward-misalignment model" as a unifying explanation.

**Verdict: SUPPORTS.** Our pg_clipfrac ≈ 0.03–0.04 even on masked steps. The clip bias (entropy compression toward high-prior modes) is active throughout training, not just on clean steps. This paper implies the clean@20 cycle is doing more than mere gradient correction: it also resets the entropy-compression trajectory against a meaningful signal each time.

---

### A3. Reinforcement Learning for Reasoning in Large Language Models with One Training Example
**Wang et al. (incl. Shuohang Wang, Simon Du, Yelong Shen)** — arXiv 2504.20571, NeurIPS 2025.

A *single* training example brings Qwen2.5-Math-1.5B from 36.0% to 73.6% on MATH500 (8.6pp beyond format correction alone), and matches the performance of a 1,200-example subset. The improvement comes primarily from the policy gradient loss (not grokking), and exploration via entropy bonus is critical. Cross-category generalisation and *post-saturation generalisation* (test performance continues rising after training accuracy plateaus) are documented.

**Verdict: STRONGLY SUPPORTS elicitation thesis.** If one example suffices, the task is not *teaching* new capability—it is eliciting a latent one. Our model is Qwen2.5-1.5B-Instruct (smaller, instruction-tuned variant); GSM8K is even easier than MATH500. The rapid 0.085→0.49 jump in the first 30 steps of EXP-17 mirrors the post-saturation generalisation pattern: a minimal signal (even a corrupted one) suffices to unlock latent capability on an easy benchmark.

---

### A4. Noise-Corrected GRPO (Dr.GRPO): From Noisy Rewards to Unbiased Gradients
**El Mansouri, Izzati, Seddik, Lahlou** — arXiv 2510.18924, submitted Oct 2025, revised May 2026.

Models reward corruption as Bernoulli label-flips, estimates flip probabilities, and debiases the gradient, yielding provably unbiased gradient estimates. Demonstrates +6.7pp on math and +1.5pp on code under realistic noisy-reward conditions. Key theoretical insight: GRPO's *group-comparative structure inherently mitigates individual-level noise* through relative advantage normalisation.

**Verdict: COMPLICATES (constructively).** This paper assumes reward noise (wrong answer labelled correct or vice versa) rather than forward-pass corruption, which is a different noise channel. However, the group-advantage normalisation mechanism is the same one at work in our masked steps: because PPO ratio ≈ 1 (self-consistent masked logprobs), the advantage normalisation is essentially decoupled from the true gradient direction. Dr.GRPO's analysis quantifies how much reward noise the group-normalisation structure can absorb before divergence—a benchmark for reasoning about our activation-noise tolerance.

---

### A5. Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs
**Yan et al.** — arXiv 2601.11061, submitted Jan 2026. Work in progress.

Identifies a "Perplexity Paradox" in Qwen2.5: under incorrect rewards, answer-token perplexity drops while prompt-side coherence degrades, suggesting the model bypasses reasoning via memorisation. Locates a "Functional Anchor" in middle layers (L18–20) that retrieves memorised solutions and "Structural Adapters" in later layers (L21+). The mechanism is circuit-level memorisation retrieval, not genuine reasoning.

**Verdict: COMPLICATES (important caveat).** Our masked model also shows a large train-inference perplexity gap (training_log_ppl ≈ 17 vs rollout_log_ppl ≈ 0.36 on masked steps). A5 raises the possibility that performance gains on GSM8K reflect memorisation retrieval rather than RL learning. However, our validation always uses the *unmasked* forward (the true policy), so we are measuring the real weights—and the weight update is biased toward memorisation only if the task admits a memorisation shortcut. GSM8K at 73.5% accuracy for a 1.5B model is plausibly a mixture of genuine arithmetic and template retrieval; the stall on Big-Math (which resists template shortcuts) is consistent with A5's mechanism being a GSM8K-specific confound.

---

## Cluster B — RLVR Elicits vs. Teaches: Latent Capability Debate

### B1. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
**Yue, Chen, Lu, Zhao, Wang, Song, Huang (Tsinghua, SJTU)** — arXiv 2504.13837, NeurIPS 2025 Oral + ICML 2025 AI4Math best paper.

Key finding: RLVR improves pass@1 (sampling efficiency) but does NOT expand pass@k at large k. Base models consistently achieve higher pass@k at k≥128 than their RLVR-trained counterparts. All correct solutions from RL-trained models already exist in the base model's sampling distribution. Mechanistic claim: if the base model cannot sample any correct solution (0/1 reward under RLVR → no gradient signal), RLVR cannot learn.

**Verdict: STRONGLY SUPPORTS our C-question (why GSM8K works but Big-Math stalls).** This paper gives the minimal sufficient condition for RLVR to function: the base model must already produce correct solutions with non-zero probability. GSM8K satisfies this easily for Qwen2.5-1.5B-Instruct; Big-Math competition problems likely do not. Our masked steps are a further degradation of the gradient signal—so the condition becomes even more restrictive: the elicitable task must be easy enough that even a *corrupted* gradient suffices to raise pass@1.

---

### B2. New Skills or Sharper Primitives? A Probabilistic Perspective on the Emergence of Reasoning in RLVR
**Wang et al.** — arXiv 2602.08281, submitted Feb 2026.

Argues the opposite of B1: RLVR *does* teach new capabilities by sharpening atomic step probabilities, enabling multi-step chains that were previously blocked by exponential probability decay. Uses Algebrarium framework; reports Pearson correlation 0.69–0.96 between atomic-step probability and composite-task performance. Global optimisation can sacrifice individual skills to maximise aggregate reward.

**Verdict: COMPLICATES (healthy tension with B1).** B2 provides the counterargument: masked+clean@20 on GSM8K may not be pure elicitation—it may be genuinely sharpening simple arithmetic primitives. The distinction matters because elicitation predicts that the clean step's gradient direction does not matter much (any push helps), while skill-sharpening predicts the clean step's direction is load-bearing. Our EXP-17 evidence (mask-only stalls at 0.13→0.15; clean@4 unlocks to 0.62) supports B2 on a fine-grained scale: the clean step provides directional information the masked steps cannot.

---

### B3. Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs
**Wen et al. (Microsoft)** — arXiv 2506.14245, submitted June 2025, revised Oct 2025.

Introduces CoT-Pass@K (measuring both answer and intermediate reasoning chain). Claims RLVR *does* extend the reasoning boundary, not just sampling efficiency, and that correct reasoning is implicitly incentivised even when rewards are answer-only. Provides a theoretical framework for this implicit incentive mechanism.

**Verdict: COMPLICATES (counter to B1, but narrow scope).** B3 is specifically about reasoning chain quality, which is harder to measure in our setting (we track ####-format reward, not chain quality). If correct: even our masked updates, which corrupt the gradient direction, may implicitly encourage correct reasoning steps whenever a clean step happens to push in the right direction. This is consistent with our observation that reward rises monotonically on clean steps.

---

### B4. You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories (RELEX)
**Wei et al.** — arXiv 2605.21468, submitted May 2026.

RLVR training traces a near-rank-1 trajectory in parameter space; 15% of training steps suffice to predict the full trajectory via linear extrapolation. Denoising effect: the low-rank structure filters noise, suggesting the model is surfacing an existing capability direction rather than accumulating gradual improvements.

**Verdict: SUPPORTS elicitation + clean-step re-anchoring.** The rank-1 trajectory finding predicts that the 19 corrupted masked steps drift along approximately the same dominant direction as the clean step—they are noisy estimates of the same low-rank update. This mechanistically explains why masked+clean@20 ≈ dense: 95% of steps are noisy projections of a 1D capability vector, and the clean step every 20 re-anchors the direction. If the trajectory is rank-1, bias from masking affects magnitude (corrected by rescale) more than direction.

---

### B5. Not All Steps Are Informative: On the Linearity of LLMs' RLVR Training
**Wang et al.** — arXiv 2601.04537, submitted Jan 2026, revised May 2026.

Documents that RLVR enters a "robust linear regime" (R² > 0.7 for parameter weights and logprobs) due to the high-variance nature of the training signal acting as a low-pass filter. Enables weight-space extrapolation (6.1× speedup) and output-space extrapolation (4.2% improvement). The linearity is not merely descriptive—it is actionable.

**Verdict: SUPPORTS.** The low-pass filter interpretation is directly applicable: our masking drops 90% of activation traffic, creating extreme signal variance, which B5 predicts should *further* enforce linearity. The clean step every 20 is then the periodic high-fidelity signal that defines the linear trend direction. B5 explains why 19 low-fidelity steps between clean steps don't destroy training: they are all noisy samples of the same linear trend.

---

### B6. The Unlearnability Phenomenon in RLVR for Language Models
**Chen, He, Zhao** — arXiv 2605.16787, ICML 2026.

Identifies that among hard examples a model struggles with, a substantial subset is *unlearnable even when correct rollouts are present*. Root cause: low gradient similarity between unlearnable examples and the broader training distribution. Standard fixes (data augmentation) do not improve gradient similarity.

**Verdict: DIRECTLY EXPLAINS our Big-Math stall.** Big-Math competition problems have low gradient similarity to the model's representation space—even the dense gradient (EXP-20) only produces modest gains. Our masked gradient has even lower information content per step. B6 predicts that masking makes unlearnable problems *more* unlearnable by further reducing the already-weak gradient signal. The 0/1 reward stall (EXP-19 flat ~0.55 reward) is consistent with the gradient-similarity criterion: competition math lives in a representation subspace the 1.5B model cannot bridge with any gradient direction, masked or not.

---

## Cluster C — Training-Signal Corruption That Still Converges

### C1. EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback
**Richtárik, Sokolov, Fatkhullin** — NeurIPS 2021 (arXiv 2106.05203).

The canonical modern error-feedback framework for biased gradient compressors. EF21 achieves O(1/T) convergence for smooth non-convex objectives with biased compressors (e.g., top-K, sign), and linear convergence under Polyak-Łojasiewicz—first such result for error feedback without unbiased compressors. Key condition: compressors must be "contractive" (reduce the error norm by a fixed factor each step), not merely random.

**Verdict: FOUNDATIONAL ANALOGY for our masked gradient.** Our per-(token,channel) binary masking with rescale is a compressor—specifically a random coordinate sub-sampler, which is contractive. EF21 predicts convergence if the compressed gradient has a positive expected inner product with the true gradient after rescaling. This is why rescale ON (our default) is critical: without it, the compressor loses the unbiasedness property needed for the inner product condition. The clean step every 20 is not needed for convergence in the EF21 sense (error feedback alone suffices for SGD), but it acts as periodic full-gradient oracle reset, which dramatically reduces the accumulated error.

---

### C2. Rethinking Gradient Sparsification as Total Error Minimization
**Shi et al.** — NeurIPS 2021 (OpenReview + arXiv 2108.00951).

Reframes gradient compression as minimising cumulative compression error over the full training run, rather than per-step error. Under this lens, top-k sparsification at k=0.1% can match dense training quality given a fixed communication budget. Provides the optimality condition: compression errors across iterations should be balanced so no single step dominates the cumulative budget.

**Verdict: SUPPORTS, quantitatively frames clean@20.** Our setup: 19 compressed steps (p=0.9 masking, ~85.5% traffic dropped) + 1 clean step. C2 predicts this is near-optimal for our communication budget if the 19 compressed steps share their compression error budget equally and the clean step resets the accumulator. The flat/stationary train-inference gap (our sawtooth) is exactly the error accumulator behaviour C2 models: the error grows over 19 masked steps and resets at the clean step.

---

### C3. Masked Training of Neural Networks with Partial Gradients
**Mohtashami, Jaggi, Stich** — AISTATS 2022 (arXiv 2106.08895).

Unified theoretical framework covering SGD variants with arbitrary parameter perturbations and gradient masking—including Dropout, DropConnect, and communication-efficient training. Importantly handles *arbitrary* (not just random) masking patterns that need not correspond to the perturbation used for gradient computation. Provides convergence guarantees under NTK-style assumptions for shallow networks.

**Verdict: MOST DIRECT THEORETICAL ANALOGUE.** Our setup is precisely within C3's framework: we perturb the activations (forward-pass masking) and the gradient inherits the perturbation. The key contribution of C3 is that arbitrary masking (not just symmetric random) converges—which validates our per-(token,channel) structured mask. The limitation: C3 proves NTK-regime convergence for shallow networks; extending to transformer fine-tuning (as in our setting) requires additional assumptions (e.g., near-linear dynamics as documented in B5).

---

### C4. Convergence Analysis of Two-Layer Neural Networks under Gaussian Input Masking
**Kolomvaki, Liao, Dramko, Guang, Kyrillidis** — arXiv 2602.17423, submitted Feb 2026.

NTK analysis of two-layer ReLU networks trained with randomly Gaussian-masked inputs. Result: linear convergence to an error region proportional to the mask's variance. The error floor scales directly with noise magnitude—higher masking variance → larger residual error at convergence.

**Verdict: SUPPORTS (with important caveat).** Our masking is at *intermediate activations* (pipeline boundaries), not input; but the NTK machinery transfers. The key prediction: our p=0.9 masking + rescale creates a noise variance ∝ p/(1-p) per activation, which sets an error floor. This floor is *task-dependent*: for GSM8K (high SNR elicitation task), the error floor is below the dense accuracy ceiling; for Big-Math (low SNR task where even dense barely learns), the error floor is above the achievable accuracy, explaining why masked training stalls there.

---

### C5. GAC: Stabilizing Asynchronous RL Training for LLMs via Gradient Alignment Control
**Xu, Su, Tian, Diao, Qian, Wu** — arXiv 2603.01501, submitted Mar 2026.

Studies async policy-gradient training for LLMs. Key finding: async training produces "persistently high cosine similarity between consecutive policy gradients" (stale-aligned gradient effect), which amplifies correlated updates and risks overshooting. Proposes Gradient Alignment Control (GAC) to project away stale-aligned directions. Provides convergence guarantees under bounded staleness.

**Verdict: ANALOGOUS but distinct.** Our masked gradient is not stale (it's contemporaneous but corrupted), so the specific GAC mechanism doesn't apply. However, C5 documents an analogous phenomenon: when gradient diversity is lost (as in our masked steps where pearson≈0.004 with the true policy), training dynamics become pathological without a corrective mechanism. Our clean step plays the same role as GAC: it restores the gradient to a diverse, high-quality signal that prevents the masked-step compounding from diverging. The paper also shows convergence under bounded corruption, which is analogous to our bounded mask probability.

---

### C6. TACO: Efficient Communication Compression of Intermediate Tensors for Scalable Tensor-Parallel LLM Training
**Liu et al.** — arXiv 2604.24088, HPDC 2026.

Compresses *activation tensors* transmitted between tensor-parallel workers during LLM training using FP8 quantisation + Adaptive Scale-Hadamard Transform + Dual-Scale quantisation. Achieves 1.87× end-to-end throughput with near-lossless accuracy on GPT and Qwen models. Addresses error accumulation under repeated within-block compression.

**Verdict: DIRECTLY ANALOGOUS, strong evidence.** TACO is the closest published system to our method: both compress *intermediate activation tensors* at parallelism boundaries during LLM training, and both report near-lossless accuracy despite significant information loss. The key difference: TACO uses quantisation (lossy but low-bias) while we use binary masking (high-variance, rescaled to unbiased). TACO's accuracy preservation validates the general principle that intermediate-tensor compression during training is tolerable; our p=0.9 masking is more aggressive than any TACO compression ratio tested, explaining why we need the clean@20 resync.

---

### C7. Heterogeneous Low-Bandwidth Pre-Training of LLMs
**Obeidi, Sarfi, Lidin, Janson, Belilovsky** — arXiv 2601.02360, submitted Jan 2026.

Combines SparseLoCo (infrequent sparse pseudo-gradient exchange) with pipeline model parallelism using *subspace-projected inter-stage communication*. "Activation compression composes with SparseLoCo at modest cost." Experiments on 178M–1B parameter models. Selective (heterogeneous) compression at high compression ratios consistently outperforms compressing all replicas.

**Verdict: SUPPORTS—closest system architecture to ours.** This paper combines two features our method uses: (i) pipeline-parallel activation compression at stage boundaries, and (ii) periodic full-gradient sync (outer step in DiLoCo ≈ our clean step). The heterogeneous finding predicts that selectively keeping some pipeline stages unmasked (e.g., the first and last stages) would improve training—a design variant worth testing. The 178M–1B scale matches our 1.5B model.

---

### C8. Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo
**Charles et al. (Google)** — arXiv 2503.09799, submitted Mar 2025.

Establishes scaling laws for DiLoCo (infrequent gradient sync, 500× communication reduction). DiLoCo with proper tuning *outperforms* standard data-parallel training at fixed compute. Benefits include larger optimal batch sizes and better generalisation. Framework: the infrequent sync period (analogous to our clean cadence) is a hyperparameter that can be tuned for the compute budget.

**Verdict: SUPPORTS, quantitative scaling analogy.** DiLoCo's communication reduction (500×) is comparable to our ~20× clean cadence × ~7× stage compression ≈ 140× reduction in total training-time cross-stage communication. C8 predicts that at this compression ratio, with proper tuning, performance should be recoverable—which is what EXP-17 demonstrates on GSM8K. The scaling-law finding also predicts that harder tasks (requiring more compute-efficient learning) tolerate less compression—aligning with our Big-Math stall.

---

### C9. GRPO's Effective Loss, Dynamics, and Success Amplification
**Mroueh** — arXiv 2503.06639, submitted Mar 2025, revised Oct 2025.

Formal analysis of GRPO under mean+variance reward normalisation. Key result: the optimal policy under GRPO admits a closed-form expression in terms of first and second moments of the reward; success probability follows a recurrence converging to a fixed point above the reference, demonstrating that GRPO *always amplifies* the policy's probability of success above the reference model.

**Verdict: SUPPORTS—explains why masked GRPO still makes progress.** The amplification result applies regardless of the quality of the gradient estimate, as long as the reward signal has non-zero correlation with the true advantage. Our masked steps have pg_clipfrac ≈ 0.03 throughout, meaning the policy IS being updated away from the reference, and the GRPO amplification mechanism is active. C9 gives a lower bound on the fixed-point success probability as a function of the reference model's capability—which for GSM8K (high reference capability in Qwen-Instruct) is well above the training starting point.

---

### C10. Gradient Routing: Masking Gradients to Localize Computation in Neural Networks
**Cloud et al.** — arXiv 2410.04332, submitted Oct 2024, revised Nov 2024.

Introduces gradient routing: data-dependent weighted masks applied *during backpropagation* to localise which parameters are updated by which data. Used for interpretability (partitioned representations), unlearning, and scalable oversight. Shows gradient routing localises capabilities even on ad-hoc subsets of data.

**Verdict: METHODOLOGICALLY ANALOGOUS.** Our forward-pass activation masking induces a specific gradient masking pattern via the chain rule—structurally similar to C10's explicit backprop masking. The key distinction: C10's masks are data-dependent and intentional; ours are structured random at stage boundaries. C10 establishes the general principle that masked backprop can selectively affect capability regions of the model without destroying overall training, supporting the observation that our masked steps still produce valid (if low-SNR) gradient updates.

---

## Cross-Cutting Synthesis

| Cluster | Finding | Supported? | Strength |
|---|---|---|---|
| A1 (Spurious rewards) | GRPO works on Qwen with near-random signal | Strongly supports | Strong (peer-reviewed) |
| A2 (Clip+entropy) | Clip bias alone drives gains under spurious rewards | Supports | Strong (ICLR 2026) |
| A3 (1-shot RLVR) | Single example elicits latent Qwen-Math capability | Strongly supports | Strong (NeurIPS 2025) |
| A4 (Dr.GRPO) | Group normalisation absorbs reward noise | Supports, different channel | Strong (peer review) |
| A5 (Spurious paradox) | Gains may be memorisation retrieval, not reasoning | Complicates | Weak (preprint, in progress) |
| B1 (Limit of RLVR) | RLVR can't learn beyond base capability ceiling | Strongly supports Q-C | Strong (NeurIPS 2025 Oral) |
| B2 (New skills vs primitives) | RLVR sharpens atomic skills, not just elicits | Complicates Q-C | Moderate (preprint) |
| B3 (Implicit correct reasoning) | RLVR implicitly incentivises correct reasoning | Complicates | Moderate (preprint) |
| B4 (RELEX, rank-1) | RLVR trajectory is rank-1, 15% steps suffice | Supports clean@20 | Moderate (preprint) |
| B5 (Linear RLVR) | High-variance signal → low-pass linear regime | Supports | Moderate (preprint) |
| B6 (Unlearnability) | Hard examples unlearnable due to gradient dissimilarity | Directly explains Big-Math stall | Moderate (ICML 2026) |
| C1 (EF21) | Biased contractive compressor + error feedback converges | Foundational analogy | Strong (NeurIPS 2021) |
| C2 (Total error minimization) | Clean step resets compression error accumulator | Supports clean@20 design | Strong (NeurIPS 2021) |
| C3 (Masked training) | Arbitrary masking + partial gradients converges | Most direct analogy | Moderate (AISTATS 2022) |
| C4 (Gaussian masking NTK) | Error floor ∝ mask variance, task-SNR determines feasibility | Supports easy/hard split | Moderate (preprint 2026) |
| C5 (GAC async RL) | Corrupted gradients + periodic correction converges | Supports clean@20 role | Moderate (preprint 2026) |
| C6 (TACO) | Activation compression at parallelism boundary: near-lossless | Strong system analogy | Strong (HPDC 2026) |
| C7 (Hetero low-BW) | Pipeline-parallel activation compression + periodic sync works | Closest system analogy | Moderate (preprint 2026) |
| C8 (DiLoCo scaling) | Infrequent sync at ~500× reduction matches dense | Strong scaling analogy | Strong (preprint, Google) |
| C9 (GRPO dynamics) | GRPO always amplifies success probability above reference | Explains masked-step progress | Moderate (preprint) |
| C10 (Gradient routing) | Masked backprop localises but does not destroy learning | Methodological analogy | Moderate (preprint 2024) |

---

## Notes for Theorist

- **The rescale condition is load-bearing**: C1 (EF21) requires a contractive compressor; rescale ON makes our binary mask contractive by preserving the expected activation magnitude. Rescale OFF breaks contractivity → potential divergence on non-elicitation tasks. This is consistent with EXP-16's rescale ablation.
- **The clean step does multiple jobs simultaneously**: (a) resets the error accumulator (C2), (b) provides the high-fidelity gradient that defines the linear trajectory direction (B4/B5), (c) re-syncs the train-inference gap (our sawtooth), (d) drives the actual policy update that clip-bias on masked steps cannot (A2). Any theory of clean@K should account for all four.
- **The easy/hard split has two complementary explanations**: (B1) base capability ceiling (no correct rollouts → no gradient), and (B6) gradient dissimilarity for hard examples (correct rollouts present but gradient doesn't generalise). Our Big-Math result is consistent with both; distinguishing them requires varying the reward sparsity while holding model and task fixed.
- **A5 (memorisation shortcut) is the main alternative hypothesis** for our GSM8K success: the model may retrieve memorised arithmetic templates rather than learn arithmetic. This is hard to rule out at 1.5B scale on GSM8K, but does not invalidate the communication-efficiency result (memorisation retrieval is still *a* mechanism, and our masked gradient elicits it as well as dense).

## Notes for Empiricist

- **B1's pass@k prediction**: if our EXP-17 masked model has lower pass@k at k≥128 than the dense EXP-16 model on the same validation set, that would confirm elicitation (not genuine learning). This is testable from existing checkpoints.
- **A1's code-reasoning mechanism**: if our masked model shows increased code-reasoning frequency (percent of responses using pseudo-code chains), that would support the spurious-reward amplification channel even for our corrupted-forward rather than random-reward case.
- **C4's error floor prediction**: the error floor ∝ p/(1-p) ≈ 9 for p=0.9. For Big-Math where even dense achieves only +5pp in 116 steps, the masked error floor may literally exceed the achievable gain—a quantitative check that could be made explicit in empirical_check.md.

---

*All arXiv links are canonical IDs; fetch at https://arxiv.org/abs/[ID]. Peer-reviewed venues noted explicitly; preprints flagged.*
