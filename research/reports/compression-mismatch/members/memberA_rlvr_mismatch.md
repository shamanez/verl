# Member A Report — The Train–Inference Mismatch in RLVR / RL Fine-Tuning of LLMs

**Author:** Member A (general / standard engine-mismatch literature)
**Date:** 2026-06-23
**Scope:** The *standard, well-known* train–inference (a.k.a. rollout–training, engine, deployment-gap) mismatch in RLVR, where the **training engine** (FSDP/Megatron/DeepSpeed forward pass) and the **rollout/inference engine** (vLLM/SGLang) compute *different* token probabilities for the *same* policy weights. **Out of scope (Member B / our project):** quantized rollouts in depth, and our project's activation-compression-at-PP-boundaries mismatch.

> **Note on shared task list:** The `TaskList`/`TaskGet`/`TaskCreate` tools named in the brief are not present in this environment (only `TaskStop` + scheduled-tasks are exposed). My top-5 findings for Member B / the lead are appended at the bottom of this file under "Top-5 findings (for the shared list)" so they are still recorded.

---

## 1. Definition — what exactly differs between the two engines

**Standard definition.** Modern RLVR frameworks use a *hybrid* design: a highly optimized **inference engine** (vLLM, SGLang) generates rollouts, and a separate **training engine** (FSDP / Megatron / DeepSpeed) computes gradients. Even though both are instantiated from the **same parameters θ**, they produce numerically different token probabilities. Writing π_rollout = π_vllm (a.k.a. μ, the *behavior* policy) and π_train = π_fsdp (a.k.a. π, the *target* policy), the mismatch is the event π_vllm(a|s) ≠ π_fsdp(a|s) despite identical θ. Because rollouts are *sampled* from μ but gradients are taken w.r.t. π, on-policy RL silently becomes **off-policy** with a biased gradient (Yao et al. 2025; Liu et al. 2025; Qi et al. 2025).

The sharpest one-line statement is from Yao et al. (OPT-ML 2025): *"despite sharing the same parameters θ, these policies can produce significantly different token probabilities, making the training implicitly off-policy."* For some tokens the disagreement is total — **π_vllm(a)=1 while π_fsdp(a)=0** (contradictory top-1 predictions).

**Concrete list of what differs between the two code-paths** (synthesized from Yao et al. 2025, Qi et al. 2025, the "Diagnosing TIM" paper 2026, and Thinking Machines / He et al. 2025):

| Axis | Inference engine (vLLM/SGLang) | Training engine (FSDP/Megatron) | Why it diverges |
|---|---|---|---|
| **Kernels / operators** | Inference-optimized (e.g. FlashInfer, custom attention, fused MoE) | Training kernels (FlashAttention bwd-capable, HF/Megatron impls) | Different libraries → different rounding even at the *same* dtype |
| **Numerical precision** | Often BF16 (sometimes FP8/INT8 quantized weights/KV) | BF16/FP16/FP32 master weights | BF16's 7 mantissa bits magnify per-op rounding; quantization amplifies it |
| **Attention impl / KV-cache** | Paged KV-cache, prefix caching, chunked prefill; KV may be split across blocks | Dense contiguous attention over the full sequence | KV splitting + paging change reduction order |
| **Batching / chunked-prefill** | Dynamic batch size from server load; **autoregressive token-by-token** decode | **Whole sequence processed in parallel** (teacher-forcing) | Batch-size-dependent tiling → different GPU launch configs → different reduction order |
| **Reduction / accumulation order** | Atomic adds, batch-dependent tiling, KV-split reductions | Different tiling/reduction | FP addition is **non-associative**; order changes the bits |
| **Sampling path** | Temperature/top-p applied in engine; logprobs may be "adjusted" vs "true" | exact softmax logprobs | vLLM historically returned adjusted, not raw, sampling probs |

**Crucial root-cause nuance (see §3):** the mismatch is **not** a weight difference — it is *two different implementations of the same math*. He et al. (Thinking Machines, 2025) localize it further: the dominant driver is **batch-size-dependent reduction order**, not floating-point precision per se — *"It is impossible to get bitwise identical results between training and inference if we can't even get bitwise identical results from two identical inference requests."*

---

## 2. Measurement — how people quantify it

Standard instruments (from Yao et al., Liu/Yingru Li, ms-swift docs, Thinking Machines, "Diagnosing TIM"):

1. **Token-level log-prob difference** δ_t = log π_train(a_t|s_t) − log π_rollout(a_t|s_t). Report **mean |δ_t|** (small) and **max |δ_t|** (large) over a batch. Argmax-flip rate = fraction of positions where the two engines' top-1 token differs.
2. **KL divergence** between the two engines' distributions, KL(π_rollout‖π_train) or KL[μ‖π], using K1 = log r, K2 = ½(log r)², or the K3 estimator (DeepSeek's low-variance, non-negative estimator). verl/ms-swift log this under `rollout_correction/` / `actor/ppo_kl`.
3. **Importance-sampling (IS) ratio distribution** ρ = π_train/π_rollout, token-level and sequence-level. Track `is_weight_mean` (ideal = 1.0), the spread, and `clipped_frac`. The **sequence-level ratio** π(y|x)/μ(y|x) is the unbiased per-response IS weight and is *notoriously high-variance* over long sequences.
4. **χ²-divergence** (`chi2_token`, `chi2_seq`) — directly measures IS-weight variance; the key theoretical handle for the *variance* failure mode (Yingru Li). **"A small TV distance can hide a massive χ²-divergence."**
5. **Effective Sample Size (ESS)** — fraction of samples actually contributing after weighting; ideal = 1.0, collapses toward 0 as variance explodes.
6. **Sequence mismatch vs length** — slope of log-ratio against response length; a steep negative slope means the mismatch grows with length.

### Concrete magnitudes reported (verified)

- **BF16 vs FP16 KL[μ‖π]** (Qi et al. 2025, Fig 2, DeepSeek-R1-Distill-Qwen-1.5B, temp 1.0, no top-p): **BF16 KL = 7.64** (seq-level log-ratio-vs-length slope = **−1.01**) versus **FP16 KL = 0.32** (slope = **−0.07**). FP16 ≈ **24× smaller** mismatch; BF16 mismatch grows *exponentially* with response length.
- **Per-precision KL between vLLM and FSDP** (Yao et al. 2025, Fig 3, Qwen2.5-0.5B, GSM8K): **BF16 ≈ 10⁻⁴–10⁻³**, **FP8 ≈ ~10⁻²**, **INT8 ≈ 2×10⁻²–10⁻¹**.
- **Token-prob difference, large model** (Yao et al. 2025, Fig 1, DAPO Qwen2.5-32B, 4×8×H100): **max-over-tokens ≈ 10⁰ (= 1.0)**, **mean-over-tokens ≈ 10⁻³**, and it **grows over training steps even after casting the lm_head to fp32**.
- **Online RL KL, sampler vs trainer** (He et al. / Thinking Machines, 2025): with IS correction, KL **≈ 0.001 with occasional spikes**; with **batch-invariant kernels and no IS, KL = exactly 0**.
- **Token-level disagreement table** ("Diagnosing TIM" 2026, Table 1): mean |δ_t| small per batch, **max |δ_t| ≈ 1.0**, individual example δ_t = −0.133, plus argmax flips.

---

## 3. Root cause = two distinct engines, NOT a weight difference

This is the load-bearing claim for our project's analogy. The strongest sources:

- **Yao et al. (OPT-ML 2025), "On the Rollout-Training Mismatch in Modern RL Systems."** They explicitly *tried the weight/precision hypothesis and ruled it out at the system level*: they **patched vLLM to (i) expose true sampling probabilities (not adjusted ones) and (ii) cast its lm_head to fp32 to match HuggingFace precision** — *"the rollout–training mismatch persists even after these fixes, suggesting the problem is fundamental to hybrid backend designs."* Same θ, two engines ⇒ mismatch. (URL below.)
- **Qi et al. (2025), "Defeating the Training-Inference Mismatch via FP16."** *"Modern RL frameworks often use different engines or optimized kernels for training and inference. Even if both are configured to use BF16, subtle differences in their implementation (e.g., CUDA kernel optimizations, parallel strategies) can lead to different rounding errors… these small discrepancies accumulate over a sequence of tokens during autoregressive sampling."* Mathematically μ = π; numerically they diverge.
- **He et al. (Thinking Machines, 2025), "Defeating Nondeterminism in LLM Inference."** Pins the dominant cause to **batch-size-dependent reduction order** in non-batch-invariant kernels (atomic adds, batch-dependent tiling), *not* precision alone. Fixing kernels to be batch-invariant ⇒ bitwise-identical sampler and trainer ⇒ **KL = 0, true on-policy**.
- **"Diagnosing TIM" (arXiv 2605.14220, 2026).** Two named sources: (1) **model/kernel implementation divergence** (e.g., FlashInfer is inference-only), (2) **non-deterministic numerical behavior** (atomic additions, batch-dependent tiling, FP non-associativity, reduction-order variation). Their fix, **VeXact**, *unifies the HuggingFace model impl across rollout and training* + deterministic batch-invariant kernels.

**Bottom line:** the canonical mismatch is an *implementation/codepath* difference between two engines holding identical weights — exactly the framing we want to import into our compression setting (where our mismatch is instead a deliberate codepath change at PP boundaries during training).

---

## 4. Typical gap magnitude — tolerable vs fatal

A consistent picture across sources: the **per-token mismatch is tiny on average but heavy-tailed**, and harm is **dynamic and length-amplified**, not a fixed constant.

- **Tolerable (silent) regime.** BF16 dense rollouts: KL ~10⁻⁴–10⁻³, mean |δ_t| small. Training often *looks* fine for hundreds of steps. But "Diagnosing TIM" shows this is dangerous: in **recompute mode**, K1/K3 KL estimators stay **"nearly flat during the first ~700 steps"** while reward is already silently degrading — because **PPO clipping asymmetrically transforms symmetric numerical noise into a sign-dependent skew** in the advantage-weighted update. KL is an *insufficient* early detector; contribution-level (zero-centered loss) analysis is needed.
- **Fatal (catastrophic) regime.** Triggered by (a) larger base mismatch — **quantized rollouts (FP8/INT8)** push KL to 10⁻²–10⁻¹; (b) **long responses** (mismatch grows ~exponentially with length under BF16; seq-log-ratio slope −1.01); (c) **MoE routing volatility** (~10% of experts flip per gradient step, Zheng et al./GSPO); (d) **extended training** (the gap and gradient noise escalate *in tandem* over steps — "Beyond Precision" 2026).

**Documented collapse magnitudes (verified):**
- **FP8 rollout collapse** (Yao et al. 2025, Fig 3): the *recompute* method's accuracy decays toward **~0 by ~step 300**; **PPO-IS and vanilla-IS reach near-0 accuracy for INT8** rollouts. TIS keeps INT8/FP8 close to BF16 (~0.5 on GSM8K).
- **Vanilla GRPO BF16 collapse** (Qi et al. 2025, Fig 3, MATH "perfectible" sanity set, DeepSeek-R1-Distill-Qwen-1.5B): peaks at only **73% (VeRL) / 84% (Oat)** then degrades; token-TIS prolongs but collapses after **82% / 88%**; **GSPO BF16 gradient norm went NaN after 1200 steps** in VeRL.
- **Online RL reward collapse without IS** (He et al. 2025): reward **collapses ~step 318** with a simultaneous KL spike.
- **MoE REINFORCE collapse** ("Diagnosing TIM" 2026, Fig 2, Qwen3-30B): vLLM reward drops **0.574 → 0.255 by step 280** then to near-zero; the exact-kernel baseline holds **0.753 train / 0.534 val**.
- **GRPO collapse** ("Diagnosing TIM" 2026, Fig 3): vLLM-recompute degrades **0.87 → 0.40 over 650 steps**, then catastrophic collapse after **step 1665**; exact baseline holds **~0.93**.

**Heuristic threshold seen in practice:** ms-swift / "Diagnosing TIM" default **token truncation τ = 2** and **sequence-rejection τ_seq ≈ 0.001**; verl's TIS example cap **C = 10.0**; Qi et al.'s experiments use IS clip **C = 3** and clip_higher = 0.28. These are the empirical "where it starts to bite" knobs.

---

## 5. Standard fixes — and how robust each is

### (A) Importance-sampling (IS) corrections — algorithm-level, cheap, *correct the gradient only*

- **Token-level Truncated IS (TIS)** — Yao et al. 2025. Multiply each token's PPO/REINFORCE term by `min(π_train/π_rollout, C)`:
  `∇J ≈ E_{a~μ}[ min(π_fsdp(a)/π_vllm(a), C) · A · ∇log π_fsdp(a) ]`.
  One-sided upper cap C trades a little bias for bounded variance. **Cheap, robust, the de-facto baseline.** Adopted in VeRL, OpenRLHF, SkyRL, OAT, Open-Instruct, slime. *Limitation:* token-level correction leaves the **prefix/state-distribution mismatch** uncorrected → **O(T²·Δ_max) residual bias** (Yingru Li); ultimately collapses under large gaps (FP8) or long runs.
- **Sequence-level IS (MIS / masked IS)** — Liu et al. 2025 ("When Speed Kills Stability"). Use one ratio ρ = π(y)/μ(y) for the whole response; **truncate** `min(ρ,C)` or **mask** `ρ·1{ρ≤C}` (drop the sequence). Unbiased in principle ⇒ more stable than token-TIS, but **high variance ⇒ slow convergence**, and **still leaves a deployment gap** (Qi et al.: seq-MIS peaks 95% vs 99% FP16; AIME24 34% vs 39%).
- **GSPO (Group Sequence Policy Optimization)** — Zheng et al. (Qwen team), arXiv 2507.18071. **Length-normalized** sequence ratio `s_i(θ) = (π_θ(y_i|x)/π_θ_old(y_i|x))^(1/|y_i|)` with sequence-level clipping. Argues token-level IS with **one sample per token-position is statistically meaningless** → high-variance noise that accumulates over length and is *exacerbated by clipping* → **irreversible collapse**. GSPO clips ~**2 orders of magnitude more** tokens than GRPO yet trains more efficiently; **much more tolerant of train–inference precision discrepancy** (can use inference-engine likelihoods directly) and **fixes MoE expert-activation volatility** without routing-replay. Used in **Qwen3** RL.
- **Off-policy / behavior-vs-target framing.** "Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm" (arXiv 2509.24203) gives the theory: GRPO admits a **native off-policy interpretation** with **rollout = behavior policy, training = target policy**; IS + clipping are the principled corrections, justifying data-weighting heuristics. This is the clean conceptual frame for "the data is off-policy because the sampler is a different policy."

**verl's implementation (verified in our checkout, PR #2953, merged 2025-08-26, `[BREAKING][vllm,fsdp]`):**
- Enable with `actor_rollout_ref.rollout.calculate_log_probs=True` (rollout returns its own logprobs as `rollout_log_probs`).
- Set the TIS cap C via `actor_rollout_ref.actor.behav_imp_weight_cap` (PR example **C = 10.0**). (Some docs/forks call this `tis_imp_ratio_cap`; the current code path uses `rollout_log_probs` + a `rollout_correction` config and applies `rollout_is_weights` inside `compute_policy_loss_*` in `verl/trainer/ppo/core_algos.py`.) Default = OFF (`calculate_log_probs: False`), so **with the fix disabled, training is byte-identical to upstream** — directly analogous to our "method OFF ⇒ byte-identical" contract.
- verl also exposes a **`bypass_recomputing_logprobs`** path (`old_log_probs = rollout_log_probs`, the "2-policy" setup) vs the default **recompute** path (training engine recomputes old_log_prob — see (C)).

### (B) Algorithm-level: clipping / off-policy correction

- **PPO/GRPO clipping** partially masks the mismatch by bounding the ratio, but it is **not a fix** — and worse, "Diagnosing TIM" shows clipping is *part of the failure mechanism* in the silent regime (it asymmetrically amplifies the sign-skew from numerical noise). **dual-clip** (`clip_ratio_c = 3.0` default in verl) and **clip_higher** (0.28) bound the upside.
- **Off-policy sequence masking** (slime; DeepSeek-V3.2): mask sequences where KL(π_old‖π_θ) > τ **and** advantage < 0 — drop the high-divergence, negative-reward samples that ignite instability.
- **LR scheduling** ("Beyond Precision," arXiv 2602.01826, 2026): reframes the mismatch as a **dynamic optimization failure** (gradient noise and mismatch escalate together), and uses **response-length as an early-warning signal** to trigger LR decay. Argues precision alone is necessary-but-not-sufficient and that **IS can fail during extended runs**.

### (C) Engine-level: make the two engines numerically consistent (removes the mismatch at its source)

- **Recompute old_log_prob in the training engine** (verl default). Instead of trusting vLLM's logprobs, the trainer recomputes them with its *own* forward pass, so `old_log_prob` and the policy gradient share a codepath. This removes the rollout-engine logprob from the gradient **but does NOT remove the off-policyness** (the *samples* still came from μ), and "Diagnosing TIM" shows recompute can *induce* the silent sign-skew. Cheap, standard, partial.
- **FP16 instead of BF16** (Qi et al. 2025). FP16's **10 mantissa bits vs BF16's 7** = **8× finer precision** (next-representable: 1.000977 vs 1.0078125). Empirically cuts the mismatch ~**24×** (KL 7.64→0.32), eliminates collapse across GRPO/GSPO/TIS/MIS/PG and Dense-14B/MoE/LoRA, lets you use the **plain unbiased policy gradient with no IS**, and **closes the deployment gap** that IS cannot. Needs loss-scaling (mature, ~1 line). Caveat: necessary-but-not-sufficient per "Beyond Precision."
- **Batch-invariant / deterministic kernels** (He et al., Thinking Machines, 2025; VeXact in "Diagnosing TIM"). Force a single universal reduction strategy independent of batch size ⇒ **bitwise-identical** sampler and trainer ⇒ **KL = exactly 0, true on-policy, no IS needed.** The cleanest fix; some inference-speed cost. Now adopted/implemented in vLLM and SGLang.
- **FP32 inference** (Qi et al. ablation): fully stable but **~3× slower** than FP16/BF16 inference ⇒ impractical at scale.

### Robustness ranking (synthesized)

| Fix | Layer | Cost | Robustness | Note |
|---|---|---|---|---|
| Batch-invariant kernels | engine | med (infer speed) | **Highest (KL=0)** | He et al.; removes need for IS |
| FP16 training | engine | ~0 (loss-scale) | **High** | Qi et al.; also closes deployment gap |
| Recompute old_log_prob | engine | +1 fwd pass | Partial | verl default; can induce silent skew |
| GSPO (seq-level, len-norm) | algo | low | High for MoE/precision | Qwen3; tolerant of engine gap |
| Sequence MIS (mask/truncate) | algo | +25% (extra fwd) | Medium | unbiased but high-variance, deployment gap remains |
| Token TIS (cap C) | algo | +25% (extra fwd) | Medium-low | cheap default; O(T²Δ) bias, fails on FP8/long runs |
| PPO/GRPO clipping alone | algo | ~0 | Low | not a fix; part of silent-failure mechanism |
| FP32 inference | engine | ~3× infer | High | impractical |

---

## Key papers table

| Title | Venue / arXiv | One-line finding | URL |
|---|---|---|---|
| On the Rollout-Training Mismatch in Modern RL Systems (Yao et al.) | OPT-ML 2025 (NeurIPS workshop) | Same θ, two engines ⇒ off-policy; **fp32-lm_head + true-prob vLLM patch fails**; TIS fix; KL 10⁻⁴(BF16)→10⁻¹(INT8) | https://opt-ml.org/papers/2025/paper116.pdf · https://openreview.net/pdf/325f91538e61ba160793adc5029888c00d06fa7a.pdf |
| Your Efficient RL Framework Secretly Brings You Off-Policy RL Training (Yao et al.) | Notion / blog 2025.08 | Public write-up of the above; token-level TIS, PPO-IS vs vanilla-IS | https://fengyao.notion.site/off-policy-rl |
| Defeating the Training-Inference Mismatch via FP16 (Qi et al., Sea AI Lab/NUS) | arXiv:2510.26788 | **BF16 is the root cause**; FP16 cuts KL 7.64→0.32 (~24×), removes collapse + closes deployment gap | https://arxiv.org/abs/2510.26788 |
| When Speed Kills Stability (Liu / Yingru Li) | blog series 2025.09 | SGA framework: token-IS has **O(T²Δ_max) bias**; seq-trunc-IS = controllable bias/variance | https://richardli.xyz/post/rl-collapse-part1/ |
| Group Sequence Policy Optimization (GSPO, Zheng et al., Qwen) | arXiv:2507.18071 | **Length-normalized sequence IS**; token-IS w/ 1 sample = high-var collapse; fixes MoE; used in Qwen3 | https://arxiv.org/abs/2507.18071 |
| Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm | arXiv:2509.24203 | GRPO is natively off-policy; rollout=behavior, train=target; justifies IS+clipping | https://arxiv.org/abs/2509.24203 |
| Defeating Nondeterminism in LLM Inference (He et al., Thinking Machines) | blog 2025 | Batch-invariant kernels ⇒ bitwise sampler=trainer ⇒ **KL=0 true on-policy**, no IS | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ |
| Diagnosing Training-Inference Mismatch in LLM RL (VeXact) | arXiv:2605.14220 (2026) | KL estimators stay flat during **silent** degradation; clipping skews symmetric noise; deterministic-kernel fix | https://arxiv.org/html/2605.14220 |
| Beyond Precision: TIM is an Optimization Problem… LR Scheduling Fixes It | arXiv:2602.01826 (2026) | Precision necessary-but-not-sufficient; dynamic failure; length-triggered LR decay | https://arxiv.org/abs/2602.01826 |
| verl PR #2953 — Rollout-Training Mismatch Fix (TIS) | github verl-project/verl | `calculate_log_probs=True` + `behav_imp_weight_cap` (C, ex.10.0); merged 2025-08-26 | https://github.com/verl-project/verl/pull/2953 |
| ms-swift GRPO Training-Inference-Mismatch docs | modelscope/ms-swift | 4 IS modes (token/seq × truncate/mask), τ=2; logs χ², ESS, IS-weight, K3 KL | https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/AdvancedResearch/training_inference_mismatch.md |
| DAPO (Yu et al.) | arXiv:2503.14476 | clip-higher decoupled clipping; the 32B RL recipe Yao et al. fix with TIS | https://arxiv.org/abs/2503.14476 |

---

## Magnitudes cheat-sheet (concrete numbers with sources)

| Quantity | Value | Setting | Source |
|---|---|---|---|
| KL[μ‖π], BF16 | **7.64** (seq log-ratio slope −1.01) | DeepSeek-R1-Distill-Qwen-1.5B, temp 1.0 | Qi et al. 2025, Fig 2 |
| KL[μ‖π], FP16 | **0.32** (slope −0.07) | same | Qi et al. 2025, Fig 2 |
| FP16 vs BF16 mismatch ratio | **~24× smaller** | same | Qi et al. 2025 |
| KL(vLLM‖FSDP), BF16 | **~10⁻⁴–10⁻³** | Qwen2.5-0.5B, GSM8K | Yao et al. 2025, Fig 3 |
| KL(vLLM‖FSDP), FP8 | **~10⁻²** | same | Yao et al. 2025, Fig 3 |
| KL(vLLM‖FSDP), INT8 | **~2×10⁻²–10⁻¹** | same | Yao et al. 2025, Fig 3 |
| Token-prob diff, max | **~10⁰ (=1.0)** | DAPO Qwen2.5-32B | Yao et al. 2025, Fig 1 |
| Token-prob diff, mean | **~10⁻³** (grows over steps) | same, even w/ fp32 lm_head | Yao et al. 2025, Fig 1 |
| Online-RL KL, with IS | **~0.001** (occasional spikes) | RLVR | He et al. 2025 |
| Online-RL KL, batch-invariant + no IS | **0 (exactly)** | RLVR | He et al. 2025 |
| max\|δ_t\| | **~1.0** (mean small) | dense | "Diagnosing TIM" 2026, Tab 1 |
| Precision: next-representable >1 | BF16 **1.0078125** (2⁻⁷) vs FP16 **1.000977** (2⁻¹⁰) | — | Qi et al. 2025, Tab 1 |
| MoE expert flips / grad step | **~10%** of experts | Qwen3-30B-A3B, 48 layers | Zheng et al. 2025 (GSPO) |
| **Collapse:** vanilla GRPO BF16 peak | **73% (VeRL) / 84% (Oat)** then degrades | MATH perfectible, 1.5B | Qi et al. 2025, Fig 3 |
| **Collapse:** token-TIS BF16 peak | **82% / 88%** then collapses | same | Qi et al. 2025, Fig 3 |
| **Collapse:** GSPO BF16 grad norm | **NaN after 1200 steps** | VeRL | Qi et al. 2025 |
| **Collapse:** online RL w/o IS | reward collapses **~step 318** + KL spike | RLVR | He et al. 2025 |
| **Collapse:** FP8 recompute | accuracy → ~0 by **~step 300** | Qwen2.5-0.5B GSM8K | Yao et al. 2025, Fig 3 |
| **Collapse:** MoE REINFORCE | **0.574→0.255 by step 280** → ~0 | Qwen3-30B, vLLM | "Diagnosing TIM" 2026, Fig 2 |
| **Collapse:** GRPO recompute | **0.87→0.40 / 650 steps**, then collapse @1665 | "Diagnosing TIM" 2026, Fig 3 |
| seq-MIS deployment gap | peak **95% vs 99%** (FP16); AIME24 **34% vs 39%** | sanity set | Qi et al. 2025 |
| IS clip C (verl example) | **10.0** | verl TIS | PR #2953 |
| IS clip C (Qi et al. expts) | **3**; clip_higher 0.28 | sanity test | Qi et al. 2025 |
| Token-trunc threshold τ | **2** (default) | ms-swift / Diagnosing-TIM | ms-swift docs |
| Seq-rejection threshold τ_seq | **0.001** | Diagnosing-TIM | arXiv:2605.14220 |
| IS overhead | **~+25%** train cost (extra fwd pass) | seq/token-IS | Qi et al. 2025 |

---

## Confidence & gaps

**High confidence (verified against primary source text/figures):**
- Yao et al. (OPT-ML 2025) — read the full 5-page PDF directly. The fp32-lm_head + true-prob patch *failing*, the TIS formula, and all KL/collapse numbers are quoted from the paper's figures/text.
- Qi et al. FP16 (arXiv 2510.26788) — read the full PDF (10 pp). KL 7.64/0.32, bit tables, collapse peaks (73/84/82/88%), NaN@1200, deployment gap (95vs99, 34vs39), Table 2 offline scores are verbatim from the paper.
- verl config keys (`calculate_log_probs`, `behav_imp_weight_cap`, `rollout_log_probs`, `bypass_recomputing_logprobs`, `clip_ratio_c=3.0`) — verified directly in our checkout (`grep` + `core_algos.py`) and PR #2953 body.
- He et al. (Thinking Machines) KL=0.001-with-IS / KL=0-with-batch-invariant / collapse ~step 318 — verbatim from the blog.
- GSPO formula, ~10% expert-flip, "2 orders of magnitude more clipping" — verbatim from arXiv 2507.18071 HTML.
- ms-swift IS modes + τ=2, "Diagnosing TIM" silent-vs-catastrophic mechanism + numbers — from the rendered docs/HTML.

**Medium confidence / minor caveats:**
- **verl config-key naming.** The earlier WebFetch summary mentioned `tis_imp_ratio_cap`; the merged PR and current code use `behav_imp_weight_cap` (cap C) + `calculate_log_probs`, and a `rollout_correction`/`rollout_is_weights` path. Forks (TRL/OpenRLHF/SkyRL) may use different key names. The *mechanism* (cap the ratio) is identical; the *exact current verl key* I'd re-confirm against the specific verl commit we pin. **[VERIFY key name per pinned commit]**
- **Yingru Li "When Speed Kills Stability"** — the SGA inequality, TV-for-bias / χ²-for-variance, and **O(T²·Δ_max)** token-IS bias are from the rendered blog (Part 1). The blog is the primary; if a peer-reviewed/arXiv version exists with exact constants, I did not pin its ID. **[UNVERIFIED arXiv id for the blog series]**

**Unverified / flagged:**
- The "Magistral" (Mistral, arXiv 2506.10910) and MiniMax-M1 (arXiv 2506.13585) tech reports *discuss* the mismatch and IS/conservative updates, but I did **not** extract specific KL/ratio numbers from their PDFs — listed only as corroborating that industry hits the same issue. **[UNVERIFIED specific magnitudes in Magistral/MiniMax]**
- Skywork / Seed tech-report specifics: **not retrieved**; flagged as a gap if the lead wants industrial confirmation beyond Qwen3/DeepSeek/Mistral.
- "Beyond Precision" (2602.01826) and "Diagnosing TIM" (2605.14220) are **2026 and recent**; I read the abstract (former) and full HTML (latter). The former's numeric tables were not fully extracted (abstract gives the argument, not the numbers). **[partial — numbers not all pinned]**

**Relevance hand-off to our project (one line, for the analogy step):** the literature's mismatch is a *training-vs-inference codepath* difference at fixed weights, corrected primarily by **(i) making engines identical (FP16 / batch-invariant kernels / recompute)** or **(ii) IS reweighting the off-policy gradient (TIS/MIS/GSPO)** — note that IS fixes the *gradient* but not the *deployment gap*, a distinction worth carrying into how we frame our compression mismatch.

---

## Top-5 findings (for the shared list)

1. **Root cause is two engines, not weights** — confirmed by Yao et al.'s *failed* fp32-lm_head+true-prob vLLM patch ("fundamental to hybrid backend designs"); He et al. localize it to **batch-size-dependent reduction order** (KL→0 with batch-invariant kernels). [opt-ml.org/papers/2025/paper116.pdf; thinkingmachines.ai]
2. **Magnitude is tiny-mean / heavy-tail / length-amplified** — BF16 KL ~10⁻⁴–10⁻³ (mean δ small, **max δ≈1.0**), but grows ~exponentially with response length (BF16 seq-log-ratio slope −1.01, KL 7.64 vs FP16 0.32 = 24×). [Qi et al. 2510.26788]
3. **Silent degradation precedes collapse and KL won't catch it** — under recompute, K1/K3 KL stay flat ~700 steps while reward rots, because **PPO clipping turns symmetric numerical noise into a sign-skewed update**; collapses seen at step 318 / ~300 (FP8) / 1200 (GSPO NaN) / 1665. [arXiv 2605.14220; He et al.; Qi et al.]
4. **Two fix families: engine-level removes it, IS-level only patches the gradient** — FP16 (8× more mantissa, KL 24×↓) and batch-invariant kernels (KL=0) **close the deployment gap**; TIS/MIS/GSPO reweight the off-policy gradient but **leave a residual deployment gap** (seq-MIS 95% vs 99%). [Qi et al.; He et al.]
5. **Cheap default = token-TIS (cap C), robust default = seq-level/GSPO** — verl: `calculate_log_probs=True` + `behav_imp_weight_cap` (C≈10), `clip_ratio_c=3.0`, default OFF ⇒ byte-identical; token-TIS has **O(T²Δ) bias** and fails on FP8/long runs; GSPO (length-normalized seq-IS) is precision-tolerant and used in Qwen3. [verl PR #2953; arXiv 2507.18071]
