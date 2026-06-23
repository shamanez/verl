# Member B — Low-Precision / Quantized Rollout as an Analogue for the Train–Inference Mismatch

**Scope.** I cover the sub-problem where the **rollout/inference engine runs at lower precision than the training engine** (FP8/INT8/INT4 weights, activations, or KV-cache), so the rollout policy `π_rollout` is a *lossy approximation* of the training policy `π_θ`. This is the closest external analogue to our own situation (lossy activation compression at pipeline-parallel boundaries injects a controlled perturbation between the policy that generated the data and the policy being trained). I am **not** analyzing our setup; I am mining the literature for mechanisms, magnitudes, and corrections we can analogize to.

**Method note / verification.** Every arXiv ID below was verified by fetching the arXiv abstract page directly (not just trusting search summaries). Numbers are quoted from the paper/blog body where I could fetch it; items I could not pull from primary text are marked `[UNVERIFIED]`. The current month is June 2026, so several core sources are 2026 preprints (arXiv `26xx.*`). I treated WebSearch *summaries* as leads only and re-verified through WebFetch.

**One-paragraph orientation for the team.** The literature has converged hard on exactly our framing in the last ~9 months: a lossy rollout engine turns nominally on-policy RL into **off-policy** RL, and the fix is **importance sampling between the lossy behavior policy and a recomputed full-precision policy**. The single most useful transferable idea: *the rollout precision only determines **which tokens were sampled**; the gradient is computed from **logprobs recomputed at training precision**, and an IS ratio `π_train/π_rollout` corrects the measure.* That is precisely the move we should consider for compressed-boundary rollouts. The single most important *caveat* for us: the low-precision perturbation is empirically **biased, not zero-mean** (sign-asymmetric on advantage-weighted updates), and at aggressive precision (INT4/INT8-uncorrected) it can **corrupt trajectories outright** (garbled/repetitive text), not merely shift probabilities.

---

## Q1. Setup — where do rollouts come from a lower-precision engine than training?

This is now a mainstream systems pattern. The canonical split is **BF16 training + FP8 (or INT8) rollout**, with the inference engine (vLLM / SGLang) quantized for throughput while the trainer (FSDP / Megatron) stays in BF16/FP32 and owns the gradient.

Precision splits observed in the wild:

- **BF16-train + FP8-rollout (linear W8A8).** The dominant baseline. vLLM/SGLang run FP8 E4M3 GEMMs for the policy forward during generation; trainer is BF16. Documented as the *default* and as the thing that breaks — "the commonly adopted BF16-train + FP8-rollout strategy suffers from severe training instability and catastrophic accuracy collapse under long-rollout generation and challenging tasks" (Jet-RL, arXiv:2601.14243; corroborated by LMSYS Unified-FP8 blog and verl FP8 docs).
- **FP8 extended to KV-cache and attention.** FP8-RL (arXiv:2601.18150) adds FP8 KV-cache with per-step QKV scale recalibration on top of W8A8 linear, in the veRL ecosystem (FSDP/Megatron + vLLM/SGLang).
- **End-to-end / "unified" FP8 (train *and* rollout FP8).** NVIDIA NeMo-RL (developer.nvidia.com FP8 RL blog), Jet-RL (2601.14243), and LMSYS Unified-FP8 (lmsys.org/blog/2025-11-25-fp8-rl) all deliberately match training precision to rollout precision to *shrink the mismatch to near-zero*. Linear layers FP8 E4M3; attention/norm/nonlinear kept BF16.
- **INT8 / INT4 (W4A16, W8A8) rollout, BF16 master weights.** FlashRL (GitHub yaof20/Flash-RL; Notion blog), QuRL (arXiv:2602.13953), QaRL (arXiv:2604.07853), AIS (arXiv:2605.13907). These run low-bit GEMMs for sampling but keep BF16 master weights and full-precision optimizer state. QaRL explicitly tests **W4A16** and **W8A8**.
- **FP4 rollout for *ranking only*, BF16 for the gradient (diffusion RL).** Sol-RL / "FP4 Explore, BF16 Train" (arXiv:2604.06916): NVFP4 rollouts generate a candidate pool, then selected seeds are *regenerated in BF16* before any gradient is taken — FP4 never touches the update. (Diffusion, not autoregressive; flagged as a boundary case.)
- **Adjacent but distinct — same-precision *numeric* mismatch.** Even with no quantization, vLLM-vs-FSDP kernel/reduction-order differences produce the same off-policy pathology (VeXact / "Diagnosing TIM," arXiv:2605.14220; "Defeating TIM via FP16," arXiv:2510.26788; slime/Miles blog; verl rollout-correction docs). The FP16 paper argues the *root cause* of even the kernel-level mismatch is **BF16's rounding error**, and reverts to FP16. This is the "zero-mismatch reference" boundary of the low-precision axis and I include it because it isolates the same mechanism without quantization.

**Where this is implemented:** verl (rollout-correction math docs + FP8 docs + TIS PR #2953 by yaof20 + Fully-Async-Policy Rollout-IS PR #3955), slime (Miles), NeMo-RL, SGLang/vLLM FP8 backends. So the correction machinery we'd want already exists in our own framework family (verl).

---

## Q2. How the mismatch behaves under low precision

### Magnitude of the distribution shift
- **Per-token logprob disagreement is usually small in the mean but heavy-tailed.** VeXact (2605.14220) reports mean `|δ_t|` (per-token logprob difference) is small per batch, but "the maximum difference can even reach 1.0 for some extreme tokens," with individual disagreements up to **−0.133 log-prob units** and observed **argmax flips** (the top-1 token differs between trainer and rollout). This is a *kernel*-mismatch study, but the tail behavior is the same shape we should expect from compression.
- **KL(train‖rollout) grows monotonically as the policy sharpens.** QuRL (2602.13953): behavior–proximal KL rises **0.002 → 0.025 (≈12×) over ~1000 steps** under INT8 rollout. AIS (2605.13907): "KL divergence grows monotonically during training as the policy sharpens and quantization artifacts interact with increasingly peaked distributions." slime/Miles: K3-KL for dense Qwen3-4B climbs from ~1e-5 over 600 steps; **MoE models sit an order or two higher (≈1e-3 to 1e-1) than dense (≈1e-5 to 1e-3)**.
- **IS-ratio spread can explode.** QuRL: "the maximum proximal-to-behavior ratio can reach up to **10⁵**, causing an extremely large gradient norm"; clip fraction jumps to ~1.5% then training collapses. AIS: the **coefficient of variation of importance weights grows** through training (uncorrected IS becomes progressively unreliable).
- **Accuracy / reward deltas (uncorrected FP8/INT rollout vs BF16):**
  - AIME25, Qwen3-8B: **36.70% → 26.81% (−9.89%)** under FP8 rollout (AIS, 2605.13907).
  - MATH500, "Qwen3.5-9B": **83.20% → 71.00% (−12.20%)** FP8 rollout (AIS).
  - GSM8K, Qwen3-8B: **89.84% → 87.65% (−2.19%)** FP8 rollout — small on an easy task (AIS). *This easy-vs-hard split matters for us; see Q3.*
  - Avg math, Qwen3-8B: **51.8% → 43.9%** under W4A16/W8A8 quantized rollout; QaRL recovers to **48.9%** (BF16 = 51.8%) (QaRL, 2604.07853).
  - INT8 (DAPO/Qwen2.5-32B), Avg@32: **30.3% with a 1.4% gap** to full precision *with FlashRL/TIS* (FlashRL repo/snippet); QuRL reports **near-0% AIME** and a DAPO case collapsing **33.33% → 0.0% Avg@1** *without* correction.

### Bias vs variance — the load-bearing question for us
The low-precision perturbation is **biased, not zero-mean** — multiple independent lines of evidence:
- **Sign-asymmetric distortion of advantage-weighted updates.** VeXact (2605.14220): recomputation "induces a **sign-dependent skew** in the advantage-weighted update signal, rather than merely adding uniform noise" — positive- and negative-advantage samples are distorted asymmetrically, giving "a **skewed, non-zero-mean distortion** of the gradient updates." Their headline framing: TIM "is **not benign numerical noise**, but a systems-level perturbation."
- **Systematic, non-recoverable bias from update-masking.** QuRL (2602.13953): weight updates are tiny (10⁻⁷–10⁻⁶) relative to INT8 quantization granularity (∝|θ|/2⁸), so "**weight quantization error is much larger than the weight update, especially at early training stages**." The quantized rollout model therefore *stops tracking* the trainer — `π_θ_old` (quantized) and `π_θold` (true) "[persist] unchanged across training steps." This is a **bias mechanism with no analogue in pure kernel mismatch**: low precision doesn't just add noise, it can make the behavior policy *deaf* to the gradient.
- **Nuance — "unbiased in expectation but cumulatively biased."** AIS (2605.13907) frames FP8 as "**unbiased in expectation** (under correct probability ratios) but introduc[ing] **high variance and cumulative bias** in policy gradients due to the product of per-token ratios." I.e. *if you compute the IS ratio correctly*, the per-step estimator is approximately unbiased, but the product-over-tokens and the truncation reintroduce bias. The verl docs quantify this precisely: **token-level TIS has `O(T²·Δ_max)` bias; sequence-level IS is unbiased but high-variance.**

**Reconciliation (important for our analogy):** the *raw* low-precision perturbation on the gradient is biased (VeXact, QuRL). The *corrected* estimator can be made approximately unbiased (AIS), but only by (a) recomputing logprobs accurately and (b) accepting a bias/variance trade in the truncation. So "is it biased?" → **yes for the raw perturbation; controllable but not free after correction.**

### Concentration and compounding
- **Concentrates in the low-probability tail.** QaRL (2604.07853): error tokens are "typically assigned **very low probability**," concentrated in "**low-probability regions of both old and current policies**," and they "**dominate gradient magnitude**" via extreme ratios. AIS: artifacts interact most with "increasingly peaked distributions."
- **Compounds over long generations.** QaRL: "noise accumulates over long generations, producing **off-trajectory repetitive and garbled tokens**." AIS: "small per-token discrepancies **compound across long generation horizons**." QuRL: divergence "**gradually diverges as training progresses**," collapse after ~1000 steps. This length-compounding is directly relevant to our 16K-response setting.

---

## Q3. Is it tolerable or fatal?

**Tolerable when:** the model is strong, the task is easy, sequences are short, and precision is FP8 (not INT4). LMSYS Unified-FP8: "BF16-train-FP8-rollout performs well when the model is strong and the task is relatively simple." Concrete: GSM8K FP8 rollout only **−2.19%** (AIS), and FP8-RL/Jet-RL/NeMo-RL all report **accuracy parity with BF16** once corrected, with 15–48% throughput gains.

**Fatal when:** task difficulty rises / model confidence drops, generations are long, the model is MoE, or training runs long. LMSYS: "as task difficulty increases and the model's confidence decreases, quantization-induced errors can substantially distort the rollout trajectory, leading to unstable optimization and degraded performance." VeXact shows even sub-1.0 mean token disagreement can **independently cause collapse** (REINFORCE: reward decays 0.574→0.255 train / 0.293→0.067 val after step 280, while the bitwise-exact reference keeps climbing to 0.753/0.534). MoE is worse: expert-selection divergence pushes KL 1–2 orders higher (slime/Miles), and Unified-FP8 shows mismatch *worsens monotonically with MoE size* (30B → 1T).

**Precision floor (synthesized across sources):**
| Precision | Verdict | Source |
|---|---|---|
| FP16/FP32 (no quant) | Safest; eliminates even kernel mismatch | 2510.26788 |
| Unified FP8 (train+rollout) | Safe, ≈parity; mismatch ≈ minimized | 2601.14243, NVIDIA, LMSYS |
| FP8 rollout-only + TIS | Safe on easy/short; needs correction on hard | 2601.18150, AIS, verl docs |
| FP8 rollout-only, **no correction** | Risky → fatal on hard/long/MoE | AIS, LMSYS |
| INT8 rollout + correction | Works (≈1–2% gap) | FlashRL, QuRL |
| INT8 rollout, **no correction** | Fatal (near-0% AIME; 33%→0%) | QuRL |
| W4A16 + strong correction (QaRL/TBPO) | Recoverable (48.9% vs 51.8%) | QaRL |
| INT4 / FP4 autoregressive, generic | Below floor; needs re-tuning, often untested | AIS ("FP4/INT4 acknowledged to require re-tuning"), QuRL (INT4 excluded) |
| FP4 *ranking-only*, BF16 gradient (diffusion) | Safe by construction (FP4 never trains) | 2604.06916 |

**Is the *bias* (not just variance) what kills it?** The strongest evidence says **yes**: VeXact explicitly attributes collapse to a sign-skewed (biased) gradient distortion, not variance; QuRL attributes collapse to a *systematic, non-recoverable* bias (update-masking) plus IS-ratio blow-up. Variance matters (it's why truncation/clipping is needed), but the *fatal* mechanism is bias — which is exactly the worry for a lossy, possibly *systematically*-biased compression codec.

---

## Q4. Corrections used

1. **Importance sampling between the lossy rollout policy and the full-precision training policy (the dominant fix).**
   - **Token-level Truncated IS (TIS):** `w_t = min(π_old(a_t|s_t)/π_rollout(a_t|s_t), C)`, multiplied into the per-token loss with `stopgrad` on the weight. Typical `C = 2` (FP8-RL, LMSYS, verl FP8 docs); verl docs cite token-level `C ∈ [1.5, 5.0]`. **Stable, outperforms no-correction, but biased** (`O(T²·Δ_max)`). Origin for *quantized* rollouts: **FlashRL** (yaof20), which became verl PR #2953.
   - **Sequence-level IS / Masked IS (MIS):** aggregate the ratio over the sequence (`C ∈ [2,10]`); MIS *rejects* whole sequences with ratio > C — "best for severe mismatch or when the distribution tail is **'toxic' (contains garbage/adversarial samples rather than signal)**" (verl docs). Sequence-level is unbiased but high-variance; MIS prevented MoE collapse in slime/Miles. **Geometric-mean IS** (slime/Miles) is the bias/variance/length-invariance compromise.
   - **Adaptive IS (AIS, 2605.13907):** keeps `C` but *gates correction strength per batch* via three diagnostics — ESS-based weight reliability, divergence severity `mean|log π_train − log π_rollout|`, and variance amplification. Recovers BF16 on 11/12 benchmarks; occasionally *beats* BF16 (+6.63% AIME25) by preserving the early-training exploration the noise provides.
   - **Trust-band / dual-clip (QaRL TBPO, 2604.07853; QuRL ACR, 2602.13953):** sequence-level trust regions and adaptive clip bounds that *mask whole corrupted responses* dominated by low-prob error tokens.

2. **Recompute logprobs at full (training) precision — the move most relevant to us.**
   This is the explicit mental model in verl's three-policy framework (rollout-corr math docs): `π_rollout` (behavior, possibly FP8) is *only* the sampler; `π_old` is **recomputed by the trainer** via `actor.compute_log_prob()`; `π_θ` is optimized. The IS ratio `π_old/π_rollout` corrects the precision/back-end gap, separate from the PPO ratio `π_θ/π_old`. FlashRL implements this as an **"RL Logprob Patch Only"** mode — *quantized sampling, but accurate logprob extraction* — so the ratio's denominator is trustworthy. verl Fully-Async PR #3955 adds "Rollout IS" that recomputes `old_log_prob` with the trainer's model. The transferable one-liner: **the rollout precision determines which tokens were sampled; the gradient uses recomputed full-precision logprobs + an IS reweight.**
   - **Caveat (ServiceNow "Correctness before Corrections"):** the *denominator must actually be correct first.* They show vLLM V1 returned **pre-logits-postprocessing** logprobs by default (needed `processed_logprobs`), which injected a *mean bias* into the ratio. And the **final projection precision is part of the correctness surface** — MiniMax-M1 (2506.13585) traced a token-prob mismatch to high-magnitude LM-head activations and fixed it with an **fp32 LM head**, lifting train/infer correlation from **~0.9 to ~0.99**; ScaleRL adopts fp32 logits as a recipe choice. Lesson for us: *get the recomputed logprob numerically clean before trusting any IS correction.*

3. **Treat the low-precision engine strictly as a behavior policy + off-policy correction.** This is the unifying frame across verl, slime/Miles, QaRL, QuRL, AIS — decoupled/3-policy PPO with the rollout as behavior policy. Note one *engineering* split: slime/Miles sometimes uses the **rollout engine's own logprob as `π_old`** (skips a forward pass) — viable only because they trust that logprob's numerics; the more conservative path (recompute) is what most FP8 papers do.

4. **Mixed-precision-consistency / "unified precision" trick (eliminate the gap instead of correcting it).** Make rollout and training the *same* precision so the perturbation cancels: **unified/end-to-end FP8** (Jet-RL 2601.14243, NVIDIA NeMo-RL, LMSYS Unified-FP8). NVIDIA: end-to-end FP8 "consistently shows a lower numerical disagreement" than FP8-generation-only, and *with IS* "completely closes the gap from BF16." This is the literature's version of "make the generator and the trainee see the same lossy view" — a possible analogue to applying the *same* compression in both forward passes.

5. **Reference/anchor-policy tricks.** Most FP8-RL works are no-KL/no-reference (like ours). The closest "anchor" ideas are the *trust-band* references (QaRL/QuRL) and three-policy TRPO extensions under behavior–reference mismatch (Xihuai Wang blog; VESPO, 2602.10693) — these add a *reference* policy distinct from both behavior and proximal. `[Partially UNVERIFIED on the TRPO-extension specifics — read only secondary descriptions.]`

---

## Q5. High-stakes question — trajectory corruption vs merely off-distribution-but-valid

**This is the most important finding for us, and the evidence is split by precision regime.** There is a clear threshold:

**FP8 rollout → trajectories are valid, just perturbed (no corruption).**
- AIS (2605.13907) is explicit: "**No trajectory corruption evidence.** Trajectories are sampled from the quantized policy (**valid samples**); the issue is that log-probability evaluations under the full-precision trainer differ from rollout probabilities. **No report of degenerate or incoherent generations under FP8.**" They even argue FP8 noise is an **"implicit exploration bonus"** early in training.
- NVIDIA NeMo-RL, FP8-RL, LMSYS, Jet-RL: all report *probability/KL-level* effects and accuracy parity once corrected — **no degenerate-generation reports** at FP8.
- VeXact (kernel mismatch, an even gentler perturbation): collapse shows up in *reward/loss curves*, with **no qualitative report of garbled text** — they don't inspect outputs, but the failure is distributional.
- The diffusion FP4-ranking case (2604.06916): NVFP4 samples "**valid but perturbed, not corrupted**" — preserved Inception/CLIP scores, Spearman ρ=0.927 ordering fidelity (but FP4 never trains, and it's diffusion).

**INT4/INT8-uncorrected / aggressive quantization → trajectories ARE corrupted (degenerate text, wrong answers).**
- **QaRL (2604.07853) is the smoking gun.** Table 6 shows literal garbled output — `"Use quantized rollout engine to to to to to to accelerate RL"` — **repetitive/garbled tokens from error accumulation over long generations**. Mechanism: low-prob error tokens accumulate autoregressively → "off-trajectory repetitive and garbled tokens." This is *corruption*, not perturbation.
- **QuRL (2602.13953):** uncorrected INT8/FP8 gives "**near 0 accuracy on AIME**," a DAPO run collapses **33.33% → 0.0%**, and the quantized actor *decouples from training dynamics* (update-masking) — i.e. it generates "invalid or low-confidence solutions." The collapse is severe enough that the generations are effectively broken, not subtly off.

**Synthesis for our analogy.** The lossy-rollout literature says there are **two regimes**, separated by how aggressive the loss is and how much it *compounds autoregressively*:
1. **Mild, bounded loss (FP8, kernel-mismatch):** trajectories stay *valid*; the harm is a **biased probability/gradient perturbation** that you correct with recomputed-logprob IS. This is the optimistic analogue.
2. **Aggressive / compounding loss (INT4, uncorrected INT8):** the loss accumulates over the autoregressive horizon and **corrupts the trajectory itself** (repetition, garbage, wrong answers). No IS reweight fully saves you because the *samples themselves* are degenerate (hence MIS *rejects* the "toxic tail" rather than reweighting it).
The pivotal variable is **whether per-step error compounds along generation faster than it's bounded.** For us, the open question this raises: does our boundary compression behave like FP8 (bounded, per-step, valid trajectories) or like INT4 (compounding into corruption)? The literature says the answer hinges on the *magnitude and autoregressive compounding* of the per-token perturbation, and on whether the perturbation is *systematic* (QuRL's update-masking is the cautionary tale: a biased codec that makes the behavior policy deaf to the gradient is worse than unbiased noise of the same size).

---

## Key papers table

| Title | Venue / arXiv id | One-line finding | URL |
|---|---|---|---|
| Diagnosing Training Inference Mismatch in LLM RL (VeXact) | arXiv:2605.14220 | Mismatch is a **biased, sign-skewed, non-zero-mean** systems perturbation; tiny token disagreements alone cause collapse | https://arxiv.org/abs/2605.14220 |
| Defeating the Training-Inference Mismatch via FP16 | arXiv:2510.26788 | Root cause is **BF16 rounding**; FP16 nearly eliminates the mismatch | https://arxiv.org/abs/2510.26788 |
| FP8-RL: Practical & Stable Low-Precision Stack for LLM RL | arXiv:2601.18150 | FP8 W8A8+KV rollout in veRL; **token TIS/MIS** keeps parity, +44% rollout throughput | https://arxiv.org/abs/2601.18150 |
| Jet-RL: On-Policy FP8 RL, Unified Train+Rollout Precision | arXiv:2601.14243 | BF16-train+FP8-rollout → "**catastrophic accuracy collapse**"; **unified FP8** fixes it | https://arxiv.org/abs/2601.14243 |
| AIS: Adaptive Importance Sampling for Quantized RL | arXiv:2605.13907 | FP8 rollout drops AIME25 **−9.89%**, MATH500 **−12.20%**; per-batch adaptive IS recovers BF16; **FP8 = valid samples, no corruption** | https://arxiv.org/abs/2605.13907 |
| QuRL: Efficient RL with Quantized Rollout | arXiv:2602.13953 | INT8 IS-ratio hits **10⁵**, KL 0.002→0.025; **update-masking = systematic bias**; uncorrected → near-0% / 33%→0% | https://arxiv.org/abs/2602.13953 |
| QaRL: Rollout-Aligned Quantization-Aware RL | arXiv:2604.07853 | W4A16/W8A8; **explicit garbled "to to to" output**; error tokens in low-prob tail; TBPO trust-band recovers 51.8%→48.9% | https://arxiv.org/abs/2604.07853 |
| FP4 Explore, BF16 Train (Sol-RL, diffusion) | arXiv:2604.06916 | FP4 for **ranking only**, BF16 regeneration for gradient; FP4 trajectories "valid but perturbed"; sidesteps mismatch | https://arxiv.org/abs/2604.06916 |
| MiniMax-M1 Technical Report | arXiv:2506.13585 | Train/infer token-prob mismatch traced to LM head; **fp32 LM head** lifts correlation **0.9→0.99** | https://arxiv.org/abs/2506.13585 |
| Process Reinforcement through Implicit Rewards (PRIME) | arXiv:2502.01456 | Process-reward RL recipe (Eurus-2-7B-PRIME); **no FP8/precision-mismatch content found** — cite as process-reward only | https://arxiv.org/abs/2502.01456 |
| FlashRL (FP8/INT8 rollout w/o perf drop) | repo (yaof20/Flash-RL); Notion blog | INT8/FP8 sampling + **accurate-logprob patch** + TIS; INT8 30.3% Avg@32, **1.4% gap**; basis of verl PR #2953 | https://github.com/yaof20/Flash-RL |
| verl — Rollout Correction Math | verl docs | 3-policy frame: rollout=behavior, **recompute π_old at train precision**, TIS bias `O(T²Δ)`, seq-IS unbiased, MIS for "toxic tail" | https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html |
| verl — FP8 RL | verl docs | FP8-rollout-only vs end-to-end; **TIS C=2**; FP8 + TIS ≈ BF16 parity; "rollout correction required even for BF16" (MoE) | https://verl.readthedocs.io/en/latest/low_precision/fp8.html |
| NVIDIA — End-to-End FP8 RL (NeMo-RL) | NVIDIA dev blog | End-to-end FP8 + IS "**completely closes the gap from BF16**"; metric = token multiplicative prob error (<1.03–1.05 safe) | https://developer.nvidia.com/blog/run-high-throughput-reinforcement-learning-training-with-end-to-end-fp8-precision/ |
| LMSYS — Unified FP8 for MoE RL | LMSYS blog | BF16-train+FP8-rollout breaks on hard/large-MoE; **unified FP8 eliminates** quantization-induced inconsistency | https://www.lmsys.org/blog/2025-11-25-fp8-rl/ |
| slime / Miles — All-in-One Mismatch | GitHub (Awesome-ML-SYS-Tutorial) | Taxonomy + **token IS biased / seq IS unbiased-high-var / geometric balanced**; MIS prevents MoE collapse; dense KL 1e-5–1e-3 vs MoE 1e-3–1e-1 | https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-en.md |
| ServiceNow — Correctness Before Corrections | HF blog | **Fix backend/logprob numerics first** (processed_logprobs; fp32 head per MiniMax-M1/ScaleRL), *then* add IS | https://huggingface.co/blog/ServiceNow-AI/correctness-before-corrections |
| verl PR #2953 — TIS rollout-mismatch fix | GitHub PR | Lands token-level TIS in verl (yaof20 / FlashRL author) | https://github.com/verl-project/verl/pull/2953 |
| verl PR #3955 — Fully-Async Rollout IS | GitHub PR | Recomputes `old_log_prob` with trainer model; Rollout-IS for async | https://github.com/verl-project/verl/pull/3955 |

---

## Magnitudes cheat-sheet (concrete numbers + sources)

- **Per-token logprob diff (kernel mismatch):** mean small; **max up to 1.0**, individual **−0.133 log-units**, argmax flips observed — VeXact (2605.14220).
- **KL(behavior‖proximal) growth, INT8:** **0.002 → 0.025 (~12×) over ~1000 steps** — QuRL (2602.13953).
- **KL magnitude, dense vs MoE:** dense ≈ **1e-5 → 1e-3**; MoE ≈ **1e-3 → 1e-1** — slime/Miles.
- **Max IS ratio, INT8 uncorrected:** **~10⁵** (huge grad norm) — QuRL.
- **Train/infer correlation, LM-head fix:** **~0.9 → ~0.99** with fp32 head — MiniMax-M1 (2506.13585).
- **"Safe" numerical-disagreement metric:** token multiplicative prob error **< 1.03–1.05** — NVIDIA NeMo-RL blog.
- **Accuracy deltas, FP8 rollout-only, uncorrected:** AIME25 Qwen3-8B **−9.89%** (36.70→26.81); MATH500 **−12.20%** (83.20→71.00); GSM8K **−2.19%** (89.84→87.65) — AIS (2605.13907).
- **Accuracy, W4A16/W8A8 quantized rollout:** BF16 **51.8%** → uncorrected **43.9%** → QaRL **48.9%** (Qwen3-8B math avg) — QaRL (2604.07853).
- **Accuracy, INT8 + FlashRL/TIS:** **30.3% Avg@32, 1.4% gap** to FP (DAPO/Qwen2.5-32B) — FlashRL.
- **Collapse, uncorrected INT8/FP8:** "**near 0% AIME**"; DAPO **33.33% → 0.0% Avg@1** — QuRL.
- **Collapse, kernel mismatch (REINFORCE):** reward **0.574→0.255** train, **0.293→0.067** val after step 280 (exact ref → 0.753/0.534) — VeXact.
- **TIS thresholds:** token `C = 2` (NVIDIA/LMSYS/verl/FP8-RL); verl ranges token `[1.5,5.0]`, seq `[2,10]`; slime token TIS `[0.5,1.5]`, geometric MIS `[0.99,1.001]`.
- **TIS bias order:** token-level **`O(T²·Δ_max)`**; sequence-level **unbiased**, higher variance — verl docs.
- **Throughput gains (the reason people accept the risk):** FP8-RL **+44%** rollout; verl FP8 ~**12–18%** dense / **>35%** MoE; NVIDIA **>15%** (→**~48%** with FP8 KV/attn); AIS **1.5–2.76×** rollout, ~50% rollout memory; Sol-RL **up to 4.64×** convergence (diffusion).

---

## Confidence & gaps

**High confidence (verified in primary text):**
- The pattern (BF16-train + low-precision-rollout) and the dominant fix (recompute-logprob IS, treat rollout as behavior policy) — confirmed in verl docs, FP8-RL, AIS, QuRL, QaRL, NVIDIA, LMSYS, slime/Miles.
- The **bias** character of the raw perturbation — VeXact (sign-skew) and QuRL (update-masking) are explicit and independent.
- The **two-regime** trajectory story: FP8 = valid samples (AIS explicit); INT4/uncorrected-INT8 = garbled/corrupted (QaRL Table 6, QuRL 33→0%). This is the cleanest, most decision-relevant finding.
- The **recompute-at-full-precision** trick and the **fp32-LM-head** correctness prerequisite (verl + FlashRL + MiniMax-M1 + ServiceNow).

**Medium confidence / partially verified:**
- Some fine-grained numbers (FlashRL Notion blog, NeurIPS workshop PDF) came from search snippets or rendered abstracts because the JS/PDF wouldn't extract cleanly. The FlashRL INT8 "30.3% / 1.4% gap" is from a search snippet of the repo, not the rendered plot. Marked accordingly.
- "Qwen3.5-9B" naming in AIS's MATH500 row is as-reported; I did not independently confirm the model card. `[low-risk UNVERIFIED]`
- VESPO (2602.10693) and the 3-policy TRPO extension: only secondary descriptions read; do not over-cite the math.

**Gaps / where the literature is thin (and where we'd have to extrapolate):**
- **No paper studies our exact perturbation** — *activation* compression at *pipeline-parallel* boundaries. The closest mechanisms are (a) FP8 *weight/activation* quantization in the rollout forward and (b) QuRL's *weight*-update-masking. None compress *cross-stage activations during training*. Our perturbation lives in a different place in the graph.
- **Bias *direction/structure* is under-characterized.** VeXact says sign-skewed; QuRL says systematic via masking; AIS says ~unbiased-after-correction. Nobody gives a clean spectral/Jacobian decomposition of the bias — which is exactly what our own work (σ(M), anchor) is trying to do. **This is a genuine white-space the team could occupy.**
- **Almost no qualitative output inspection.** Only QaRL actually *shows* corrupted text; everyone else infers from reward/KL. So "valid vs corrupted" is well-evidenced at the extremes but thinly evidenced in the middle band — which is plausibly where our compression sits.
- **Almost all RL+FP8 work uses no-KL/no-reference GRPO/DAPO** (like us), so the "anchor/reference" correction column is sparse — there's an opening to test an anchor *as a precision-mismatch corrector*, which connects to our own anchor lineage.
- **The "recompute logprobs at full precision" move assumes you can cheaply recompute** an exact `π_old`. For us the analogue is subtler: if the *training* forward is the lossy one (compressed activations), there may be no clean full-precision logprob to put in the IS denominator — the lossy view *is* the trainee's view. The unified-FP8 result (make both sides see the same lossy view) may be the more apt analogue than recompute-IS. **Flag for the cross-member debate.**

---

*Prepared by Member B (low-precision/quantized-rollout analogue). Ready to challenge Member A's general-RLVR-mismatch findings — in particular I'll press on (1) whether the mismatch is bias or variance [my evidence says bias dominates the fatal cases], (2) whether "recompute logprobs at full precision" is even available in our compressed-training setting, and (3) the valid-vs-corrupted trajectory threshold.*
