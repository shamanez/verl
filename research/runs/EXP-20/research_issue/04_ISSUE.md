> **Interactive research issue (briefing + analysis + open-questions seed — not a closed report).**
> It starts from the PowerSGD-style PP **activation** compression *as implemented* + the every-`k`-steps
> **clean (dense) refresh** + every observation and implementation assumption from EXP-20, and is meant to be
> good enough for a research agent to **read the code + all the WandB metrics and derive its own insights**
> about how RL/GRPO gradients behave under pipeline-parallel compression, and what to improve. The open
> questions in §8 are the part you pick up.
>
> Source artifacts (all in-repo): `research/runs/EXP-20/research_issue/{01_wandb_metrics,02_math_interpretation,03_theory}.md`,
> `research/runs/EXP-20/POWERSGD_IMPLEMENTATION.md`, `research/runs/EXP-20/{verdict,RESULTS_ANALYSIS}.md`,
> `research/runs/FIXED_CONTROL_SURFACE.md`, and `CODE_WALKTHROUGH.md`. Codec: `verl/workers/comm_eff/powersgd_activation.py`.

# 1. TL;DR / thesis

We built a **PowerSGD-style low-rank projection of the activation crossing each pipeline-stage boundary** (transmit `r` coordinates `Y = M@Q` instead of the full `H=1536`-dim activation), trained vanilla GRPO on Qwen2.5-1.5B-Instruct + GSM8K, and ran the codec as the **only** axis against the PRF random-mask at equal communication budget, with an every-5-steps dense ("clean") refresh.

**The central question this issue must answer — "is the ~0.74 final accuracy just because of the 10 full-grad (clean) steps, or something else?"** (`clean_cadence=5` over 50 steps = exactly 10 dense-grad steps) — is answered **No**. The compressed steps carry **57–95%** of the train-reward gain, reward climbs steeply *between* the clean steps, and the compressed-step learning slope is *steeper* than the clean-step slope. The clean step is a small periodic **bias-flush**, not where reward is made.

The thesis, in one line: **the compressed step learns because projecting the boundary gradient onto the top-`r` activation subspace is a low-variance, small-bias descent direction (the boundary gradient is low-rank, so the discarded off-subspace part `(I−P)g` is tiny); the clean step is a periodic full-rank flush of the off-subspace bias that the no-error-feedback codec drops — and it contributes little to reward precisely because that bias is small.** Consequences: `clean_cadence` can likely be relaxed or removed, error feedback is the principled way to remove it, and the right success metric is the dense-vs-compressed **update cosine** (which EXP-20 never logged). All of this is **settled by internal codec-vs-codec data**; the in-flight dense run (§9) only calibrates the absolute ceiling and cannot overturn it.

---

# 2. What was built (the codec + clean-cadence mechanism), with the implementation assumptions

**The idea.** At each PP boundary, replace the block's output activation `M` (shape `(N tokens, H)`) with its projection onto a **shared, frozen, low-rank orthonormal basis** `Q` (shape `(H, r)`):

```
Y   = M @ Q       # (N, r)  — the r coordinates actually transmitted across the boundary
M̂  = Y @ Qᵀ      # (N, H)  — reconstruction; M̂ = M P,  P := Q Qᵀ  (orthogonal projector, P² = P = Pᵀ)
```

Only the `r`-dim `Y` crosses the boundary; `Q` is a communication-free shared codebook (bootstrapped from a seed, refreshed by power iteration). With the codec OFF the path is **byte-identical to dense GRPO**. (In EXP-20 there is no real pipeline parallelism — the boundary compression is *simulated in place* via forward hooks on selected decoder blocks while the model runs under FSDP data-parallel across 4 GPUs.)

**The implementation assumptions actually used (each one is load-bearing for the analysis):**

1. **Exact self-adjoint projector backward — no straight-through.** `Q` is a detached fp32 buffer (`self._basis`, never an `nn.Parameter`, never `requires_grad`); `M` stays in-graph. So PyTorch's own backward of the two matmuls *is* the exact projector: `dL/dM = (dL/dM̂)·QQᵀ`. No custom `autograd.Function`, no surrogate. Verified at `verl/workers/comm_eff/powersgd_activation.py:338-340` (forward `Y=M@Q`, `M̂=Y@Qᵀ`). This is **why** the upstream gradient is exactly `P·g` (the projected boundary gradient), the object §5 reasons about.
2. **NO error feedback.** The discarded residual `M − M̂` (equivalently the off-subspace gradient `(I−P)g`) is **dropped every step** — there is no accumulator. The only state carried across steps is the basis `Q`, not a residual (`maybe_update_basis` → `_reset_sketch()`; verified `powersgd_activation.py:516, 641-644`). Classic PowerSGD re-injects this residual; **this fork does not.** This is *the* reason the clean step exists (§5.4) and the top improvement lever (§7.2).
3. **Basis update = block power iteration on the activation Gram, post-backward.** Off-graph during the forward, accumulate `V += (MᵀM)Q` (fp32, deduped against grad-checkpoint recompute via a per-microbatch `_fwd_generation` counter). Then once per step *after* the backward, `Q ← orthonormalize(V)` — i.e. `Q_{t+1} = orth(C·Q_t)`, `C = MᵀM`, which drives `Q` toward the **top-`r` right-singular subspace of the activations** (Eckart–Young-optimal rank-`r` reconstruction). Verified `powersgd_activation.py:374, 512`; post-backward call site `engine_workers.py:928-943` (in `update_actor`'s `finally:`).
4. **Frozen-Q within a step ⇒ ρ≈1.** `Q` is held constant for the entire global step: both the old-logprob recompute (`compute_log_prob`, with `compress_recompute=true` stamping the *same* `Q_t`) and the actor-train forward (`update_actor`) apply the identical operator `Q_t Q_tᵀ`. The basis advances to `Q_{t+1}` only after the gradient is applied. So the GRPO importance ratio `ρ = exp(logπ_new − logπ_old)` is not corrupted by codec drift (the projector's bias shifts *both* log-probs the same way and largely cancels in the ratio). Verified `engine_workers.py:697-706, 928-943`.
5. **`sync_basis=true` — one shared consensus codebook across DP.** Each of the 4 DP ranks sees a different shard, so each local sketch `V_i = C_i Q` differs; `maybe_update_basis` **all-reduces the raw `V` over the DP group before** `orth`, so `Q` is **bit-identical on every rank** (averaging orthonormal frames would be meaningless — reduce the raw sketch, then orthonormalize). Deadlock-safe: every rank iterates the fixed `sorted(boundary_indices)` and zero-fills missing boundaries. Verified end-to-end: `powersgd_q_cross_rank_max_rel_dev = 0.0` at every step. (The DP axis is *not* compressed — only the PP boundary is; the basis-sync is an `H·r` DP-axis traffic per non-clean step, separate from and uncounted by the headline `logical_pp_bytes` budget.)
6. **Deterministic per-layer seed bootstrap (zero-communication).** `Q_L = orth(randn(H, r))` drawn on **CPU in fp32** with a seeded generator `seed_L = (base_seed·1_000_003 + layer_idx·7919) & 0x7FFFFFFF`, making `Q_L` bit-identical on every rank without any communication; persists across steps (warm start), not cleared by `unregister()`.
7. **fp32 QR, bf16 projection.** Orthonormalize/store `Q` in fp32 (bf16-QR loses orthogonality), project (`M@Q`, `Y@Qᵀ`) in the activation dtype (bf16). `q_cond ≈ 1` confirms orthonormality (caveat: it is measured on the QR output, so it detects *collapse*, not a poorly-*fit* basis — `reconstruction_rel_error` is the real fit metric). Projection shrinks activation norm (`‖M̂‖ ≤ ‖M‖`) but the next block's input RMSNorm absorbs it, so PowerSGD needs **no rescale knob** (unlike the mask).
8. **The clean step bypasses BOTH codecs.** `is_clean_step = clean_cadence>0 AND global_step % clean_cadence == 0` (`clean_cadence=5` ⇒ steps {5,10,…,50}). On a clean step `mask_active = not clean_step = False` ⇒ no hooks register ⇒ the actor-train forward AND the old-logprob recompute are byte-identical dense; `maybe_update_basis(is_clean_step=True)` skips (Q held, no V). So AdamW refreshes its moments on the **full-rank** gradient every `k` steps. Verified `engine_workers.py:917` + `transformer_impl.py` lifecycle.

For depth, read **`research/runs/EXP-20/POWERSGD_IMPLEMENTATION.md`** (end-to-end, §0–§12) and **`CODE_WALKTHROUGH.md`** (FSDP integration + the explicit "not yet implemented" gap list).

---

# 3. The 4 EXP-20 arms + setup

**Fixed control surface** (LOCKED operator directive — the *only* axis that varies is the codec; full table in `research/runs/FIXED_CONTROL_SURFACE.md`): Qwen2.5-1.5B-Instruct (**H = 1536**), GSM8K (test set = 1319), vanilla GRPO no-KL/no-entropy, lr `1e-6`, train_batch_size 128 × `rollout.n`=8, `ppo_mini_batch_size`=64, `max_response_length`=16384, `total_training_steps`=50 / `total_epochs`=2, **`clean_cadence`=5** (⇒ 10 clean + 40 compressed), `seed`=0, 4×H200. **A full diff of the three resolved configs (`set -x` ground truth) shows the only differing lines are `compression_type`, `powersgd.rank`/`mask.p`, and `experiment_name` — the codec is genuinely the only axis.**

| Arm | exp name | WandB id | codec | rank / p | logical PP bytes/tok | **val-acc@50** |
|---|---|---|---|---|---|---|
| **mask p=0.95** (baseline-of-record) | `ce_mask_p95_clean5_50s_gsm8k` | `3yxzzwn3` | `prf_mask` | p=0.95 | **76.8** | **0.7384** |
| **PowerSGD r=102** | `ce_powersgd_r102_clean5_50s_gsm8k` | `kqozxfr0` | `powersgd` | r=102 (+33% budget) | **102.0** | **0.7437** (+0.0053) |
| **PowerSGD r=77** (byte-matched) | `ce_powersgd_r77_clean5_50s_gsm8k` | `oquyeic3` | `powersgd` | r=77 | **77.0** | **0.7415** (+0.0031) |
| **dense control** | `ce_dense_50s_gsm8k` | `5e2jpho9` | (off) | — | H=1536 | **0.7536** (+1.0–1.5 pp over the compressed arms — see §9 + dense-results comment) |

**Budget note (H=1536, not the issue-assumed 2048):** the mask at p=0.95 keeps `0.05·1536 = 76.8` coords/tok, so **r=77 is the byte-matched arm** (the equal-communication comparison, +0.26%); r=102 is +33% budget. `logical_pp_bytes/tok` is the per-token coordinate count crossing each boundary (mask: surviving fraction of H; PowerSGD: the rank r). At equal budget, PowerSGD (principal-subspace projection) **edges** the PRF mask (random sparsification) — verdict = **PASS** (`research/runs/EXP-20/verdict.md`), decisive on the byte-matched r=77 arm. Honest calibration: deltas are small (+0.003–0.005), single seed, 50 steps — a directional curve-match, not a variance study. The accuracy spread across all three codecs is **0.53 pp**, within RL noise: **the three codecs are accuracy-equivalent at this budget.**

---

# 4. The central analysis: *"is the ~0.74 just the 10 clean steps?"*

**Setup.** `clean_cadence=5` ⇒ steps {5,10,…,50} take a full **dense** boundary gradient (10 clean steps); the other 40 steps take the **compressed** boundary gradient. `reward(t)` = `critic/score/mean` (train), logged every step. Define `Δ_t = reward(t) − reward(t−1)`.

**Verdict: NO — the compressed steps carry the learning.** Four independent cuts of the data all agree:

**(a) Net reward climb on compressed steps only.** Summing the rise *within* the 9 inter-clean segments (the climb produced purely on compressed steps, clean-step deltas excised):

| arm | total train-reward gain (step 1→50) | rise on the 40 inter-clean **compressed** steps | as % of total |
|---|---|---|---|
| mask p=0.95 | +0.6680 | **+0.5186** | **78%** |
| PowerSGD r=102 | +0.6660 | **+0.6016** | **90%** |
| PowerSGD r=77 | +0.6357 | **+0.5986** | **94%** |

**(b) Two attributions, same verdict.** Strict "Δ booked *at* the clean step" (Attribution A) → clean-step share **4.8–19.6%**. Crediting the gradient *type that produced the new policy* (Attribution B, gives the dense step its most generous share) → clean-driven **27.5–42.7%**. **Compressed steps book the majority (57–95%) under either attribution.** Per-clean-step Δreward is small and frequently *negative* (e.g. mask clean Δ = {5:−0.023, 10:+0.090, 15:+0.035, 20:+0.021, 25:−0.050, 30:+0.016, 35:+0.017, 40:+0.003, 45:−0.023, 50:+0.047}); only the *first* clean step (step 10) is a large positive jump.

**(c) Reward rises monotonically between clean steps**, especially in the high-gradient early segments 5→19 which alone supply +0.32 to +0.45; segments turn negative only twice (late-training noise after the plateau near 0.75).

**(d) The compressed-step slope is *steeper* than the clean-step slope.** OLS fit of reward vs step on each subsequence:

| arm | slope on clean-step samples | slope on compressed-step samples |
|---|---|---|
| mask | +0.01317 /step | **+0.01478 /step** |
| r=102 | +0.01329 /step | **+0.01513 /step** |
| r=77 | +0.01303 /step | **+0.01499 /step** |

The two subsequences track the *same* underlying learning curve, and the compressed steps advance the policy at least as fast as the dense steps. **The "10 clean steps doing all the work" hypothesis is falsified by the data.** (73–82% of the gain is in the first 20 steps; the codec is stress-tested precisely during the steep early phase and tracks fine.)

---

# 5. How RL/GRPO gradients behave under PP compression (the theory)

This is the *why* behind §4 — the mechanism a downstream agent should test and extend.

**5.1 The projected gradient is a structured biased-but-aligned estimator.** Because `Q` is detached and orthonormal, the upstream gradient is exactly `g_codec = g_hat·P` where `P = QQᵀ`; to first order (`g_hat ≈ g`, exact on a clean step), `g_codec ≈ P·g = g − (I−P)g`. The codec **deterministically drops the off-subspace component `(I−P)g` every step** (no error feedback). The **bias** is therefore exactly the off-subspace energy:

```
‖bias‖ / ‖g‖ = ‖(I−P)g‖ / ‖g‖ = sqrt(1 − ‖Pg‖²/‖g‖²) .
```

**5.2 The bias is small because the boundary gradient is low-rank.** Two facts: (i) `Q` is fit to the activation second moment, where the gradient energy concentrates, and its measured fidelity `reconstruction_rel_error = ‖(I−P)M‖/‖M‖` converges to **~0.02** within ~9 steps (so `col(Q)` captures ~98% of the activation energy); (ii) the boundary gradient inherits the activations' low-rank structure — per-layer recon is **<4% at every depth** and the depth profile is near-flat (L24/L3 ≈ 1.6–2.0×). The rank–fidelity curve is **flat across [77, 102]** (−25% rank ⇒ only +10.8% relative recon error), so the spectral knee is *below* 77. Hence `‖(I−P)g‖/‖g‖` is on the order of `sqrt(0.02) ≈ 0.14` in norm, and **almost certainly smaller for the gradient than for the raw activation** because GRPO's advantage weighting concentrates energy on the decisive answer/reasoning tokens (a lower-dimensional, more structured set; §5.6). *Falsifiable corollary:* the dense-vs-compressed **update cosine** should be `≈ sqrt(1 − recon²) ≳ 0.98` once the basis warms (steps ≥ ~9) — see §8 Q1, §7.1.

**5.3 PowerSGD (biased, low-variance) vs the PRF mask (unbiased, high-variance) — the bias–variance contrast that explains the data.** The mask multiplies each coordinate by an independent Bernoulli keep/drop and rescales survivors by `1/(1−p)=20×`: **unbiased in expectation** but **high variance** (∝ `p/(1−p) ≈ 19`). PowerSGD's projection is **biased** (drops `(I−P)g`) but **low-variance** (energy-preserving, `‖Pg‖ ≤ ‖g‖`; consensus `Q` ⇒ no per-rank stochasticity). They sit at opposite corners of the bias–variance plane:

| | systematic error (bias) | stochastic error (variance) | grad it produces |
|---|---|---|---|
| **PRF mask** | ~0 (unbiased, rescaled) | **high** (∝ p/(1−p) ≈ 19) | large & noisy |
| **PowerSGD** | small (`(I−P)g`, ~14% norm) | **low** (energy-preserving, consensus Q) | small & smooth |

**5.4 The grad-norm gap is the direct fingerprint of that contrast.** Clean-step grad-norm is **~0.4 in all three arms** (necessarily — the clean step bypasses the codec and computes the same dense gradient). The compressed-step grad-norm differs sharply by codec:

```
                 clean-step grad   compressed-step grad (steady)   ratio
  mask p=0.95         0.399              ~11.8  (median 10.9)       ~27×
  PowerSGD r=102      0.408              ~1.7   (median 1.57)       ~3.8×
  PowerSGD r=77       0.390              ~2.1   (median 1.87)       ~4.8×
```

The mask's `20×` rescale of a sparse survivor set inflates the grad-norm ~27× over the clean norm; PowerSGD's energy-preserving projection inflates it only ~4×. (PowerSGD's step-1 grad-norm spikes to 166/194 — a benign one-step **cold-basis** transient, co-incident with `recon ≈ 0.97`; a random rank-77 subspace in H=1536 captures only ~5% of a generic vector, so the step-1 projector is near-arbitrary. It decays to the 1–3 band by step ~4, before reward begins climbing.)

**5.5 The headline RL insight: progress is direction-driven, not magnitude-driven.** Despite the **6–7× compressed grad-norm difference, the two codecs reach identical accuracy.** lr `1e-6` + grad-clip absorb the magnitude; what both codecs preserve is the *direction* — the mask in expectation (averaged over its high variance across 143360 per-microbatch applications), PowerSGD per step (small bias, low variance). In this regime **the learning signal lives in the gradient's direction / subspace alignment, not its norm; a codec is "good" iff it preserves the dominant gradient subspace (high `‖Pg‖/‖g‖`)**, and a 6–7× norm difference between two subspace-preserving codecs is immaterial. This is why the right success metric is the update cosine (§7.1), not the grad-norm.

**5.6 RL safety: ρ≈1 from frozen-Q-within-step.** GRPO's importance ratio `ρ = exp(logπ_new − logπ_old)` is a *ratio*, and the frozen-Q rule (assumption #4) applies the **identical** projector to both paired forwards, so the projector's bias shifts numerator and denominator the same way and largely **cancels**. Confirmed by `rollout_actor_probs_pearson_corr`, which snaps to **0.999 by step 5 and stays there, identically across all three arms** — the codec does not corrupt on-policyness after warm-up. If `Q` instead drifted *between* the old-logprob recompute and the train forward, `ρ` would partly measure codec drift and corrupt the objective — the frozen-Q rule is the load-bearing guard. (The step-5 snap is the *warm-up/policy-sync* clean step, codec-independent; it also explains the anomalous `val@0 ≈ 0.08`, a pre-alignment artifact, not a real capability measurement.)

**5.7 The clean step is a periodic full-rank bias-flush — and it flushes little.** With no error feedback, the off-subspace gradient `(I−P)g` is silently zeroed every compressed step, so a small bias can accumulate in those directions. The clean step bypasses the codec and refreshes AdamW's moments on the full-rank gradient, flushing that accumulated bias (and resetting the grad-norm — the visible ~27× heartbeat). But §5.2 showed the per-step off-subspace gradient is *small*, so the accumulated drift is small, so the flush corrects little — hence its **4.8–19.6% reward share** and frequently-negative per-clean-step Δ (§4b). The clean step is doing real work, but that work is small because the bias is small. *(Cross-codec contrast: for the **mask**, prior project work shows pure-masked GRPO **stalls** without clean steps — the mask's bias is zero in expectation but has enormous variance the optimizer cannot average away fast enough, so its clean step is a **variance reset**, not a bias flush. The *same* knob plays *different* roles for the two codecs, which is why PowerSGD should tolerate clean-step removal far better than the mask — a sharp testable prediction.)*

**5.8 Two well-separated timescales.** **Basis learning is fast** — `recon` 0.97 → <0.025 in ~9 steps, then flat (the activation subspace is *slowly varying*, so one power-iteration/step tracks it). **Policy learning is slower** — reward keeps climbing past the basis-lock to ~step 30+. Two consequences: (i) *accurate reconstruction is not a precondition for useful descent* (reward already climbs at steps 3–4 while `recon = 17–39%` — the projector captures the *advantage-relevant* directions before the full activation spectrum); (ii) once `recon` plateaus, `Q` is effectively a fixed codebook and the *same* small `(I−P)g` is dropped each step, so the no-EF drift is slow and bounded and a single clean step suffices to flush it. (If the subspace drifted fast, the dropped component would rotate and the no-EF bias would be far more dangerous — see §8 Q6.)

---

# 6. WandB pointers (what to read)

- **Entity / project:** `shamanework-pl` / `verl_compression_research`. Run URL pattern: `https://wandb.ai/shamanework-pl/verl_compression_research/runs/<id>`.
  (All runs show state=`crashed` — an egress/heartbeat artifact; runs finished locally. For the 3 EXP-20 arms, WandB history is downsampled to ~48 rows, so the **per-step finals are authoritative from the local logs** `research/runs/EXP-20/ce_*_50s_gsm8k.log`; WandB is the right source for full-trajectory `scan_history()` on the other runs.)

| arm | WandB id | run link |
|---|---|---|
| mask p=0.95 | `3yxzzwn3` | https://wandb.ai/shamanework-pl/verl_compression_research/runs/3yxzzwn3 |
| PowerSGD r=102 | `kqozxfr0` | https://wandb.ai/shamanework-pl/verl_compression_research/runs/kqozxfr0 |
| PowerSGD r=77 | `oquyeic3` | https://wandb.ai/shamanework-pl/verl_compression_research/runs/oquyeic3 |
| **dense control** `ce_dense_50s_gsm8k` | `5e2jpho9` | https://wandb.ai/shamanework-pl/verl_compression_research/runs/5e2jpho9 |

**Key metric keys to pull** (and what each tells you):

| metric key | reads |
|---|---|
| `critic/score/mean` | per-step train reward — the §4 decomposition input (every step) |
| `val-core/openai/gsm8k/acc/mean@1` | the headline val accuracy (logged steps 0/25/50 here; `val-aux/.../reward/mean@1` is identical) |
| `actor/grad_norm` | the clean-vs-compressed grad-norm contrast (§5.4); clean steps ≈0.4 |
| `comm_eff/powersgd_reconstruction_rel_error` (+ per-layer) | basis fidelity `‖(I−P)M‖/‖M‖`; 0.97→0.02 in ~9 steps — **the real basis-health signal** |
| `comm_eff/powersgd_q_cond` | projector validity (≈1 ⇒ orthonormal; detects *collapse*, not poor *fit*) |
| `comm_eff/powersgd_q_cross_rank_max_rel_dev` | cross-DP consensus (=0 ⇒ one shared codebook across all 4 ranks) |
| `comm_eff/powersgd_basis_updates` / `clean_steps` | 40 / 10 — codec fired on the 40 compressed steps, bypassed on the 10 clean |
| `comm_eff/logical_pp_bytes_powersgd_y_only` (= r) / `logical_pp_bytes_prf` | the per-token forward byte budget |
| `rollout_actor_probs_pearson_corr` | train↔inference (vLLM) log-prob agreement; ~0 at steps 1–2 → 0.999 by step 5 (§5.6) |

Useful adjacent runs (full trajectories via WandB): `grpo_dense_bigmath_baseline` (`lwl9yk4y`) is a *genuine dense* run but **Big-Math/MATH-lighteval, not GSM8K** (its only val key is `val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1`); `grpo_mask_channel_p0p9_rescale_clean_every20_2epoch` (`t03dn4nh`) is the closest GSM8K long run but is *compressed* (mask p=0.9, clean_cadence=20), not dense. Neither is a same-config dense GSM8K control — see §9.

---

# 7. Improvement levers (prioritized — mechanism + how to test)

**7.1 ★ Instrument the dense-vs-compressed update cosine (do this first — lowest effort, highest diagnostic value).** This is the **direct measurement of the alignment §5 argues for** and is exactly **EXP-20 success criterion #7, which was UNMEASURABLE because it was never instrumented** (`verdict.md:21,44,111` — direction-agreement was rested on reward + reconstruction + jaggedness instead). The theory *predicts its value*: `cos(Δθ_compressed, Δθ_dense) ≈ sqrt(1 − recon²) ≳ 0.98` post-warm-up. **How:** on a periodic step compute *both* the compressed and the dense update on the *same* minibatch (one extra dense forward/backward, like a clean step but without applying it) and log `cos` per-layer and globally. It (a) confirms/falsifies the projected-gradient theory directly, (b) is the trigger signal for §7.4, and (c) makes the head-to-head machine-checkable before any launcher promotion.

**7.2 ★ Error feedback / residual accumulation (highest leverage — currently ABSENT).** Classic PowerSGD keeps a residual `e` and compresses `g + e`, then `e ← (g+e) − P(g+e)`; the off-subspace component is never discarded — remembered and re-injected, so the estimator becomes **asymptotically unbiased**. The codec **explicitly has none** (assumption #2). EF would apply the off-subspace gradient *continuously* rather than in periodic dense bursts, and **could remove the clean step entirely** — converting "10 dense steps" from a structural requirement into an optional accelerator. **Test:** add an EF buffer; run the 2×2 {EF, no-EF} × {clean, no-clean}; track val@50 and the off-subspace energy `‖(I−P)g‖/‖g‖` (→0 under EF, flat ~0.14 without). **Three real subtleties (flag, don't hand-wave):** (i) it is an *activation/gradient* residual across a boundary, `O(N·H)` per boundary per micro-batch — at N = packed 16K-token sequences this needs a memory budget check; (ii) the residual must be applied **identically** to the old-logprob recompute and the train forward or it reintroduces a within-step inconsistency that breaks ρ≈1 (assumption #4); (iii) the residual was computed in the *old* `col(Q)` — re-injecting after `Q` rotates mixes frames (simplest correct version: EF only in steady state where the basis is ~fixed, or rotate the residual into the new basis). Deserves its own experiment.

**7.3 Downward rank sweep + relaxed `clean_cadence`.** The rank–fidelity curve is **flat across [77,102]** so the knee is *below* 77 (r=102 is wasted budget): sweep `r ∈ {16,32,48,64,77}` downward, track steady recon + the §7.1 cosine + val@50, find the true minimal byte budget `r*`. Separately, the clean step is near-free for PowerSGD (§5.7): sweep `clean_cadence ∈ {5,10,25,∞}` for *both* codecs and plot val@50 vs cadence (theory predicts shallow PowerSGD degradation, steep mask degradation). **Depth-adaptive rank** is a third cheap win: the deepest layer L24 is rank-*independent* (recon 0.038 at both ranks ⇒ neither resolves it) while the extra 25 ranks of r=102 land on shallow layers that don't need them — at *fixed total* `Σ r_layer`, moving budget to deep boundaries should lower the max per-layer recon at no extra bytes. (The codec already keys the basis per-`layer_idx`, so per-layer rank is a small extension.) A natural follow-on, **adaptive/triggered clean steps**, fires a dense step only when a drift signal (recon spike, off-subspace energy, or the §7.1 cosine) crosses a threshold — the bridge between "relax cadence" and "EF" (EF removes the *average* need; a trigger keeps a cheap safety valve for rare large-drift events).

---

# 8. Open research questions (the interactive core — each falsifiable, each with the metric/experiment that answers it)

A downstream research agent reading the code + the WandB metrics should pick these up. Each states a prediction, the decisive measurement, and an explicit falsifier.

1. **Is the projected gradient actually aligned with the dense gradient?** *Predicted:* `cos(Δθ_compressed, Δθ_dense) ≈ sqrt(1 − recon²) ≳ 0.98` for steps ≥ ~9 (§5.2). *Measure:* instrument the dense-vs-compressed update cosine per-layer + global (§7.1) — criterion #7 that was never logged. *Falsified if* the cosine is materially below the recon-implied bound (⇒ `g_hat ≉ g`, the compressed-forward loss is not a first-order-faithful proxy for the dense loss).
2. **Can error feedback remove the clean step?** *Predicted:* EF + `clean_cadence=∞` ≈ current accuracy; no-EF + `∞` degrades by a small-but-nonzero floor (§7.2, §5.7). *Measure:* the {EF, no-EF} × {clean, no-clean} 2×2, tracking val@50 and `‖(I−P)g‖/‖g‖` (→0 under EF). *Falsified if* EF+no-clean underperforms current (⇒ the clean step does more than flush off-subspace bias — its variance-reset / re-alignment roles matter more than the theory assigns).
3. **How far can `clean_cadence` relax before accuracy falls?** *Predicted:* shallow degradation; PowerSGD tolerates removal far better than the mask (low variance, §5.7). *Measure:* `clean_cadence ∈ {5,10,25,∞}` for both codecs; val@50 vs cadence. *Falsified if* PowerSGD falls off as steeply as the mask (⇒ its clean step is also a variance reset, not just a bias flush).
4. **Where is the true rank knee?** *Predicted:* knee at some `r* < 77`; accuracy holds until `recon` climbs / cosine drops (§7.3). *Measure:* `r ∈ {16,32,48,64,77}` downward; steady recon + cosine + val@50. *Falsified if* accuracy degrades smoothly from r=77 with no plateau (⇒ no sharp knee; every bit of rank buys accuracy).
5. **Does depth-adaptive rank beat uniform rank at fixed total budget?** *Predicted:* yes — moving budget from over-provisioned shallow layers to the higher-effective-rank L24 lowers the max per-layer recon at equal total bytes (§7.3). *Measure:* depth-allocated vs uniform `Σ r_layer`; per-layer recon + val@50. *Falsified if* uniform matches or beats adaptive (⇒ deep-layer recon is not the binding constraint on accuracy).
6. **Is the basis subspace actually slowly varying (justifying one power-iteration/step + a fixed steady codebook)?** *Predicted:* yes — the principal-angle drift between `col(Q_t)` and `col(Q_{t+1})` is small after warm-up (§5.8). *Measure:* log per-step principal angles / `‖Q_{t+1}Q_{t+1}ᵀ − Q_t Q_tᵀ‖` after step 9. *Falsified if* the subspace rotates fast (⇒ one power-iteration/step under-tracks it, the no-EF dropped component rotates, the bias is more dangerous — argues for `update_cadence>1` or EF-with-rotation).
7. **Does the advantage weighting make the *gradient* lower-rank than the activation (RL-specific)?** *Predicted:* the advantage-weighted policy gradient is *more* concentrated than the raw activation (§5.6), so the gradient's off-subspace energy < the activation's recon error. *Measure:* compute `‖(I−P)g‖/‖g‖` for the **gradient** directly (the codec currently logs only the *activation* recon) and compare to `recon`. *Falsified if* gradient off-subspace energy ≥ activation recon (⇒ advantage weighting does not sharpen the low-rank structure; the projector's safety margin is smaller than §5.2 assumes).
8. **Is compression accuracy-*free* or accuracy-*cheap* against the true dense ceiling?** *Predicted:* |gap| ≲ ~0.5–1 pp (within the 0.53 pp inter-arm spread) ⇒ free (§9, Comparison 2). *Measure:* the in-flight dense run, val@50. *Falsified if* gap ≫ 1 pp (⇒ a real tax ⇒ EF becomes the lever to close it).

---

# 9. Dense baseline — ✅ RESULTS (dense@50 = 0.7536)

A same-config **DENSE** GSM8K 50-step run — `ce_dense_50s_gsm8k`, comm-eff **OFF**, identical lr / batch / 2-epoch surface, `test_freq=10` (val at 0/10/20/30/40/50) — has **COMPLETED** (WandB [`5e2jpho9`](https://wandb.ai/shamanework-pl/verl_compression_research/runs/5e2jpho9)). **Results: dense@10 = 0.7324, dense@50 = 0.7536** (full trajectory + the three comparisons filled in the [dense-results comment](https://github.com/shamanez/verl-compression-research/issues/21#issuecomment-4619991952)). **Headline:** dense@10 ≈ the compressed ceiling (≈10 full grads suffice), and dense@50 sits **~1–1.5 pp above** all three compressed arms — a small but consistent compression tax that *sharpens* [#22](https://github.com/shamanez/verl-compression-research/issues/22). The `<TBD>` placeholders below are resolved in that comment.

This is necessary because **no usable ≥50-step dense GSM8K trajectory currently exists**: the only genuine dense run in the project is `grpo_dense_bigmath_baseline` (`lwl9yk4y`), which is **Big-Math / MATH-lighteval, not GSM8K** (it shows the same post-step-10 improvement shape the operator noticed, but on MATH-eval: val 0.536→0.558→0.584 over steps 0/10/20 — wrong dataset, wrong eval set, 1 epoch / 120 steps, base capability ~0.54); the only DENSE+GSM8K runs are two empty 2-step probes; and the surviving dense-GSM8K parity figure ≈0.741 is **prose-only from a different-config EXP-17 era** (mask p=0.9, clean_cadence=20). The project-fixed **baseline-of-record for the EXP-20 comparison is therefore the mask arm (0.7384)**, not dense.

**Be explicit about what the dense number can and cannot do:** the internal codec-vs-codec decomposition (§4–§5 — that the compressed steps carry 57–95% of the learning, the bias–variance contrast, the clean-step-as-small-bias-flush) is **ALREADY SETTLED** by data internal to each arm; the dense run only adds the **absolute parity ceiling**. **It cannot overturn the "compressed steps carry the learning" finding.** Three framed comparisons (placeholders to be filled in the appended comment):

**Comparison 1 — dense@10 vs compressed@50** (do 10 full grads alone already reach ~0.74?):
- **DENSE val@10 = `<TBD>`** vs compressed val@50 ≈ 0.738–0.744.
- *If DENSE@10 ≳ 0.73:* 10 dense gradients nearly reach the ceiling ⇒ the *quantity* of dense signal could dominate. **Confound:** EXP-20's 10 clean steps are *interleaved with and build on* 40 compressed updates, so a true *pure*-10-dense-step control is the honest test. Even if true, it would **not** contradict §4 (compressed steps carry 57–95% of *this run's* gain) — it would say "fewer total dense steps could also get there," a different claim. *If DENSE@10 ≪ 0.73:* the 40 compressed steps materially advance the policy (consistent with §4–§5).

**Comparison 2 — dense@50 vs compressed@50** (the parity ceiling / is compression ~free?):
- **DENSE val@50 = `<TBD>`**; gaps `DENSE − {mask 0.7384, r77 0.7415, r102 0.7437} = <TBD>`.
- *If |gap| ≲ 0.5–1 pp* (the inter-arm spread is 0.53 pp): **compression is accuracy-free at this budget** — the projected gradient is ceiling-matching, the strongest version of the §5 thesis. *If gap ≫ 1 pp:* a real compression tax exists, which **directly motivates EF (§7.2)** as the mechanism to recover it (the tax = the un-flushed off-subspace bias).

**Comparison 3 — dense's own post-step-10 slope** (shape, not just endpoints):
- **DENSE val@{10,20,30,40,50} = `<TBD,…>`** ⇒ dense post-10 slope = `<TBD>`/step; compare to the compressed arms' val@25→50 slope (≈ +0.0007/step for mask) and the compressed train-reward late slope (~flat after step 30).
- *If dense shares the steep-to-~step-15 then-flat diminishing-returns shape:* compression preserves the **learning *dynamics***, not just the endpoint. *If dense is materially steeper late:* compression slows late-stage learning (an off-subspace-bias drag that grows as the in-subspace signal is exhausted — again an EF target).

---

*Compiled from EXP-20 (`exp/20-powersgd-activation`). Data: wandb-archivist (`01_wandb_metrics.md`). Decomposition: metrics-interpreter (`02_math_interpretation.md`). Theory: theory-analyst (`03_theory.md`). Implementation: `POWERSGD_IMPLEMENTATION.md` + `CODE_WALKTHROUGH.md`. Verdict: `verdict.md` (PASS).*
