# EXP-20 — Results Analysis & Construction Metrics

Operator-facing synthesis (companion to the analyst's formal `verdict.md`). All numbers from `runs/EXP-20/` logs; box torn down + verified.

## Part A — The results

### 1. Head-to-head (val-core/openai/gsm8k/acc/mean@1 @ step 50)

| arm | coords/token | compression | val@25 | **val@50** | vs mask |
|---|---|---|---|---|---|
| mask p=0.95 (bar) | 76.8 | ~20× | 0.7194 | **0.7384** | — |
| **PowerSGD r=77 (matched)** | 77 | ~20× | 0.7104 | **0.7415** | **+0.0031** |
| PowerSGD r=102 | 102 | ~15× (+33%) | 0.7316 | **0.7437** | +0.0053 |

**Headline:** at *equal* communication budget (~20×), PowerSGD (principal-subspace projection) edges the PRF mask (random sparsification). The +33%-budget arm is consistently a touch higher, as expected.

**Honest calibration:** the deltas are small (+0.003 to +0.005) and this is a **single seed**, so the rigorous claim is *"PowerSGD matches the mask at equal budget, with a slight edge plausibly within single-seed noise."* The plan's bar was `PowerSGD ≥ mask − 0.02`; both arms clear it comfortably (they're *above* the mask, not merely within tolerance) — so the hypothesis ("tracks or beats") is confirmed, but I would not assert a definitive "beats by X" margin on one seed.

**Trajectory note:** r=77 started *behind* (val@25 = 0.7104, the tightest budget) and had the **steepest second-half climb** (+0.031 from step 25→50) to cross above the bar. Its lower-rank basis took a little longer to warm, but once warmed it matched. r=102 was ahead throughout (more budget).

### 2. Reconstruction story — the robust construction finding

Final per-layer reconstruction error `‖M−M̂‖/‖M‖` @ step 50 (M̂ = MQQᵀ):

| boundary layer | r=102 | r=77 |
|---|---|---|
| 3 | 0.018 | 0.023 |
| 7 | 0.015 | 0.016 |
| 11 | 0.016 | 0.017 |
| 15 | 0.017 | 0.019 |
| 18 | 0.020 | 0.022 |
| 21 | 0.026 | 0.029 |
| 24 (deepest) | 0.038 | 0.038 |
| **aggregate** | **~0.021** | **~0.024** |

- **(a) The boundary activations are genuinely low-rank.** Even r=77 (~5% of H=1536) reconstructs **~98%** of the activation energy. This is the INF-20 spectral precondition holding strongly — and it is *why* PowerSGD works on this model.
- **(b) r=77 ≈ r=102 in fidelity.** Dropping 25% of the rank (102→77) barely moves the error (+0.003 aggregate; layer_24 *identical* at 0.038). The top ~77 directions already span the subspace — the extra 25 ranks in r=102 are nearly redundant. **This is the mechanistic reason r=77's val-acc nearly equals r=102's.**
- **(c) Depth pattern:** the deepest boundary (layer_24) is consistently hardest (~0.038, ~2× the shallow layers) — its activations are slightly higher-rank — but still very low. (At the 2-step probe layer_24 was 0.92; that was warm-start, not a real weakness — it converged to 0.038.)

### 3. Why the reward gap is small (the codec advantage hides in reconstruction, not reward)

PowerSGD is a far higher-fidelity codec than the mask: it keeps the top-r *principal* directions (~2% error), while the mask keeps a *random* 5% of dims (discarding ~95% of each token's energy at random). Yet their val-acc is nearly equal. Why:
- **`clean_cadence=5` dominates the reward.** Every 5th step both arms take a full *dense* gradient step (10 of them), re-anchoring to the true trajectory — this fork's established "clean step is the corrector." Both codecs only need to not derail between refreshes, and both do.
- GSM8K + Qwen-1.5B over 50 steps is forgiving enough that even the lossy random mask trains fine.
⇒ **The reward metric does not strongly separate the codecs here; the reconstruction error is where PowerSGD's superiority is visible.** To make *reward* discriminate, lower/remove `clean_cadence` so the compressed steps carry more.

### 4. Distributed correctness (the codebook held)
`powersgd_q_cross_rank_max_rel_dev = 0.0` at **every step** on both PowerSGD arms ⇒ the `sync_basis=true` consensus kept the basis Q **bit-identical across all 4 DP ranks** for all 50 steps (the operator-flagged concern — the fix held end-to-end). `q_cond ≈ 1.0000003` throughout ⇒ the basis stayed orthonormal (no collapse).

### 5. Caveats / what this does NOT show
- **Single seed** — small deltas lack a variance estimate; directional, not a definitive margin.
- **50 steps** — a directional curve-match, not converged training.
- **`clean_cadence=5` dominates** — the codecs aren't stress-tested on reward; reconstruction is the discriminating signal.
- **One model + task** (Qwen-1.5B + GSM8K).
- **Measurement gap:** the plan's *dense-vs-compressed update cosine* success criterion was **never implemented/logged** (confirmed: no cosine metric in any arm). We can't directly measure gradient-direction agreement; we infer it (low reconstruction ⇒ the projected gradient ≈ the dense gradient on the kept subspace; + reward tracks). **Follow-up: add the cosine metric to measure it directly.**

## Part B — What we measure about the construction (and why)

The codec must satisfy four properties; each logged diagnostic is a direct test of one.

| Metric | What it is | What it tells us | EXP-20 |
|---|---|---|---|
| **`powersgd_q_cond`** | σ_max(Q)/σ_min(Q) | **Valid projector?** ≈1 ⇒ Q orthonormal (QᵀQ=I ⇒ QQᵀ idempotent, P²=P). Non-finite/large ⇒ basis collapse. *Correctness guard, NOT fidelity* — it's measured on the QR output so it's ~1 by construction; catches a degenerate basis, never a poorly-fit one. | 1.0000003 ✓ |
| **`powersgd_reconstruction_rel_error`** (+per-layer) | ‖M−M̂‖/‖M‖ | **Does the basis capture the activation?** THE fidelity / basis-health metric. →0 ⇒ Q spans the activation's principal subspace (activations are low-rank at rank r); →1 ⇒ discarding everything. The real test of whether rank r suffices (INF-20). Per-layer = *where* it struggles. | 0.97→0.02 ✓ |
| **`powersgd_q_cross_rank_max_rel_dev`** | max rel. deviation of Q across DP ranks | **One shared codebook?** 0 ⇒ every rank holds the bit-identical basis (consensus held). >0 ⇒ per-rank codebooks diverged (the bug fixed by `sync_basis=true`). Distributed-correctness / invariant #4. | 0.0 ✓ |
| **`logical_pp_bytes_powersgd_y_only`** | = r | **On budget?** The per-token payload `Y=MQ` actually sent across the boundary, vs the mask's `(1−p)·H`, to assert matched budget. | 77 / 102 ✓ |
| **`powersgd_basis_updates`** | count of `orth(V)` | **Is the subspace being tracked?** One block-power-iteration step per non-clean step. 40 over 50 (40 compressed + 10 clean) ⇒ the basis is actively learned, not frozen. | 40 ✓ |
| **`powersgd_applications`** | count of hook fires | The projector actually ran (sanity). | large ✓ |
| **`clean_steps`** | count of dense-refresh steps | **Debiaser firing on schedule?** 10 at cadence=5 (steps 5,10,…,50). | 10 ✓ |
| **`mask_applications`** (on powersgd arms) | mask-hook fire count | **Codec exclusivity** — 0 ⇒ the mask is OFF, only PowerSGD fires (the check that proved this run wasn't accidentally masking). | 0 ✓ |
| **`grad_norm`** | gradient magnitude | **Stability** — warm-start spike (166→0.4 over the first steps) + clean-step drops; *not climbing* = stable. | stable ✓ |

**Together they answer:** is the construction (a) numerically valid [`q_cond`, no NaN], (b) high-fidelity [`reconstruction_rel_error`], (c) a single shared codebook [`q_cross_rank`], (d) on-budget [`logical_pp_bytes`], (e) actively learning the subspace [`basis_updates`], and (f) correctly debiased [`clean_steps`]. For EXP-20, **all six = yes**. The one thing we *intended* to measure but didn't is the gradient-direction agreement (cosine) — a logged-metric gap to close next.
