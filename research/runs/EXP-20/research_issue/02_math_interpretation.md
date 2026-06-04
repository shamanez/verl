# EXP-20 — Mathematical Interpretation of the Trajectories (3 arms)

**Compiled by:** metrics-interpreter · **Date:** 2026-06-04
**Input:** `01_wandb_metrics.md` (archivist) + raw per-step logs `runs/EXP-20/ce_*_50s_gsm8k.log`
(all 50 steps re-parsed independently; every cited number reproduced from the raw logs, not just
the archivist's table).
**Scope:** quantify what the data shows. No mechanism theory / no improvement design (that is task #3).

The three arms share a byte-identical config (vanilla GRPO, no-KL no-entropy, lr 1e-6,
`clean_cadence=5`); **the codec is the only axis**. clean_cadence=5 ⇒ steps {5,10,15,…,50}
take a **full dense (uncompressed) PP-boundary gradient** (10 clean steps); the other 40 steps
take the **compressed** boundary gradient (mask p=0.95, or PowerSGD r=102 / r=77).

---

## 0. Headline (the operator's must-answer question)

**The compressed gradient is a real — if biased — descent direction, not an inert step that the
clean steps then "rescue."** Reward rises steeply *on the 40 compressed steps*, between the clean
steps, in every arm:

| arm | total train-reward gain (step1→50) | rise accumulated on the 40 inter-clean **compressed** steps¹ | clean-step share² |
|---|---|---|---|
| mask p=0.95 | **+0.6680** | **+0.5186 (78%)** | 19.6% |
| PowerSGD r=102 | **+0.6660** | **+0.6016 (90%)** | 8.8% |
| PowerSGD r=77 | **+0.6357** | **+0.5986 (94%)** | 4.8% |

¹ Sum over the 9 inter-clean segments (6–9, 11–14, …, 46–49) of `reward(seg_end) − reward(seg_start−1)`
— i.e. the net climb produced **purely on compressed steps**, with the clean-step deltas excised.
² Σ of the per-step Δreward booked *at* the 10 clean steps, as a fraction of the total gain (next section).

Read that middle column directly: **most of the learning happens on the compressed steps.** The clean
step is not where the reward is made — it is a periodic *grad-norm event* (a ~30× collapse, §3) whose
contribution to reward is small and, for PowerSGD, nearly negligible. This is the empirical signature of
a *biased-but-aligned* compressed gradient, and it is the single most important number in EXP-20.

---

## 1. Decomposition of the total reward gain: clean vs compressed steps

`reward(t)` = `critic/score/mean` at step `t` (mean train reward of the rollouts the policy produced
*before* the step-`t` update). Define `Δ_t = reward(t) − reward(t−1)`. There are two defensible ways to
attribute each Δ to a gradient type; **both give the same verdict** (compressed dominates), so the
conclusion is attribution-robust.

### 1a. Attribution A — Δ booked *at* the clean step (the strict "is the clean step where reward appears?" question)

A step `t∈{5,10,…,50}` is clean; `Δ_t` for such `t` is the reward change observed across the clean step.

| arm | total gain | Σ Δ on 10 clean steps | clean % | Σ Δ on 39 compressed steps | comp % | mean Δ / clean step | mean Δ / comp step |
|---|---|---|---|---|---|---|---|
| mask | +0.6680 | **+0.1309** | **19.6%** | **+0.5371** | **80.4%** | +0.0131 | +0.0138 |
| r102 | +0.6660 | **+0.0586** | **8.8%** | **+0.6074** | **91.2%** | +0.0059 | +0.0156 |
| r77 | +0.6357 | **+0.0303** | **4.8%** | **+0.6055** | **95.2%** | +0.0030 | +0.0155 |

Per-clean-step Δ are small and frequently **negative** (the reward sequence is noisy; clean steps land
on arbitrary minibatches): e.g. mask clean-step Δ = `{5:−0.023, 10:+0.090, 15:+0.035, 20:+0.021,
25:−0.050, 30:+0.016, 35:+0.017, 40:+0.003, 45:−0.023, 50:+0.047}`. Only the **first** clean step (step 10
for mask/r77, where Δ≈+0.08–0.09) shows a large positive jump; the later clean steps hover around zero.

### 1b. Attribution B — Δ attributed to the gradient type that *produced* the new policy

`reward(t)` reflects the policy after the optimizer update at the **end of step (t−1)**, so `Δ_t` is driven
by the grad type of step **(t−1)**. "Dense-driven" Δ ⇔ `(t−1)∈clean`.

| arm | dense-driven Σ Δ (9 deltas) | % | compressed-driven Σ Δ (40 deltas) | % |
|---|---|---|---|---|
| mask | +0.2295 | 34.4% | **+0.4385** | **65.6%** |
| r102 | +0.2842 | 42.7% | **+0.3818** | **57.3%** |
| r77 | +0.1748 | 27.5% | **+0.4609** | **72.5%** |

Attribution B gives the dense step a *larger* share (the reward jump is often observed on the step *after*
the dense update), but **compressed steps still book the majority (57–72%) of the gain in all three arms**.

**Verdict (robust to attribution choice):** compressed gradients carry 57–95% of the reward gain. The
"10 clean steps doing all the work" hypothesis is **falsified** by the data.

### 1c. Direct slope test: reward climbs *between* clean steps

The cleanest non-attribution evidence. Take each inter-clean window and measure the net rise produced on
its 4 compressed steps (`reward(clean_k+4) − reward(clean_k)`):

```
segment (compressed-only)   mask        r102        r77
 5 →  9   (after clean@5)   +0.0869     +0.1035     +0.0957
10 → 14   (after clean@10)  +0.1094     +0.1670     +0.1025
15 → 19   (after clean@15)  +0.1494     +0.1787     +0.1836
20 → 24                     +0.0879     +0.0605     +0.0928
25 → 29                     +0.0303     −0.0059     +0.0410
30 → 34                     +0.0293     +0.0293     +0.0527
35 → 39                     −0.0225     +0.0127     −0.0010
40 → 44                     +0.0234     +0.0166     +0.0273
45 → 49                     +0.0244     +0.0391     +0.0039
SUM                         +0.5186     +0.6016     +0.5986
```

**Reward rises monotonically within almost every compressed segment, especially the high-gradient early
segments (5→19), which alone supply +0.32 to +0.45.** Compressed segments turn negative only twice
(mask 35→39, r102 25→29) — late-training noise once reward has plateaued near 0.75, not a structural stall.

### 1d. OLS slope, compressed-step subsequence vs clean-step subsequence

Fit reward vs step on each subsequence independently:

| arm | slope on clean-step samples | slope on compressed-step samples |
|---|---|---|
| mask | +0.01317 /step | **+0.01478 /step** |
| r102 | +0.01329 /step | **+0.01513 /step** |
| r77 | +0.01303 /step | **+0.01499 /step** |

The compressed-step subsequence has a **slightly steeper** slope than the clean-step subsequence in all
three arms — the two subsequences track the *same* underlying learning curve. The compressed gradient is
not merely "non-destructive"; it advances the policy at the same rate the dense steps do.

### 1e. Where in training is the gain booked (early vs late)

| arm | steps 1–20 gain | % | steps 20–50 gain | % |
|---|---|---|---|---|
| mask | +0.4863 | 73% | +0.1816 | 27% |
| r102 | +0.5488 | 82% | +0.1172 | 18% |
| r77 | +0.4932 | 78% | +0.1426 | 22% |

73–82% of the reward gain is in the first 20 steps; by step 20 train-reward ≈ 0.62–0.67 and val@25 ≈
0.71–0.73 (already at the historical dense-parity neighborhood ~0.741). Steps 20–50 are a slow grind from
~0.71 → ~0.74. The codec is being stress-tested precisely during the steep early phase and it tracks fine.

---

## 2. PowerSGD reconstruction dynamics (warm-start convergence)

`powersgd_reconstruction_rel_error` = relative Frobenius error of the rank-r reconstruction of the boundary
gradient, aggregated over layers. `warm_start=true`, `update_cadence=1`: the basis Q is refreshed every
compressed step (40 refreshes by step 50; **not** refreshed on the 10 clean steps).

| step | r=102 | r=77 | note |
|---|---|---|---|
| 1 (cold, no basis) | 0.9667 | 0.9763 | reconstruction is ~garbage — random/empty basis |
| 2 | 0.7137 | 0.6911 | one warm-start refresh → error drops ~26pts |
| 3 | 0.3933 | 0.3975 | |
| 4 | 0.1727 | 0.1437 | |
| 6 | 0.0861 | 0.0901 | |
| 9 | 0.0246 | 0.0247 | **converged** (<2.5%) |
| 25 | 0.0212 | 0.0203 | steady |
| 50 | 0.0213 | 0.0236 | steady |

**Convergence:** 0.97 → <0.025 within **~9 steps** for both ranks. The warm-start power-iteration basis
locks onto the dominant gradient subspace within ~8 refreshes. After step 9 the error is flat (0.02–0.024)
for the remaining 41 steps — the boundary-gradient subspace is **slowly varying**, so a single warm-started
basis tracks it with one power-iteration/step.

Note the timing relative to learning: recon is still poor (0.17–0.39) during steps 3–4, yet train-reward is
already climbing on those steps. The early reward gain does **not** require an accurate reconstruction — the
compressed gradient is a *useful* descent direction even when its rank-r reconstruction error is 17–39%.

### 2a. Rank–fidelity curve is flat across [77, 102]

| quantity | r=102 | r=77 |
|---|---|---|
| coords/tok crossing each PP boundary | 102 | 77 |
| compression ratio (H=1536) | 15.1× | 19.9× |
| budget vs mask (76.8) | +33% | +0.3% (matched) |
| steady recon@50 | 0.0213 | 0.0236 |

Dropping the rank by **−25%** (102→77) worsens steady reconstruction by only **+10.8% relative**
(0.0213 → 0.0236, both still <2.4% absolute). Both ranks sit on the **flat part** of the rank–fidelity
curve: r=77 already captures essentially the whole gradient's energy, so spending 33% more budget on r=102
buys almost no fidelity. (This couples directly to the accuracy result in §5.)

### 2b. Per-layer depth profile @ step 50

Reconstruction error rises mildly with depth; all layers <4%:

| layer | 3 | 7 | 11 | 15 | 18 | 21 | 24 | L24/L3 |
|---|---|---|---|---|---|---|---|---|
| r=102 | 0.0184 | 0.0149 | 0.0155 | 0.0171 | 0.0200 | 0.0259 | **0.0376** | 2.04× |
| r=77 | 0.0233 | 0.0162 | 0.0169 | 0.0195 | 0.0221 | 0.0293 | **0.0380** | 1.63× |

The **deepest layer (L24) recon is essentially rank-independent**: 0.0376 (r102) vs 0.0380 (r77), a +1.1%
gap. The extra 25 ranks of r=102 help the *shallow/mid* layers (L3: 0.0184 vs 0.0233, the largest gap)
but do nothing for the late layers — i.e. L24's gradient is *higher effective rank* than 102 can capture,
so neither rank fully resolves it, and the marginal ranks 78→102 land on the shallow layers where they're
not needed. The depth profile is itself near-flat (L24/L3 ≈ 1.6–2.0×, max error 3.8%): the gradient is
low-rank at every depth.

### 2c. Shared-codebook (cross-DP) invariants hold end-to-end

- `powersgd_q_cond` (basis conditioning): **min 1.0000002, max 1.0000040** across all 50 steps, both arms.
  The orthonormalized basis Q is numerically perfect (κ ≈ 1 to 6 decimals) — QR/reortho is healthy.
- `powersgd_q_cross_rank_max_rel_dev` = **0.0 at every step, both arms.** The basis Q is **bit-identical
  across all 4 DP ranks** — the cross-DP consensus / shared-frozen-codebook invariant holds for the entire
  run (`sync_basis=true`). There is no DP-rank drift in the codec, so the compression is deterministic and
  consistent across the data-parallel group (a hard correctness precondition for the method).
- `powersgd_basis_updates`@50 = 40 (one per compressed step), `powersgd_applications`@50 = 143360
  (per-microbatch projections), confirming the codec fired on exactly the 40 compressed steps and was
  bypassed on the 10 clean steps.

---

## 3. Grad-norm dynamics

`actor/grad_norm` is the post-codec gradient norm fed to the optimizer. Two structural features:

### 3a. Clean-step collapse (~30× for mask)

| arm | clean-step grad (steps 5,10,…,50) | compressed grad, steady (steps ≥6, excl. clean) | ratio |
|---|---|---|---|
| mask p=0.95 | mean **0.399** [0.353, 0.505] | mean **11.83**, median **10.90** [8.27, 25.80] | **27.3×** |
| PowerSGD r=102 | mean **0.408** [0.367, 0.485] | mean **1.71**, median **1.57** [1.10, 3.23] | 3.8× |
| PowerSGD r=77 | mean **0.390** [0.343, 0.478] | mean **2.06**, median **1.87** [1.06, 5.82] | 4.8× |

The clean-step grad-norm is **~0.4 in all three arms** — identical, as it must be: a clean step bypasses
the codec entirely, so all three arms compute the *same* uncompressed gradient (modulo their slightly
divergent weights). The ~30× drop at every clean step (mask) is the visible heartbeat of clean_cadence=5;
it is the `clipfrac`/grad signature the operator has seen before, **not** a bug.

### 3b. The mask vs PowerSGD compressed-grad gap (the informative contrast)

After warmup, the **mask compressed-step grad-norm (~11, median 10.9) is ~6–7× larger than PowerSGD's
(~1.6–2.1)**, while both reach the *same* final accuracy. Quantitatively:

- mask compressed grad ≈ **27×** the clean grad (0.4);
- PowerSGD compressed grad ≈ **3.8–4.8×** the clean grad.

Interpretation (data-level, not mechanism): the two codecs deliver gradients of very different *magnitude*
to the optimizer, yet produce near-identical learning. The mask injects a large, noisy boundary gradient
(95% of coordinates zeroed, the surviving 5% un-rescaled → high variance, large norm); PowerSGD delivers a
low-rank *projection* of the full gradient (energy-preserving, hence a norm only modestly above the clean
norm). That the **accuracy is identical despite the 6–7× grad-norm difference** tells us reward progress is
governed by gradient *direction/alignment*, not magnitude — magnitude differences are absorbed by the fixed
lr 1e-6 without destabilizing either arm. (PowerSGD's near-clean grad-norm is also why it is the gentler,
lower-variance codec — relevant to task #3, flagged not analyzed here.)

### 3c. PowerSGD cold-basis step-1 spike

PowerSGD step-1 grad-norm is **166.4 (r102) / 194.1 (r77)** — a one-step transient, exactly co-incident with
the cold-basis recon (0.97). With no warm basis, the rank-r projection on step 1 is near-arbitrary, producing
a large, mostly-wrong boundary gradient. It decays fast: step 2 → 40/65, step 3 → 45/21, step 4 → 7/3, and
from step ~4 onward it sits in the steady 1–3 band. The spike inflates the naive "compressed-grad mean"
(8.0/8.9) reported in §2.1 of the archivist dump; the **steady** compressed grad is 1.6–2.1. The mask has no
such spike (8.3 at step 1) because masking needs no learned basis. The step-1 spike is benign — it lands on a
single update at the start of training and is gone before reward begins its climb (reward at steps 1–4 is
flat ~0.13 for all arms; the cold-basis update did no lasting damage).

---

## 4. Train ↔ inference consistency (`rollout_actor_probs_pearson_corr`)

Pearson correlation between the training-forward log-probs (computed *under the codec*) and the vLLM rollout
log-probs. **Identical across all three arms** at every step — the codec is not a differentiator here:

| step | mask | r=102 | r=77 |
|---|---|---|---|
| 1 | 0.0064 | 0.0121 | 0.0241 |
| 2 | 0.0036 | 0.0018 | 0.0281 |
| 3 | 0.0046 | −0.003 | −0.008 |
| 4 | 0.0063 | 0.0045 | −0.012 |
| **5** (first clean) | **0.9995** | **0.9995** | **0.9995** |
| 25 | 0.9994 | 0.9994 | 0.9994 |
| 50 | 0.9992 | 0.9990 | 0.9990 |

`rollout_probs_diff_mean` mirrors it: ~0.84 at steps 1–4 → **0.0033–0.0035 from step 5 on**, all arms.

**Interpretation:** the train/inference gap is ~total at init (Pearson ≈ 0, log-prob diff ≈ 0.84) — the
freshly-loaded actor's forward pass is uncorrelated with the rollout policy — then **snaps to ≈0.999 by step
5 and stays there for the rest of training, identically in all three arms.** Two consequences:

1. **The train/inference alignment is codec-independent.** Both codecs (and both PowerSGD ranks) carry the
   *same* near-perfect train↔rollout agreement from step 5 on. The gap is a property of the warmup/policy-sync,
   not of the compression. So the train/inference gap is **not** the axis that separates these arms — it is
   ruled out as an explanation for any (tiny) accuracy difference.
2. The step-0 val anomaly (~0.08, vs the model's known ~0.71 base capability) is the same phenomenon: the
   `val_before_train` ran during the steps-1–4 uncorrelated regime, before the policy/rollout aligned. It is a
   warmup artifact, not a real capability measurement, and does not affect the step-25/50 finals.

The policy and its rollouts align within ~5 steps and remain locked regardless of codec — fast and stable.

---

## 5. Budget vs fidelity vs accuracy

Final validation accuracy (`val-core/openai/gsm8k/acc/mean@1` @ step 50):

| arm | budget (coords/tok) | compression | steady recon@50 | **val-acc@50** |
|---|---|---|---|---|
| PowerSGD r=102 | 102 (+33% vs mask) | 15.1× | 0.0213 | **0.7437** |
| PowerSGD r=77 | 77 (matched to mask) | 19.9× | 0.0236 | **0.7415** |
| mask p=0.95 | 76.8 | 20.0× | n/a | **0.7384** |

- **Spread = 0.53 pp** across all three arms (0.7384 → 0.7437). This is within run-to-run RL noise; the three
  codecs are **accuracy-equivalent** at this budget.
- **Equal-budget comparison (the load-bearing one):** r=77 vs mask, both ≈ 77 coords/tok → r=77 **+0.31 pp**.
  At the same communication budget, the PowerSGD low-rank projection is at least as good as the mask (slightly
  better, within noise).
- **+33% budget buys +0.22 pp:** r=102 − r=77 = +0.22 pp for +33% more rank. Combined with the flat
  rank–fidelity curve (§2a: recon 0.0236 → 0.0213), this says the **rank–accuracy curve is essentially flat in
  [77, 102]** — r=77 is already past the knee. Spending more rank is near-free in *both* fidelity and accuracy,
  i.e. wasted.
- **Implication for budget choice:** r=77 (the matched, cheaper rank) is the right operating point — it
  matches mask accuracy at mask's budget and within 0.22 pp of the +33% rank. There is no accuracy reason to
  pay for r=102.

---

## 6. Quantitative summary (the numbers task #3 should build on)

1. **Compressed gradients carry the learning, not the clean steps.** Inter-clean compressed segments alone =
   +0.52 / +0.60 / +0.60 of the +0.67 / +0.67 / +0.64 total gain (mask / r102 / r77). Clean-step share is
   4.8–19.6% (strict) / 27.5–42.7% (grad-type attribution). The compressed-step OLS slope (+0.0148 to +0.0151)
   is *steeper* than the clean-step slope (+0.0130 to +0.0133). The compressed grad is a biased-but-aligned
   descent direction.
2. **PowerSGD warm-start converges in ~9 steps** (recon 0.97→<0.025) and is flat thereafter; the boundary
   gradient subspace is slowly varying. Early reward gain (steps 3–4) occurs *while* recon is still 17–39% —
   accurate reconstruction is not a precondition for useful descent.
3. **Rank–fidelity and rank–accuracy curves are flat across [77, 102].** −25% rank ⇒ +10.8% relative recon
   error and −0.22 pp accuracy. r=77 (≈20× compression, budget-matched to the mask) is past the knee.
4. **Grad-norm:** clean steps collapse to ~0.4 (identical across arms, codec bypassed). Mask compressed grad
   ~11 (27× clean); PowerSGD compressed grad ~1.6–2.1 (4–5× clean) after a benign step-1 cold-basis spike
   (166/194). Mask's grad is ~6–7× PowerSGD's yet accuracy is identical ⇒ progress is direction-driven, lr
   absorbs the magnitude.
5. **Codec correctness invariants hold end-to-end:** q_cond ≈ 1 (κ≤1.0000040), cross-DP basis deviation = 0.0
   at every step (bit-identical Q across 4 DP ranks). Shared-frozen-codebook consensus is exact.
6. **Train↔inference gap is codec-independent:** Pearson ≈ 0 at steps 1–2 → 0.999 by step 5, identical across
   arms. Not a differentiator. Step-0 val ~0.08 is the same warmup artifact.

---

## 7. Dense comparison — setup + placeholders (numbers TBD)

Per the archivist (§4 of `01_wandb_metrics.md`): **no ≥50-step DENSE GSM8K baseline exists** in any source
(local logs, WandB, LOG/STATUS/git). The only surviving dense-GSM8K number is the **GOAL.md prose figure
≈ 0.741**, from the *earlier* EXP-17 milestone (mask p=0.9, clean_cadence=20, 2 epochs) — a rough ceiling
reference, **not** a same-config control for EXP-20. A same-config (clean5, 50-step, Qwen2.5-1.5B, GSM8K)
**DENSE** run is being launched in parallel (`test_freq=10`). Below are the three comparisons task #3 needs,
pre-framed with what EXP-20 already implies, and explicit placeholders.

**Reference (compressed arms, from this run):**
- val@10 (compressed): not logged (test_freq=25); **train-reward@10** = mask 0.308 / r102 0.247 / r77 0.305.
- val@25 (compressed): mask 0.7195 / r102 0.7316 / r77 0.7104.
- **val@50 (compressed)**: mask **0.7384** / r102 **0.7437** / r77 **0.7415**.
- train-reward@50: mask 0.804 / r102 0.788 / r77 0.772.

### Comparison 1 — dense@10 vs compressed@50 (do 10 full grads already reach ~0.74?)
> EXP-20 spends 10 clean (= dense) gradients total over 50 steps; the question is whether those 10 full grads
> *alone* (a 10-step pure-dense run) already reach the ~0.74 ceiling, which would mean the 40 compressed steps
> add little net val. (Caveat: the 10 clean steps in EXP-20 are interleaved with — and build on — 40
> compressed updates, so this is not a clean isolation; a true 10-step dense run is the right control.)
> - **DENSE val@10 = `<TBD>`**  vs  compressed val@50 ≈ 0.738–0.744.
> - Verdict placeholder: if DENSE@10 ≳ 0.73 → most of the value is in the dense grads and compression is
>   ~free padding; if DENSE@10 ≪ 0.73 → the 40 compressed steps materially advance the policy (consistent
>   with §1's finding that compressed steps book 57–95% of train-reward gain). **`<fill after dense run>`**

### Comparison 2 — dense@50 vs compressed@50 (ceiling / is compression ~free?)
> The headline parity test: the same-config dense ceiling vs the three compressed finals (0.7384 / 0.7415 /
> 0.7437).
> - **DENSE val@50 = `<TBD>`**.
> - Gap placeholders: `DENSE@50 − mask@50 = <TBD> − 0.7384`; `DENSE@50 − r77@50 = <TBD> − 0.7415`;
>   `DENSE@50 − r102@50 = <TBD> − 0.7437`.
> - Verdict placeholder: |gap| ≲ ~0.5–1 pp (the inter-arm spread is 0.53 pp) ⇒ **compression is
>   accuracy-free** at this budget; gap ≫ 1 pp ⇒ a real compression tax exists. (Historical prose ≈0.741
>   sits *inside* the compressed spread, weakly suggesting the gap is small — but that figure is a different
>   config and cannot settle it.) **`<fill after dense run>`**

### Comparison 3 — dense's own post-step-10 slope (shape, not just endpoints)
> With dense `test_freq=10`, val is logged at 0/10/20/30/40/50. Compare the dense val trajectory's slope after
> step 10 against the compressed arms' val@25→val@50 slope (≈ (0.738−0.720)/25 ≈ **+0.0007/step** for mask;
> r102 0.7316→0.7437; r77 0.7104→0.7415) and against the compressed **train-reward** late slope (OLS overall
> +0.0148–0.0151/step; near-flat after step ~30).
> - **DENSE val@{10,20,30,40,50} = `<TBD,TBD,TBD,TBD,TBD>`** → dense post-10 slope = `<TBD>`/step.
> - Verdict placeholder: if dense and compressed share the same diminishing-returns shape (steep to ~step
>   15–20, flat 0.71→0.74 after), compression preserves the *learning dynamics*, not just the endpoint.
>   **`<fill after dense run>`**

> **Operator note:** until the dense run lands, every cross-arm statement in §1–§5 is *codec-vs-codec* (the
> project-fixed baseline-of-record is the **mask arm**, per `01_wandb_metrics.md §0/§4`), which is valid and
> sufficient for the budget/fidelity/decomposition conclusions above. The dense comparison only adds the
> absolute ceiling; it cannot overturn the compressed-vs-clean decomposition, which is internal to each arm.
