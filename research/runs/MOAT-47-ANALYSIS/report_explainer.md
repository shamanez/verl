# EXP-47 report explainer — verified explanatory content

## What this analysis is

We recorded the full weight trajectory of a GRPO run — every one of the 338 weight
matrices of Qwen2.5-1.5B-Instruct, saved in fp32 at 160 optimizer-tick snapshots
during training on GSM8K (EXP-57). This study never trains anything; it *replays*
that saved trajectory offline and asks one question that matters for
communication-efficient distributed training: **if a worker is holding a stale copy of
the weights, can it do better by locally extrapolating those weights forward than by
doing nothing (or than by waiting for a fresh copy)?** Each method is graded by the
`weight_proj_ratio` = ‖predicted − true‖ / ‖stale − true‖. A ratio **below 1 means the
prediction beats holding the stale weights**; exactly 1 is "no better than doing
nothing"; above 1 is actively harmful. The lane PASSES by producing a *correct,
schema-verified, decisive answer in both cadence regimes* — a negative result (nothing
beats stale) would be an equally valid PASS. The direction here happened to be
favorable.

## Where Δ comes from: the sliding-window protocol

This is the single most misread part of the report, so read it slowly. **Δ is not a
property of one fixed checkpoint pair.** It is the *spacing between the two anchor
snapshots* that a linear method uses, and for every cell `(Δ, h)` we slide that anchor
pair across the *entire* trajectory and average the result.

Concretely, for a cell `(Δ, h)`: let `t` be the position of the *later* anchor. A valid
window needs `t − Δ ≥ 0` (the earlier anchor exists) and `t + h ≤ n − 1` (the true
future target exists). So `t` ranges over `Δ … n − 1 − h`, giving
`n_windows = n − h − Δ` windows. Each window:

1. reads the two snapshots `θ_{t−Δ}` and `θ_t`,
2. extrapolates a straight line `h` steps forward to get `θ̂_{t+h}`,
3. scores it against the *true* snapshot `θ_{t+h}` (never seen by the predictor).

The reported number for the cell is the **median `weight_proj_ratio` over all those
windows** (p10 and p90 are stored too, for the error bands).

**Worked example — the real regime-S operating point.** n = 80 global steps,
Δ = 10, h = 10. The later anchor `t` runs 10 … 69 → **60 windows**. But the damped
method also has to *choose its damping λ out-of-sample*: for the window at anchor `t` it
may only look at earlier windows whose target `t′+h` already lies strictly before `t`.
The first anchors have no such earlier evidence, so the first `h + 1 = 11` windows are
**warm-up** and are dropped (not scored with a leaked λ). That leaves
**49 out-of-sample-scored windows**, whose median ratio is **0.940** at the selected
**λ\* = 0.3**. (All four numbers — 60, 11, 49, 0.940 — were reconfirmed directly from
`scorecard-perstep/scorecard.jsonl`.)

```
global step:  0    10        20        ...              69   79
              |----Δ=10----> t                                 (anchor t slides →)
              θ_{t-10}     θ_t  --extrapolate h=10-->  θ_{t+10}   (scored vs truth)

 t = 10,11,...,20,21, ................................ ,69     → 60 windows
      \___ first 11 (t=10..20) are OOS warm-up: no earlier target < t ___/
      then t = 21 ... 69                                        → 49 scored
```

## Which parameters were used

**All 338 weight tensors of Qwen2.5-1.5B-Instruct — nothing is subsampled.** By
`block_type` the counts are: `q_proj`/`k_proj`/`v_proj`/`o_proj` = 28 each (28 decoder
layers), `gate_proj`/`up_proj`/`down_proj` = 28 each, `norm` = 57 (28 input + 28
post-attention layernorms + 1 final norm), `bias` = 84 (the Qwen2.5 q/k/v projection
biases, 3 × 28), and `embed` = 1. `lm_head` is **tied to the embedding**, so it carries
no separate tensor (`lm_head` count = 0) and is reported as an explicit `tied=true` row
that mirrors the embedding. Grouped by `super_block`: attention 112, mlp 84, norm 57,
bias 84, embed 1. Total ≈ 1.54 billion scalar parameters.

The **`global` row concatenates every one of those 1.54B scalars into a single vector**
and computes one ratio on it — a *true joint* ‖·‖-ratio, **not** an average of
per-matrix ratios (group sums are sums of member Gram statistics, per the module's
block-sum design). Every group row (`block_type`, `super_block`, `layer`,
`layer_block`, `special`) is the same joint ratio over its members. For the per-scalar
linearity R², **every individual scalar gets its own R²**; scalars that never move
(`SS_tot ≤ 1e-300` under fp32) have undefined R² and are **excluded and counted** —
**4,227,167** in regime S (≈ 0.27 % of all scalars), 4,220,204 in regime T.

## Glossary

- **weight_proj_ratio** — ‖θ̂_{t+h} − θ_{t+h}‖ / ‖θ_t − θ_{t+h}‖. The core score;
  lower is better, `< 1` beats the stale copy. Reported as median over windows.
- **stale_error** — ‖θ_t − θ_{t+h}‖: how wrong you are if you just keep the stale
  weights (the ratio's denominator). **proj_error** — ‖θ̂_{t+h} − θ_{t+h}‖: how wrong
  the *prediction* is (the numerator).
- **skill** — a monotone rephrasing of the ratio (0 when ratio = 1); positive = the
  predictor helped. Same information as the ratio.
- **h (horizon)** — how many steps ahead of the latest anchor we predict/score.
- **Δ (anchor lag)** — spacing between the two anchor snapshots the linear method uses.
- **window / anchor** — one placement of the anchor pair at position `t`; the analysis
  slides `t` over all valid positions.
- **n_windows** — number of valid windows for a cell = `n − h − Δ`.
- **h_star / h_safe** — the largest `h` at which the global median ratio is still `< 1`
  (the farthest you can safely project). Reported per regime, in that regime's unit.
- **λ (damping)** — shrink factor on the extrapolation step: `θ̂ = θ_t + λ·(h/Δ)·(θ_t −
  θ_{t−Δ})`. λ=0 ⇒ hold stale; λ=1 ⇒ full (naive) linear.
- **λ\* (OOS-selected λ)** — the λ chosen per scored window from strictly-earlier
  windows only. **oracle λ** — the λ that minimizes the *in-sample* ratio (cheating;
  used only to show the honesty gap).
- **hold_stale** — do nothing, θ̂ = θ_t; ratio ≡ 1 by construction.
- **naive_linear** — full linear extrapolation, κ = h/Δ (i.e. λ=1).
- **damped_linear** — naive_linear with an out-of-sample-selected λ ∈ [0,1].
- **paper_linear** — Wang et al. 2026's protocol: same secant, but the *anchor rule*
  differs (see below).
- **β = 1 + h/Δ_resolved** — the extrapolation factor; the "how far past the endpoint"
  ratio. It is **not** a fitted coefficient — it is `1 + h/Δ`.
- **Δ_resolved** — the *effective* Δ actually used by a window. For fixed-Δ methods it
  equals Δ; for paper_linear it grows with `t`.
- **anchor_mode** — `fixed` (naive/damped, fixed short Δ) or `frac25` (paper_linear,
  anchor at 25 % of elapsed training).
- **cadence** — which snapshots we treat as the time axis. **per-step** (regime S) =
  the even ticks [0,2,…,158], 80 global steps, PRIMARY. **per-tick** (regime T) = all
  160 optimizer ticks. The trace has **2 optimizer ticks per global step**.
- **unit** — `global_step` (regime S) vs `tick` (regime T); (Δ,h) are measured in it.
- **band** — the Gram cache holds all pairwise delta-inner-products up to a lag of
  `band = max(Δ)+max(h)` (60 in S, 80 in T); this is why every cell's score is an
  exact block-sum with no re-reading of the trace.
- **per-scalar linearity R²** — for each scalar, the R² of its value vs step index
  (`1 − SS_res/SS_tot`); the PRIMARY diagnostic. **traj_r2** — a **different, legacy**
  object: one variance-weighted *per-matrix* vector-fit R² (median ≈ 0.68 here). Never
  read traj_r2 against the paper's per-scalar anchors.
- **Pr(R²>0.7)** — fraction of (non-constant) scalars that are "strongly linear."
- **n_excluded_const** — count of never-moving scalars dropped from the R² stats.
- **Spearman ρ** — rank correlation between per-group median R² and per-group damped
  ratio; the "does linearity predict projectability" coupling.
- **breakers** — groups whose ratio ≥ 1 at the operating point (would break the global
  "projection helps" conclusion). Here: none.
- **block_type / super_block / layer_block** — taxonomy axes: fine tensor type (10),
  coarse role (5), and per-(layer,type) cells (252 decoder cells).
- **tied lm_head** — the output head shares the embedding tensor; reported as `tied`.
- **operating point** — the headline cell: (Δ=10,h=10) in S, (Δ=20,h=20) in T.
- **dir_cos** — cosine between the predicted-error direction and the stale displacement
  (a geometry diagnostic, reported alongside the ratio).

## What was verified before trusting these numbers

`selftest.log` records 18 invariants plus a real-trace battery, all PASS (`SELFTEST:
GO`), grouped by what they defend against:

- **Math correctness.** damped λ=1 reproduces naive_linear *bit-for-bit* and λ=0
  reproduces hold_stale, in **both** regimes (worst |diff| = 0.0). The fast block-sum
  scoring path was re-checked against a direct tensor recompute
  (`predictors.Order1` + `metrics.full_metric_row`): worst relative diff **0.0** on
  the real trace, **0.0** on a damped λ=0.4 cell. The per-scalar R² accumulator was
  checked against a direct numpy per-element fit: worst diff **≤ 4.4e-16**. paper_linear
  was proven *formula-identical* to naive_linear at matched anchors: direct-vs-block-sum
  agreement **4.37e-14** relative.
- **No cheating.** The OOS λ selector can only see windows whose target lies strictly
  before the current anchor; an intentionally leaking split **trips the assert** (the
  guard fires). The score point `t+h` is never in the selection set.
- **Data integrity.** All 338 matrices partition exactly with `other = 0`; every row's
  `n_windows == n − h − Δ` (0 violations); the only NaNs come from the denominator guard
  (0 stray NaNs); every emitted R² lies in [0,1].
- **Reproducibility.** Re-streamed statistics and recomputed rows are **byte-identical**;
  the cache is keyed by a fingerprint that folds in cadence + tick-set + schema version,
  so regime S and regime T can never share a cache file.
- **Cross-check vs the earlier #45 study.** per-tick naive_linear at (Δ=20,h=20) =
  **1.1580**, matching #45's 1.158 (the band-80 rebuild did not perturb the shared cell).

## Reading each figure

- **R² histogram** — x: per-scalar R² in [0,1]; y: count of scalars. A right-heavy
  distribution = strongly linear weights. Ours peaks in the middle: **median 0.535**,
  with **33.5 %** of scalars above 0.7 — moderate linearity, not the near-linear regime
  of hard-math RL.
- **R²-vs-ratio coupling scatter** — x: per-group median R²; y: per-group OOS-damped
  ratio at the operating point; one dot per block_type/super_block/layer group (43
  dots). Down-and-to-the-right = high-R² groups project better. **Spearman ρ = −0.75**
  (per-step), −0.67 (per-tick): a strong negative coupling — linearity predicts
  projectability, the signal future optimization can steer by.
- **Depth×block R² heatmap** — rows: layer 0…27; cols: block_type; color: per-scalar
  median R². This is the *per-scalar* map — visually and numerically distinct from the
  legacy `traj_r2` heatmap. R² is fairly uniform across depth/type (super-block medians
  span only 0.515 embed → 0.575 bias), i.e. no layer is dramatically more linear.
- **Accuracy-vs-horizon curves (both regimes)** — x: horizon h; y: median ratio, one
  line per method, dashed line at ratio = 1. naive_linear crosses 1 early and shoots up
  (per-step 0.89→1.16 by h=10→1.78 by h=40); OOS-damped stays under 1 far longer
  (per-step 0.85→0.94→0.978 at h=30). Damped is the only line still under 1 at large h.
- **Δ-sensitivity curve** — x: Δ (extended to 40 in regime T); y: median ratio at the
  operating h. The line **rises monotonically** with Δ (per-tick @h=20: 0.871, 0.906,
  0.940, 0.951, 0.967, 0.971 for Δ = 5,10,20,25,35,40). Takeaway: **wider anchors hurt;
  best_delta = 5**, the smallest.
- **λ-selection curve** — x: λ from 0 to 1; y: in-sample median ratio for a cell. A
  clean bowl: at (10,10) it runs 1.000 (λ=0) → **min 0.937 at λ=0.3** → 1.158 (λ=1).
  The dip location is why the selector picks moderate damping, not λ=0 or λ=1.
- **Layer×block ratio / h_star heatmaps** — rows: layer, cols: block_type; color: the
  OOS-damped ratio (or the safe horizon) at the operating point. Uniformly below 1 (and
  h_star uniformly positive) — no dark "breaker" cell anywhere.
- **Special-groups table** — embed/norm/bias/lm_head rows. All project (ratios embed
  0.943, norm 0.935, bias 0.924, lm_head 0.943 — lm_head = embed because it is tied);
  each also below 1, so no special group breaks the conclusion.
- **Paper-equivalence table/panel** — matched-(t,h) comparison of paper_linear vs
  fixed-Δ naive vs OOS-damped, plus the per-window β distribution. paper_linear starts
  *worse* than naive at short h (h=1: 1.009 vs 0.888) but **crosses over to beat naive by
  h≈10** (1.111 < 1.158) as its wide, noise-averaged slope pays off; β climbs
  monotonically with h over **[1.02, 3.67]**. OOS-damped still beats both at every h.

## The findings in one paragraph

Our GSM8K GRPO run is **moderately linear**: per-scalar R² median **0.535**,
Pr(R²>0.7) **0.335** (identical across regimes) — sitting between the Wang et al. SFT
floor (0.426 / 0.259) and the strong-RL analog (0.845 / 0.794), and R² **predicts
projectability** (Spearman ρ = **−0.75** per-step). At the primary operating point
(Δ=10, h=10 global steps), OOS-**damped_linear = 0.940** (λ\*=0.3) beats both
naive_linear (**1.158**, actively harmful) and hold-stale (1.0): **local projection
helps.** **Wider Δ never helps** — best_delta = **5**, ratio rising monotonically to
Δ=40. Damped stays under 1 out to **h_safe = 30 global steps** (40 ticks), roughly 15×
farther than naive (safe only to h=2 / h=5). **No group breaks** the conclusion.
Finally, paper_linear's wide proportional window is worse than fixed-short-Δ naive at
short horizons but overtakes it by h≈10 (β ∈ [1.02, 3.67]); OOS-damped beats both at
every horizon.
