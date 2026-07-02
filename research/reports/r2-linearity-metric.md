# R² Trajectory-Linearity — canonical metric rationale + contract

Single source of truth for the **per-scalar R² linearity metric**. Other lanes/issues LINK here.
Grounded in Wang et al. 2026, *Linear Dynamics in the RLVR Training of Large Language Models*
([arXiv:2601.04537](https://arxiv.org/abs/2601.04537), code `Miaow-Lab/RLVR-Linearity`, training
codebase adapted from verl — our own substrate). Repo evidence: `runs/MOAT-45-ANALYSIS/report.html`
(§"How this compares to Wang et al."), `runs/MOAT-45-ANALYSIS/verdict.md`, `LOG.md` (EXP-45),
`scripts/moat_scorecard.py` (`stream_stats`/`group_diagnostics`), `scripts/weight_proj/metrics.py`.

## 1. TL;DR directive

- **`linearity_r2` (per-scalar OLS R² vs step) is the PROGRAM'S PRIMARY linearity metric** — a MUST-measure
  in EVERY projection lane, reported beside the paper anchors, and the quantity we optimize the projection
  mechanism against toward the MOAT goal.
- **`weight_proj_ratio` stays the projection-PERFORMANCE metric** (out-of-sample, lower=better). Unchanged.
- **`traj_r2` is a LEGACY per-matrix proxy** kept for continuity. Do **not** conflate it with "linearity"
  in the paper's sense — it is a different statistic (see §2, §5).

## 2. Exact definition

Per **individual scalar parameter** `θ[i]`, fit an OLS straight line of its value against the training-step
index `t`, and take the coefficient of determination:

```
R²[i] = 1 − SS_res / SS_tot                (0 ≤ R² ≤ 1, higher = more linear)
```

- **Threshold:** `R² > 0.7` ≡ "strongly linear" (Wang et al.'s cut).
- **Reporting convention:** median R² over scalars **and** `Pr(R² > 0.7)` (fraction of strongly-linear scalars).
- **Constant-scalar exclusion:** scalars with `SS_tot ≈ 0` (never move) have undefined R² and are EXCLUDED;
  **count** how many are dropped.
- **Same recipe** applies per-token to log-probs (their §B.2) and per-neuron to activations (their Fig 11);
  weights are our first target.
- Stated by the paper in prose (their §4.1) — never as a numbered equation.

**Per-scalar vs per-matrix — the load-bearing distinction.** The paper's R² is *per individual scalar*.
Our existing `traj_r2` is a *per-matrix* variance-weighted single-slope *vector* fit (one slope vector for
the whole flattened matrix), so it is dominated by the largest-moving components and is **not comparable
1:1** to the per-scalar number. They are related statistics, not the same test.

## 3. Why it is a MUST

1. **Comparability with the domain's most influential result.** Wang et al. is the reference paper for
   RLVR weight dynamics; their Table 7 anchors (§5) only mean something for us if we compute the *same*
   per-scalar statistic. `traj_r2` cannot be read against those anchors.
2. **Method-agnostic diagnostic of the trajectory itself.** R² is computed from the weight path alone,
   before any predictor is chosen. It predicts *where* and *for how long* ANY linear projection can work —
   it is the governing quantity, not an artifact of one method.
3. **Their mechanism says our regime may have LESS of it.** Their §5 mechanism: high-variance verifiable
   reward acts as a low-pass filter, so the gradient hugs a fixed direction — linearity is RL-specific and
   noise-driven (SFT ≈ 0.4 vs RL ≈ 0.8). GRPO on *easy* GSM8K → higher pass rate → less reward noise →
   plausibly WEAKER linearity. We must **quantify our own regime**, not assume theirs.
4. **It is the optimization target/monitor toward MOAT.** Optimizing the projection mechanism needs the
   governing quantity per-layer and per-block: where R² is high → project; where low → re-ground or skip.
5. **It closes EXP-45's open question.** Our 0.685 per-matrix number is explicitly flagged as *not*
   apples-to-apples with the paper's 0.845 / 0.426 anchors. Per-scalar R² is the missing measurement.

## 4. The three-metric hierarchy

| Metric | Statistic | Sample | Direction | Role | Question it answers |
|---|---|---|---|---|---|
| **`linearity_r2`** | per-**scalar** OLS R² vs step | in-sample | higher=better | **PRIMARY diagnostic** | Is the weight path itself linear? |
| **`weight_proj_ratio`** | `‖θ̂(t+h)−θ(t+h)‖ / ‖θ(t)−θ(t+h)‖` | out-of-sample | lower=better (hold-stale ≡ 1) | performance | Can a chosen predictor exploit it? |
| **`traj_r2`** | per-**matrix** variance-weighted vector fit | in-sample | higher=better | legacy proxy (continuity) | Roughly how straight is the matrix as a whole? |

## 5. Reference anchors

Wang et al. Table 7 (weight-R² median / `Pr(R²>0.7)`), beside ours:

| Source · config | weight-R² median | Pr(R² > 0.7) | note |
|---|---|---|---|
| Paper · RL runs (range) | 0.732 – 0.868 | 0.702 – 0.824 | reasoning-distilled, hard math |
| Paper · R1-Distill-Qwen-1.5B · GRPO / DeepScaleR | **0.845** | **0.794** | **nearest analog to ours** |
| Paper · Qwen2.5-1.5B + GSM8K · SFT | 0.426 | 0.259 | their ONLY matching-base row |
| EXP-45 · Qwen2.5-1.5B + GRPO + GSM8K — `traj_r2` | 0.685 | 0.34 | **different statistic** (per-matrix) — not 1:1 |
| **EXP-45 · same — per-scalar `linearity_r2`** | **unmeasured** | **unmeasured** | **EXP-47's job to fill** |

Our `traj_r2` 0.685 plausibly sits *between* their SFT ≈ 0.43 and RL ≈ 0.8 — consistent with the
noise-driven mechanism predicting weaker linearity on easy GSM8K — but this is qualitative until the
per-scalar number exists.

**Bridge to their extrapolation payoff.** Their extrapolation factor `β = (t′−t0)/(t1−t0) = 1 + κ`, where
`κ = h/Δ` is our horizon/spacing ratio. Their accuracy peak `β ≈ 3` maps to our `κ ≈ 2`. Both see a
sweet-spot-then-degradation shape (their Fig 5 inverted-U; our `weight_proj_ratio < 1` only at short h,
1.404 at Δ=20,h=40, `h* = 5`).

## 6. How to measure it on our trace (implementation note for the lanes)

Per-scalar R² rides the **SAME one-time streaming pass** as the stats-cache build in
`moat_scorecard.stream_stats` — **no extra trace read**. That pass already keeps, over the `N` streamed
snapshots, per-element f64 accumulators (`φ_t = θ_t − θ_0`; subtracting `θ_0` does not change R²):

```
V = Σ_t φ_t                 (already accumulated — u.V)
W = Σ_t (t − t̄) φ_t         (already accumulated — u.W)
```

Add **ONE** more per-element accumulator:

```
P = Σ_t φ_t²                (NEW — per element)
```

Then per element, with `S_tt = N(N²−1)/12` (constant, same for all scalars at a fixed cadence):

```
SS_tot = P − V²/N
SS_reg = W² / S_tt
R²     = SS_reg / SS_tot = (W² / S_tt) / (P − V²/N)
```

- **Exclude** scalars with `SS_tot ≈ 0` (constant weights); **count** the excluded.
- **Reduce per matrix** to `{median R², Pr(R²>0.7), histogram}` and store those in the stats cache. Do **NOT**
  persist the ~1.5B raw per-scalar R² values.
- **Cost:** one extra per-element vector accumulator + ~2n flops/tick (square-and-add). No extra I/O.
- `N` / cadence is a parameter: the tick pass gives per-tick R² for free; the per-step (PRIMARY) R² uses the
  step-cadence snapshots (1 step = 2 ticks in the EXP-57 trace). Contrast: today's `traj_r2` in
  `group_diagnostics` uses the *aggregated* `sum_phi2 / v2 / wc2` scalars (norms over elements), which is why
  it collapses to one per-matrix vector fit rather than a per-scalar distribution.

## 7. Reporting convention

Every projection lane reports:

- **Global** median R² + `Pr(R² > 0.7)`, always beside the paper anchors (§5), plus the excluded-constant count.
- **Per `block_type` / `super_block` / `layer_idx`** (where R² is high → project; low → re-ground/skip).
- **Both cadence regimes:** per-step (**PRIMARY**, matches the paper's "vs step") and per-tick (secondary).
- **R²-vs-ratio coupling:** do high-R² groups project better (lower `weight_proj_ratio`)? This is the link
  between the diagnostic and the performance metric.
</content>
</invoke>
