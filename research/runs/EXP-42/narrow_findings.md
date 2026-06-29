# EXP-42 narrow study (196 decoder matrices) — interim findings

Source: `research/scripts/weight_proj_sweep.py runs/EXP-42` → `report.html` + `sweep_narrow.json`.
Runs: regimeA `er0syc3n` (val@80=0.7695), regimeB `0tpez2fz` (val@80=0.0788, codec-only r=77 collapse).
Both 160 ticks, 196 matrices, k=4096. WandB tails backfilled (both lastHistoryStep=80).

## Operating point Δ=10 (median over 196 matrices × valid anchors)
| regime | metric | h=5 (α0.5) | h=10 (α1.0=K) | h=20 (α2.0) |
|---|---|---|---|---|
| A (clean) | w1_p50 | 0.8492 HELPS | 0.9718 HELPS | 1.1729 no |
| A | dir_cos | 0.6261 | 0.5490 | 0.4432 |
| A | w1 [p10,p90] | [0.80,0.90] | [0.92,1.05] | [1.10,1.34] |
| B (codec) | w1_p50 | 0.9298 HELPS | 1.0829 no | 1.3493 no |
| B | dir_cos | 0.5708 | 0.4799 | 0.3654 |
| B | w1 [p10,p90] | [0.66,1.19] | [0.75,1.39] | [0.92,1.73] |

Crossover h* (largest h with median w1<1): A Δ10=10, Δ5=13 ; B Δ10=5, Δ5=8.

## Verdicts
- **H1 (premise) PASS for clean regime**: at h=K=10, median w1<1 (0.972) AND dir_cos>0 (0.549).
- **H1 for compressed regime**: at h=10 w1>1 (1.083) — projection does NOT help at the operating
  horizon; helps only to h≤5.
- **H2 (overshoot) confirmed**: w1 rises past 1 as h grows (A>1 at h≥20, B>1 at h≥10). BUT
  **dir_cos stays POSITIVE everywhere (0.37–0.63)** → NO sign-flip in weight space at these
  horizons. The overshoot is MAGNITUDE (α scales the step past θ_now along a positively-correlated
  direction), not direction reversal. Refines the prior-collapse "sign-flip" hypothesis: the
  sign-flip seen in prior ANCHOR-COSINE runs is not a weight-space direction reversal at h≤20.
- **H3 (regime effect) PASS**: compression halves the crossover (10→5 at Δ10) and lifts w1 at every
  horizon — the codec makes the weight trajectory less linearly predictable. regimeB's wider
  [p10,p90] band (e.g. [0.75,1.39] at h10 vs A's [0.92,1.05]) shows much higher per-matrix variance.
- **fixed ≈ learned**: identical to 4 dp — the per-matrix scalar-mean residual is inert (a uniform
  shift barely moves norm/direction), as predicted.

## Gradient-study GATE (per Analyst predicate)
- Worth doing for the CLEAN regime at fixed_linear, h≤10 (Δ=10): w1<1 AND dir_cos>0 at the
  operating horizon. Winning (method,Δ,h) = (fixed_linear, 10, ≤10).
- NOT worth doing for the COMPRESSED regime at h=10 (w1>1); only marginally at h≤5.

## RLVR-linear-paper check (per-matrix linearity R squared, decoder matrices, preliminary)
Per-matrix R squared = mean over ticks of cos squared between the previous displacement
(theta[t] - theta[t-s]) and the next displacement (theta[t+s] - theta[t]), at scale s ticks
(2 ticks per global step). Computed from the narrow decoder sketches (196 matrices).

| scale | regime A (normal GRPO) R2 p50 | regime B (codec) R2 p50 |
|---|---|---|
| 1 tick (0.5 step) | 0.800 | 0.593 |
| 5 ticks (2.5 steps) | 0.532 | 0.389 |
| 10 ticks (5 steps, = K) | 0.319 | 0.182 |

Read: the RLVR-linear claim "weights move roughly linearly, R squared about 0.9" is APPROXIMATELY
true for the normal GRPO run only at the FINEST scale (consecutive ticks, R2 about 0.80, in the
ballpark of but below 0.9), and it DECAYS sharply with scale: by the K=10 staleness scale the
decoder trajectory is no longer linear (R2 about 0.32). This is the mechanism behind the crossover:
the look-ahead can project a few steps (where the path is locally straight) but overshoots once the
horizon reaches K, because the path has curved by then. Compression (regime B) lowers the linearity
at every scale (R2 0.59 at 1 tick), consistent with its earlier crossover. NOTE: this is the decoder
set only; the full dense-only report over all groups (decoder, embed, norm, bias) is produced in the
deferred new session (see NEW_SESSION_PROMPT.md), which also re-collects the dense run widened.

## Sketch fidelity (calib, tol 5%)
- regimeA: all_pass (rel_err 0.49% / 1.08% / 1.47% at h=5/10/20).
- regimeB: 2/3 pass — h5 rel_err 5.85% (marginally over), h10 3.85%, h20 0.10%. The sketch and the
  exact calib AGREE on every helps/no-help verdict (calib h5=0.988<1, h10=1.043>1), so the crossover
  h*=5 is robust; the 5.85% is attributable to anchor-sampling (15 calib anchors vs ~28k sweep
  samples) in the collapsed regime, not a count-sketch reconstruction error (CPU parity is bit-exact
  and regimeA passes cleanly with identical machinery).

## Dense-run v2 deeper analysis (regimeA, GPU-free, report_dense_v2.html)
Builder runs/EXP-42/build_dense_report_v2.py (sibling of build_dense_report.py). All five studies
computed from the regimeA decoder sketch only; rel std about 1/sqrt(k) about 1.6 percent stated where
it matters. Coherent read: the dense GRPO trajectory is globally near-linear and low-rank, but the
look-ahead's two-point slope overshoots, so a damped coefficient is the lever.

(a) Low-rank displacement subspace + global linearity:
| measure | p10 | p50 | p90 |
|---|---|---|---|
| participation ratio (per-tick increments, ceiling 159) | 5.5 | 7.6 | 10.7 |
| components for 90% increment energy | 20 | 26 | 32 |
| participation ratio (cumulative displacement) | - | 1.2 | - |
| top-PC energy fraction (centered trajectory) | - | 0.69 | - |
| global line-fit R^2 (centered) | 0.66 | 0.68 | 0.71 |
| global line R^2 (through-origin extrapolation) | 0.83 | 0.85 | 0.86 |
RLVR low-rank claim SUPPORTED (temporal sense). The global line-fit R^2 (0.85 through-origin)
reconciles the prior local consecutive-step R^2 (0.80 at 1 tick, 0.32 at K=10) with the cited paper's
global about-0.9: same data, one slow drift plus per-step noise; the local metric sees the noise, the
global metric sees the drift. Sketch noise inflates the residual so 0.85 is a lower bound. NOT
computable, not claimed: matrix-native (LoRA) singular rank of a single weight matrix (flatten+sketch
destroys it); embeddings / RMSNorm gains / biases (not collected, decoder only).

(b) Per-matrix projectability: crossover h* tight 9 to 14 ticks (median 11); ratio@10 spans
[0.956, 0.985]. Furthest: attention v_proj / o_proj in mid-to-late layers (h* 13 to 14). Least: MLP
and attention k/q_proj (h* 9 to 10). h* histogram {9:1, 10:56, 11:114, 12:21, 13:2, 14:2}.

(c) Optimal-coefficient sweep (Delta=10): naive alpha=h/Delta overshoots at every horizon.
| h | naive alpha | ratio@naive | opt alpha | ratio@opt | gain |
|---|---|---|---|---|---|
| 5 | 0.50 | 0.849 | 0.33 | 0.780 | 0.069 |
| 10 | 1.00 | 0.972 | 0.53 | 0.836 | 0.135 |
| 20 | 2.00 | 1.173 | 0.74 | 0.897 | 0.276 |
Optimal alpha is well below naive (about 0.5 to 0.75), and at h=20 it keeps the ratio below 1 where
naive fails. Mechanism: the two-point slope over-states the persistent drift (it also captures
per-step noise), so naive extrapolates too far; damping corrects it. Gains far exceed the 1.6 percent
sketch floor. Caveat: in-sample oracle, but alpha is stable, so a fixed damped coefficient near 0.5 is
a deployable change to the look-ahead rule (validate in the compressed regime, h*=5 there).

(d) Linearity vs projectability: more-linear -> more-projectable. Spearman 0.45 (R^2@1 vs h*),
-0.51 (R^2@1 vs ratio@10, negative is better). Significant at n=196. Real but loose.

(e) Learned vs fixed residual: inert on the dense run. max|resid| 2.7e-9 (clip 1e-3), residual-term
rel-norm about 1e-4, |delta ratio| at most 6.5e-5, all below the 1.6 percent sketch floor. A scalar
shift barely moves a high-dim displacement; the per-matrix mean drifts smoothly so the fixed
extrapolation has near-zero retrospective error.
