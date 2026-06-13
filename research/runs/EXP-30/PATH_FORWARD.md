# EXP-30 → Path Forward (team synthesis)

> Synthesized by the orchestrator from the exp30-pathforward team's three reviews:
> `pathforward/mechanist.md` (mechanism, numbers recomputed from the per-target sidecar),
> `pathforward/critic.md` (adversarial validity), `pathforward/strategist.md` (ranked program).
> Companion to `verdict.md` (PASS — unchanged). Written while the de-censoring run
> `exp30_B2_ext100` is in flight; its outcome slots into §4 without rewriting this doc.

## 1. Where the program stands (one paragraph)

EXP-30 produced the program's first emission-free, converting correction: B2 (K-delayed exact codec
residual, λ=1, β_anc=0 on the EXP-29 paired-replay substrate) reached best val@50 = 0.7528 vs the
0.7210 realistic floor, with zero post-warmup emission — while the geometry gate retired the entire
blend-on-valid-M class for ~4 GPU-hr without spending a training cell. The honest statistical read
(critic T2): 0.7528 clears the pre-registered floor rule at the point-estimate level (~1.9σ as a
difference of proportions). **DENSE BASELINE CORRECTED 2026-06-13:** the apples-to-apples dense is now
the same-code, same-hyperparameter rerun `exp30_dense_rerun` (`73ntu76u`) = **0.7839** (old `5e2jpho9`
0.7536 was old code); the dense val@50 is a **band ≈ 0.75–0.78** (rollout nondeterminism ≈ ±0.024/draw).
Against the same-config dense (0.7839), B2 is **−0.031 (≈96% of dense)** — so the honest claim is
**"near-parity, NOT established"**: B2 reaches the old-dense draw but sits ~3 pts below the current-code
draw, within ~1.3 nondeterminism-σ. Parity-vs-dense is unresolved at n=1 seed; the binding fix is seed
replicates of BOTH B2 and dense (R2/R3). The 50-step stability result is censored (EXP-27 ignited at ~61);
ext100 cleared that band for seed 0.

## 2. What we now know (mechanism — numbers recomputed, not inherited)

- **F1 sharpened (the prize).** At identical (batch, θ): pooled cos(G_anc_rep, G_comp_ring) = **+0.007**
  (per-target uniform — only 6.9% of the 196 matrices have |cos| > 0.2; min −1.000; NOT a median
  artifact), settled norm ratio **‖G_true‖/‖G_comp‖ ≈ 0.29**. The codec error is **~92% of the
  compressed gradient's energy**. The fast circuit's gradient is dominated by codec artifact;
  the true gradient rides on it at ~⅓ norm, orthogonally. (Weight-space confirmation of EXP-26's 0.318.)
- **Why the residual wins and the blend can't (the selector).** m1 ≈ 0.012 (cross-batch TRUE-gradient
  correlation: dead) vs m4 j4 ≈ 0.295 (within-circuit CODEC-ERROR autocorrelation: alive). A K-delayed
  correction can only transport structure that survives the delay — the codec error does, the true
  gradient doesn't. λ=1 ring telescoping is exact subtraction of that persistent artifact; an
  orthogonal 92%-energy contamination can be *subtracted* but never *offset* by adding a partner
  (blend algebra). Lever flagged: shrinking delay_K tightens cancellation further.
- **Why plain PowerSGD still trains** (open question, now ranked): primary — Adam's per-coordinate
  normalization on the concentrated true coordinates (top-1% mass ≈ 0.60); secondary — running-moment
  cancellation of the rotating codec error. Cheap closing probe: top-1%-coordinate cos(A, C).
- **m7 re-frames GOAL-3.** The valid gradient has stable rank ≈ 1.9 and top-1% mass ≈ 0.60: r=77 is
  over-provisioned for a rank-~2 object; the defect is basis MISMATCH, not capacity. Live lever:
  compress the (low-rank) RESIDUAL to ≪77 columns — potentially better-than-current savings.
  (Hybrid/update-energy Q stays falsified — EXP-26 Step C.)
- **Carrier law made quantitative (F2).** m6 ≈ 0.62 ⇒ injected-carrier autocorrelation time τ ≈ 10.5
  ticks ≈ 2× cadence — marginal, below EXP-27's compounding τ ≈ 19.5 but NOT memoryless.
  **Falsifiable ext100 forecast (mechanist, registered before the outcome):** no ignition through
  step ~61; P(ignite in 61–100) ≈ 20–35%; m6 crossing 0.85 = trip-wire; val holds/improves absent
  ignition. If the endogenous residual stays emission-free past the EXP-27 horizon, that is the first
  positive evidence for the **endogenous(residual)-vs-exogenous(EMA) carrier discriminator** — the most
  reusable mechanism finding available here.

## 3. Threats to validity (critic — severity · cheapest resolving measurement)

| # | threat | severity | cheapest resolution |
|---|---|---|---|
| T2 | "parity reached" overclaimed: B2−dense z ≈ −0.05 at SE 0.0119; single seed, single val step | HIGH | +1 B2 seed @50 (~5 GPU-hr) — **higher decision value than any single-seed extension**; needs operator approval (outside tonight's authorization) |
| T1 | 50-step emission-free is censored; ext100 clears only the 51–66 band for seed 0 — not >100-step stability, not seed-generality, not EMA successors | HIGH | ext100 (running); seed replicate shares cost with T2 |
| T3 | B2−plain attribution is a 2-delta read (replay knob postdates `u1v94opv`) | MED | #28's plain@100 control doubles as the clean 1-delta reference |
| T6 | bytes_ratio 0.0505 counts only the fast path; anchor-side M-allreduce + Q-broadcast traffic unmeasured — GOAL-3 "parity+savings" not yet closed | MED | paper accounting from existing logs (zero GPU) |
| — | cleared false alarms: KL **off** (runtime dump `'use_kl_loss': False`; the True token is a pre-override launcher echo — resolved_params is the only authority), merger hygiene held, A→B2 diff exact | — | — |

## 4. ext100 decision tree (rank 0 — gates everything; strategist)

| outcome | reading | next move |
|---|---|---|
| (i) emission-free + val ≥ ~0.74 @75/100 | de-censored for seed 0; endogenous-carrier evidence | harden: seed replicate (T2) + honest-bytes (R1) → milestone write-up |
| (ii) emission-free + val decays | stable but Q/anchor staleness binds at horizon | diagnose Q-freeze (Q-rotation telemetry) before any new cell |
| (iii) ignites ~61 | carrier law extends to endogenous residuals; forecast falsified | STOP δ-residual at λ=1; do NOT λ-cap (EXP-27: dose-capping delays, not prevents) |
| (iv) ignites 61–100 late | persistence marginal as τ≈10.5 predicted | same as (iii) + m6 trip-wire analysis feeds successor design |

**No new training cell launches until ext100 returns.**

> **OUTCOME (2026-06-13): branch (i)/(ii) hybrid.** Emission-free through 100 (de-censored for seed 0;
> two isolated 1/1024-rollout cap-pins at steps 94/99 — benign, no P1/P2/P3). Val: 0.7536@50 (= dense
> ceiling) → 0.7475@75 (> parity) → 0.7400@100 (mild decay, > floor). Next per tree: R1 honest-bytes +
> R3 plain@100 (the decay comparator) + R4 Q-rotation telemetry (decay-diagnosis); R2 seed replicate
> still the binding statistical fix. B1 paper run (operator-directed) completes the operator row.

## 5. Ranked program (decision-value per GPU-hr; strategist + critic merged)

1. **R1 — honest-bytes accounting** (≈0 GPU-hr, laptop): count anchor-circuit traffic
   (M-allreduce, Q-broadcast, cadence-amortized) into the savings claim. Converts "parity" into the
   GOAL-3 deliverable. Do regardless of ext100.
2. **R2 — B2 seed replicate @50** (~5 GPU-hr): resolves T2 (the binding statistical threat) and
   extends T1 seed-generality. Recommend bundling with R1 into one issue. *Needs operator approval.*
3. **R3 — plain@100 (#28 Cell B)** (~5 GPU-hr): approve now — dual-purpose: 100-step no-carrier
   drift control (for ext100 interpretation) + clean 1-delta attribution base (T3).
4. **R4 — Q-rotation telemetry** (cheap probe): gates #28 Cell A (current-step EF) — m7 rank-2 +
   recon-flat suggest Q may be frozen ⇒ Cell A risks inert-by-Q-convergence; spend its ~20 GPU-hr
   only if Q rotates.
5. **R5 — K-delayed additive replay-gradient sub-basis** (highest ceiling, scope now / run later):
   route the anchor's rank-~2 true-gradient directions as an ADDITIVE off-principal sub-basis
   (avoids the EXP-26 Step-C falsified corner; respects the carrier law). Also the residual-compression
   variant (δ at ≪77 columns) as the GOAL-3 upside.
- **BLOCKED:** small-β_anc EMA smoothing — m6 ≈ 0.62 was the pre-registered safety number and it came
  back unsafe (τ ≈ 2× cadence). Blend-class on any M: retired (GATE-B1 + EXP-23/25 lineage).
  signed_ema: dead. Dense re-run: forbidden. λ/η sweeps: forbidden (dose-chasing).

## 6. Next 3 issues to open (strategist; one-line hypotheses)

1. **B2 honest inter-stage byte accounting** — H: anchor-circuit traffic amortizes to keep total
   savings ≥ ~15× (vs the fast-path-only 19.8×); closes GOAL-3 honestly. (+fold in R2 seed replicate
   if approved.)
2. **Q-rotation telemetry + #28 Cell-A inert-gate** — H: act-basis Q is effectively frozen after
   warm-up; if so, #28's current-step EF inherits B2's mechanism with no added value and should be
   re-scoped. (Split plain@100 out of #28 and approve it immediately.)
3. **K-delayed additive replay-gradient sub-basis (residual-compression variant included)** —
   H: routing the rank-~2 valid-gradient directions additively (≪77 columns) preserves B2-level val
   at strictly better bytes, without touching the falsified hybrid-Q corner.

## 7. Corrections log

- mechanist §6 KL flag: **resolved as false alarm** (launcher echo vs Hydra last-wins; runtime dump +
  resolved_params = `use_kl_loss=False`, `use_kl_in_reward=False`, `entropy_coeff=0`). The ladder is
  literally-vanilla no-KL GRPO. Standing lesson: only `resolved_params.txt`/runtime dump are authority
  for launch knobs, never the `set -x` echo.
- verdict.md "parity REACHED" stands as the pre-registered point-estimate report; this doc carries the
  statistical precision (T2). PASS itself is unaffected (driven by the >0.7210 floor rule).
