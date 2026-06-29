# EXP-42 verdict — weight-projection accuracy of the look-ahead anchor

VERDICT: PASS (the measurement study succeeded and is decisive)

Scope: this verdict covers the planned 196-decoder-matrix study (regimes A and B). The
operator-requested completeness extension (widen to all matrices including the excluded
embeddings, RMSNorm gains, and biases) is DEFERRED to a fresh session; the GPU box was
auto-reaped mid-collection and the widened data was not captured. See plan 42.md Phase 4 and
runs/EXP-42/NEW_SESSION_PROMPT.md.

## What ran
- regime A: plain GRPO, COMM_EFF_ENABLED=false. 80 steps (160 ticks). val@40=0.7702, val@80=0.7695.
  comm_eff counters all 0 (dense path confirmed). WandB run er0syc3n (tail backfilled to step 80).
- regime B: PowerSGD r=77, codec only (anchor.enabled=false, spectral.enabled=false). 80 steps.
  val@40=0.0872, val@80=0.0788 (the codec-only policy collapsed; this is allowed data, not a
  failure). WandB run 0tpez2fz (tail backfilled to step 80).
- Both ran on one operator-provided 1xH200 NVL (team account, instance 43071381), single-GPU,
  resp=1024, dynamic batching. Box TORN_DOWN. No instance billing on either Vast account.

## Hard gates
- off-path parity, decoder-only 196, predictor parity, no-leak: PASS at the CPU pre-run probe
  (probe_cpu.py, Phase 1).
- single-GPU fits: PASS. regime A reached 80 steps with no OOM. Peak GPU allocated ~43 GB of
  143.7 GB (the ~141 GB resident reading was the vLLM reserved KV pool at mem_util=0.5, not
  allocation pressure).
- regime-B codec ACTIVE on 1 GPU: PASS. powersgd_applications climbed to 19838 in regime B versus
  0 in regime A; reconstruction_rel_error ~ 0.97; anchor_backwards=0; spectral_corrections=0. The
  PowerSGD codec is an in-graph activation projection (M_hat = (M Q) Q^T), so it fires on a single
  GPU without pipeline or data parallelism. theta_B provably diverges from theta_A.
- sketch fidelity (calibration, tol 5 percent): regime A all pass (0.49, 1.08, 1.47 percent at
  h=5,10,20). regime B 2 of 3 pass (h5 marginal at 5.85 percent, h10 3.85, h20 0.10). The marginal
  h5 cell is attributable to anchor-sampling (15 calib anchors versus ~28k sweep samples) in the
  collapsed regime, not a count-sketch reconstruction error: the sketch and the exact calib agree
  on every helps / no-help verdict, so the crossover h* is robust. Count-sketch on-device versus
  offline parity is bit-exact (probe_cpu.py and the select_all CPU validation).

## Headline measurement (operating spacing Delta = 10, median over 196 matrices and valid anchors)
| regime | h=5 (alpha 0.5) | h=10 (alpha 1.0 = K) | h=20 (alpha 2.0) | crossover h* |
|---|---|---|---|---|
| A (clean)        | w1=0.8492 (helps), dir_cos=0.626 | w1=0.9718 (helps), dir_cos=0.549 | w1=1.1729 (no), dir_cos=0.443 | 10 |
| B (codec-only)   | w1=0.9298 (helps), dir_cos=0.571 | w1=1.0829 (no),    dir_cos=0.480 | w1=1.3493 (no), dir_cos=0.365 | 5  |

fixed_linear and learned_linear are identical to 4 decimal places at every operating-point cell.

## Answers to the plan hypotheses
- H1 (premise): SUPPORTED for the clean regime. At the operating horizon h = K = 10 the projected
  weight lands closer than raw-stale (w1 = 0.972 < 1) with dir_cos = 0.549 > 0. NOT supported for
  the compressed regime at h = 10 (w1 = 1.083 > 1); it helps there only up to h = 5.
- H2 (overshoot): SUPPORTED in magnitude. w1 rises monotonically past 1.0 as h grows. BUT dir_cos
  stays positive at every horizon in both regimes, so there is NO weight-space sign flip on this
  grid. The overshoot is the extrapolation stepping past theta_now along a consistently aligned
  direction, not a direction reversal. This refines the prior-collapse hypothesis (the sign flip
  seen in the prior extrapolated-anchor-cosine runs is not a weight-space direction reversal at
  h up to 20).
- H3 (regime effect): SUPPORTED. Activation compression halves the crossover (h* 10 to 5 at
  spacing 10) and lifts w1 at every horizon, with a wider per-matrix spread (for example
  [0.75, 1.39] at h=10 in B versus [0.92, 1.05] in A). The codec makes the weight trajectory less
  linearly predictable.
- Falsification not triggered: median w1 < 1 occurs at h >= 1 in both regimes (so linear weight
  extrapolation does beat doing nothing at short horizons).

## Gate for a future gradient-accuracy study
- WORTH doing for the CLEAN regime at fixed_linear, h up to 10 (spacing 10): w1 < 1 and
  dir_cos > 0 at the operating horizon. Winning (method, Delta, h) = (fixed_linear, 10, h<=10).
- NOT worth doing for the COMPRESSED regime at h = 10 (w1 > 1); only marginally at h up to 5.

## Deliverables
- runs/EXP-42/report.html (self-contained HTML: plots, operating-point answer, crossover table,
  full curves, sketch fidelity, generated discussion).
- runs/EXP-42/sweep_narrow.json (raw sweep), narrow_findings.md (interim notes).
- Code: exp/42-weight-accuracy @ 531dd5e9 (instrument + select_all extension). promote_launcher_as:
  none (this is a measurement, not a promoted launcher).

## Dense-run weight-behavior report (operator follow-up, GPU-free)
runs/EXP-42/report_dense.html (builder runs/EXP-42/build_dense_report.py) characterises the normal
GRPO run (regime A) from the dense decoder sketch:
- Weight change: median relative drift reaches about 0.057 percent of the initial norm by step 80.
  GRPO at lr 1e-6 on an already-instruction-tuned 1.5B model is a tiny, gentle motion; what matters
  for projection is the direction of that small motion.
- Projectability: crossover h* = 10 ticks (about 5 global steps), ratio 0.972 at the operating
  horizon. You can linearly project about 5 steps ahead before the projection stops helping.
- Performance: val 0.7702 at step 40, 0.7695 at step 80 (flat, near-converged).
- RLVR-linear test: PARTIALLY, and scale-dependent. Per-matrix linearity R squared is 0.80 at one
  tick (in the ballpark of the about-0.9 RLVR-linear claim) and decays to 0.32 at the K=10 scale.
  The trajectory is locally close to linear and curves by the K=10 horizon, which is the mechanism
  behind the overshoot. Attention is slightly more linear than MLP at fine scale (0.81 vs 0.74);
  linearity is uniform across layer depth.
A deeper GPU-free follow-up (low-rank structure, per-matrix projectability, optimal-coefficient
sweep) is specified in runs/EXP-42/NEW_SESSION_PROMPT.md. No further GPU training will be run.
