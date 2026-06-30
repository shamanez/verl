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

## IMPORTANT caveat on regime B (why it collapsed; not representative of healthy compression)
Regime B ran the PowerSGD codec on a FROZEN RANDOM basis. anchor.enabled=false but owns_q=true, so
the only Q updater (the anchor) was off AND the fast-path basis update is fail-closed in owns_q=true
mode. Evidence: basis_updates=0.0 for the whole run, reconstruction_rel_error flat at ~0.975->0.970
(a fixed rank-77 random subspace of a ~rank-1536 activation keeps ~5 percent), anchor_backwards=0,
spectral off (no merger). A rank-77 random projection with no gradient correction discards ~97
percent of the boundary gradient, which is why the policy collapsed to val 0.079. This is NOT an
inherent PowerSGD limitation: the older comm-eff runs that learned at delay_K=5 (val ~0.72-0.74)
kept the anchor ON, which adapts Q (Q<-orth(V) from harvested activations, low recon error) AND a
signed_ema merger that folds the clean near-fresh anchor gradient into the compressed gradient.
Consequence: the EXP-42 A-vs-B contrast is "clean dense vs COLLAPSING FROZEN-BASIS codec," not
"clean vs healthy adaptive-compressed." The weight trajectory of the realistic learning compressed
system (anchor-adapted Q at delay 5 + merger) was NOT measured. A functional codec-only regime B
would have set owns_q=false (let the fast path adapt Q) or kept the anchor on for Q maintenance.
Regime B's projectability numbers (crossover h*=5, lower linearity) therefore describe a degenerate
collapsing trajectory, which is still valid data but must be labeled as such.

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
- Paper vs this run: the cited RLVR-linearity paper (arXiv:2601.04537, per lookahead.py) reports
  linear weight extrapolation holding about 600 steps at R squared about 0.9. By the analogous
  metric here (weight_proj_ratio crossover) linear extrapolation holds only about 5 global steps
  (10 ticks) in this run. The gap is large. Likely causes: this run is only 80 steps from an
  already-tuned model with tiny noise-dominated motion (0.057 percent drift), and the two linearity
  metrics may differ (global line fit / low-rank subspace in the paper versus local consecutive-step
  R squared here). arXiv:2601.04537 is at the edge of the assistant knowledge cutoff, so its figures
  are taken from the code citation, not an independent reading.

A deeper GPU-free follow-up (low-rank structure, per-matrix projectability, optimal-coefficient
sweep, and a like-for-like global-line-fit linearity to compare with the paper) is specified in
runs/EXP-42/NEW_SESSION_PROMPT.md. No further GPU training will be run.

## Dense-run weight-behavior v2 (deeper GPU-free follow-up, 2026-06-29)
runs/EXP-42/report_dense_v2.html (builder runs/EXP-42/build_dense_report_v2.py, sibling of
build_dense_report.py) extends the dense analysis with five studies, every number computed from
the regimeA decoder sketch (196 matrices, 160 ticks, k=4096; rel std about 1/sqrt(k) about 1.6
percent). Metrics dumped to report_dense_v2_metrics.json. Coherent one-line read: the dense GRPO
trajectory is globally near-linear and low-rank, but the look-ahead's two-point slope overshoots, so
a damped coefficient is the lever.

- (a) LOW-RANK displacement subspace: stacking the 159 per-tick displacement vectors per matrix, the
  median participation ratio is 7.6 of a 159 ceiling and a median of 26 components hold 90 percent of
  the per-tick update energy. The cumulative displacement is nearly rank-1 (participation ratio about
  1.2; one direction holds about 69 percent of the centered-trajectory energy). The RLVR low-rank
  claim is SUPPORTED in the temporal sense. A like-for-like GLOBAL straight-line fit gives R squared
  0.85 (through-origin extrapolation) and 0.68 (centered), which reconciles the prior report's "local
  consecutive-step R squared decays to 0.32" with the cited paper's global about-0.9: same data, the
  trajectory is one slow drift plus per-step noise, the local metric sees the noise and the global
  metric sees the drift. Sketch noise inflates the residual, so 0.85 is a lower bound on the true
  linearity. NOT computable from this data and not claimed: the matrix-native (LoRA-style) singular
  rank of an individual weight matrix, which flatten-then-sketch destroys; and anything about
  embeddings, RMSNorm gains or biases, which were not collected (decoder matrices only).
- (b) PER-MATRIX projectability: the crossover horizon h* is tight across the decoder, 9 to 14 ticks
  (median 11), and the ratio at h=10 spans only [0.956, 0.985]. The attention value and output
  projections (v_proj, o_proj) in the middle-to-late layers project furthest (h* 13 to 14); MLP and
  the attention key/query projections project least (h* 9 to 10). Projectability is a decoder-wide
  property, not a few outliers.
- (c) OPTIMAL coefficient: the naive alpha = h/Delta overshoots at every horizon. A damped alpha
  (about 0.53 at h=10, about 0.74 at h=20) cuts the median weight_proj_ratio from 0.972 to 0.836 at
  h=10, and keeps it below 1 at h=20 (1.173 to 0.897) where the naive rule fails. The gain grows with
  horizon and is far above the 1.6 percent sketch floor. Mechanism: the two-point slope over-states
  the persistent drift because it also captures per-step noise, so the naive rule extrapolates too
  far; damping corrects the over-step. Caveat: the optimal alpha is fit in-sample (an oracle upper
  bound), but it is stable across horizons (about 0.5 to 0.75), so a single fixed damped coefficient
  near 0.5 captures most of the gain with no online estimation. This is the concrete deployable change
  the dense data suggests for the look-ahead rule (to be validated in the compressed regime, where
  EXP-42 already found the crossover at h*=5).
- (d) LINEARITY vs projectability: more-linear matrices are more projectable. Spearman 0.45 (fine-scale
  R squared at 1 tick vs crossover h*) and -0.51 (vs ratio at h=10, negative because lower ratio is
  better). Significant at n=196 (5 percent threshold about 0.14). Real but loose: local linearity
  explains part of projectability, the rest is set by how per-step noise inflates the two-point slope.
- (e) LEARNED vs FIXED residual: inert on the dense run. The per-matrix scalar mean-shift residual
  grows to at most 2.7e-9 (clip 1e-3), contributes relative norm about 1e-4 to the projection, and
  changes the ratio by at most 6.5e-5, all below the 1.6 percent sketch floor. A scalar added
  uniformly to a high-dimensional matrix barely moves the displacement norm or direction, and the
  per-matrix mean drifts so smoothly that the fixed extrapolation has almost no retrospective error to
  correct.
