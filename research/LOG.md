# Research Log

The detailed historical log is folded into `runs/SUMMARY.md`, W&B, git history, and merged code.
Use method names + settings, not old run labels.

## EXP-42 · 2026-06-29T11:00:00Z · M4 · PASS (measurement)
M4: Measure weight-projection accuracy of the look-ahead anchor (does theta_hat land closer to theta_now than the raw-stale theta_stale) as a function of horizon, in two regimes, on one operator 1xH200 (single-GPU, operator-authorised).
- hypothesis: at the anchor operating horizon (h = K = 10 ticks, alpha = 1), the projected weight is closer to theta_now than raw-stale (median weight_proj_ratio < 1) in both regimes for at least fixed_linear; overshoot raises the ratio past 1 as h grows; activation compression changes predictability.
- result: PASS as a decisive measurement. regime A (plain GRPO, val@80=0.7695): crossover h*=10, at h=10 ratio=0.972 (helps), dir_cos=0.549. regime B (PowerSGD r=77 codec-only, val@80=0.0788, collapsed = allowed data): crossover h*=5, at h=10 ratio=1.083 (no help). dir_cos stays positive at every horizon in both regimes (0.37 to 0.63), so the overshoot is a MAGNITUDE effect (alpha extrapolates past theta_now along an aligned direction), NOT a weight-space sign flip; this refines the prior-collapse picture. fixed_linear == learned_linear at the operating point (scalar-mean residual inert). H1 holds for the clean regime, H3 (regime effect) confirmed: compression halves the useful horizon. Codec-active hard gate on 1 GPU CONFIRMED: powersgd_applications=19838 (B) vs 0 (A), reconstruction_rel_error ~0.97; the in-graph PowerSGD projection fires without PP/DP. Sketch fidelity vs on-box exact calib within 5 percent except regime B h5 (5.85 percent, anchor-sampling in the collapsed regime; sketch and exact agree on every helps/no-help verdict so h* is robust). Gradient-accuracy follow-up GATE: worth doing for the clean regime at fixed_linear h<=10; not worth for the compressed regime at h=10. Operator-requested completeness extension (widen to all matrices incl. embeddings/RMSNorm/biases) DEFERRED to a fresh session after the box was auto-reaped mid-collection (instrument built + pushed at 531dd5e9, CPU-validated).
- code: exp/42-weight-accuracy @ 531dd5e9 (weight-trajectory sketch instrument + select_all extension); promote_launcher_as none.
- deliverables (run dir de-bloated 2026-06-30): reports/exp42-{weight-projection-accuracy,dense-weight-behavior,dense-deep-analysis,prior-gradient-probe}.html; tooling research/scripts/{weight_proj_sweep,build_report,build_dense_report,build_dense_report_v2}.py. WandB: er0syc3n (A), 0tpez2fz (B), both backfilled to step 80.
- verdict: .claude/plans/42-verdict.md

## EXP-41 · 2026-06-25T18:05:00Z · M4 · STOP
M4: Implement linear weight extrapolation for look-ahead anchor training at delay_K=20
- hypothesis: fixed-linear look-ahead anchor (theta_hat=2*theta[t-20]-theta[t-40]) at cadence/delay_K=20/20 on the fast-1K surface survives 100 steps without collapse, lifts anchor_align_cos vs the raw-stale 5/5 baseline, and reaches cell A's 5/5 reference band (val@100 ~[0.7066,0.7255]); STOP is a scientific falsification — pre-run fire-forcing probe PASSED all 10 hard correctness invariants (implementation is correct)
- result: lift IS present (+0.0267, 6/8 true fires positive, peak +0.131) and catastrophic entropy ignition did NOT recur, but cell B collapsed (response_length/mean breached 2x threshold at 8 steps, peak 552 vs 496; val crashed 0.498->0.115->0.048; val@100=0.0478 vs band [0.7066,0.7255]); hypothesis falsified on this surface; cell C (learned) gated off per plan's on_fail; deferred direction: lower beta_anc (0.50->0.10-0.25) with fixed-linear held on (merger over-amplifying the now-fresher projected anchor gradient); WandB: cell A=7tbzm9kl, cell B=g6dt6bza; cell A reference band: val@25/50/75/100=0.6998/0.7255/0.7233/0.7066, raw-stale cos mean +0.0063
- run dir: runs/EXP-41/
- verdict: runs/EXP-41/verdict.md

## Current state (2026-06-25)

- **Basic setup / operating base:** the **EMA merger** — `signed_ema` (α=0.25, β_anc=0.50) — on the
  **fast 1K surface** (resp 1024, dynamic-bsz, TP=1, gpu_mem 0.55, 50 steps) at HIGH anchor latency
  (cadence/delay_K = **20/20**, the k-collapse regime), on the locked **PowerSGD r=77 anchor
  substrate** (anchor owns `Q`, clean=0, paired replay).
- **Baseline = the problem state:** at 20/20 the method collapses (Priority 1). At LOW latency (5/5)
  the same merger reached parity (val@50 ≈ 0.736 vs dense ≈ 0.766, n=1, older 2K surface).
- **Status:** stable + parity at ≈5% gradient-comm at LOW latency (Goals 1–3 met); holding it at
  realistic high latency is the open problem.
- Repo de-bloated to the two active priorities — runs + prior reports removed; durable record =
  `runs/SUMMARY.md` + the two `reports/*.html` summaries + W&B + git.

## Next-phase rule

Work the **two priorities only**, both from the EMA/2K base above, one knob at a time:

1. **Project the weights to fix the k-collapse** (milestone M4) — start with the GPU-free offline
   cosine-lift kill-gate before any GPU spend.
2. **Reduce the compression-induced train–inference mismatch (Gap A)** (milestone M6).

Do not import old anchor-gradient claims or run labels into new plans.
