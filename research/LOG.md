# Research Log

The detailed historical log is folded into `runs/SUMMARY.md`, W&B, git history, and merged code.
Use method names + settings, not old run labels.

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
