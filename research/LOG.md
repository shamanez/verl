# Research Log

The detailed historical log is folded into `runs/SUMMARY.md`, W&B, git history, and merged code.
Use method names + settings, not old run labels.

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
