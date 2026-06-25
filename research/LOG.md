# Research Log

The detailed historical log is folded into `runs/SUMMARY.md`, W&B, git history, and merged code.
Use method names + settings, not old run labels.

## Current state (2026-06-25)

- **Basic setup / operating base:** the **EMA merger** — `signed_ema` (α=0.25, β_anc=0.50) — on the
  **2K accel surface** (resp 2048, dynamic-bsz, TP=1, gpu_mem 0.55, 50 steps), on the locked
  **PowerSGD r=77 anchor substrate** (anchor owns `Q`, cadence/delay_K=5, clean=0, paired replay).
  `delayed_ef` (λ=1, β_anc=0) is kept only as a parity floor / compatibility reference, not a planning target.
- **Reference val@50 (n=1, accel surface):** dense 0.7657 (EXP-36C) · comm-eff `signed_ema` 0.7362 (EXP-36B).
- **Status:** stable + parity at ≈5% gradient-comm (Goals 1–3 met); the bar to beat is dense ≈0.75–0.78.
- Repo de-bloated to the two active priorities — runs + prior reports removed; durable record =
  `runs/SUMMARY.md` + the two `reports/*.html` summaries + W&B + git.

## Next-phase rule

Work the **two priorities only**, both from the EMA/2K base above, one knob at a time:

1. **Project the weights to fix the k-collapse** (milestone M4) — start with the GPU-free offline
   cosine-lift kill-gate before any GPU spend.
2. **Reduce the compression-induced train–inference mismatch (Gap A)** (milestone M6).

Do not import old anchor-gradient claims or run labels into new plans.
