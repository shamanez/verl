# EXP-42 key numbers — look-ahead anchor HORIZON sweep

Data miner extract. Single source of truth: `runs/EXP-42/report/series.json`.
All numbers come from the per-cell training logs (no fabrication; gaps marked).

**Surface (held fixed across cells):** 1K GSM8K, Qwen2.5-1.5B-Instruct, 100 steps
target, `test_freq=25` (val@25/50/75/100), resp 1024, `delay_K=cadence=20`,
signed_ema α=0.25 β_anc=0.50, PowerSGD r=77, n=8 rollouts, 4 GPU.

**Step-unit note:** metric `step:` lines are **global steps**. The
`[comm_eff][lookahead] step=N` diagnostics are in **optimizer ticks** (2 ticks
per global step at batch128/mini64), so tick 60 = global step ~30, tick 80 =
global step ~40. The "first extrapolated fire" column below reports the tick and
its global-step equivalent. anchor fires every 20 ticks (delay_K=20).

## EXP-42 sweep cells (look-ahead ON)

| Cell | mode | strength | val@25 | later val | collapse onset (global step) | collapse thr (rl/mean) | 1st extrapolated fire | 1st-ext cos sign | last step | outcome |
|------|------|----------|--------|-----------|------------------------------|------------------------|-----------------------|------------------|-----------|---------|
| **A25** | fixed_linear | 0.25 | **0.5724** | — | **38** (rl 519 > 491) | 490.98 | tick 60 (gstep~30) | **+0.0157** (positive) | 39 | Collapsed @38 via length explosion (rl 110→665 over steps 32–39); never reached val@50 |
| **A50** | fixed_linear | 0.50 | **0.6459** | val@50=0.5694, val@75=0.3124 | **83** (rl 543 > 500) | 500.46 | tick 60 (gstep~30) | **+0.0081** (positive) | 85 | Best HORIZON cell; slow decay then length explosion @83 (rl 211→842); collapsed before val@100 |
| **A75** | fixed_linear | 0.75 | **0.1873** | — | none in window | 479.56 | none (log truncated) | — (no ext fire) | 27 | **DATA GAP:** internal log truncated at global step 27 (Training Progress 28%), BEFORE first extrapolated fire (tick 60=gstep 30) and before any collapse; rl was *declining* (200→83) at truncation; val@25 already low (0.187) |
| **L** | learned_linear_with_fixed_linear_cold_start | 1.0 | **0.3965** | — | **43** (rl 511 > 509) | 509.11 | tick 80 (gstep~40) | **−0.0138** (negative) | 45 | Collapsed @43 via length explosion (rl 365→603); deeper warmup ring (3 snaps) ⇒ first extrapolation later (tick 80); first true look-ahead cos is NEGATIVE |

## EXP-41 reference anchors (same surface)

| Cell | role | val@25 | val@50 | val@75 | val@100 | collapse onset | 1st extrapolated fire | 1st-ext cos | outcome |
|------|------|--------|--------|--------|---------|----------------|-----------------------|-------------|---------|
| **EXP41_ref_5over5** | 5/5 lookahead-DISABLED raw-stale anchor reference (cadence=5) | 0.6998 | 0.7255 | 0.7233 | **0.7066** | none (stable 100 steps) | none (lookahead disabled; all 40 fires raw_stale) | n/a | Stable reference; raw-stale anchor_align_cos hovers ~0.0–0.03 |
| **EXP41_alpha1p0** | α=1.0 full-catch-up fixed_linear @20/20 | 0.3616 | 0.4981 | 0.1145 | **0.0478** | **57** (rl 599 > 496) | tick 60 (gstep~30) | **+0.0325** (positive) | 100 | Collapsed ~step 57 via length explosion; ran to step 100 but val stayed collapsed (0.0478); rl partially recovers by step 100 (842-peak → 162) but reward does not |

## Headline reads (numbers only)

- **No HORIZON cell beats the 5/5 raw-stale reference** (val@100=0.7066). Every
  look-ahead-ON cell collapsed via length explosion (rl/mean breaching 2× its
  own first-25-step baseline), except A75 whose log truncated before any verdict.
- **Monotone-ish in strength among the runnable fixed_linear cells:** A50 (str
  0.50) is the most robust HORIZON cell (val@25=0.646, survives to step ~83);
  A25 (str 0.25) collapses earliest (@38); A75 (str 0.75) val@25 already lowest
  (0.187) before truncation.
- **First true look-ahead (extrapolated) cosine is near-zero / sign-unstable**
  across cells: A25 +0.016, A50 +0.008, L −0.014, EXP41 α1.0 +0.033 — i.e.
  extrapolated g(θ̂) is essentially uncorrelated with g_live, same regime as the
  raw-stale baseline (~0.0–0.03). Extrapolation does not lift alignment.
- **anchor_backwards / lookahead_fires (cumulative, last):** A25 3/1, A50 8/6,
  A75 2/0 (truncated), L 4/1, EXP41-ref 40/0, EXP41-α1.0 10/8.

## Data gaps / caveats

1. **A75 log truncated at global step 27** (`train_A75_internal.log`). The cell
   was still progressing (Training Progress 28%, no error/Traceback/OOM), but
   the captured internal log ends before the first extrapolated fire (tick
   60=gstep 30) and before any collapse. So A75 has: val@25=0.187 only, NO
   collapse verdict, NO extrapolated cos. mode/strength recovered from the
   resolved launch command (`lookahead_mode=fixed_linear`,
   `lookahead_strength=0.75`), not from a fire diagnostic.
2. **EXP41_alpha1p0 strength** is not printed in its (older-code) diagnostic
   lines; set to 1.0 from context (EXP-41 verdict.md: "α=1.0 full catch-up").
   Recorded in series.json as `strength_source`.
3. **collapse_onset_step** uses the prompt's definition: first step where
   rl/mean > 2× mean(first-25-step rl/mean), sustained ≥2 consecutive logged
   steps. Reported breach step + threshold per cell above.
4. **No other gaps.** `critic/score/mean`, `response_length/mean`,
   `actor/entropy`, `actor/grad_norm`, `response_length/clip_ratio` are all
   present and captured for every logged global step in every cell (counts
   match per cell: A25=39, A50=85, A75=27, L=45, both EXP-41=100).
