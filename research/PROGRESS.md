# Progress

Durable record: `runs/SUMMARY.md` · `runs/FIXED_CONTROL_SURFACE.md` · `reports/*.html` · W&B · git.
North-star: `.claude/GOAL.md`. Per-run verdicts: `LOG.md`. Tick history pruned (recoverable from
git + LOG.md); the harness appends new ticks below.

**Base:** `signed_ema` (α=0.25, β_anc=0.50), fast 1K surface, HIGH anchor latency
(cadence/delay_K=20/20, the k-collapse regime), locked PowerSGD r=77 anchor. Values:
`runs/FIXED_CONTROL_SURFACE.md`.

**Two priorities:** (1) M4 — solve the k-collapse by projecting WEIGHTS; the dense weight trajectory
is collected → R2; GPU-free analysis spine #44–#56, entry **#44**. (2) M6 — shrink the ~0.04
compression train–inference mismatch.

**Where we are (2026-07-02):**
- **#44 PASS** — offline sweep engine accepted. (A mid-run STOP from a bf16 noise-floor category error was
  overturned by an adversarial-verify workflow: corrected differenced floor + directedness p≈1.05, core
  blocks clear the floor at h≥5. Full trail: `LOG.md`, git.)
- **EXP-57 (fp32 dense weight-traj)** drained 2026-07-01: 160/160 ticks, manifests R2-verified byte-exact,
  val gsm8k acc@1=0.782 — the fp32 trace #45–#56 now consume.
- **Analysis refactor landed 2026-07-02 (fp32 + download-first).** Engine reads a PRE-DOWNLOADED local trace
  (`weight_proj_sweep.py --trace-root`) or streams (few-snapshot passes / #46); bounded-footprint released
  for analysis, collection unchanged; on fp32 the bf16 noise-floor gate is off (reliability = projection
  accuracy + linearity). New: `weight_proj_fetch_trace.py`, `synth_exp57_manifests.py`. Verified locally +
  adversarially reviewed; issues #44–#56 realigned. SoT: `reports/r2-access-pattern-for-analysis.md`.
