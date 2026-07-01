# Research Runs Summary

Durable record (run dirs de-bloated). North-star: `../.claude/GOAL.md`. Detail: `../reports/*.html`, `LOG.md`, git.

**Baseline** (`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`):
`signed_ema` (α=0.25, β_anc=0.50), PowerSGD r=77 anchor (owns Q), at 20/20 anchor latency — the
**k-collapse regime** where the method fails (parity holds only at 5/5). Values: `FIXED_CONTROL_SURFACE.md`.

**M4 (PASS):** the shared dense full-weight per-tick trajectory (160 bf16 snapshots, all params) is
the spine root every analysis issue reads —
`s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/`. Next: **#44** (GPU-free offline
sweep engine). Access: `reports/r2-access-pattern-for-analysis.md`.
