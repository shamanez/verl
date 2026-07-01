# Research Runs Summary

Durable record (run dirs de-bloated). North-star: `../.claude/GOAL.md`. Detail: `../reports/*.html`, `LOG.md`, git.

**Baseline** (`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`):
`signed_ema` (α=0.25, β_anc=0.50), PowerSGD r=77 anchor (owns Q), at 20/20 anchor latency — the
**k-collapse regime** where the method fails (parity holds only at 5/5). Values: `FIXED_CONTROL_SURFACE.md`.

**M4 (PASS) — dense full-weight per-tick trajectory (160 snapshots, all 338 matrices).** The spine root
every analysis issue reads. Available in **both precisions** (160/160 R2-verified, `verify_full_weight_dump.py --r2` PASS):
- **bf16** (EXP-43, ~3.1 GB/snap): `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/tick_<N>/tick_<N>.pt`
- **fp32** (EXP-57, ~6.17 GB/snap — the true fp32 master weights; **use this for #44–#56**): `s3://shamane-pluralis/verl-research/EXP-57/regimeA/weights/full/tick_<N>/tick_<N>.pt`  (N = 0..159)

Next: **#44** (GPU-free offline sweep engine). Access: `reports/r2-access-pattern-for-analysis.md`.
