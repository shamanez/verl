# Progress

Historical tick-by-tick orchestration output has been de-bloated. The next phase
starts from the compact handoff:

- `research/runs/SUMMARY.md`
- `research/.claude/plans/SUMMARY.md`
- `research/runs/FIXED_CONTROL_SURFACE.md`

Current working state:

- The **accelerated comm-eff loop** is the locked base: `signed_ema(α=0.25, β_anc=0.50)`,
  accel surface @ gpu_mem 0.55, `diagnostics=false`, PowerSGD r=77 anchor substrate.
  Launcher: `examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`.
- ~25 min train / ~28 min wall per 50-step run. Reference val@50 (n=1): comm-eff 0.7362
  (EXP-36B) vs dense 0.7657 (EXP-36C).
- The vLLM speed knobs (gpu_mem 0.75 / chunked_prefill / forward_prefetch) were tried and
  dropped — no speedup, added noise.
- Prior bulky run directories and execution plans are removed.
