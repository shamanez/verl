# Research Log (newest first)

## Two permanent references

This research operates against two baselines, kept as permanent reference
runs:

### `baseline` — dense GRPO (the control)

- Qwen2.5-1.5B-Instruct on GSM8K, verl unmodified, 100 steps on 4×H200.
- `val/test_score` 0.087 → 0.789.
- Run dir: `runs/baseline/`
- Launcher: `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- Plan: `.claude/plans/baseline.md`

### `communication-baseline` — the comm-eff method's smoke-scale verification

- Qwen2.5-1.5B-Instruct on GSM8K with the full comm-eff pipeline enabled:
  PRF activation mask `p=0.9` on both gradient-feeding forwards
  (`mask_recompute=true`), hookless K-stale anchor circuit
  (cadence=5/delay=5), two-sided Tikhonov spectral correction
  (`α=0.5, τ=0.01, β_anc=0.9`). No KL, no entropy.
- 20-step smoke on 4×H200; mean reward steps 11-20 = +82% above mean steps
  1-10; all six anchor guards held; visible learning trend.
- Run dir: `runs/communication-baseline/`
- Launcher: `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- Plan: `.claude/plans/communication-baseline.md`
- Reproducibility manifest: `runs/communication-baseline/REPRODUCIBILITY.md`

## Active investigation

Before scaling the comm-eff method past smoke shape, a paper-scale dry run
surfaced symptoms that need explaining:

- `grad_norm` starts large (≈ 1100 at step 1, *before* any policy drift
  could matter) and climbs from there.
- `entropy` collapses across the run; `ppo_kl` per-step grows past PPO's
  trust-region assumption.
- `response_length/max` repeatedly hits the truncation cap — typical
  policy-collapse output pattern.

Investigation queued in `notes/investigation-prompt-grad-norm.md` — paste
that file into a fresh session to draft the GitHub issue. The prompt
enumerates the candidate root causes (importance-sampling variance under
independent PRF masks, empty-`M_anchor` spectral degeneracy, FSDP / DTensor
integration audit, anchor harvest correctness, mini-batch + token-wedge
variance amplification) and a four-test discriminating plan. KL stays
off across all tests (operator constraint, design of the method).

## Older history

The implementation arrived through a sequence of incremental experiments
whose artifacts have been folded into `runs/SUMMARY.md`. The full
provenance is in git log + the merged PRs on `vast-ai-workload`.
