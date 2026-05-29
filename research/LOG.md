# Research Log (newest first)

## Two permanent references

This research operates against two baselines, kept as permanent reference
runs:

### `baseline` — dense GRPO (the control)

- Qwen2.5-1.5B-Instruct on GSM8K, verl unmodified, 100 steps on 4×H200.
- **NO KL, no entropy** (pg_loss only) — launcher standardized to no-KL so it
  matches the comm-eff objective for apples-to-apples; EXP-14 reconfirmed
  no-KL dense learns cleanly (val 0.083 → 0.721 in 10 steps).
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
- ⚠️ **Superseded at paper scale by EXP-14 (below).** This smoke PASS used the
  old mask+anchor+spectral config; the paper-scale investigation found the
  masked path does not yet learn. The launcher now defaults to a simpler
  mask+rescale+per-channel config (anchor/spectral OFF) — see EXP-14.

## EXP-14 — paper-scale grad_norm explosion: RESOLVED (diagnosis), GitHub #14 (closed), #15 follow-ups

The paper-scale dry run's symptoms (step-1 `grad_norm` ≈ 1100, entropy issues)
are now explained and the launcher knobs updated:

- **Root cause:** the mask's **magnitude collapse** — `h*mask` at p=0.9 with no
  rescale drops boundary-block RMS to √(1-p) ≈ 0.32× → out-of-distribution
  shift → step-1 grad_norm ~771 (not an IS/RNG artifact). Confirmed by peeling:
  pure masked GRPO (anchor/spectral OFF) explodes on its own.
- **`rescale` (inverted-dropout `h*mask/(1-p)`)** tames grad_norm (771 → 1.5,
  ppo_kl ≈ 0) — but it does **NOT** by itself recover learning (val flat at
  p=0.9; the masked forward is a near-random surrogate). New default ON.
- **Refuted:** `consistent_across_forwards` alone (positional mask + differing
  phase packing). Cross-pass IS consistency is instead structural via the new
  **`granularity=channel`** (per-channel) default.
- **`clean_cadence` (naive periodic clean step) is NOT sustainable:** masked
  steps keep exploding and PPO `pg_clipfrac` climbs toward saturation
  (cellF c=2: 0.26 → 0.44, rising) → clipped tokens contribute zero gradient →
  learning dies. The apparent early score rise is the clean steps alone.
  OFF by default.
- **New knob surface** in the comm-eff launcher (clean_cadence / rescale /
  granularity / consistent_across_forwards / mask.seed), all env-toggleable;
  anchor + spectral OFF by default. See `runs/SUMMARY.md` knob table.
- **Open question → #15:** can masked GRPO learn at all? Acceptance bar is a
  stable low `pg_clipfrac` + sustained val/score (mask-rate sweep p=0.9→0.5→0.1),
  not a bounded grad_norm. KL stays off (operator constraint).

Verdict + per-cell metrics: `runs/EXP-14/verdict.md` (folded to `runs/SUMMARY.md`
on de-bloat). Code merged via PR #8 on `shamanez/verl`.

## Older history

The implementation arrived through a sequence of incremental experiments
whose artifacts have been folded into `runs/SUMMARY.md`. The full
provenance is in git log + the merged PRs on `vast-ai-workload`.
