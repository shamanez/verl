# Research Log (newest first)

## The one baseline + the method under test

This research has a single baseline and one method under test. Both are the
same launcher; the baseline is just the method switched off.

### Baseline = dense GRPO == comm-eff OFF

- The dense control *is* `vast_comm_eff_baseline_*.sh` with
  `COMM_EFF_ENABLED=false` — byte-identical to unmodified verl (PR #1).
  No-KL, no-entropy (pg_loss only). (The convenience dense launcher
  `vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` has the identical objective.)
- **Proof the codebase trains dense-perfect:** EXP-14 `test1_cellA`
  (comm-eff OFF), **10 steps**, 4×H200 — `val/test_score` **0.083 → 0.721**,
  clean monotone improvement. 10 steps is enough to see clear learning, so
  that's the standing control horizon.
- We **no longer keep the old 100-step dense baseline run** (artifacts +
  plan + finding pruned). The dense proof is now this 10-step cellA.

### Comm-eff method — implementation correct, masking still under test

- **Implementation is correct:** comm-eff OFF ⇒ byte-identical dense (PR #1);
  masking ON fires on exactly the gradient-feeding forwards, `mask_ratio`
  tracks `p`, grads finite (PRs #2–#6, 127 unit tests).
- **The masking side still needs a lot of testing** — at paper scale pure
  masked GRPO does not yet learn (see EXP-14 below). Open → #15
  (mask-rate sweep p=0.9→0.5→0.1).
- **Anchor + spectral correction are layered fixes for later** — bring them
  in only when the plain masked path isn't enough. They default OFF and
  should stay OFF to start.
- The old **communication-baseline** smoke run (old mask+anchor+spectral
  config, 20-step PASS) is superseded by EXP-14; its artifacts/plan/finding
  were pruned.
- Launcher: `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`.

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
