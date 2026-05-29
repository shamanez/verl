# Research Log (newest first)

## State of the project

One baseline and one method under test — both from the same launcher.

### Baseline = dense GRPO == method OFF

The dense control is the comm-eff launcher with the master switch off
(`COMM_EFF_ENABLED=false`) — byte-identical to unmodified verl, no-KL
no-entropy (pg_loss only). It learns cleanly on GSM8K in a short run; this is
the bar every compression run must match. No standalone baseline run artifacts
are kept — the control is just the method switched off.

### Comm-eff method — implementation correct, masking under test

- **Implementation is correct:** method OFF ⇒ byte-identical dense; with masking
  ON the mask fires on exactly the gradient-feeding forwards and grads stay
  finite (unit-tested).
- **Masking still needs a lot of testing.** At high mask rates plain masked GRPO
  does not yet learn: dropping ~90% of boundary activations collapses their
  magnitude, which without rescaling blows up grad_norm; and even with rescaling
  the masked policy is too far off-distribution to learn — the PPO ratio clips
  out a growing fraction of tokens. The open question is whether, and at what
  mask rate, masked GRPO learns; the next step is a mask-rate sweep judged on
  val/score (and a stable, low clip fraction), not on a bounded grad_norm.
- **Anchor + spectral correction are layered fixes for later** — brought in only
  if masking alone is not enough. Default OFF; kept OFF to start.

The knob surface and the things tried so far are in `runs/SUMMARY.md`; the
engineering map is `CODE_WALKTHROUGH.md`.
