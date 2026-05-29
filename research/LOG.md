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
  does not yet learn — and the thing to judge on is **val/score, not grad_norm**
  (Adam's update is scale-invariant and grad-clipped, so the raw norm is a
  symptom at best). Two *distinct* problems. **Without rescale** the mask is
  biased (`E[h⊙mask] = (1-p)·h`): the forward sits off-distribution, the GRPO
  importance ratio is corrupted, and the gradient carries a systematic bias Adam
  cannot correct → a stalled trajectory with a non-vanishing floor. **With
  rescale** the mask is unbiased (`E[h̃] = h`) but high-variance (`p/(1-p)`),
  and with no denoising (grad-clip / EMA / spectral / anchor) the
  unbiased-but-noisy gradient is a random walk → val still flat. So rescale is a
  correctness knob, not a fix; the path forward is variance control + a lower
  mask rate. *(We do not attribute observed grad_norm differences across cells
  to any one knob — those cells stacked changes and the artifacts are pruned. An
  earlier "rescale reduces grad_norm" claim was a mistake and has been removed.)*
- **Anchor + spectral correction are layered fixes for later** — brought in only
  if masking alone is not enough. Default OFF; kept OFF to start.

The knob surface and the things tried so far are in `runs/SUMMARY.md`; the
engineering map is `CODE_WALKTHROUGH.md`.
