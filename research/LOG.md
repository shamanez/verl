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
  does not yet learn — for two *distinct* reasons, neither of which is a
  grad_norm explosion. **Without rescale** the mask is biased
  (`E[h⊙mask] = (1-p)·h`): the forward sits off-distribution, the GRPO
  importance ratio is corrupted, and the gradient carries a systematic bias Adam
  cannot correct → a stalled trajectory with a non-vanishing floor. (The large
  *measured* grad_norm there is a symptom of the corrupted ratio; Adam's update
  is scale-invariant and grad-clipped, so "explosion" is the wrong model.)
  **With rescale** the estimator is unbiased (`E[h̃] = h`, `r ≈ 1`) but
  high-variance (`p/(1-p)`), and with no denoising (grad-clip / EMA / spectral /
  anchor) the unbiased-but-noisy gradient is a random walk → val still flat. The
  fix is variance control + a lower mask rate, not the rescale alone. The next
  step is a mask-rate sweep judged on val/score, not on a bounded grad_norm.
- **Anchor + spectral correction are layered fixes for later** — brought in only
  if masking alone is not enough. Default OFF; kept OFF to start.

The knob surface and the things tried so far are in `runs/SUMMARY.md`; the
engineering map is `CODE_WALKTHROUGH.md`.
