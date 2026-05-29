# Research runs — summary

Concise, self-contained record of the method, what's been tried, and the knob
surface. No per-run artifacts are kept; this is the durable record.

## Baseline = dense GRPO == method OFF

The dense control is the comm-eff launcher with `COMM_EFF_ENABLED=false` —
byte-identical to unmodified verl, no-KL no-entropy. It learns cleanly on GSM8K
in a short run and is the bar every compression run must match.

## Comm-eff method — implementation correct, masking under test

The implementation is correct (OFF ⇒ dense parity; the mask fires on exactly
the gradient-feeding forwards; unit-tested). The masking side still needs a lot
of testing — at high mask rates plain masked GRPO does not yet learn.

### What we tried, and what it told us

Headline up front: **a large grad_norm was never the disease.** Adam's
per-coordinate update is scale-invariant (`m̂/√v̂` cancels any constant scaling
of the gradient) and bounded to order `η`, and verl grad-clips on top — so a big
raw norm cannot, by itself, produce a large or "exploding" update. The two real
failure modes are **bias** and **variance**.

| tried | result |
|---|---|
| mask only, **no rescale** (p=0.9) | **biased.** With no `1/(1-p)`, `E[h⊙mask] = (1-p)·h` — the masked forward sits systematically off-distribution, so the GRPO importance ratio `r = exp(Δlogp)` is corrupted (and *that* is what inflates the *measured* grad_norm; the activations themselves *shrink*, they do not grow). The harm is the bias — a direction error Adam cannot correct → a stalled trajectory with a non-vanishing floor. |
| **`rescale`** (inverted-dropout `h⊙mask/(1-p)`) | **unbiased, but high-variance.** Restores `E[h̃] = h` → forward back on-distribution → `r ≈ 1`, `ppo_kl ≈ 0`, grad_norm back to dense order. **Still does not learn:** rescale trades bias for variance (`p/(1-p)`), and with no denoising (hard grad-clip / slow EMA / spectral contraction / keeping the anchor gradient out of Adam's moments) the unbiased-but-noisy gradient is a random walk in a short run; val stays flat. The *variance* — not the rescale — is what the anchor + spectral machinery exists to tame. |
| `consistent_across_forwards` (same seed across forwards) | refuted on its own — the per-element mask is positional and the two forwards pack tokens differently, so equal seed ≠ equal mask. Cross-pass consistency is instead structural via per-channel masking. |
| naive `clean_cadence` (periodic unmasked step) | **not sustainable** — the masked steps stay corrupted and the PPO clip fraction climbs toward saturation, so clipped tokens stop contributing gradient and learning dies; any early score rise is the clean steps alone. |

**Open question:** can masked GRPO learn at all, and at what mask rate? Next is a
mask-rate sweep (p = 0.9 → 0.5 → 0.1) judged on val/score and a stable, low PPO
clip fraction. **Anchor + spectral correction stay OFF** until a masked config
is shown to actually learn.

## Knob surface (in `vast_comm_eff_baseline_*.sh`)

All independently env-toggleable; defaults = the mask-only baseline to start the
sweep from.

| knob | default | meaning |
|---|---|---|
| `COMM_EFF_ENABLED` | true | master switch (false ⇒ byte-identical dense) |
| `COMM_EFF_MASK_ENABLED` | true | activation mask on pipeline-boundary blocks |
| `COMM_EFF_MASK_P` | 0.9 | masked fraction (sweep target) |
| `COMM_EFF_MASK_GRANULARITY` | channel | per-channel (packing-invariant ⇒ identical mask across forwards) vs `element` (legacy) |
| `COMM_EFF_MASK_RESCALE` | true | inverted-dropout `h*mask/(1-p)` — keeps grad_norm at dense order (not a learning fix on its own) |
| `COMM_EFF_CLEAN_CADENCE` | 0 (OFF) | naive periodic unmasked step — unsustainable, opt-in only |
| `COMM_EFF_ANCHOR_ENABLED` | false | K-stale anchor circuit (layer on later) |
| `COMM_EFF_SPECTRAL_ENABLED` | false | two-sided Tikhonov spectral correction (layer on later) |

## Implementation locus (on `vast-ai-workload`)

- `verl/workers/config/comm_eff.py` — config schema
- `verl/workers/comm_eff/{state.py, activation_mask.py, anchor.py, spectral_filter.py}` — runtime
- `verl/workers/engine_workers.py` — `compute_log_prob` mask stamp
- `verl/workers/engine/fsdp/transformer_impl.py` — boundary-block mask gating
- `tests/workers/comm_eff/` — CPU unit tests

## Conceptual notes

- `notes/anchor-memory-cost.md` — why the anchor clone is memory-heavy
- `notes/fast-circuit-vs-anchor-pass.md` — which of the GRPO forwards get masked
