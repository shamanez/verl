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

Headline up front: **judge on val/score, not grad_norm.** Adam's per-coordinate
update is scale-invariant (`m̂/√v̂` cancels any constant scaling of the gradient)
and bounded to order `η`, and verl grad-clips on top — so a large raw norm
cannot, by itself, produce a large or "exploding" update. The two real failure
modes are **bias** and **variance**.

> ⚠️ **Correction — a mistake we made.** An earlier version of this record
> claimed *"rescale reduces / tames the grad_norm."* **That is wrong and has
> been removed.** Theory says the opposite: rescale *adds* variance. We do
> **not** attribute the grad_norm differences seen across cells to rescale, or
> to any single knob — those cells stacked several changes (consistency +
> rescale + packing/seed differences), the artifacts are pruned, and the
> comparison is confounded. The lower grad_norm we observed most likely came
> from something else (e.g. mask consistency / per-token effects / run
> randomness), not from rescale.

| tried | what it is / what it told us |
|---|---|
| mask only, **no rescale** (p=0.9) | **biased mask.** With no `1/(1-p)`, `E[h⊙mask] = (1-p)·h` — the masked forward sits systematically off-distribution, so the GRPO importance ratio is corrupted. The harm is the bias: a direction error Adam cannot correct → a stalled trajectory with a non-vanishing floor. (Dropping rescale makes activations *shrink*, not grow.) |
| **`rescale`** (inverted-dropout `h⊙mask/(1-p)`) | **a knob, not a fix.** Its only job is to restore `E[h̃] = h` — an *unbiased* mask. It trades bias for variance (`p/(1-p)`): necessary for correctness, not sufficient. Plain masked GRPO with rescale still did not learn in a short run (val flat). The variance is what the anchor + spectral + grad-clip machinery exists to tame. |
| `consistent_across_forwards` (same seed across forwards) | refuted on its own — keying the mask positionally over each forward's packing gave a token a different mask in the two differently-packed forwards, so equal seed ≠ equal mask. Cross-pass consistency is now achieved by keying the per-element mask on each token's stable `(sample_id, position_id)`; this knob was removed. |
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
| `COMM_EFF_MASK_RESCALE` | true | inverted-dropout `h*mask/(1-p)` — restores `E[h̃]=h` (unbiased mask; not a learning fix on its own) |
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
