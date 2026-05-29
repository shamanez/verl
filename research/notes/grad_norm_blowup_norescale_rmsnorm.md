# Why no-rescale activation masking explodes the GRPO grad-norm (and rescale fixes it)

**EXP-16, 2026-05-30. Qwen2.5-1.5B-Instruct / GSM8K / vanilla GRPO, 4×B200.**

## TL;DR

Masking ~90% of the residual-stream activations at pipeline boundaries **without**
the inverted-dropout `1/(1-p)` rescale shrinks the residual-stream RMS. The
transformer's **RMSNorm layers are scale-invariant in the forward pass but their
backward Jacobian scales like `1/RMS(input)`** — so a smaller forward RMS
**amplifies every gradient that flows backward through the norm**, and that
amplification **compounds over the 7 masked boundaries**. Result: `grad_norm`
≈ 2700 for mask-no-rescale vs ≈ 0.38 dense. Rescale restores the activation
magnitude, so there is nothing to amplify (`grad_norm` ≈ 4.4).

**It is NOT an FSDP bug, NOT a gradient-checkpointing bug, NOT an in-place /
fused-kernel artifact.** It is the autograd math of RMSNorm + no-rescale masking,
and it reproduces on a single GPU with no FSDP.

**Picture:** `research/runs/EXP-16/grad_sketch.png` — left: `grad_norm` by config
(dense 0.38 / rescale 4.5 / no-rescale 2700, log scale); right: the gradient leaving
RMSNorm vs its input RMS, sitting on the `1/RMS` line.

Forward/backward round trip at ONE masked boundary (what the whole note is about):

```
FORWARD
  h  ──[mask: zero 90% of channels]──►  h̃   (RMS collapses ~50× if NO rescale)
  h̃ ──[RMSNorm: ÷ RMS(h̃)]──►  y            (y looks fine — the norm re-normalizes magnitude)
  y  ──► rest of network ──► loss
BACKWARD
  g_y ──[RMSNorm backward: × ~1/RMS(h̃)]──►  g_h̃   ◄── BLOWS UP when RMS(h̃) is tiny
  g_h̃ ──[mask backward: × m  (× 1/(1-p) if rescale)]──►  g_h
  g_h ──► upstream layers 0,1,2…  ◄── enormous  ⇒  big grad_norm
```

We zero **activations** (the residual-stream hidden state), NOT parameters. The small
activation RMS is what the downstream norm divides by, so its backward multiplies the
gradient by `1/RMS` → bigger gradient. That is the entire effect.

## Evidence (triangulated)

Isolated single-GPU probe (`research/runs/EXP-16/grad_diag.py`; no FSDP, no Ray,
raw cross-entropy on a fixed batch):

| config | grad_norm, ckpt ON | grad_norm, ckpt OFF |
|---|---|---|
| dense | 30.7 | 30.7 |
| mask p=0.9 **no-rescale** | 172,850 (**5620×**) | 172,850 (5620×) |
| mask p=0.9 **rescale** | 71.7 (**2.3×**) | 71.7 (2.3×) |

Real harness (FSDP + GRPO + GSM8K, 3-step diagnostics):

| run | grad_norm | entropy | pearson(actor vs rollout) |
|---|---|---|---|
| dense (comm_eff off) | **0.38** | 0.39 | **0.9996** |
| mask p=0.9 **rescale** | **~4.4** | 5.92 | 0.006 |
| mask p=0.9 **no-rescale** (cell 1) | **~2700** | 6.35 | 0.02 |

Two control facts from the probe that matter:
- `grad-ckpt effect on masked (ON/OFF) = 1.000` → gradient-checkpoint recompute is
  fully consistent (it re-applies the same mask). **Not a checkpointing bug.**
- `mask.requires_grad = False`, `mask unique = [0,1]`, applied out-of-place as
  `h̃ = h * mask` → autograd gates the backward by the same mask. **Mask is correct.**
- The 5620× reproduces with **no FSDP at all**. **Not an FSDP issue.**

## The mechanism, step by step

### 1. Masking shrinks the residual-stream magnitude

At a boundary: `h̃ = h ⊙ m`, with `m∈{0,1}` keeping each element with prob `(1-p)=0.1`.

- **No-rescale:** `E[h̃] = (1-p)·h = 0.1·h`; `mean(h̃²) = (1-p)·mean(h²) = 0.1·σ²`, so
  `RMS(h̃) ≈ 0.316·σ` — the residual-stream RMS **drops ~3.2×**.
- **Rescale (inverted dropout):** `h̃ = h⊙m/(1-p) = h⊙m × 10`; `E[h̃] = h` — expected
  activation preserved (exactly the train-vs-test dropout trick). RMS back near nominal.

### 2. RMSNorm: forward scale-invariant, backward ∝ 1/RMS

Qwen is pre-norm; the next block applies `y = RMSNorm(x) = x / RMS(x) · γ`,
`RMS(x) = sqrt(mean(x²))`.

- **Forward** is scale-invariant: `RMSNorm(c·x) = RMSNorm(x)`. So masking does **not**
  change the forward output scale — which is exactly why this is invisible as a
  forward problem.
  *Intuition for the backward:* if scaling the input by `c` leaves the output unchanged,
  the Jacobian MUST scale by `1/c`. Shrink the input 50× and the output is identical, so
  the output is "50× more sensitive" per unit of input → the backward gradient is ×50.
  That `1/c = 1/RMS` is the whole effect; the formula below just makes it exact.
- **Backward** Jacobian:
  `∂y_i/∂x_k = (γ_i / RMS(x)) · [ δ_ik − x_i x_k /(H·RMS(x)²) ]`.
  The leading factor is **`1/RMS(x)`**. Gradient flowing back into the (masked)
  residual stream is multiplied by `1/RMS`.

Plugging in:
- dense: backward factor `≈ 1/σ`
- no-rescale: `RMS ≈ 0.316σ` → factor `≈ 3.16/σ` → **~3.2× larger** per boundary
- rescale: RMS restored → factor `≈ 1/σ` → no amplification

### 3. It compounds over 7 boundaries

Boundaries at layers **3, 7, 11, 15, 18, 21, 24** (7 of them). A gradient
propagating from the loss toward the **early** layers crosses all the downstream
boundary-norms, each multiplying by ~3.2×:

```
no-rescale ≈ 3.16^7 ≈ 3,900×   (measured 5,620×)
rescale    ≈ ~1×               (measured 2.3×)
```

The order-of-magnitude match is good, but the real fingerprint is **where** the
gradient lives: the probe's per-layer breakdown for no-rescale is dominated by
**layers 0,1,2,4** — the earliest layers, whose backward path crosses the most
amplifying norms. Dense/rescale show no such early-layer pile-up. That layer-wise
signature is the strongest confirmation of the mechanism.

### 4. Why rescale is the fix

Rescale by `1/(1-p)` puts the surviving activations back at the right magnitude →
`RMS(h̃) ≈ σ` → the `1/RMS` factor stays ~1 → no per-boundary amplification →
nothing to compound. grad_norm returns to dense scale. The forward was always fine
(norm is scale-invariant); rescale's real job is keeping the **backward** pass
well-conditioned.

## This is architecture-general: every pre-norm transformer (Llama 3.2, Mistral, Gemma, …)

This is **not** a Qwen quirk. The blow-up needs only two ingredients, and both hold for
essentially every modern LLM:

1. the mask is applied to the **residual-stream activation**, and
2. a **scale-invariant normalization** (RMSNorm *or* LayerNorm) sits **downstream** of it.

Pre-norm architectures place a normalization at the input of *every* block, so a masked
residual point *always* feeds a norm immediately — they are the maximally-exposed case.
**Llama (all versions incl. 3.2), Qwen2.5, Mistral, Gemma, Phi are pre-RMSNorm → identical
behaviour.** GPT-2/3-style **pre-LayerNorm** models hit the same thing via LayerNorm's
`1/std` backward (LayerNorm is also scale-invariant). Even **post-norm** models are not
immune: any masked activation that subsequently flows through a norm gets the same
`1/scale` backward amplification — pre-norm just guarantees one is always right there.

Precise statement: *masking an activation that is later normalized by a scale-invariant
op ⇒ backward gain ∝ `1/scale`.* The **fix (inverted-dropout rescale) is equally
architecture-general** — restore `E[activation]` and the collapse disappears on any of
these models. Only the *magnitude* shifts with depth, hidden size, mask-rate `p`, and
where the boundaries land; the law and its fix do not. (So a Llama-3.2
dense-vs-norescale-vs-rescale probe would show the same dense « rescale « no-rescale
ordering we measured on Qwen2.5-1.5B.)

## Why rescale still sits ~12× above dense on the FSDP run (0.38 → 4.5)

Rescale removes the *catastrophic* part (the RMS-collapse `1/RMS` blow-up: 7100× → gone),
but a **bounded** gap remains — on the real FSDP/GRPO run, dense `grad_norm ≈ 0.38` vs
rescale `≈ 4.5`, i.e. **~12×**. Two honest, non-bug sources:

1. **Mask backward gain.** `g_h = (m/(1-p))·g_h̃` amplifies the 10% kept channels ×10; in
   L2 that is `√(1/(1-p)) ≈ 3.2×` vs dense, independent of any norm.
2. **Stochastic-mask variance.** A *fresh random* 90% mask every step makes the forward
   (hence loss and gradient) a **noisy** estimate of the dense one — variance inflated
   `~p/(1-p) ≈ 9×` at `p=0.9`.

Together ≈ 12×. This is the documented **variance cost** of masking (bounded, shrinks
with lower `p`), not an unbounded bias-driven explosion. It is exactly what anchor +
spectral correction + lower-`p` exist to reduce — rescale's only job is to kill the
RMS-collapse blow-up, which it does. (NB: rescale's second moment *overshoots* —
`RMS ≈ 3042` vs dense `52.6`, measured — so the `1/RMS` factor actually goes slightly
*below* 1; the residual 12× is the variance/backward-gain above, not a leftover `1/RMS`.)

## Wrong hypotheses, ruled out

- **"It's an FSDP kernel / sharding thing."** No — reproduced on a single GPU with
  no FSDP. FSDP all-gathers *parameters* (and writes them into a flat buffer
  in-place), but that does not touch the per-layer activation autograd. The RMSNorm
  `1/RMS` backward is identical with or without FSDP.
- **"It's gradient checkpointing dropping the mask on recompute."** No — ckpt ON ==
  ckpt OFF exactly (ratio 1.000). The counter coincidence
  (`mask_applications/train == /old_logprob`) was a red herring.
- **"It's an in-place op / fused kernel keeping the wrong input."** The mask multiply
  and RMSNorm are out-of-place; a bad in-place op would trip autograd's version
  counter and error, not silently inflate. Fused norm kernels compute the *same*
  mathematical gradient (fusion is a perf optimization, not a different function).
- **"The mask is inconsistent across passes (IS ratio broken)."** No — mask is a
  deterministic PRF, bit-identical across old_logprob and train forwards
  (cell-0 gate + `ppo_kl ≈ 0`). Consistency makes the IS ratio ≈ 1; it does **not**
  make the gradient small. The no-rescale magnitude bias does the damage.

## Caveats / open questions

1. **Rescale matches the mean, not the RMS.** Only ~10% of elements survive, each ×10,
   so the *second moment* overshoots (`RMS ≈ 3.16σ`) and per-element variance is high.
   That's why rescale lands at ~2.3× dense, not exactly 1×. The variance is large at
   `p=0.9`; milder at lower `p`.
2. **Rescale fixes the gradient *scale*, not the policy *mismatch*.** Even rescaled,
   the masked actor's per-token probs stay nearly uncorrelated with the vLLM rollout
   (pearson ~0.006 vs 0.9996 dense). The masked subnetwork is still a different
   function; closing that gap is the job of anchor + spectral correction, not rescale.

## The actual levers (if we want to reduce the blow-up)

- **`rescale=true`** (the inverted-dropout fix) — primary lever; already cell 2.
- **Lower `p`** — less RMS suppression per boundary, smaller amplification.
- **Fewer / different boundaries** — the compounding is `~3.16^(#boundaries)`.
- **Where the mask is applied** — masking the pre-norm residual stream is what couples
  it to the `1/RMS` backward. Masking post-norm (or a normalized quantity) would
  change this coupling. (Design question, not a quick toggle.)

## Direct tests answering the "FSDP / in-place / fused-kernel" hypothesis (2026-05-30)

Three probes under `research/runs/EXP-16/`, all CPU / single-GPU, no FSDP. They
settle whether the blow-up is an infra artifact (FSDP kernel, in-place op, fused
norm, gradient-checkpoint recompute) or the RMSNorm math.

**`grad_diag3.py` — elementary autograd ground truth + RMSNorm 1/RMS sweep:**
- `mask.requires_grad=False`, kept_frac=0.10.
- `[no-rescale] h.grad == mask -> True`. If the backward had "kept the input as is"
  (unmasked), `h.grad` would be all-ones; instead only 10% nonzero (the kept
  elements). The backward IS correctly masked. **No in-place / keep-input artifact.**
- `[rescale]   h2.grad == mask*10 -> True`.
- `[ckpt]      h3.grad == mask -> True` through `torch.utils.checkpoint(use_reentrant=False)`
  — recompute preserves the mask in the backward graph. **Not a checkpointing bug.**
- RMSNorm backward sweep (the mechanism, measured): input_RMS 0.02 -> grad 15642;
  RMS 1.00 -> grad 313; RMS 58 -> grad 5.4. The product `RMS × grad ≈ 313` is
  constant -> gradient through RMSNorm scales **exactly as 1/RMS(input)**.

**`grad_diag2.py` — full Qwen2.5-1.5B (CPU, fp32, eager), boundary-fire count + RMS:**
| config | boundary_fires ON/OFF | mean boundary RMS | grad_norm ON/OFF |
|---|---|---|---|
| dense | 7 / 7 | 52.56 | 114.6 / 114.6 |
| mask no-rescale | 7 / 7 | 1.09 (0.021× dense) | 39440 / 39440 |
| mask rescale | 7 / 7 | 3042 (58× dense) | 334 / 334 |
- grad inflation: no-rescale 344× dense, rescale 2.9× dense.
- `boundary_fires == 7` for ckpt ON and OFF (so the hook fires once — this is why the
  training counter was `mask_applications/train == /old_logprob == 1792`, NOT 2×).
  **But `grad_norm` is identical ON vs OFF (39440==39440)** -> checkpoint reconstructs
  the masked activation in the backward graph; firing once is benign. The earlier
  "1792 ⟹ recompute unmasked ⟹ bug" inference is FALSE.

**`grad_diag.py` — single-GPU bf16, raw CE:** dense 30.7, no-rescale 172850 (5620×),
rescale 71.7 (2.3×), `grad-ckpt effect on masked ON/OFF = 1.000`.

**`grad_psweep.py` — grad_norm vs mask-rate `p` × rescale (GPU bf16, cross-entropy / SFT-style, no FSDP):**
| p | no-rescale (× dense) | rescale (× dense) |
|---|---|---|
| 0.1 | 36 (1.2×) | 26 (0.8×) |
| 0.3 | 1507 (49×) | 25 (0.8×) |
| 0.5 | 11190 (364×) | 31 (1.0×) |
| 0.7 | 22648 (737×) | 39 (1.3×) |
| 0.9 | 172850 (5621×) | 72 (2.3×) |
- No-rescale explodes super-linearly with `p` (normal at p=0.1, 5621× at p=0.9); rescale
  stays ~normal (0.8–2.3× dense) across the WHOLE range. This is the SFT regime (CE loss,
  no FSDP) — so a supervised mask demo with a normal grad norm simply used rescale, or a
  low `p`, or fewer/post-norm boundaries. "SFT vs GRPO" and "FSDP vs not" are NOT the
  variable; rescale + p + boundary placement are.

All four methods (GPU-bf16, CPU-fp32-full-model, elementary-autograd, p-sweep) agree:
forward/backward mask is consistent (not FSDP, not checkpointing, not in-place, not
fusion); the blow-up is RMSNorm's `1/RMS` backward × the no-rescale residual-RMS
collapse; rescale restores the RMS and removes it.

**Answering the specific questions:**
- *"backward scales like 1/RMS — is that the FSDP kernel?"* No — it's RMSNorm's autograd
  (`x·rsqrt(mean(x²))`), reproduced on CPU with no FSDP and in elementary autograd.
- *"in-place op / fused kernel keeping the input as is?"* No — `h.grad == mask` exactly
  (not all-ones); Qwen2RMSNorm is plain out-of-place ops; a bad in-place would trip
  autograd's version counter and error, not silently inflate.
- *"is the huge thing coming from the actual layer inputs?"* Yes — the amplified
  quantity is `dL/d(layer input)` (the masked residual), via `1/RMS`; it propagates to
  the early-layer weights (probe shows layers 0–4 dominate).
- *"a way to switch off in-place / change this?"* There is no in-place op to switch off;
  the gradient is mathematically correct. The real levers are `rescale=true` (restores
  RMS — already shown to fix it), lower `p`, fewer boundaries, or masking a
  post-norm / normalized quantity instead of the pre-norm residual.
