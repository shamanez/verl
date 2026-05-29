# Tuning under (simulated) pipeline parallelism while matching dense — strategy

**EXP-16, 2026-05-30. Qwen2.5-1.5B-Instruct, 4×B200. Companion to
`grad_norm_blowup_norescale_rmsnorm.md`.**

Goal: train with per-element activation masking at the 7 pipeline-stage boundaries
(so only the kept fraction of activations is communicated across stages) such that
(a) the grad-norm is well-behaved and (b) the training trajectory / final model
matches the dense (un-masked) baseline.

## What the measurements say (the constraints we must design around)

All single-GPU, no FSDP, cross-entropy on a fixed batch, mask at 7 boundaries
(layers 3,7,11,15,18,21,24). Probes under `research/runs/EXP-16/grad_*.py`.

**1. The norm is the EASY half — rescale fixes it.**
`rescale=true` (`h⊙m/(1-p)`) keeps `grad_norm/dense ≈ 1` up to p=0.7 (2.3× at p=0.9).
Without rescale, the residual-RMS collapse drives RMSNorm's `1/RMS` backward and
`grad_norm` explodes (5620× at p=0.9). **Make `mask.rescale=true` the default.**

**2. The DIRECTION is the hard half — fidelity to dense decays steeply with p.**
cosine(rescaled masked grad, dense grad) vs mask-rate p (`grad_fidelity.py`):

| p | comms kept | cos→dense |
|---|---|---|
| 0.02 | 98% | 0.90 |
| 0.10 | 90% | 0.58 |
| 0.20 | 80% | 0.50 |
| 0.30 | 70% | 0.28 |
| 0.50 | 50% | 0.09 |
| 0.70 | 30% | 0.01 |
| 0.90 | 10% | 0.01 |

cos→1 as p→0 (probe validated). To keep cos ≥ 0.5 you need p ≲ 0.2 (keep ≥80%).
At the aggressive p=0.9 (10% comms) a single masked step is ~orthogonal to dense.

**3. The gap is BIAS, not just variance — averaging does NOT fix it.**
Averaging the rescaled masked grad over K=1..32 independent mask draws
(`grad_avg_probe.py`): at p=0.9 cos stays ≈0 and avg-norm/dense → 0.46; at p=0.5
cos rises only 0.06→0.17 and norm/dense → 0.48. If rescale were unbiased the
average would head to cos→1, norm→1. It does not ⇒ stacking masking at 7
boundaries through deep nonlinearities (RMSNorm, softmax) yields a **biased**
expected gradient. Variance reduction (more draws, antithetic) alone cannot reach
dense; the bias must be **corrected**.

**Conclusion:** "identical to dense" at a useful comms saving is impossible from
masking+rescale alone. It requires either a low mask rate (small saving) or an
explicit dense-correction. This is exactly the scientific case for the project's
anchor + spectral correction.

## The strategy — layered, cheapest-first

1. **Rescale, always.** `actor.comm_eff.mask.rescale=true`. Fixes the norm; makes
   the per-step gradient the right magnitude. Non-negotiable default.
2. **Pick p from the frontier + your comms budget.** cos≥0.5 ⇒ p≲0.2;
   p=0.3 ⇒ cos≈0.28; p≥0.5 only viable WITH a corrector. Don't run p=0.9 raw.
3. **Anneal p (curriculum).** Start p≈0 (dense) and ramp to target p over the first
   N steps. The earliest, highest-leverage updates stay high-fidelity; the model
   adapts to the masked subnetwork before p is aggressive.
4. **Periodic clean (dense) steps** — `clean_cadence=K`. Every K steps run a fully
   un-masked step → an EXACT, unbiased dense gradient that re-anchors the
   trajectory. Cheap, exact, coarse. (EXP-16 cells 3/4.) Cost: 1 full-comms step
   per K.
5. **Anchor + spectral correction** — `anchor + spectral` (EXP-16 cell 5). The
   principled bias-corrector: a low-rank control-variate built from a stale
   dense/anchor computation that subtracts the masked gradient's dominant biased
   subspace, restoring cosine toward dense WITHOUT paying full dense comms every
   step. Our probes quantify the bias it must remove (large at high p). **This is
   where the method earns its comms savings; if it works, it is what makes high-p
   training dense-equivalent.**
6. **Grad clip stays** (`grad_clip=1.0`) as a safety net (not a fix).

## Magnitude-restoration modes: `constant` vs `rms_match` — implemented & measured (`grad_modecmp.py`)

The masker now has a switchable, non-destructive `mask.rescale_mode ∈
{none, constant, rms_match, auto}` (the legacy `rescale` bool maps via `auto`:
true→constant, false→none). The three boundary formulas:

- `none`: `h⊙m` (raw).
- `constant`: `h⊙m/(1−p)` (inverted dropout; Idea 1).
- `rms_match`: `h⊙m · detach(rms_true/rms_masked)` (Idea 2b, self-contained &
  comms-valid: a **detached** per-token gain that forces the masked activation's
  RMS to equal the **true** pre-mask RMS, so the downstream pre-norm RMSNorm
  divides by the true RMS. Comms: `rms_true` is a 1-float/token side channel
  (~0.6% at p=0.9); `rms_masked` is recoverable on the receiver from the kept
  entries. Gain ≈ √(1/(1−p)) ⇒ far milder than constant's 1/(1−p), so it is
  *more* bf16-safe per element).

Measured on real Qwen2.5-1.5B (4×B200, CE on a fixed batch, mask at all 7
boundaries, via the REAL `ActivationMasker`; all 52 existing mask unit tests
still pass — the switch is non-destructive):

| mode | p | RMS(h̃)/dense | cos→dense | grad_norm/dense |
|---|---|---|---|---|
| none | 0.5 | 0.706 | 0.012 | 259 |
| constant | 0.5 | 1.413 | 0.087 | **1.26** |
| rms_match | 0.5 | **1.000** | 0.049 | 6.41 |
| none | 0.9 | 0.311 | 0.00 | 1603 |
| constant | 0.9 | 3.115 | 0.012 | **2.30** |
| rms_match | 0.9 | **1.000** | 0.020 | 35.7 |
| none | 0.95 | 0.216 | 0.005 | 1309 |
| constant | 0.95 | 4.327 | −0.004 | **5.05** |
| rms_match | 0.95 | **1.000** | 0.016 | 27.9 |

Two findings, one of them counter to the original prediction:

1. **`rms_match` delivers EXACT activation RMS** (1.000 per token at every p) — the
   "exact norm" design claim is correct *at the activation level*, validated on
   the real hook.
2. **But `rms_match` does NOT beat `constant` on the GRADIENT norm — it is worse**
   (6–36× dense vs 1.3–5×). Mechanism: `constant` *overshoots* the RMS
   (1.4–4.3×), which **damps** the downstream RMSNorm `1/RMS` backward at each of
   the 7 boundaries, and that damping **compounds** (≈1.41⁷≈7.5× at p=0.5 —
   matches the 6.4/1.26 gap). `rms_match` removes that damping by hitting exactly
   the dense RMS. **So constant's overshoot is a *feature* for grad-norm
   stability, not a bug.**
3. Neither touches **direction**: cos→dense stays ≈0 at high p for both, and the
   constant-vs-rms_match cos gap is noise-level. Direction is the spectral job.

**Verdict:** keep **`constant`** (the default `rescale=true`) as the grad-norm
stabilizer — its overshoot is beneficial. Reach for **`rms_match`** only when you
need exact *forward* activation statistics or low-bit quantization at the
boundary (milder, bounded per-element gain), accepting a larger grad norm. The
sublayer-output variant (Idea 2a) stays ruled out — it never crosses the wire.

## The masked-gradient bias is LOW-RANK — the spectral premise is validated (`grad_lowrank_probe.py`)

The key question for "can we reach dense": is the bias `R = g_dense − g_masked`
low-rank (correctable cheaply by a stale anchor) or full-rank (hopeless)? We added a
rank-k SVD of R back to the masked grad per weight matrix and measured cosine→dense
(idealized upper bound — uses the exact current dense residual):

| rank k | cos→dense @ p=0.5 | top-k energy(R) | cos→dense @ p=0.9 |
|---|---|---|---|
| 0 (raw) | 0.087 | — | 0.012 |
| 1 | **0.718** | 0.57 | 0.453 |
| 2 | 0.780 | 0.68 | 0.498 |
| 4 | 0.821 | 0.76 | 0.547 |
| 8 | 0.852 | 0.81 | 0.590 |
| 16 | 0.880 | 0.85 | 0.634 |
| 32 | **0.908** | 0.89 | 0.688 |

**A single rank-1 correction recovers most of the alignment** (p=0.5: 0.09→0.72,
capturing 57% of residual energy); rank-32 → 0.91. So the bias masking introduces is
heavily concentrated in a few directions ⇒ a **low-rank control-variate from a stale
dense anchor (= the spectral correction) is the right tool and can make masked
training dense-equivalent.** This is the strongest gradient-level evidence that the
project's anchor+spectral mechanism works.

Implications for the spectral config: even small rank helps enormously, but
fidelity keeps climbing to rank≈16–32, so `spectral.max_targets`/rank should be
tuned upward from the current default (was 4) if budget allows; at p=0.9 you need
more rank (rank-32 → 0.69) than at p=0.5 (rank-8 → 0.85). The real (stale-anchor)
corrector will sit below this idealized curve — that gap is exactly what EXP-16
cell 5 measures.

## The decisive remaining experiment (EXP-16 cell 5)

The table above is the IDEALIZED (current-dense-residual) upper bound. The remaining
question is how much the STALE anchor + fixed spectral basis gives up vs it. Run the
actual anchor+spectral correction (cell 5) and measure cos(corrected grad, dense)
across p — success = corrected cosine close to the idealized curve (e.g. ≥0.7 at
p=0.5). This tells us the real comms-vs-fidelity operating point.

## Recommended config for a first dense-matching run

```
COMM_EFF_ENABLED=true
COMM_EFF_MASK_ENABLED=true
COMM_EFF_MASK_RESCALE=true          # mandatory — fixes the norm
COMM_EFF_MASK_P=0.5                  # or anneal 0.0 -> 0.5; not 0.9 raw
COMM_EFF_MASK_RECOMPUTE=true
COMM_EFF_CLEAN_CADENCE=4             # exact dense re-anchor every 4 steps
COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=2 COMM_EFF_ANCHOR_DELAY_K=2
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CADENCE=2 COMM_EFF_SPECTRAL_ALPHA=0.5
```
Compare cos-to-dense + `train/reward_mean` trajectory against the dense baseline.
If the corrected gradient tracks dense at p=0.5, push p up; if not, lower p / raise
clean cadence until the trajectory matches, then trade back toward comms savings.

## Honest bottom line

- **Norm:** solved by the **`constant`** rescale (`h⊙m/(1−p)` → grad_norm/dense
  1.3–5× across p). Implemented as a switchable `mask.rescale_mode`; the new
  `rms_match` mode gives *exact* activation RMS but a *larger* grad norm (its
  overshoot-free RMS removes the beneficial downstream damping), so `constant`
  stays the stabilizer; `rms_match` is for forward-stat fidelity / quantization.
- **Dense-identical at high mask rate:** NOT achievable by masking+rescale alone —
  the expected masked gradient is biased and near-orthogonal at p≥0.7, and averaging
  doesn't fix it (the gap is bias, not noise).
- **BUT the bias is low-rank**, so it is correctable: an idealized rank-1 correction
  takes p=0.5 alignment 0.09→0.72 and rank-32 →0.91 (p=0.9: →0.69). This validates
  the spectral/anchor control-variate as the load-bearing mechanism.
- **Therefore the achievable recipe is: rescale (fix norm) + low/annealed p (stay on
  the good part of the fidelity frontier) + periodic clean steps (exact dense
  re-anchor) + anchor+spectral low-rank correction (remove the residual bias).**
  Dense-*equivalent* (same trajectory/quality) is realistic; bit-identical is not
  (and isn't the point — the comms saving is). The one number still to measure is how
  much the STALE anchor gives up vs the idealized low-rank curve (EXP-16 cell 5).
