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

## The decisive next experiment (extends EXP-16)

The probes above measure the RAW masked gradient. The open question — *does the
corrector close the cosine gap?* — is answered by measuring
**cos(spectral/anchor-corrected grad, dense grad)** vs p, the same way:

- Build a `grad_corrected_cosine.py` that applies the anchor+spectral correction to
  the masked gradient and reports cos→dense and norm/dense across p.
- Success = the corrected cosine sits well above the raw curve at the target p
  (e.g. corrected cos ≥ 0.7 at p=0.5), with norm/dense ≈ 1.
- This is the single most informative run for the goal — it directly tests whether
  the method can make high-mask training dense-equivalent, BEFORE spending full
  multi-step GRPO budget. It also tells the analyst exactly what to expect from
  EXP-16 cell 5.

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

- **Norm:** solved (rescale).
- **Dense-identical at high mask rate:** not achievable by masking+rescale alone —
  the expected gradient is biased and near-orthogonal at p≥0.7. Achievable
  *trajectory/quality* equivalence is plausible via rescale + low/annealed p +
  clean-step cadence + a working spectral/anchor corrector. The spectral corrector
  is the load-bearing piece, and the corrected-cosine probe is how we prove it
  before burning GRPO hours.
