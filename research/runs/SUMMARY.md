# Research runs — summary

Durable, self-contained operational record: the method, the proven result, the
settled decisions, and the knob surface. No per-run artifacts are kept; the
next-cycle plan is `findings/NEXT_RESEARCH.md`.

## Baseline = dense GRPO == method OFF

The comm-eff launcher with `COMM_EFF_ENABLED=false` — byte-identical to unmodified
verl, no-KL no-entropy. Learns cleanly on GSM8K; the bar every compression run must match.

## The settled comm-eff base

`mask (p=0.9, per-(token,channel) at the 7 pipeline boundaries [3,7,11,15,18,21,24])`
**+ rescale (inverted-dropout `1/(1-p)`) + a true dense "clean" gradient every K
steps (`clean_cadence`)**. All three settled (see "Settled decisions"). Anchor +
spectral as implemented are **not** part of the base.

## Proven result — stable; GSM8K parity (elicitation), Big-Math stall (fidelity limit)

With `clean_cadence` the masked path learns and does not diverge: the
train-inference gap is a **clean-resettable sawtooth** (masked steps `kl≈17 /
pearson≈0.004` → clean steps `kl≈0.0004 / pearson≈0.9996`, resets fully every clean
step), clean-step grad_norm trends **down**, true-policy entropy stays sharp — no
ratchet, no drift to random (EXP-17 PASS). The PPO ratio is masked-old-vs-masked-new
(self-consistent, ≈1), so the gap never enters the loss and the unmasked deployed
policy still improves.

| run | dataset | method | val (start→end) | reading |
|---|---|---|---|---|
| dense (EXP-16 cell6) | GSM8K | dense | → 0.741 | bar |
| **EXP-17** | GSM8K | mask p=0.9 + **clean@20** | 0.085 → **0.735** | ≈ parity (−0.8%, ~85.5% comm cut) → **elicitation** |
| **EXP-19** | Big-Math | mask p=0.9 + **clean@20** | 0.56 → **0.55 flat** | **stalls** = gradient-fidelity limit |
| **EXP-20** | Big-Math | dense | 0.558 → **~0.59–0.61** | learns → headroom exists |

**Why** (base Qwen2.5-1.5B, zero RL, `\boxed`+`math_reward`, 200 each): **GSM8K 0.715
vs Big-Math 0.480**. GSM8K is easy → RL only *elicits* a latent skill (a coarse
compressed gradient suffices → parity). Big-Math is hard → genuine learning needed;
dense finds +0.06, the lossy masked gradient cannot → flat. Mask information loss is
tolerable for elicitation, fatal for learning. Dense reaching 0.61 proves headroom
*exists* → the stall is a fidelity limit, not a ceiling.

## Anchor + spectral, as implemented — did NOT work

EXP-16 `anchor@2+spectral@2` (no clean steps): **GSM8K val 0.080 ≈ random**, pearson
still ~0.004, inert. Root cause **orthogonality** — spectral is a *linear reweighting
of the masked gradient* in the unmasked anchor's SVD basis, but masking rotates that
gradient nearly orthogonal to the true direction (cos≈0), so no linear projection of
it manufactures the missing direction; and the clean anchor gradient is **never
applied** (only feeds the EMA basis). The clean step works because it *applies* the
true gradient; anchor+spectral only *used* it as projection geometry. This is the
load-bearing lesson for the frontier (`findings/NEXT_RESEARCH.md`).

## Settled decisions (do not relitigate)

- **Rescale (`h⊙mask/(1-p)`) is ON, permanent.** Job = unbias the masked activation
  (`E[h̃]=h`); without it the forward sits off-distribution and RMSNorm's `1/RMS`
  backward compounds over 7 boundaries → grad_norm ~2700 (vs ~0.38 dense). It trades
  bias for bounded variance (`p/(1-p)≈9×`); **necessary for correctness, not a
  learning fix** (plain masked+rescale still doesn't learn — `clean_cadence` is what
  made it learn). Do **not** attribute grad_norm differences to rescale (confounded).
  **Judge on val/score, not grad_norm** — Adam's `m̂/√v̂` is scale-invariant + verl
  grad-clips, so a large raw norm cannot by itself produce a large update.
- **Mask cross-pass consistency is solved.** Keyed on each token's stable
  `(sample_id, position_id)` (+ layer + global_step), no per-call/packing term → the
  old-logprob and train forwards see the **bit-identical** mask → IS ratio ≈ 1.
  Test-locked (`tests/workers/comm_eff/test_activation_mask.py`).
- **The every-K clipfrac drop is the clean step, not a bug** — both forwards unmasked
  → ratio ≡ 1 → `pg_clipfrac → ~4e-4`, grad_norm → dense ~0.4. A *spike* there would
  signal mask inconsistency; the *collapse* is the positive correctness signal.
- **`clean_cadence` scales to sparse K.** K=4/5/20 all reach GSM8K parity; larger K is
  only *slower* (steps-to-reward≥0.5 = 17/18/44) — the classic Local-SGD speed/comm
  tradeoff. `pg_clipfrac` stays ~0.035 (≪0.15), never saturates.

## Big-Math dataset (in inventory)

`gshasiri/Big-Math-RL-Verified-filtered`, prepped by `scripts/bigmath_dapo.py` → verl
parquet with `data_source=DigitalLearningGmbH/MATH-lighteval` (→ `math_reward`: last
`\boxed{}` + `is_equiv`; 20k train / 500 val). **Do not** use `data_source=math_dapo`
for `\boxed` prompts (its `is_correct_minerva` scrapes "Answer:" → biased reward); a
custom route returning `pred=None` crashes `process_validation_metrics`.

## Knob surface (in `vast_comm_eff_baseline_*.sh`)

| knob | default | meaning |
|---|---|---|
| `COMM_EFF_ENABLED` | true | master switch (false ⇒ byte-identical dense) |
| `COMM_EFF_MASK_ENABLED` | true | activation mask on pipeline-boundary blocks |
| `COMM_EFF_MASK_P` | 0.9 | masked fraction (p-sweep target on the frontier) |
| `COMM_EFF_MASK_RESCALE` | true | inverted-dropout `h*mask/(1-p)` — **settled ON** (unbias) |
| `COMM_EFF_MASK_RECOMPUTE` | true | mask the old-logprob forward too (keeps IS ratio ≈ 1) |
| `COMM_EFF_CLEAN_CADENCE` | 0 | true dense gradient every K steps — **the lever that makes masked GRPO learn** (K≤20 reaches GSM8K parity) |
| `COMM_EFF_ANCHOR_ENABLED` | false | K-stale anchor circuit — implemented form inert (frontier redesign target) |
| `COMM_EFF_SPECTRAL_ENABLED` | false | two-sided Tikhonov spectral correction — implemented form inert by orthogonality (frontier redesign target) |

## Implementation locus (on `vast-ai-workload`)

- `verl/workers/config/comm_eff.py` — config schema
- `verl/workers/comm_eff/{state.py, activation_mask.py, anchor.py, spectral_filter.py}` — runtime
- `verl/workers/engine_workers.py` — `compute_log_prob` mask stamp
- `verl/workers/engine/fsdp/transformer_impl.py` — boundary-block mask gating
- `tests/workers/comm_eff/` — CPU unit tests
