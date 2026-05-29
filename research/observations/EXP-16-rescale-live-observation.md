# EXP-16 live observation — `grpo_mask_channel_p0p9_rescale_10steps`

Observer session. Box: `ssh -p 33732 root@3.144.230.17` (4×B200, manual lifecycle,
not in runs.jsonl). Run started 2026-05-29 22:10 UTC, Ray session
`2026-05-29_22-10-11`, TaskRunner pid 94866. Watching until the instance is down.

Directive: (1) verify that **per step, per prompt, regardless of how many forward
passes, the mask is identical**; (2) track divergence/convergence via clip
fractions, scores, grad-norm, ratios.

---

## Run config (from the live process cmdline)

- Model Qwen2.5-1.5B-Instruct, GSM8K, GRPO, **no-KL no-entropy** (`use_kl_loss=False`,
  `kl_loss_coef` overridden off, `entropy_coeff=0`). lr 1e-6, train_batch 128, n=8,
  mini_batch 64, micro_batch_size_per_gpu=1, dynamic_bsz off (max_token_len 36864),
  max_response 16384, 10 steps, val_before_train off, 4 GPUs.
- comm_eff: `enabled=true`, `mask.enabled=true`, **`p=0.9`**, **`rescale=true`**
  (→ rescale_mode `constant`, gain = 1/(1−p) = **10×**), **`mask_recompute=true`**,
  `seed=0`, `pp_size=8`, `clean_cadence=0`. anchor OFF, spectral OFF.
- This is the rescale twin of the already-finished `..._no_rescale_10steps`.

---

## 1. Mask determinism — VERIFIED (source + empirical)

### Source-level proof (the strongest result)
`prf_token_mask` (verl/workers/comm_eff/activation_mask.py) computes each mask
entry as a **counter-based splitmix64 PRF** keyed on the tuple
`(base_seed, layer_idx, global_step, sample_id, position_id, channel)`:

- **No forward-pass counter, no mutable RNG/Generator state.** ⇒ for a fixed
  `global_step` and a fixed token identity, the mask is *bit-identical no matter
  how many forwards run* — old-logprob recompute, the train forward, every PPO
  mini-batch and epoch, and gradient-checkpoint recomputation (the backward
  re-forward). This is determinism *by construction*, not by luck.
- Keyed on **stable token identity** `(sample_id, position_id)`, not packed
  position ⇒ packing-invariant across dynamic-bsz repacking / mini-batch shuffle.
- Integer-only splitmix64 ⇒ identical on CPU/GPU and across ranks.

### The single practical fragility point (verified safe here)
`comm_eff_sample_id` is **re-stamped as `arange(bsz)` independently** in
`compute_log_prob` and `update_actor` (engine_workers.py
`_comm_eff_stamp_sample_ids`); the column is *not* persisted between the two
worker calls. Consistency therefore relies on both calls receiving the per-rank
batch in **identical row order**. In standard verl GRPO this holds (balance_batch,
if any, runs once before both; advantage computation adds columns without
reordering). `global_step` is threaded identically via the private
`comm_eff_global_step` meta key. ⇒ The id→sample mapping is reproduced
identically, the mask key follows the physical sample through PPO shuffling.
**If anything ever reorders the batch between compute_log_prob and update_actor,
the masks would silently desync** — worth a guard/assert, but not a problem in
this run's config.

### Empirical confirmation (from the completed no_rescale twin, same mask path)
Per-path mask-application counters at every step:
- `mask_applications/train` == `mask_applications/old_logprob` (exactly equal,
  both nonzero; +1792/step each = 7 boundaries × 256 micro-forwards).
- `mask_applications/{rollout,ref_logprob,val,infer,ckpt}` == 0 (strict
  confinement; EXP-6 falsifier passes).
- `comm_eff/mask_ratio` ≈ 0.900 on all 7 boundaries, every step (stable, calibrated).

Equal train/old_logprob counts + identical PRF key ⇒ the old-policy log-prob and
the new-policy train forward see the **same mask** for the same token, so the PPO
importance ratio is computed on a consistent footing (the whole point of
`mask_recompute=true`). The behavioral tell to watch on the rescale run: ratio≈1
/ pg_clipfrac small at the very first inner micro-step (params unchanged + same
mask ⇒ ρ=1). A desync would show as nonzero clip from the first step.

---

## 2. no_rescale baseline (completed, pid 25828) — for comparison

| step | grad_norm | pg_clipfrac | pg_clipfrac_lower | ppo_kl | score/mean | entropy |
|---|---|---|---|---|---|---|
| 1 | 2698.1 | 0.163 | 0.103 | −0.0152 | 0.147 | 6.350 |
| 2 | 2360.2 | 0.156 | 0.106 | +0.0165 | 0.139 | 6.366 |
| 3 | 2391.0 | 0.148 | 0.096 | +0.0101 | 0.131 | 6.345 |
| 4 | 3536.6 | 0.150 | 0.096 | +0.0056 | 0.133 | 6.345 |
| 5 | 2515.6 | 0.156 | 0.098 | +0.0036 | 0.153 | 6.352 |
| 6 | 2721.2 | 0.156 | 0.104 | +0.0007 | 0.139 | 6.350 |

Signature: **grad_norm in the thousands** (grad-clip at 1.0 is saturated every
step ⇒ update is the corrupted masked direction renormalised to 1), reward flat
~0.13, clip fraction high but stable, ppo_kl small. Not blowing up, but not
learning — consistent with "rescale fixes scale not correlation": the masked
gradient *direction* is wrong w.r.t. the dense gradient.

### Expectation for the rescale run (to be checked as steps land)
constant rescale (gain 10×) makes E[h_tilde]=h. Expect grad_norm to drop to a
much saner range than ~2700 (RMS-overshoot of constant rescale also *damps*
grad). Reward improvement is the open question — if direction-correlation is the
real problem, reward stays flat even with sane grad_norm.

---

## SCOPE UPDATE — this is a 4-cell rescale-only sequence

Driver `run_rescale_sequence.sh` runs cells **strictly sequentially**, skipping any
with a `done.flag` (warm resume). Operator directive baked in: *"do not do anything
without grad rescaling"* — the old no-rescale cells (1,3) are permanently removed.

| cell | name | config | steps |
|---|---|---|---|
| **2** (running) | grpo_mask_channel_p0p9_rescale_10steps | mask p0.9 + rescale | 10 |
| 4 | ..._rescale_clean_every4_20steps | + clean-step @ cadence 4 | 20 |
| 5 | ..._rescale_anchor2_spectral2_20steps | + anchor@2 + spectral@2 | 20 |
| 6 | dense_grpo_comm_eff_off_25step_reference | DENSE control (mask OFF) | 25 |

"Until the instance is down" therefore spans all four. Mask-consistency directive
applies to cells 2/4/5; cell 6 is the dense reference.

---

## Live results — CELL 2 (rescale), steps 1–5  [22:10–22:18 UTC, ~96 s/step]

| step | grad_norm | pg_clipfrac | clipfrac_lower | ppo_kl | score/mean | entropy | mask_ratio | train==old_lp |
|---|---|---|---|---|---|---|---|---|
| 1 | **4.374** | 0.0282 | 0.00079 | −1.5e-4 | 0.1260 | 5.925 | 0.9003 | 1792 = 1792 ✓ |
| 2 | **4.387** | 0.0274 | 0.00065 | +2.7e-4 | 0.1328 | 5.923 | 0.9002 | 3584 = 3584 ✓ |
| 3 | **5.323** | 0.0258 | 0.00063 | +9.7e-4 | 0.1318 | 5.926 | 0.9003 | 5376 = 5376 ✓ |
| 4 | **4.760** | 0.0290 | 0.00085 | +3.3e-4 | 0.1338 | 5.919 | 0.8998 | 7168 = 7168 ✓ |
| 5 | **4.595** | 0.0299 | 0.00081 | −1.4e-4 | 0.1367 | 5.920 | 0.9003 | 8960 = 8960 ✓ |

All other mask paths (rollout/ref_logprob/val/infer/ckpt) = 0 every step;
anchor_*/spectral_* = 0 (correct, OFF in cell 2).

### Insight A — rescale tames the gradient (CONVERGING, not diverging)
Constant rescale (gain 10×) drops grad_norm from no_rescale's **~2400–3500** to
**~4.4–5.3** — a ~500–600× reduction, into a sane band just above the clip
threshold. The PPO health metrics follow:
- **pg_clipfrac ~0.03** (rescale) vs **~0.16** (no_rescale): ~5× fewer clipped
  tokens ⇒ importance ratios sit much closer to 1.
- **ppo_kl ~±3e-4** (rescale) vs ±0.01–0.016 (no_rescale): per-update policy move
  is tiny and controlled.
- **entropy 5.92, flat** across all 5 steps — no entropy collapse, no blow-up.
- **score/mean 0.126→0.133→0.132→0.134→0.137** — a gentle *upward* drift (vs
  no_rescale's flat ~0.13 noise). Small (lr 1e-6, 10 steps) but the right sign.

Verdict so far: the rescale cell is **stable / mildly converging**, not diverging.
This is the cleanest behaviour seen for the masked path. Whether the upward score
drift is real learning or batch noise needs more steps / the longer cells (4,5).

### Insight B — mask consistency confirmed LIVE (behavioural fingerprint)
The directive's core claim is now backed by live evidence on the rescale run:
1. **Counters:** `mask_applications/train` == `mask_applications/old_logprob`
   *exactly*, every step (+1792/step each). Same fire-count on both
   gradient-feeding forwards is the mask_recompute=true signature; the source PRF
   guarantees they are the *same* masks (identical key tuple).
2. **mask_ratio = 0.900 ± 0.0005** every step, all 7 boundaries — the per-element
   Bernoulli draw is stable and calibrated to p.
3. **Behavioural tell:** pg_clipfrac ≈ 0.03 and ppo_kl ≈ 0 are exactly what a
   *consistent* old↔new mask produces (ratio≈1 because the same tokens are zeroed
   in both log-prob computations). A per-pass mask *desync* would inflate the
   ratio spread and clip fraction from step 1 — not observed.

(Note: the lone earlier "no_rescale step:1 grad_norm 4.37" line was a stale
pre-amendment attempt sharing a recycled Ray pid; the completed no_rescale run
that defines the baseline is pid 25828 with grad_norm ~2400–3500.)

## Live log (running tally)
- 22:10 start; FSDP loaded; cell-2 stepping at ~96 s/step. At 22:18 → step 5/10.
- Robust single-SSH monitor `boktw39kl` armed (per-step stream across all cells).
  First monitor (`bs5my088x`) false-positived "instance down" from concurrent-SSH
  handshake collisions — box was up throughout; replaced.
- (cells 4/5/6 + steps 6–10 appended as they land)
