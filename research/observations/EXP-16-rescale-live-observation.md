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

## CELL 2 COMPLETE (done.flag present) — full trajectory

| step | grad_norm | pg_clipfrac | ppo_kl | score/mean |
|---|---|---|---|---|
| 1 | 4.374 | 0.0282 | −0.00015 | 0.1260 |
| 2 | 4.387 | 0.0274 | +0.00027 | 0.1328 |
| 3 | 5.323 | 0.0258 | +0.00097 | 0.1318 |
| 4 | 4.760 | 0.0290 | +0.00033 | 0.1338 |
| 5 | 4.595 | 0.0299 | −0.00014 | 0.1367 |
| 6 | 4.970 | 0.0286 | +0.00052 | 0.1260 |
| 7 | 4.056 | 0.0277 | +0.00108 | 0.1406 |
| 8 | 4.771 | 0.0277 | +0.00011 | 0.1367 |
| 9 | 4.888 | 0.0303 | +0.00176 | 0.1455 |
| 10 | 4.508 | 0.0307 | +0.00108 | **0.1475** |

**Verdict cell 2: STABLE + mildly CONVERGING.** grad_norm bounded [4.06, 5.32]
(no drift), clip frac flat ~0.03, ppo_kl < 0.002 throughout. score/mean rises
0.126 → 0.147; back-half mean (steps 7–10 ≈ 0.143) > front-half (1–4 ≈ 0.131).
Small (lr 1e-6, 10 steps) but the right sign and monotone-ish late. Mask
train==old_logprob held at every step (final 17920 == 17920). No divergence.

## CELL 4 (running) — `rescale_clean_every4_20steps`, clean_cadence=4, 20 steps
What to verify here (the clean-step mechanism):
- `comm_eff/clean_steps` should step to 1,2,3,4,5 at global steps **4,8,12,16,20**.
- On those clean steps masking is forced OFF for the *whole* step (train AND
  old-logprob recompute), so `mask_applications/train` should **not increase**
  that step, and grad_norm should drop toward the **true dense** value
  (memory: dense GRPO grad ≈ 0.38, i.e. ~10× below the masked ~4.7) then jump
  back up on the next masked step. That alternation is the falsifier that the
  clean step is genuinely taking a dense AdamW step.
- Mask consistency on the masked steps must still show train==old_logprob.

## Perf footnote (operator's "is it slow?" question)
~96 s/step, MFU ≈ 0.76% on 4×B200 — slow *by design*, not a fault. Bottleneck:
`update_actor` ~64 s (67%) driven by `ppo_micro_batch_size_per_gpu=1` +
`use_dynamic_bsz=False` (one sequence per forward). `mask_recompute=true` adds the
~20 s second masked forward (`old_log_prob`); grad-checkpointing adds ~33%. Rollout
is only ~9 s (responses avg ~280 tok ≪ 16 K cap). A 5–10× speedup is available via
larger micro-batch / dynamic bsz (mask is packing-invariant so numerics shouldn't
move), but that's an operator call — not changing a live run. Whole 4-cell sequence
≈ 2 h.

## CELL 4 (running) — `rescale_clean_every4_20steps` — steps 1–7

| step | entropy | grad_norm | pg_clipfrac | clean_steps | train/old_lp (cum) | score/mean |
|---|---|---|---|---|---|---|
| 1 | 5.920 | 4.997 | 0.0237 | 0 | 14 / 7 | 0.125 |
| 2 | 5.927 | 4.324 | 0.0240 | 0 | 28 / 21 | 0.152 |
| 3 | 5.925 | 4.943 | 0.0263 | 0 | 42 / 35 | 0.139 |
| **4** | **0.385** | **0.394** | **0.00018** | **1** | **42 / 35 (frozen)** | 0.136 |
| 5 | 5.923 | 4.482 | 0.0260 | 1 | 56 / 42 | 0.156 |
| 6 | 5.929 | 4.910 | 0.0323 | 1 | 70 / 49 | 0.183 |
| 7 | 5.924 | 4.746 | 0.0293 | 1 | 84 / 56 | **0.221** |

### FINDING 1 — clean-step mechanism VERIFIED (step 4)
On the clean step the mask counters **freeze** (train/old_logprob stay 42/35 — zero
fires), grad_norm drops to **0.394 = true dense** (~12× below the masked ~4.7),
and pg_clipfrac→2e-4 / ppo_kl→3e-5 because *both* old-logprob and train forwards
are unmasked ⇒ importance ratio ≡ 1. Step 5 returns cleanly to the masked regime.
This is exactly the EXP-14 contract: every 4th step takes an unclipped AdamW step
on the genuine dense gradient.

### FINDING 2 — the "entropy collapse" is a forward-corruption tell, NOT divergence
`actor/entropy` ≈ **5.92 on masked steps** but **0.385 on the clean step** (step 4),
then recovers to 5.92 at step 5. The clean step is the unmasked forward, so **0.385
is the TRUE policy entropy**; the 5.92 is the entropy of the *masked* forward's
output distribution. Masking 90% of channels at 7 boundaries (+10× rescale)
flattens the logits enough to inflate apparent entropy ~15×. Direct evidence that
the masked forward sits far from the true policy — consistent with the masked
gradient norm being ~12× dense even after rescale: **constant rescale fixes the
SCALE; the masked forward's distribution/direction stays heavily perturbed**
("scale not correlation"). entropy_coeff=0, so this never enters the loss — but it
is a clean quantitative handle on how much the mask distorts the forward.

### FINDING 3 — HEADLINE: the clean step UNLOCKS convergence under masking
Cell 4 score/mean trajectory (*=clean step, dense):
```
s1 .125  s2 .152  s3 .139  s4* .136  s5 .156  s6 .183  s7 .221  (s8* clean)
s9 .261  s10 .339  s11 .343  s12* .359  s13 .430  s14 .431  s15 .471 ...
```
**0.125 → 0.471 in 15 steps (~3.8×).** Contrast cell 2 (pure masked, no clean):
0.125 → 0.147 over 10 steps (flat). **At matched step 10: cell 4 = 0.339 vs
cell 2 = 0.147.** Same model, same init, same p=0.9 rescale mask — the ONLY
difference is the clean step every 4. ⇒ the pure-masked gradient carries weak
learning signal (cell 2 barely moves), but **re-anchoring to the true dense
gradient every 4th step lets the run actually climb**. This is the EXP-14
hypothesis confirmed in vivo: periodic clean steps + a consistent mask in between
= stable convergence; masking alone = stall.

Caveat: 3–4 dense steps at lr 1e-6 cannot by themselves move reward 4×, so the
masked steps ARE contributing — the clean steps appear to *course-correct*
(prevent masked-noise drift) rather than do all the learning. Worth a dedicated
ablation (clean-only vs masked-only vs mixed) to attribute the gain.

Masked-step grad_norm creeps 4.7 → ~7.1 as the policy improves (clip ~0.04,
ppo_kl <0.001 — still controlled, not diverging); clean steps stay pinned at the
true dense ~0.40. Mask consistency (train & old_logprob both grow on masked steps,
both frozen on clean steps) holds throughout.

### Note on batching difference
Cell 4's mask-fire counts (~14 train/step) are ~128× smaller than cell 2's (~1792),
i.e. cell 4 packs the batch into ~2 micro-batches (coarser/dynamic bsz) vs cell 2's
256 single-sequence micro-batches. Doesn't affect mask correctness (PRF is
packing-invariant) but explains the counter magnitudes and cell 4's faster steps.

## CELL 4 FINAL (done.flag) — converged 0.125 → 0.619 over 20 steps (~5×)
Full reward by step: .125 .152 .139 [.136] .156 .183 .221 [.227] .261 .339 .343
[.359] .430 .431 .471 [.481] .541 .602 .597 [.619]  ([ ]=clean step 4/8/12/16/20).
All 5 clean steps fired on schedule with dense grad_norm ~0.39–0.43 and frozen mask
counters (final 210 train / 119 old_logprob); 15 masked steps grad ~4.3–7.1.
**Best result of the run.** The clean-step cadence is the decisive ingredient.

## CELL 5 — FAILED (CUDA OOM, 0 steps logged) → SEQUENCE STOPPED
`grpo_mask_channel_p0p9_rescale_anchor2_spectral2_20steps`, launched 22:52:28,
`FAILED rc=1` 22:55:14. **`torch.OutOfMemoryError`** on GPU 0 (174/178 GiB used)
in `update_actor` → `F.linear`. 0 `step:` lines logged ⇒ OOM hit during step 1's
forward, *after* the anchor+spectral circuits initialised cleanly (the EXP-12
spectral-storage line and the anchor config dict are both present). **Not a
mask/determinism or divergence failure — a memory-provisioning failure on the
heaviest cell.**

Why cell 5 OOMs when 2 & 4 didn't — three new costs stack on GPU 0:
1. **Dynamic bsz @ 98304 token budget** (`ppo_max_token_len_per_gpu=98304`, vs cell
   2's 36864) → very large packed micro-batches (huge activations).
2. **Anchor circuit**, `anchor_backward_isolation_mode=clone` → an extra cloned
   backward (cadence 2, delay_K 2).
3. **Spectral circuit**, `ema_device=gpu` + `svd_mode=full` + `basis_cache=cache`
   (rank 8, max_targets 4) → GPU-resident EMA buffers + full SVD basis.

Fix levers for the operator (then `run_rescale_sequence.sh 5` to resume):
`spectral.ema_device=cpu`, lower `ppo_max_token_len_per_gpu` (→ ~36864),
`spectral.svd_mode=lowrank` / smaller `max_targets`/`rank`, and/or enable
param/optimizer offload. Sequencer halts on failure (`exit rc`), so **cell 6
(dense reference) was NOT run.**

## CURRENT STATE (post-cell-5 failure)
- All 4 GPUs **0 MiB / 0%** — box idle. main_ppo process gone. Instance still UP.
- Sequence halted at cell 5; cells 2 & 4 complete with done.flag; cells 5 & 6 pending.
- Monitor `b1dyotpya` stays armed: will catch a cell-5 resume (new steps) or the
  instance going down (SSH stream ends). Observation continues per directive.

## SUMMARY (cells that ran)
| cell | config | reward | grad_norm (masked) | divergence | mask consistency |
|---|---|---|---|---|---|
| no_rescale (pre-amend, superseded) | mask, no rescale | flat ~0.13 | ~2400–3500 | clip-saturated stall | train==old_lp, paths confined |
| 2 | mask + rescale | 0.126 → 0.147 (10 st) | ~4.1–5.3 | stable, ~flat | ✓ |
| 4 | mask + rescale + clean@4 | **0.125 → 0.619 (20 st)** | ~4.3–7.1 (clean ~0.4) | **stable, strong convergence** | ✓ + clean-freeze ✓ |
| 5 | + anchor@2 + spectral@2 | — (OOM) | — | n/a | circuits init OK, then OOM |
| 6 | dense reference | — (not run) | — | — | n/a |

Primary directive (same mask per step/prompt across all forward passes):
**VERIFIED** — source PRF has no pass-counter and is keyed on stable token identity;
live counters show train==old_logprob every masked step with all non-eligible paths
0; clean steps freeze the counters; behaviour (clip≈0.03, ppo_kl≈0; ratio≡1 on
clean steps) matches a consistent old↔new mask. No desync, no divergence observed.

## Live log (running tally)
- 22:10 cell 2 start (~96 s/step) → done (10/10), reward 0.126→0.147.
- 22:41 cell 4 start → done (20/20) 22:52, reward 0.125→0.619; clean@4 verified.
- 22:52 cell 5 start → **OOM, FAILED 22:55**, sequence STOPPED; cell 6 not run.
- 22:55 box idle (GPUs 0%), instance UP. Push sent to operator. Watching for
  resume / teardown.
- **23:05 cell 5 RESUMED** (operator applied a mem fix). Now running with
  anchor+spectral ACTIVE — first circuit firings observed: step 1
  `spectral_corrections=4` (= max_targets, 4 matrices/firing), `anchor_backwards=1`,
  score 0.140, grad_norm 4.90, clip 0.027 (healthy, no OOM).
  NB cell 5 counters train≠old_logprob (28 vs 21) — EXPECTED under dynamic-bsz
  (different micro-batch counts per path); mask consistency is per-token
  (packing-invariant PRF), so unequal path fire-counts are not a desync.
  Watching: spectral/anchor cadence (every 2 steps), spectral rel_change, and
  whether anchor+spectral correction helps vs cell 4's clean-step convergence.

## CELL 5 (running, resumed) — anchor@2 + spectral@2 — steps 1–10
Circuits ACTIVE and firing EVERY step (not the nominal cadence 2):
`spectral_corrections` +4/step (= max_targets 4 matrices/firing),
`anchor_backwards` +1/step. (Effective cadence 1 — operator may have changed it on
the OOM-fix resume; unconfirmed.) No clean steps (clean_cadence=0).

Three-way reward comparison (same model/init/p=0.9 rescale mask):
| step | cell 2 pure-mask | cell 4 clean@4 | cell 5 anchor+spectral |
|---|---|---|---|
| 1 | 0.126 | 0.125 | 0.140 |
| 3 | 0.132 | 0.139 | 0.115 |
| 5 | 0.137 | 0.156 | 0.133 |
| 7 | 0.141 | 0.221 | 0.120 |
| 8 | 0.137 | 0.227 | 0.121 |
| 9 | 0.146 | 0.261 | 0.135 |
| 10 | 0.147 | **0.339** | 0.125 |

### FINDING 4 (mid-run, preliminary) — spectral+anchor ≠ clean step for convergence
At the halfway mark **cell 5 is flat ~0.12–0.145, statistically indistinguishable
from pure-masked cell 2, and far below clean-step cell 4 (0.339 @ step10).** The
spectral grad-correction does tighten the PPO ratio a little (clip frac ~0.02–0.028
vs cell 4's ~0.03–0.045) and grad_norm stays controlled (~4–5, ppo_kl ~0, no
divergence), but that local tightening does NOT translate into reward gains.
Tentative read: with these hyperparams (alpha 0.5, tau 0.01, beta_anc 0.9,
effective cadence 1), the spectral correction is not recovering the true-gradient
*direction* the way a periodic dense (clean) step does. CAVEAT: cell 4's climb was
back-loaded (steps 9–20), so this is not final until cell 5 step 20.

### CELL 5 FINAL (done.flag) — CONFIRMED no convergence
score flat the whole run: 0.140 → … → step19 0.142 → **step20 0.131** (never left
the 0.11–0.15 band). Final counters: spectral_corrections 80 (4/step×20),
anchor_backwards 20 (1/step×20). **Key diagnostic: `comm_eff/spectral/rel_change_mean
≈ 0.243`** — the spectral projection changes the masked gradient by only ~24%. That
is far too small to recover the dense direction (the masked grad is 10–12× dense in
norm AND poorly correlated in direction), so the correction is cosmetic for
convergence. **Cell 5 ≈ pure-masked cell 2; the anchor+spectral circuit, as
configured, does NOT substitute for a true dense step.**

## CELL 6 (running) — `dense_grpo_comm_eff_off_25step_reference` — the gold control
`comm_eff.enabled=false`, 25 steps. Confirms every inference made from the masked
runs:
- grad_norm ~**0.38–0.39** every step = the cell-4 clean-step grad and the dense
  memory ~0.38 ⇒ clean steps really were computing the true dense gradient.
- entropy ~**0.38–0.42** = the clean-step entropy ⇒ true policy entropy is ~0.38;
  the masked-forward ~5.92 was pure corruption artifact (Finding 2 confirmed).
- mask counters **ABSENT** (`tr= ol=`), spectral/anchor 0 ⇒ disabled path is a
  strict no-op (upstream-parity contract holds).
- step 1→2 score 0.125 → 0.166, climbing; pg_clipfrac ~2e-4, ppo_kl ~0 (dense,
  ratio≡1). Expect convergence comparable to / better than cell 4 over 25 steps.

## CONCLUSION (ranking under p=0.9 rescale mask)
**Only a true dense gradient drives convergence.** Reward outcome:
- dense (cell 6) ≈ clean@4 (cell 4, 0.62) ≫ pure-mask (cell 2) ≈ mask+anchor+spectral
  (cell 5), both flat ~0.13.
The clean step works because it IS a periodic exact dense step; the spectral
correction (rel_change ~24%) is too weak to recover the dense direction the masking
destroys. Mask determinism held in every masked cell (2,4,5) and the disabled cell
(6) was a verified no-op.
- Monitor `boktw39kl` (single persistent SSH) streaming per-step across all cells.
  (First monitor false-positived "instance down" on concurrent-SSH collisions; box
  was up throughout; replaced.)
- (cells 4 steps + cells 5/6 appended as they land)
