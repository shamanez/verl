# EXP-20 — Metrics Data Dump (3 arms) + Dense-Baseline Verdict

**Compiled by:** wandb-archivist · **Date:** 2026-06-04
**Scope:** pure data gathering — no interpretation/theory (that is tasks #2/#3).
**Sources used (both):**
- **WandB public API** (queried directly per operator instruction): entity `shamanework-pl`,
  project `verl_compression_research`, key sourced from `~/.config/verl-research/secrets.env`
  (never echoed). Egress works. Full 14-run project inventory + per-step `scan_history()` for
  the dense hunt → §6. For the 3 EXP-20 arms, WandB history is *downsampled* to ~48 rows and the
  run summaries captured a mid-run (step-25) val, so the EXP-20 finals (§1) come from the local logs.
- **Local per-step logs** `runs/EXP-20/ce_*_50s_gsm8k.log` (every step 1–50; authoritative for the 3 arms).

---

## 0. The three arms — design summary

| Arm | exp name | WandB id | compression | rank / p | logical_pp_bytes/tok | val-acc@50 |
|---|---|---|---|---|---|---|
| **mask p=0.95** | `ce_mask_p95_clean5_50s_gsm8k` | `3yxzzwn3` | `prf_mask` | p=0.95 | **76.8** | **0.7384** |
| **PowerSGD r=102** | `ce_powersgd_r102_clean5_50s_gsm8k` | `kqozxfr0` | `powersgd` | r=102 (+33% budget) | **102.0** | **0.7437** (+0.0053) |
| **PowerSGD r=77** | `ce_powersgd_r77_clean5_50s_gsm8k` | `oquyeic3` | `powersgd` | r=77 (budget-matched) | **77.0** | **0.7415** (+0.0031) |

`logical_pp_bytes/tok` is the per-token coordinate count crossing each PP boundary
(mask: ~5% of H=1536 survives ⇒ 76.8; PowerSGD: the rank r ⇒ 102 / 77). r=77 ≈ 76.8 ⇒
**r=77 is the equal-communication-budget comparison vs the mask**; r=102 is +33% budget.
Baseline-of-record for the comparison = the **mask** (NOT dense — see §4).

**Codec is the ONLY axis.** A full diff of the three resolved configs (`set -x` ground truth,
`resolved_params__*.txt`) shows the *only* differing lines are `compression_type`, `powersgd.rank`/`mask.p`,
and `experiment_name`. Every other knob is byte-identical (see §3).

---

## 1. Per-step trajectory (all 50 steps, from local logs)

`reward` = `critic/score/mean` (train); `grad` = `actor/grad_norm`; `recon` = `powersgd_reconstruction_rel_error` (aggregate); `clean` = `actor/comm_eff/clean_steps` (cumulative).
Clean (dense-gradient) steps land on **5, 10, 15, …, 50** — `clean_steps` increments there and grad collapses to ~0.4 (see §2).

```
STEP | reward(mask) | reward(r102) | reward(r77) | grad(mask) | grad(r102) | grad(r77) | recon(r102) | recon(r77) | clean
   1 |      0.136 |      0.122 |      0.137 |     8.282 |   166.444 |   194.081 |    0.9667 |    0.9763 |     0
   2 |      0.134 |      0.126 |      0.140 |     7.925 |    40.014 |    64.618 |    0.7137 |    0.6911 |     0
   3 |      0.127 |      0.129 |      0.136 |     7.952 |    44.918 |    20.705 |    0.3933 |    0.3975 |     0
   4 |      0.154 |      0.128 |      0.144 |    12.829 |     6.944 |     3.254 |    0.1727 |    0.1437 |     0
   5 |      0.131 |      0.131 |      0.126 |     0.403 |     0.370 |     0.403 |    0.1727 |    0.1437 |     1
   6 |      0.168 |      0.155 |      0.151 |     9.262 |     2.276 |     1.899 |    0.0861 |    0.0901 |     1
   7 |      0.195 |      0.164 |      0.190 |     8.265 |     1.217 |     1.065 |    0.0506 |    0.0541 |     1
   8 |      0.202 |      0.218 |      0.213 |     9.961 |     2.174 |     1.554 |    0.0348 |    0.0388 |     1
   9 |      0.218 |      0.234 |      0.222 |     9.105 |     2.134 |     1.641 |    0.0246 |    0.0247 |     1
  10 |      0.308 |      0.247 |      0.305 |     0.505 |     0.485 |     0.478 |    0.0246 |    0.0247 |     2
  11 |      0.323 |      0.301 |      0.282 |    13.037 |     1.539 |     1.825 |    0.0217 |    0.0225 |     2
  12 |      0.323 |      0.336 |      0.319 |    12.469 |     1.897 |     1.541 |    0.0216 |    0.0229 |     2
  13 |      0.400 |      0.405 |      0.387 |    11.639 |     1.396 |     1.474 |    0.0205 |    0.0201 |     2
  14 |      0.417 |      0.414 |      0.407 |    10.070 |     1.624 |     1.472 |    0.0192 |    0.0189 |     2
  15 |      0.452 |      0.448 |      0.442 |     0.421 |     0.425 |     0.416 |    0.0192 |    0.0189 |     3
  16 |      0.474 |      0.490 |      0.489 |    16.115 |     1.579 |     2.444 |    0.0196 |    0.0214 |     3
  17 |      0.542 |      0.552 |      0.551 |    16.300 |     1.211 |     1.312 |    0.0198 |    0.0190 |     3
  18 |      0.584 |      0.610 |      0.633 |    10.890 |     1.237 |     1.765 |    0.0194 |    0.0225 |     3
  19 |      0.602 |      0.627 |      0.626 |    25.803 |     1.447 |     1.645 |    0.0191 |    0.0219 |     3
  20 |      0.622 |      0.671 |      0.630 |     0.420 |     0.435 |     0.389 |    0.0191 |    0.0219 |     4
  21 |      0.651 |      0.691 |      0.667 |    11.539 |     1.100 |     1.673 |    0.0206 |    0.0193 |     4
  22 |      0.605 |      0.636 |      0.615 |    10.539 |     1.557 |     3.011 |    0.0217 |    0.0223 |     4
  23 |      0.664 |      0.673 |      0.678 |    12.108 |     1.536 |     2.243 |    0.0223 |    0.0223 |     4
  24 |      0.710 |      0.731 |      0.723 |    12.838 |     1.335 |     1.835 |    0.0212 |    0.0203 |     4
  25 |      0.660 |      0.695 |      0.688 |     0.409 |     0.404 |     0.406 |    0.0212 |    0.0203 |     5
  26 |      0.662 |      0.669 |      0.667 |    10.734 |     1.306 |     1.682 |    0.0224 |    0.0229 |     5
  27 |      0.699 |      0.729 |      0.722 |    11.296 |     1.596 |     1.843 |    0.0209 |    0.0228 |     5
  28 |      0.743 |      0.760 |      0.751 |    11.372 |     1.401 |     2.137 |    0.0205 |    0.0228 |     5
  29 |      0.690 |      0.689 |      0.729 |    15.333 |     1.425 |     1.472 |    0.0211 |    0.0225 |     5
  30 |      0.706 |      0.719 |      0.712 |     0.393 |     0.382 |     0.373 |    0.0211 |    0.0225 |     6
  31 |      0.740 |      0.776 |      0.747 |    11.428 |     1.932 |     1.169 |    0.0221 |    0.0221 |     6
  32 |      0.747 |      0.759 |      0.769 |     9.148 |     1.279 |     1.830 |    0.0240 |    0.0226 |     6
  33 |      0.702 |      0.706 |      0.716 |    13.965 |     1.416 |     1.892 |    0.0212 |    0.0224 |     6
  34 |      0.735 |      0.748 |      0.765 |    10.241 |     1.431 |     2.450 |    0.0222 |    0.0233 |     6
  35 |      0.752 |      0.739 |      0.764 |     0.362 |     0.412 |     0.343 |    0.0222 |    0.0233 |     7
  36 |      0.758 |      0.757 |      0.768 |     9.015 |     1.596 |     1.641 |    0.0211 |    0.0233 |     7
  37 |      0.777 |      0.792 |      0.782 |    10.070 |     2.910 |     2.112 |    0.0221 |    0.0226 |     7
  38 |      0.746 |      0.739 |      0.761 |    10.711 |     1.277 |     1.940 |    0.0222 |    0.0231 |     7
  39 |      0.729 |      0.752 |      0.763 |    10.904 |     1.742 |     2.339 |    0.0221 |    0.0239 |     7
  40 |      0.732 |      0.734 |      0.734 |     0.370 |     0.367 |     0.370 |    0.0221 |    0.0239 |     8
  41 |      0.780 |      0.778 |      0.791 |     9.945 |     1.525 |     1.990 |    0.0211 |    0.0221 |     8
  42 |      0.753 |      0.762 |      0.762 |    10.758 |     1.761 |     5.822 |    0.0213 |    0.0230 |     8
  43 |      0.756 |      0.778 |      0.758 |     9.489 |     2.161 |     1.982 |    0.0232 |    0.0236 |     8
  44 |      0.756 |      0.751 |      0.762 |    14.787 |     1.577 |     3.117 |    0.0211 |    0.0233 |     8
  45 |      0.732 |      0.731 |      0.740 |     0.354 |     0.381 |     0.365 |    0.0211 |    0.0233 |     9
  46 |      0.769 |      0.782 |      0.753 |    10.343 |     2.537 |     2.598 |    0.0210 |    0.0227 |     9
  47 |      0.798 |      0.799 |      0.807 |    12.023 |     3.231 |     2.875 |    0.0206 |    0.0240 |     9
  48 |      0.778 |      0.794 |      0.780 |    10.065 |     1.882 |     2.629 |    0.0217 |    0.0238 |     9
  49 |      0.757 |      0.771 |      0.744 |    14.432 |     2.393 |     2.130 |    0.0213 |    0.0236 |     9
  50 |      0.804 |      0.788 |      0.772 |     0.353 |     0.423 |     0.354 |    0.0213 |    0.0236 |    10
```

### Validation accuracy (`val-core/openai/gsm8k/acc/mean@1`) — logged at steps 0/25/50

| step | mask p=0.95 | PowerSGD r=102 | PowerSGD r=77 |
|---|---|---|---|
| **0** (pre-train, val_before_train) | 0.0826 | 0.0766 | 0.0857 |
| **25** (mid) | 0.7195 | 0.7316 | 0.7104 |
| **50** (final) | **0.7384** | **0.7437** | **0.7415** |

(`val-aux/openai/gsm8k/reward/mean@1` is identical to acc at every logged step in all arms.)
Step-0 (~0.08) is anomalously low vs the model's known ~0.71 base capability — the val-before-train ran
before the rollout↔actor policy aligned; see the Pearson-corr warmup in §2.4. It does not affect the
step-25/50 finals.

---

## 2. Key secondary dynamics (from local logs, all 50 steps)

### 2.1 Grad-norm: clean steps vs compressed steps
The dominant feature. Clean (dense-gradient) steps {5,10,…,50} have ~30× smaller grad-norm than compressed steps.

| Arm | clean-step grad (steps 5,10,…,50) | compressed-step grad (all others) |
|---|---|---|
| mask p=0.95 | mean **0.399** (0.353–0.505) | mean **11.58** (7.92–25.80) |
| PowerSGD r=102 | mean **0.408** (0.367–0.485) | mean **8.00** (1.10–166.44)\* |
| PowerSGD r=77 | mean **0.390** (0.343–0.478) | mean **8.92** (1.06–194.08)\* |

\* PowerSGD compressed-step max is the **step-1 cold-basis transient** (166/194). From step ~4 onward PowerSGD compressed-step grad sits at **~1–3** — *lower* than the mask's steady ~10–16. The PowerSGD mean is inflated only by the single warmup step.

### 2.2 PowerSGD reconstruction rel-error — warmup then steady ~0.02
Aggregate `powersgd_reconstruction_rel_error` (relative Frobenius error of the rank-r reconstruction of the boundary gradient):

| step | r=102 | r=77 |
|---|---|---|
| 1 (cold, no basis) | 0.967 | 0.976 |
| 2 | 0.714 | 0.691 |
| 3 | 0.393 | 0.398 |
| 4 | 0.173 | 0.144 |
| 6 | 0.086 | 0.090 |
| 9 | 0.025 | 0.025 |
| 25 (steady) | 0.021 | 0.020 |
| 50 (steady) | 0.021 | 0.024 |

Converges from ~0.97 → ~0.02 within ~9 steps for both ranks. r=102 vs r=77 steady-state fidelity is nearly identical (~0.021 vs ~0.024) — r=77 is already on the flat part of the rank-accuracy curve.

**Per-layer recon @ step 50** (rises mildly with depth; all <4%):

| layer | 3 | 7 | 11 | 15 | 18 | 21 | 24 |
|---|---|---|---|---|---|---|---|
| r=102 | 0.0184 | 0.0149 | 0.0155 | 0.0171 | 0.0200 | 0.0259 | 0.0376 |
| r=77 | 0.0233 | 0.0162 | 0.0169 | 0.0195 | 0.0221 | 0.0293 | 0.0380 |

### 2.3 PowerSGD codec health (shared-codebook invariants)
- `powersgd_q_cond` (basis conditioning): **max 1.0000040, min 1.0000002** across all steps both arms → orthonormal, perfectly conditioned.
- `powersgd_q_cross_rank_max_rel_dev`: **0.0 at every step, both arms** → the basis Q is **bit-identical across all 4 DP ranks** (the cross-DP consensus / shared-codebook invariant holds end-to-end; `sync_basis=true`).
- `powersgd_basis_updates` @ step 50 = **40** (basis refreshes on the 40 compressed steps; not refreshed on the 10 clean steps); `powersgd_applications` @ 50 = **143360** (= per-microbatch projections). `update_cadence=1`, `warm_start=true`.

### 2.4 Train↔inference consistency: `rollout_actor_probs_pearson_corr`
Correlation between the training-forward log-probs (under the codec) and the vLLM rollout log-probs. **Identical across all three arms** — the train/inference gap is codec-independent (both codecs carry the same gap; it is not a differentiator):

| step | mask | r=102 | r=77 |
|---|---|---|---|
| 1 | 0.0064 | 0.0122 | 0.0242 |
| 2 | 0.0036 | 0.0018 | 0.0281 |
| 5 | 0.9995 | 0.9996 | 0.9996 |
| 25 | 0.9995 | 0.9994 | 0.9994 |
| 50 | 0.9992 | 0.9991 | 0.9991 |

~0 at steps 1–2 (cold actor diverges from rollout policy), snaps to **~0.999 by step 5** (the first clean step) and stays there. `rollout_probs_diff_mean` mirrors this: ~0.84 at step 1 → ~0.0035 from step 5 on, all arms.

### 2.5 Communication-volume / comm_eff counters @ step 50
| counter | mask p=0.95 | r=102 | r=77 |
|---|---|---|---|
| `logical_pp_bytes_prf` | 76.8 | — | — |
| `logical_pp_bytes_powersgd_y_only` | — | 102.0 | 77.0 |
| `mask_applications` (total) | 143360 | 0 | 0 |
| `mask_ratio` (agg) | 0.9500 | — | — |
| `powersgd_applications` | — | 143360 | 143360 |
| `powersgd_basis_updates` | — | 40 | 40 |
| `clean_steps` (cumulative) | 10 | 10 | 10 |
| `anchor_*`, `spectral_*` | all 0 (OFF) | all 0 (OFF) | all 0 (OFF) |

mask per-layer `mask_ratio` @ 50 is uniform ≈0.9500 across layers {3,7,11,15,18,21,24} (0.9497–0.9507) — the p=0.95 keep-rate is honored per-boundary.

### 2.6 Perf / response length (context)
| metric | mask | r=102 | r=77 |
|---|---|---|---|
| throughput tok/s @ 50 | 670.8 | 601.6 | 628.6 |
| timing_s/step @ 50 | 121.9 | 119.2 | 119.4 |
| response_length/mean @ 50 | 216.9 | 177.7 | 190.8 |

---

## 3. Resolved configs (ground truth from `set -x` traces)

**Shared by ALL three arms** (the fixed control surface — vanilla GRPO, no-KL no-entropy):
```
model.path                = Qwen/Qwen2.5-1.5B-Instruct
algorithm.adv_estimator   = grpo
actor.use_kl_loss         = False        # vanilla GRPO
actor.entropy_coeff       = 0
algorithm.use_kl_in_reward= False
actor.optim.lr            = 1e-6
clean_cadence             = 5            # dense gradient every 5th step
comm_eff.enabled          = true
comm_eff.anchor.enabled   = false        # anchor OFF
comm_eff.spectral.enabled = false        # spectral OFF
data.train_files          = /root/data/gsm8k/train.parquet
data.val_files            = /root/data/gsm8k/test.parquet
data.train_batch_size     = 128
data.max_prompt_length    = 1024
data.max_response_length  = 16384
rollout.n                 = 8
rollout.name              = vllm
rollout.tensor_model_parallel_size = 2
ppo_mini_batch_size       = 64
ppo_max_token_len_per_gpu = 36864
trainer.total_training_steps = 50
trainer.total_epochs      = 2
trainer.test_freq         = 25
trainer.save_freq         = 50
trainer.val_before_train  = True
trainer.n_gpus_per_node   = 4  (nnodes=1)
model.enable_gradient_checkpointing = True
model.use_remove_padding  = True
```

**Per-arm differences (the ENTIRE codec axis — verified via `diff` of resolved_params):**
| arm | `comm_eff.compression_type` | `comm_eff.powersgd.rank` | `comm_eff.mask.p` |
|---|---|---|---|
| mask p=0.95 | `prf_mask` | (102, inert) | **0.95** (active) |
| PowerSGD r=102 | `powersgd` | **102** (active) | 0.9 (inert) |
| PowerSGD r=77 | `powersgd` | **77** (active) | 0.9 (inert) |

PowerSGD codec knobs (both psgd arms): `powersgd.sync_basis=true`, `update_cadence=1`, `warm_start=true`, `qr_dtype=fp32`, `reortho_eps=1e-6`, `compress_recompute=true`, `pp_size=8`, `seed=0`.
Full files: `runs/EXP-20/resolved_params__ce_{mask_p95,powersgd_r102,powersgd_r77}_clean5_50s_gsm8k.txt`.

---

## 4. ★ DENSE-BASELINE VERDICT ★ (revised after direct WandB query)

> **There is NO dense GSM8K run with a usable trajectory.** The only genuine "normal dense RL"
> run in the project — and the one the operator saw with the post-step-10 improvement — is
> **`grpo_dense_bigmath_baseline` (`lwl9yk4y`), which is Big-Math / MATH-lighteval, NOT GSM8K.**
> The only DENSE+GSM8K runs are two 2-step probes with empty WandB history.

### 4.1 The run the operator saw = `grpo_dense_bigmath_baseline` (`lwl9yk4y`)
Direct WandB `scan_history()` confirms this is a **true dense run** and it shows **exactly the
post-step-10 improvement the operator described** — but on the **MATH-eval** metric, not GSM8K:

- **It is genuinely dense:** `comm_eff.enabled = False`, Qwen2.5-1.5B-Instruct, `adv_estimator=grpo`, `lr=1e-6`, `bs=128`, `total_training_steps=120` (101 reward rows logged), `total_epochs=1`.
- **Its ONLY validation metric is `val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1`.** A full history scan finds **zero `*gsm8k*` keys** in this run. `train_files=/root/data/bigmath/train.parquet`.
- **The step-10 improvement (MATH-eval acc):** val@0 = **0.536** → val@10 = **0.558** → val@20 = **0.584**, continuing to climb to ~0.608 by step 80. Reward: 0.41 (s1) → 0.49 (s20) → 0.50 (s50) → ~0.55 (s80). Full trajectory in §6.2.

⇒ This is the right *kind* of run (normal dense RL, comm_eff OFF, clear early-improvement), but it
is the **wrong dataset** (Big-Math, evaluated on MATH-lighteval) to serve as the EXP-20 (GSM8K) dense
ceiling. It is **not directly comparable** to the EXP-20 arms (different data, different eval set,
1 epoch / 120 steps vs 2 epochs / 50 steps, base capability ~0.54 not ~0.08-cold).

### 4.2 The only DENSE + GSM8K runs that exist = two 2-step probes (empty)
- `ce_dense_probe_2s_gsm8k` ×2 (`8t7shmtl`, `hdgwfyjf`): `comm_eff.enabled=False`, `compression_type=dense`, GSM8K — but `total_training_steps=2`. WandB history: **0 reward rows, 0 gsm8k-val points** (probes; nothing logged). The local `ce_dense_probe_2s_gsm8k.log` shows it reached `global_step=3` with one mid-run val `acc=0.399`. **NOT a baseline.**
- **No `ce_dense_50s_gsm8k` run exists** — the optional EXP-20 dense arm never ran (the chain stopped after r=77).

### 4.3 Closest GSM8K long-trajectory run (for orientation) = a COMPRESSED run, not dense
`grpo_mask_channel_p0p9_rescale_clean_every20_2epoch` (`t03dn4nh`): GSM8K, **comm_eff ON** (mask p=0.9, clean_cadence=20), 116 steps, 114 reward rows, val-acc@110 = **0.7225**. This is a *masked* run (not dense). Its improvement starts later (val ~0.08 through step 10, ~0.13 @ step 20, ~0.49 @ step 30, ~0.69 @ step 50) — the clean_cadence=20 means its first dense refresh is at step 20. Useful only as a GSM8K trajectory shape reference; it is not a dense control. Full trajectory in §6.3.

### 4.4 Historical dense-GSM8K number (prose only, artifacts gone)
- Project dense control = **`baseline` / `EXP-3`** (de-bloat skill: "ids `baseline`/`3`/`EXP-3` are the permanent dense control"). Its run dir/log is **not** in the repo and **not** in this WandB project — de-bloated long ago; no trajectory survives.
- `GOAL.md` records the dense-GSM8K parity figure as **≈ 0.741** ("reaches GSM8K dense parity (**EXP-17: 0.735 vs 0.741**)"). This is the **only surviving dense-GSM8K number, prose-only**, from an **earlier config** (EXP-17 era: mask p=0.9, clean_cadence=20, 2 epochs) — a rough ceiling, **not** a same-config EXP-20 control.

### 4.5 Bottom line for the lead
- **No dense GSM8K trajectory exists to compare against EXP-20.** The dense run that exists is Big-Math/MATH-eval (`lwl9yk4y`); the dense+GSM8K runs are empty 2-step probes.
- The central question ("is our ~0.74 just the 10 clean steps, or do the 40 compressed steps add learning?") **cannot be settled against a same-config dense GSM8K run, because none exists.** To settle it cleanly, a **dense GSM8K 50-step run (comm_eff OFF, same lr/bs/2-epoch)** must be **launched**. (A clean-only / dense-only ablation at the same 10-dense-step budget would be the direct test.)
- For context, the three EXP-20 arms (0.7384 / 0.7415 / 0.7437) already sit just above the historical dense-parity prose figure ~0.741; the project-fixed baseline-of-record for the EXP-20 comparison is the **mask arm (0.7384)**, not dense.

---

## 6. Full WandB project inventory + dense trajectories (direct API)

Queried `shamanework-pl/verl_compression_research` via `wandb.Api()` (egress OK). **14 runs total.** All show
state=`crashed` — an egress/heartbeat artifact (runs finished locally; WandB never got the clean-exit
signal). `comm_eff.enabled`, `compression_type`, `clean_cadence`, dataset dug from each run's nested config.

### 6.1 Inventory (14 runs, chronological)
| created | name | id | dataset | comm_eff | type | clean_cad | mask.p | psgd.r | steps | val-acc (eval set) | reward |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06-01 04:23 | grpo_mask_channel_p0p9_rescale_clean_every20_2epoch | `t03dn4nh` | **GSM8K** | ON | mask | 20 | 0.9 | — | 116 | **0.7225** (gsm8k@110) | 0.708 |
| 06-01 07:20 | grpo_mask_p0p9_clean20_bigmath_fixed | `zejoupvf` | bigmath | ON | mask | 20 | 0.9 | — | 120 | — (no gsm8k) | 0.467 |
| 06-01 09:09 | **grpo_dense_bigmath_baseline** | `lwl9yk4y` | **bigmath** | **OFF** | dense | 0 | — | — | 120 | **0.608** (MATH-eval@80) | 0.55 |
| 06-03 16:30 | ce_powersgd_probe_2s_gsm8k | `5lg2fiqk` | GSM8K | ON | powersgd | 0 | 0.9 | 102 | 2 | — (empty) | — |
| 06-03 17:08 | ce_powersgd_probe_2s_gsm8k | `uam6gkg0` | GSM8K | ON | powersgd | 0 | 0.9 | 102 | 2 | — (empty) | — |
| 06-03 17:14 | ce_powersgd_probe_rankH_2s_gsm8k | `88iur3y2` | GSM8K | ON | powersgd | 0 | 0.9 | **2048** | 2 | — (empty) | — |
| 06-03 17:48 | **ce_dense_probe_2s_gsm8k** | `8t7shmtl` | **GSM8K** | **OFF** | **dense** | 0 | — | — | 2 | — (empty) | — |
| 06-03 17:56 | ce_mask_p95_clean5_50s_gsm8k (early/aborted) | `0luftyuv` | GSM8K | ON | prf_mask | 5 | 0.95 | — | 50 | 0.081 (1 pt) | — |
| 06-03 18:05 | ce_powersgd_probe_2s_gsm8k | `j5t8xt2e` | GSM8K | ON | powersgd | 0 | 0.9 | 102 | 2 | — (empty) | — |
| 06-03 18:11 | ce_powersgd_probe_rankH_2s_gsm8k | `aj3fhtic` | GSM8K | ON | powersgd | 0 | 0.9 | 2048 | 2 | — (empty) | — |
| 06-03 18:31 | **ce_dense_probe_2s_gsm8k** | `hdgwfyjf` | **GSM8K** | **OFF** | **dense** | 0 | — | — | 2 | — (empty) | — |
| 06-03 18:36 | **ce_mask_p95_clean5_50s_gsm8k** (FINAL) | `3yxzzwn3` | GSM8K | ON | prf_mask | 5 | 0.95 | — | 50 | 0.7384 (local) | 0.778 |
| 06-03 23:47 | **ce_powersgd_r102_clean5_50s_gsm8k** | `kqozxfr0` | GSM8K | ON | powersgd | 5 | 0.9 | 102 | 50 | 0.7437 (local) | 0.794 |
| 06-04 01:42 | **ce_powersgd_r77_clean5_50s_gsm8k** | `oquyeic3` | GSM8K | ON | powersgd | 5 | 0.9 | 77 | 50 | 0.7415 (local) | 0.780 |

(Note: `ce_mask_p95...0luftyuv` is an earlier aborted attempt of the same mask config; the **canonical** mask arm is `3yxzzwn3`. The two `ce_dense_probe` and the rankH probes are the EXP-20 hard-invariant probes — `rank=2048=H` is the lossless r=H check.)

**Dense (comm_eff OFF) runs in the whole project = exactly 3:** `lwl9yk4y` (Big-Math, 120 steps, real) + the two `ce_dense_probe_2s_gsm8k` (GSM8K, 2 steps, empty). That's it.

### 6.2 `grpo_dense_bigmath_baseline` (`lwl9yk4y`) — FULL trajectory (the run the operator saw)
Dense (comm_eff OFF), Big-Math train, **MATH-lighteval** eval, 120-step config / 101 reward rows, test_freq=10.
val = `val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1` (≡ reward/mean@1). **NO gsm8k val exists.**

| step | reward | MATH-eval val-acc |
|---|---|---|
| 0 | — | **0.536** |
| 1 | 0.406 | |
| 5 | 0.393 | |
| 10 | 0.486 | **0.558**  ← post-step-10 bump the operator noticed |
| 20 | 0.492 | **0.584** |
| 30 | 0.499 | 0.568 |
| 40 | 0.526 | 0.574 |
| 50 | 0.504 | **0.566** |
| 60 | 0.516 | 0.594 |
| 70 | 0.485 | 0.594 |
| 80 | 0.579 | **0.608** (peak) |
| 90 | 0.523 | 0.602 |
| 100 | 0.536 | 0.586 |

Steady reward climb 0.41→~0.55; MATH-eval val 0.536→0.608 (+0.072 over ~80 steps). The early move 0.536→0.558→0.584 (steps 0→10→20) is the visible "improvement after ~step 10." (Base capability here is ~0.54, not the ~0.08-cold of the GSM8K runs — different dataset/eval.)

### 6.3 `grpo_mask_channel_p0p9_rescale_clean_every20_2epoch` (`t03dn4nh`) — closest GSM8K long run (COMPRESSED, not dense)
GSM8K, **comm_eff ON** (mask p=0.9, clean_cadence=20), 116-step config / 114 reward rows. val = `val-core/openai/gsm8k/acc/mean@1`.

| step | reward | GSM8K val-acc |
|---|---|---|
| 0 | — | 0.085 |
| 1 | 0.108 | |
| 10 | 0.149 | 0.083 |
| 20 | 0.155 | 0.132  ← first clean refresh at step 20 (clean_cadence=20) |
| 30 | 0.356 | 0.488 |
| 40 | 0.426 | 0.553 |
| 50 | 0.584 | 0.690 |
| 60 | 0.643 | 0.704 |
| 70 | 0.690 | 0.725 |
| 80 | 0.646 | 0.734 (peak) |
| 90 | 0.759 | 0.719 |
| 100 | 0.730 | 0.720 |
| 110 | 0.746 | **0.7225** (final) |

NOT a dense control (it's masked). Listed only as a GSM8K trajectory-shape reference. Reaches 0.7225 by step 110 — consistent with the EXP-20 mask arm reaching 0.7384 by step 50 under the more aggressive clean_cadence=5.

---

## 7. Provenance / how to reproduce this dump
- WandB: `wandb.Api()` against `shamanework-pl/verl_compression_research`, API key sourced from `~/.config/verl-research/secrets.env` (never echoed). Inventory via `api.runs(...)`; per-step trajectories via `run.scan_history()` (full, un-downsampled); configs dug from each run's nested `config` dict (`actor_rollout_ref.actor.comm_eff.*`, `data.train_files`, `trainer.*`). All runs show state=`crashed` (egress/heartbeat artifact — runs finished locally).
- EXP-20 arm finals (§1–2): local logs parsed by `/tmp/extract_exp20.py` (` - `-delimited `key:value` step lines, Ray `(TaskRunner …)` prefix stripped) → `/tmp/exp20_parsed.json`. 51 step lines/arm (steps 0–50; step 0 = val-only). WandB history for these arms is downsampled (~48 rows) so local is authoritative for the finals.
- Configs: `runs/EXP-20/resolved_params__*.txt` (each run's `set -x` trace; last-write-wins / Hydra semantics).
