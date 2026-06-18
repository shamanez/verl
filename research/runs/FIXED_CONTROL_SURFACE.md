# Fixed Control Surface — GSM8K comm-eff experiments

**Status: LOCKED (operator directive, 2026-06-04; substrate extended 2026-06-09 / #25).**
Every experiment in this project holds these constant. As of issue #25 the comm-eff
**substrate** (PowerSGD r=77 + a mandatory anchor that owns `Q`) is also locked, and the
**only** axis that may vary between arms is the **merger** (the ☆ section). Changing
anything else requires a separate, explicit justification — same bar as the
model/loss/hardware controls in `CLAUDE.md §1`. This file extends those three with the
*training* hyperparameters + the locked comm-eff substrate.

Why this exists: small RL deltas can differ by only a few thousandths of val-acc and are
only interpretable if the rest of the run is byte-for-byte comparable. Silent
drift in batch size / cadence / steps between experiments destroys that. Pin it
once, here, and read it off for every launch.

## The control surface (held constant)

| Dimension | Value | Notes |
|---|---|---|
| **Model** | `Qwen/Qwen2.5-1.5B-Instruct` | dense control anchored to it; **H (hidden) = 1536** |
| **Data** | GSM8K (`openai/gsm8k`) | `train.parquet` / `test.parquet`; test set = 1319 |
| **RL objective** | vanilla GRPO, **no-KL, no-entropy** | `use_kl_loss=False`, `use_kl_in_reward=False`, `entropy_coeff=0`, `adv_estimator=grpo` |
| **Learning rate** | `1e-6` | AdamW (verl default betas) |
| **train_batch_size** | **128** prompts | ×`rollout.n`=8 ⇒ 1024 sequences/step |
| **ppo_mini_batch_size** | **64** | |
| **ppo_micro_batch_size_per_gpu** | **1** | static batching (`use_dynamic_bsz=False`) for trackability |
| **rollout.n** | **8** | rollouts per prompt |
| **rollout.tensor_model_parallel_size** | **2** | vLLM TP |
| **rollout.gpu_memory_utilization** | **0.4** | |
| **max_prompt_length** | **1024** | |
| **max_response_length** | **16384** | the 16K-response headroom that forces multi-GPU |
| **ppo_max_token_len_per_gpu** | **18432** actor / **36864** log_prob+ref | actor halved (anchor's ~3 GB no-hook clone/rank must fit — launcher default since #25). **NON-BINDING under static batching**: with `ppo_micro_batch_size_per_gpu=1` + `use_dynamic_bsz=False`, each micro-batch is exactly 1 sequence (≤16384+prompt ≈ 16.6K < 18432), so this cap NEVER triggers and does NOT affect the result — 18432 vs 36864 is mathematically identical here. **Keep 18432 on B2-style comparison cells INCLUDING the dense reference** so the dense baseline is apples-to-apples (only the codec/merger varies); raising to 36864 on an anchor-OFF ablation is allowed but then it is NOT one-knob vs the comm-eff cells |
| **total_training_steps** | **50** (→100 extended) | **50, NOT 55** (operator 2026-06-17: no end-of-training val@55). verl validates at `is_last_step OR global_steps % test_freq == 0` (`ray_trainer.py:1720-1721`), so `total=55, test_freq=25` fired a 3rd val at step 55 (is_last_step); `total=50` makes the last step coincide with the test_freq val ⇒ **val@25 / val@50 ONLY**. val@50 flush is handled by the launcher's `wandb sync` final-flush daemon + the authoritative local train.log (the old 5-step buffer is obsolete). **Comparison number = val@50**, identical across total=50/55 (same step-50 model). Extend to 100 (a test_freq multiple → no spurious val) for a winner. |
| **total_epochs** | **2+** | a ceiling sized to reach the step target (≈59 steps/epoch) |
| **comm-eff substrate** (LOCKED — see ☆ below) | anchor on + owns `Q`, cadence 5 / delay_K 5, `clean_cadence`=0 | the anchor is **mandatory** and is the **only** thing that updates `Q`; it refreshes `M`+`Q` every 5 ticks from stale, delayed weights and REPLACES any periodic dense clean step (do **not** assume a `clean_cadence`) |
| **save_freq** | **50** | |
| **val_before_train** | **True** | the step-0 val point |
| **calculate_log_probs** | **True** | train-inference mismatch diagnostic; rollout CORRECTION stays OFF (old_log_prob always recomputed) |
| **Hardware** | 4×H200 (pref) or 8×H100 | Vast `verl-research-vllm020`; `max_dph=24` |
| **seeds** | comm_eff `seed=0` (mask + powersgd) | |

## ☆ The locked substrate + the variable axis (the merger)

As of issue #25 the comm-eff base is the **anchor circuit on a PowerSGD codec**, and the
substrate is **locked**. The single axis that may vary between arms is now the **merger**
(how the anchor `M` corrects the fast gradient).

**Locked substrate** — held constant across every comm-eff arm (exact values are the
launcher `${VAR:-default}`; the ground truth of any run is its `resolved_params.txt`):

| knob | value | note |
|---|---|---|
| `compression_type` | `powersgd` | the locked codec (only one compatible with anchor-owns-`Q`) |
| `powersgd.rank` | `77` | byte-matched to mask p=0.95 (H=1536: `0.05·1536 ≈ 77`) |
| `anchor.enabled` | `true` | MANDATORY — the stale full-grad reference `M` |
| `anchor.owns_q` | `true` | the anchor is the ONLY thing that updates `Q` |
| `anchor.cadence` / `delay_K` | `5` / `5` | refresh + staleness, in optimizer ticks |
| `clean_cadence` | `0` | DEAD — the anchor replaced the periodic dense step |
| `spectral.correction_mode` | `delayed_ef` (replicated base) | B2 baseline merger — error-feedback, `λ=1`, `β_anc=0`, val@50 **0.7528**. **Current leading result (provisional): `signed_ema` α=0.5 `beta_anc=0.50` → val@50 0.7635** (EXP-34, verdict REVISE — pending a β=0.50 replicate; see `runs/SUMMARY.md`). Until that confirms, `delayed_ef` stays the locked control base. |
| `replay_paired_batch` / `snapshot_device` | `true` / `cpu` | valid on-policy anchor `M` — part of the B2 substrate |
| vLLM `disable_custom_all_reduce` | `true` | **required** for the box to init (CUDA-IPC under the mp executor); greedy-val-neutral → a controlled var, not a knob |

**The variable axis — how the anchor `M` is USED.** Current **leading result: `signed_ema` α=0.5
`beta_anc=0.50` → val@50 0.7635** (EXP-34) — the highest measured, edging B2 `delayed_ef` (0.7528)
but **provisional** (verdict REVISE: single draw + best-of-3, within ±0.024 noise of B2; pending a
β=0.50 replicate). `delayed_ef` (B2) remains the established, replicated dense-parity baseline. EXP-34
showed `beta_anc` is NON-flat on `signed_ema` (peaks at 0.50), unlike the flat `delayed_ef` β curve
(EXP-33). Anchor-usage levers (EXP-31) were all null. Compact planning handoff: `.claude/plans/SUMMARY.md`.

**Reference codecs (NOT the base; ablation only):** the dense control
(`comm_eff.enabled=false`, the learning ceiling) and the legacy `prf_mask`
(`mask.p`; cannot anchor-own-`Q`, so run it with `anchor.owns_q=false`).

**☆ DENSE BASELINE (val@50) — CORRECTED 2026-06-13.** The dense "ceiling" is **run-variance-dominated**;
report it as a **band ≈ 0.75–0.78**, not a single point (rollout nondeterminism ≈ ±0.024/draw even at
seed 0). Two draws on record: **current-code, same-static-batch-config dense rerun
(`73ntu76u`) = 0.7839** — the APPLES-TO-APPLES baseline for any B2-config comm-eff cell (proof: all
comm_eff counters 0) — and the old-code `5e2jpho9` = 0.7536 (historical). Always compare a comm-eff cell
to a dense run sharing its code + hyperparameters; the current-code rerun confirmed our valid-M merges
did **not** regress dense (≥ old). Dense is **never re-run for production**, but an apples-to-apples dense
control alongside a comm-eff sweep is sanctioned.

## Measurement knobs (NOT control variables — may vary freely)

These don't touch the trained model (validation is a separate, read-only pass), so
they can differ between runs without breaking comparability:

- **`test_freq`** — validation cadence. **= 25** (val at 0/25/50 on a 50-step run;
  0/25/50/75/100 at 100 steps). The per-step **train** reward (`critic/score/mean`,
  logged every step regardless of `test_freq`) is the fine-grained signal between
  validations.

## Diagnostics policy — production runs ship with capture OFF (operator directive, 2026-06-11)

Diagnostic instrumentation that **holds tensors in memory** is an OOM hazard and is
risk-tiered. Tensor capture must stay opt-in, bounded, and justified by the active
plan. Tiers:

1. **Scalar telemetry — always-on safe.** Norms, cosines, byte counters, residual-dose
   `rel_change`, corr-style watch metrics: negligible memory, log them on every run.
2. **Bounded tensor captures — only when the plan's success criteria require them.**
   `capture.enabled=true` MUST come with the bounds: `max_ticks` (≈10–12), `min_tick`
   (post-warm window), `stratified_targets`, `rank0_only=true`, fp32-dump-then-free.
   The plan must name which criterion consumes the captures; otherwise OFF.
3. **Extra-backward probes — NEVER on a production arm.** `capture_g_dense` and
   `capture_fresh_anchor` each add a parallel full backward + held fp32 tensors (the r1
   OOM). They run only on **dedicated short diagnostic runs** whose val number is NOT a
   deliverable (e.g. a 5–10-step geometry probe), never on the arm that produces the
   comparison val.

A **production arm** = any run whose val/score lands in a verdict, SUMMARY, or
comparison table. Default posture for production: `COMM_EFF_CAPTURE_ENABLED=false`
(the launcher default) — flipping it on requires the tier-2 justification in the plan.
Standing OOM guards on EVERY run regardless of tier:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`spectral.ema_device=cpu` (keeps the ~6 GB fp32 M/EF state off-GPU), and the
18432 actor token budget above while the anchor is on.

## How to launch on this surface

**THE canonical launcher every cell runs on top of is
`examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh`**. It is self-contained: it pins the **entire B2 substrate explicitly** (delayed_ef λ=1, β_anc=0,
PowerSGD r=77, anchor on + owns `Q`, cadence=delay_K=5, clean=0, replay, the OOM guards) and then execs
the generic `vast_comm_eff_baseline_*.sh` engine — so a **bare run reproduces B2 = the SOTA comm-eff reference
= the B2 reference** (no knobs to set). Do **not** invoke the generic `vast_comm_eff_baseline_*.sh` directly:
its `${VAR:-default}` defaults are *where the values live* (and the ground truth of any run is its
`resolved_params.txt`), but the b2_sota wrapper is the audited entry point and keeps every arm
one-knob-from-B2. You override only the run length + the ONE axis you're varying (an anchor-usage lever —
all default OFF ⇒ bitwise B2):

```bash
# B2 = the SOTA comm-eff base — a BARE run, 50 steps, val every 25 (nothing else to set):
TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 EXPERIMENT_NAME=b2_repro \
  bash examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh

# the issue-#31 variable axis = anchor-gradient USAGE on top of B2 (e.g. the built L4 perturbation lever):
COMM_EFF_SPECTRAL_PERTURB_SIGMA=0.03 EXPERIMENT_NAME=L4_perturb_0p03 \
  bash examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh   # all 4 levers OFF = bitwise B2

# disable_custom_all_reduce=true is the B2 wrapper DEFAULT (locked-surface controlled var; the bare
# run above already sets it). Override =false ONLY to opt out on a box that does not hit the crash:
DISABLE_CUSTOM_ALL_REDUCE=false TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 EXPERIMENT_NAME=b2_custom_ar \
  bash examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh

# dense reference (band ≈ 0.75–0.78) — one-knob OFF, shares the comm-eff code path (NOT via b2_sota,
# which force-enables comm-eff): set the master switch on the GENERIC launcher.
COMM_EFF_ENABLED=false TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 EXPERIMENT_NAME=dense_ref \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

Pass `TOTAL_TRAINING_STEPS` (50, then 100 for an extended winner) + `TEST_FREQ=25` per launch. The full B2
substrate is baked into the b2_sota launcher — do **not** re-type it; the ground truth of any run is its
`resolved_params.txt` (the SOTA settings are summarized in `runs/SUMMARY.md`). Closed anchor-usage
levers and their tested knobs are summarized in `.claude/plans/SUMMARY.md`.

See also: `CLAUDE.md §1` (model/loss/hardware controls), `examples/grpo_trainer/VAST_README.md`
(launcher stability contract), `research/.claude/project.yaml` (`default_compute`, provisioning).
