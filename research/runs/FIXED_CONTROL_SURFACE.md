# Fixed Control Surface — GSM8K comm-eff experiments

**Status: LOCKED (operator directive, 2026-06-04; substrate locked 2026-06-09; accelerated base 2026-06-18).**
The locked base is now the **accelerated comm-eff loop** via
`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`:
**accel surface** (resp 2048, dynamic-bsz, rollout TP=1, gpu_mem_util 0.55,
ppo_max_token 24576, 50 steps, test_freq 25, no val-before-train) + **signed_ema**
(α=0.25, β_anc=0.50) core merger + **diagnostics=false** speed knob (math-neutral,
EXP-36B/NEUTRALITY_REVIEW.md), on the unchanged **PowerSGD r=77 anchor substrate**.
~25 min train / ~28 min wall per 50-step run. Reference points (n=1, this surface):
dense ≈0.7657 (EXP-36C); comm-eff signed_ema(0.25,0.50) ≈0.7362 (EXP-36B). The **only**
axis that may vary between arms is the **merger** (the ☆ section). Anything else needs separate justification —
same bar as `CLAUDE.md §1`. Values live in the launcher, not duplicated here.

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
| **use_dynamic_bsz** | **True** (accel base) | token-balanced dynamic batching (the per-element mask is packing-invariant) |
| **rollout.n** | **8** | rollouts per prompt |
| **rollout.tensor_model_parallel_size** | **1** (accel base) | vLLM TP |
| **rollout.gpu_memory_utilization** | **0.55** (accel base) | |
| **max_prompt_length** | **1024** | |
| **max_response_length** | **2048** (accel base) | accel surface — short responses for fast turnaround (was 16384) |
| **ppo_max_token_len_per_gpu** | **24576** actor (accel base) / **36864** log_prob+ref | actor budget under dynamic-bsz on the accel surface |
| **total_training_steps** | **50** (→100 extended) | **50, NOT 55** (operator 2026-06-17: no end-of-training val@55). verl validates at `is_last_step OR global_steps % test_freq == 0` (`ray_trainer.py:1720-1721`), so `total=55, test_freq=25` fired a 3rd val at step 55 (is_last_step); `total=50` makes the last step coincide with the test_freq val ⇒ **val@25 / val@50 ONLY**. val@50 flush is handled by the launcher's `wandb sync` final-flush daemon + the authoritative local train.log (the old 5-step buffer is obsolete). **Comparison number = val@50**, identical across total=50/55 (same step-50 model). Extend to 100 (a test_freq multiple → no spurious val) for a winner. |
| **total_epochs** | **2+** | a ceiling sized to reach the step target (≈59 steps/epoch) |
| **comm-eff substrate** (LOCKED — see ☆ below) | anchor on + owns `Q`, cadence 5 / delay_K 5, `clean_cadence`=0 | the anchor is **mandatory** and is the **only** thing that updates `Q`; it refreshes `M`+`Q` every 5 ticks from stale, delayed weights and REPLACES any periodic dense clean step (do **not** assume a `clean_cadence`) |
| **save_freq** | **50** | |
| **val_before_train** | **False** (accel base) | skip step-0 val (untrained base is constant) |
| **calculate_log_probs** | **True** | train-inference mismatch diagnostic; rollout CORRECTION stays OFF (old_log_prob always recomputed) |
| **Hardware** | 4×H200 (pref) or 8×H100 | Vast `verl-research-vllm020`; `max_dph=24` |
| **seeds** | comm_eff `seed=0` (mask + powersgd) | |

## ☆ The locked substrate + the variable axis (the merger)

The comm-eff base is the **anchor circuit on a PowerSGD codec**, and the
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
| `spectral.correction_mode` | `signed_ema` (accel base default) | core merger — α=0.25, `β_anc=0.50`. Prior `delayed_ef` (λ=1, β_anc=0, val@50 0.7528) is the legacy replicated control. |
| `spectral.diagnostics` | `false` (accel base) | proven math-neutral speed knob (EXP-36B/NEUTRALITY_REVIEW.md) — skips per-step rel_change syncs + diag prints, optimizer path unchanged |
| `replay_paired_batch` / `snapshot_device` | `true` / `cpu` | valid on-policy anchor `M` — part of the substrate |
| vLLM `disable_custom_all_reduce` | `true` | **required** for the box to init (CUDA-IPC under the mp executor); greedy-val-neutral → a controlled var, not a knob |

**The variable axis — how the anchor `M` is USED.** The accel base merger is `signed_ema`
(α=0.25, β_anc=0.50). Anchor-usage levers (EXP-31) were all null; `beta_anc` is NON-flat on
`signed_ema` (peaks ~0.50) vs the flat `delayed_ef` β curve (EXP-33). Compact planning handoff:
`.claude/plans/SUMMARY.md`.

**Reference codecs (NOT the base; ablation only):** the dense control
(`comm_eff.enabled=false`, the learning ceiling) and the legacy `prf_mask`
(`mask.p`; cannot anchor-own-`Q`, so run it with `anchor.owns_q=false`).

**☆ DENSE BASELINE (val@50).** The dense "ceiling" is **run-variance-dominated**; report it
as a **band ≈ 0.75–0.78**, not a single point (rollout nondeterminism ≈ ±0.024/draw even at
seed 0). On the **current accel surface @0.55**, the dense control (comm-eff OFF) is
**EXP-36C = 0.7657** (val@25 0.7627; all comm_eff counters 0) — the apples-to-apples
reference for an accel-surface comm-eff cell. Always compare a comm-eff cell to a dense run
sharing its code + surface. Dense is **never re-run for production**, but an apples-to-apples
dense control alongside a comm-eff sweep is sanctioned.

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
accel actor token budget above while the anchor is on.

## How to launch on this surface

**THE canonical launcher every cell runs on top of is
`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`** — the
accelerated comm-eff base. It pins the whole surface + substrate + signed_ema(0.25,0.50)
+ diagnostics=false explicitly, then execs the generic `vast_comm_eff_baseline_*.sh`
engine, so a **bare run reproduces the accel base** (nothing else to set). Override only
the run length + the ONE axis you're varying.

```bash
# accel comm-eff base — bare run, 50 steps, val@25 (nothing else to set):
EXPERIMENT_NAME=accel_repro \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh

# dense reference on the accel surface (≈0.7657, EXP-36C) — one-knob OFF on the GENERIC launcher
# (the accel base force-enables comm-eff), mirroring the accel surface envs:
COMM_EFF_ENABLED=false USE_DYNAMIC_BSZ=True MAX_RESPONSE_LENGTH=2048 ROLLOUT_TP=1 \
  ROLLOUT_GPU_MEM_UTIL=0.55 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
  TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 VAL_BEFORE_TRAIN=False EXPERIMENT_NAME=dense_ref \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

Pass `TOTAL_TRAINING_STEPS` (50, then 100 for an extended winner) per launch; the rest is
baked in. Ground truth of any run is its `resolved_params.txt`. Closed anchor-usage levers:
`.claude/plans/SUMMARY.md`.

See also: `CLAUDE.md §1` (model/loss/hardware controls), `examples/grpo_trainer/VAST_README.md`
(launcher stability contract), `research/.claude/project.yaml` (`default_compute`, provisioning).
