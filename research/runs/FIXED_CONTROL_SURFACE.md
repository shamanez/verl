# Fixed Control Surface — GSM8K comm-eff experiments

**Status: LOCKED (operator directive, 2026-06-04; substrate extended 2026-06-09 / #25).**
Every experiment in this project holds these constant. As of issue #25 the comm-eff
**substrate** (PowerSGD r=77 + a mandatory anchor that owns `Q`) is also locked, and the
**only** axis that may vary between arms is the **merger** (the ☆ section). Changing
anything else requires a separate, explicit justification — same bar as the
model/loss/hardware controls in `CLAUDE.md §1`. This file extends those three with the
*training* hyperparameters + the locked comm-eff substrate.

Why this exists: small RL deltas (the EXP-20 arms differ by ±0.005 val-acc) are
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
| **ppo_max_token_len_per_gpu** | **18432** actor / **36864** log_prob+ref | actor halved (anchor's ~3 GB no-hook clone/rank must fit — launcher default since #25). **NON-BINDING under static batching**: with `ppo_micro_batch_size_per_gpu=1` + `use_dynamic_bsz=False`, each micro-batch is exactly 1 sequence (≤16384+prompt ≈ 16.6K < 18432), so this cap NEVER triggers and does NOT affect the result — 18432 vs 36864 is mathematically identical here. **Keep 18432 on EXP-30-style comparison cells INCLUDING the dense reference** so the dense baseline is apples-to-apples (only the codec/merger varies); raising to 36864 on an anchor-OFF ablation is allowed but then it is NOT one-knob vs the comm-eff cells |
| **total_training_steps** | **50 → 100** | start at 50; extend to 100 once 50 trains cleanly |
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

**The variable axis — the merger** (`spectral.correction_mode` + its weight, e.g.
`signed_ema_alpha`): the research axis going forward. `signed_ema` is wired but
**falsified** (net-harmful — result + why in `SUMMARY.md`); error-feedback on the
PowerSGD residual (#24) is the next candidate.

**Reference codecs (NOT the base; ablation only):** the dense control
(`comm_eff.enabled=false`, the learning ceiling) and the legacy `prf_mask`
(`mask.p`; cannot anchor-own-`Q`, so run it with `anchor.owns_q=false`).

**☆ DENSE BASELINE (val@50) — CORRECTED 2026-06-13.** The dense "ceiling" is **run-variance-dominated**;
report it as a **band ≈ 0.75–0.78**, not a single point (rollout nondeterminism ≈ ±0.024/draw even at
seed 0). Two draws on record: **current-code, same-static-batch-config rerun `exp30_dense_rerun`
(`73ntu76u`) = 0.7839** — the APPLES-TO-APPLES baseline for any EXP-30-config comm-eff cell (proof: all
comm_eff counters 0) — and the old-code `5e2jpho9` = 0.7536 (historical). Always compare a comm-eff cell
to a dense run sharing its code + hyperparameters; the current-code rerun confirmed our EXP-29/30 merges
did **not** regress dense (≥ old). Dense is **never re-run for production**, but an apples-to-apples dense
control alongside a comm-eff sweep is sanctioned and was operator-directed in EXP-30.

## Measurement knobs (NOT control variables — may vary freely)

These don't touch the trained model (validation is a separate, read-only pass), so
they can differ between runs without breaking comparability:

- **`test_freq`** — validation cadence. **= 25** (val at 0/25/50 on a 50-step run;
  0/25/50/75/100 at 100 steps). The per-step **train** reward (`critic/score/mean`,
  logged every step regardless of `test_freq`) is the fine-grained signal between
  validations.

## Diagnostics policy — production runs ship with capture OFF (operator directive, 2026-06-11)

Diagnostic instrumentation that **holds tensors in memory** is an OOM hazard and is
risk-tiered. Provenance: EXP-26 B-ef **r1** OOM'd at step ~42 inside the anchor backward
with `capture_fresh_anchor=true` co-resident; the **r2** re-run with captures off (plus
`expandable_segments`) completed cleanly. Tiers:

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
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (standard since the r1 OOM),
`spectral.ema_device=cpu` (keeps the ~6 GB fp32 M/EF state off-GPU), and the
18432 actor token budget above while the anchor is on.

## How to launch on this surface

One canonical launcher (`examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`)
— its `${VAR:-default}` defaults ARE the anchor base + the core surface (batch, lr, rollout
shape, contexts, objective, the substrate above). A bare comm-eff launch is the base; you
override only the run length + the axis you're varying:

```bash
# the anchor base, 50 steps, validation every 25 (nothing else to set):
TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 EXPERIMENT_NAME=ce_anchor_base_50s \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# sweep the merger axis:
COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.7 EXPERIMENT_NAME=ce_a0p7 \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# dense reference: same launch with COMM_EFF_ENABLED=false.
```

Pass `TOTAL_TRAINING_STEPS` (50, then 100) + `TEST_FREQ=25` per launch. The substrate
defaults (anchor on, owns `Q`, PowerSGD r=77, signed_ema merger, no clean step) are baked
into the launcher — do **not** re-type them; the ground truth of any run is its
`resolved_params.txt`.

See also: `CLAUDE.md §1` (model/loss/hardware controls), `examples/grpo_trainer/VAST_README.md`
(launcher stability contract), `research/.claude/project.yaml` (`default_compute`, provisioning).
