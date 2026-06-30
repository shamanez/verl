# Post-Experiment Summary Plan

Compact handoff for future planning. Execution plan files are deleted after each
issue; this file persists. Results live in `research/runs/SUMMARY.md` + each run's
`verdict.md` + W&B.

## Current base (the baseline = the problem state)

| item | value |
|---|---|
| launcher | `examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh` |
| method | `signed_ema`, α=0.25, β_anc=0.50 |
| surface | fast 1K: resp 1024, dyn-bsz, rollout TP=1, gpu_mem 0.55, ppo_max_token 24576, 50 steps, val@25/50, diagnostics off |
| substrate | PowerSGD r=77 + anchor (owns Q, clean=0, paired replay, `disable_custom_all_reduce`) |
| anchor latency | cadence/delay_K = **20/20** — the k-collapse regime (Priority 1); the baseline collapses here |

At LOW latency (5/5) the same merger reached parity (val@50 ≈ 0.736 vs dense ≈ 0.766,
n=1, older 2K surface); the open problem is holding parity at realistic high latency.

## Settled knobs (do not re-sweep)

| knob family | takeaway |
|---|---|
| merger | `signed_ema` is the core merger family |
| `beta_anc` on signed_ema | non-flat, peaks ≈ 0.50 |
| `signed_ema_alpha` | peaks ≈ 0.25; α=0.0 does NOT ignite |
| δ-momentum / adaptive-λ / perturbation / control-variate / sub-basis | all null vs baseline |

## Planning rule

Start from the base launcher **unchanged** and vary a SINGLE knob. Everything else is
locked (`runs/FIXED_CONTROL_SURFACE.md`). The two active fronts are **GPU-free offline
kill-gates** — `../../reports/priority-1-anchor-staleness-k-collapse.html` and
`../../reports/priority-2-compression-train-inference-mismatch.html`; gate before any GPU spend.

Do not import invalid (pre-paired-replay) anchor claims or rebuild deleted plan files.

## M4 weight-projection track (separate from the comm-eff fronts above)

Dense weight-trajectory analysis, GPU-free. Shared trace = **EXP-43** (PASS, issue closed):
per-tick full weights in R2 `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/`
(key `tick_<N>/tick_<N>.pt`, 160 bf16 snapshots, n_matrices=338, ~492 GB R2-only) + index
manifests (R2 + `runs/EXP-43/regimeA/weights/`). Entry point = **#44** (the offline sweep
engine); #45-#56 build on it. All `kind:analysis`. Stream the trace layer/block-wise
(`reports/r2-access-pattern-for-analysis.md`) - never bulk-download. Reports go to
`research/reports/`; results to `runs/SUMMARY.md` + each verdict.
