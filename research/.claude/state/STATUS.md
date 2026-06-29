# Research Status — 2026-06-29

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 42 | M4 weight-projection accuracy vs horizon (fixed vs learned · plain GRPO vs +compression) | PLAN_READY | none yet (single 1×H200, observe-only) | — | **REFRAMED 2026-06-29 (operator):** measure how accurately `θ̂` projects the WEIGHTS (not gradients) as a function of steps-ahead, in 2 regimes. Cost-minimised: ordinary training emits a tiny per-tick weight SKETCH (~320 MB/regime); look-ahead replayed OFFLINE on the MacBook across every method×horizon. 1 box, 2 sequential runs, then teardown. Plan `.claude/plans/42.md`; runbook `runs/EXP-42/RUNBOOK.md`. Single-GPU = documented deviation (needs operator sign-off on `num_gpus=1` filter). |
| 43 | M4 gradient-projection accuracy (same-batch grad_proj_gain) | PLANNED · BLOCKED on 42 | none | — | Split from old EXP-42 on 2026-06-29. Instrument already BUILT + 14/14 CPU probes PASS on `exp/42-lookahead-horizon`. Gated: launch ONLY if EXP-42 shows `weight_proj_ratio<1` & `dir_cos>0` at a useful horizon; else STOP (a gradient can't be more accurate than the weight it's computed at). Plan `.claude/plans/43.md`. |
| 41 | M4 look-ahead anchor (delay_K=20, fixed-linear) | DONE | external 4×H200 i_42465843 (team) TORN_DOWN | STOP | probe PASSED 10/10 (code correct); cell A 5/5 ref clean (val@100=0.7066); cell B collapsed (val@100=0.0478) via length-explosion; lift +0.0267 present but merger over-amplified. WandB A=7tbzm9kl B=g6dt6bza |

## Last tick
2026-06-29 · running=[] · analyzing=[] · logging=[] · blocked=[43 (on 42)]

## Pipeline state
EXP-42 **reframed** to a weight-projection-accuracy measurement (operator ruling: weights MUST be
verified accurate before any gradient claim). EXP-43 created to carry the (already-built) gradient
study, gated on EXP-42. No experiment in flight; GPU OFF (operator-managed). Next GPU step =
EXP-42's 2-regime single-1×H200 observe run — **awaiting operator sign-off on the single-GPU
deviation** before provisioning.

## Open notes for operator
- `runs.jsonl` still has a line-4 `EXP-42-run1` BYO box (RUNNING, operator-managed) from the OLD
  gradient run — tear down or repurpose at will; the reframe does not touch it.
- Single-GPU `num_gpus=1 gpu_name=H200` is NOT in the sanctioned `gpu_filter_chain` (4×H200/8×H100
  only). Sign off before provisioning, or attach an operator-provided 1×H200.

## EXP-41 close-out summary (most recent CLOSED experiment)
- VERDICT: STOP — fixed-linear θ̂=2θ[t-20]-θ[t-40] at 20/20: cell A 5/5 ref clean (0.6998/0.7255/0.7233/0.7066),
  cell B collapsed (0.36→0.50→0.11→0.048) via response-length explosion; cos-lift present but merger over-amplified.
- Deferred direction: lower beta_anc — SUPERSEDED by the EXP-42 reframe (measure the weight projection first).

## Budget
0 live instances (harness-provisioned). No active spend. Operator-managed BYO box may still exist (see notes).
