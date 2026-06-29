# Research Status — 2026-06-29

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 42 | M4 weight-projection accuracy vs horizon (fixed vs learned · plain GRPO vs +compression) | PLAN_READY | none yet (single 1×H200, observe-only) | — | **REFRAMED 2026-06-29 (operator):** measure how accurately `θ̂` projects the WEIGHTS (not gradients) as a function of steps-ahead, at the **K=10** operating point, in 2 regimes. Collection on GPU (incl. exact on-box headline); horizon sweep + report on the MacBook from the tiny sketch (~320 MB/regime). 1 box, 2 sequential runs, then teardown. Plan `.claude/plans/42.md`; runbook `runs/EXP-42/RUNBOOK.md`. Single-GPU **operator-authorized**: 1×H200, fall back to 1×B200 ONLY on OOM; full permission to fit to H200. |
| 41 | M4 look-ahead anchor (delay_K=20, fixed-linear) | DONE | external 4×H200 i_42465843 (team) TORN_DOWN | STOP | probe PASSED 10/10 (code correct); cell A 5/5 ref clean (val@100=0.7066); cell B collapsed (val@100=0.0478) via length-explosion; lift +0.0267 present but merger over-amplified. WandB A=7tbzm9kl B=g6dt6bza |

## Last tick
2026-06-29 · running=[] · analyzing=[] · logging=[] · blocked=[]

## Pipeline state
EXP-42 **reframed** to a weight-projection-accuracy measurement at the **K=10** operating point
(operator ruling: weights MUST be verified accurate before any gradient claim). A gradient-accuracy
follow-up is **deferred to a separate future session** — no plan kept (avoids agent pollution). No
experiment in flight; GPU OFF (operator-managed). Next GPU step = EXP-42's 2-regime single-GPU
observe run (1×H200, B200 fallback on OOM); single-GPU is operator-authorized.

## Open notes for operator
- `runs.jsonl` still has a line-4 `EXP-42-run1` BYO box (RUNNING, operator-managed) from the OLD
  gradient run — tear down or repurpose at will; the reframe does not touch it.
- Single-GPU `num_gpus=1` is a documented (operator-authorized) deviation from the sanctioned
  `gpu_filter_chain` (4×H200/8×H100); 1×H200 primary, 1×B200 fallback on OOM.

## EXP-41 close-out summary (most recent CLOSED experiment)
- VERDICT: STOP — fixed-linear θ̂=2θ[t-20]-θ[t-40] at 20/20: cell A 5/5 ref clean (0.6998/0.7255/0.7233/0.7066),
  cell B collapsed (0.36→0.50→0.11→0.048) via response-length explosion; cos-lift present but merger over-amplified.
- Deferred direction: lower beta_anc — SUPERSEDED by the EXP-42 reframe (measure the weight projection first).

## Budget
0 live instances (harness-provisioned). No active spend. Operator-managed BYO box may still exist (see notes).
