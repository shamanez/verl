# Research Status — 2026-06-05

## Active
**EXP-25** anchor-circuit default (stale-M + anchor-owns-Q + signed_ema merger) — code complete on `vast-ai-workload` (R1+R2+R3 + dead-spectral path removed; `exp/25-anchor-default` merged `--no-ff` + deleted). Stale run scaffold deleted → re-materialize from `.claude/plans/25.md`. **Ready to resume** on existing box `39602487`.

## Issue pipeline
| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 25 | Anchor-circuit default | READY_TO_RUN | box 39602487 (4×H200) | — | Reuse the box: verify SSH first, do NOT provision. Code on `vast-ai-workload`. |
| 24 | Error-feedback + basis-aligned anchor | BLOCKED | — | — | `depends_on:#25` (needs PASS). |

## Resume path
Reuse box 39602487 → id-0 anchor-M probe (cadence=1/delay_K=1) → id-1 all-flags-ON probe → α∈{0,0.3,0.5} sweep (50 steps, delay_K=5/cadence=5, max_targets=-1, powersgd r=77) → analyst → log-writer. Locked hyperparams: `runs/FIXED_CONTROL_SURFACE.md`.

## Last tick
2026-06-05 · ready=[25] · blocked=[24 dep]

## Budget
1 live box 39602487 (4×H200, ~$15.23/hr). Tear down at verdict / if SSH unreachable.
