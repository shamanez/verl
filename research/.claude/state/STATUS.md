# Research Status — 2026-07-02T19:29:44+10:00

## Issue pipeline
| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 47 | MOAT: Linear projection baseline and best Delta for ANCHOR | PLAN_READY (blocked on approval) | none (GPU-free analysis) | — | **label is `status:planned`, NOT `status:approved`** — orchestrator hard gate: cannot dispatch analyst. Box 43511290 reachable, EXP-57 trace present, but #47 additive code (moat_report.py + damped_linear/paper_linear/--cadence in moat_scorecard.py) NOT yet on box. Awaiting operator label flip. |
| 45 | MOAT: Minimal projection scorecard + block/layer structure | DONE | none (GPU-free analysis) | PASS · M4 | contract GO; shared EXTERNAL box 43511290 (operator-managed, NOT torn down by any issue) |
| 58 | Big-Math 1000-step GRPO — fp32 weights + full ckpts→R2 | PROBE_GREEN / HANDED_OFF | 1×H200 43387501 | probe 5/5 PASS | OTHER session's lane — NOT touched by this /goal |

## Last tick
2026-07-02T19:29:44+10:00 · EXP-47 blocked at human-approval gate (`status:planned`) · analyst NOT dispatched (sacred gate) · EXTERNAL box 43511290 reachable + trace present, #47 code not yet applied · EXP-58 left alone (other session) · no EXP-47-owned box to reap

## Budget
EXP-47: $0 GPU-hours (GPU-free replay on operator's $0.11/hr box 43511290, operator-managed, outside this issue's teardown scope). No provisioning, no teardown by this issue.
