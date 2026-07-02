# Research Status — 2026-07-02T16:46:40+10:00

## Issue pipeline
| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 45 | MOAT: Minimal projection scorecard + block/layer structure for EXP-57 | DONE | none (GPU-free analysis) | PASS · M4 | contract GO for #47/#48/#49/#56; harness landed on vast-ai-workload (moat_scorecard.py + weight_proj/structure.py); analysis box 43511290 stays EXTERNAL (operator-managed, shared, NOT torn down) |
| 58 | Big-Math 1000-step GRPO — fp32 weights + full ckpts→R2 | PROBE_GREEN / HANDED_OFF | 1×H200 43387501 (RUNNING, held) | probe 5/5 PASS | collection NOT launched; another agent owns launch; box held (keepalive active); see runs/EXP-58/HANDOFF.md |

## Queued (unblocked by #45 contract)
#47 / #48 / #49 (kind:analysis, M4) register their projection methods against `moat_scorecard.py`; #56 renders the MOAT verdict from the schema. All open, awaiting plan/approval. They share the EXTERNAL analysis box 43511290.

## Last tick
2026-07-02T16:46:40+10:00 · EXP-45 verdict=PASS logged (M4) · analysis box 43511290 EXTERNAL/operator-managed (NOT torn down) · EXP-58 box 43387501 HELD (do-not-teardown, keepalive active) · #47/#48/#49/#56 unblocked

## Budget
EXP-45: $0 GPU-hours (GPU-free replay on operator's $0.11/hr big-disk box; not billed to this issue).
EXP-58: $3.03/hr on 1×H200, held. Backstop = 96 GPU-hr budget cap (heartbeat reaper neutralized by keepalive per operator hold).
