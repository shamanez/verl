# Research Status — 2026-06-18 (EXP-35 DONE → REVISE; box held warm)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 35 | signed_ema α-sweep {0,.25,.5,.75,1} @ β_anc=0.50 (accel 4×H200) | **DONE** | 1×4H200 (i_41420622, team, **HELD WARM**) | **REVISE** | 5/5 cells; σ(M)-parity confirmed (peak α=0.25=0.7528); α=0.0 NO ignition (headline). 2 housekeeping boxes unchecked (H100 mem-band on H200; C1-ignition clause falsified). All step-50 rows backfilled to W&B. |

## EXP-35 α-curve (val@50)
α=0.0 → 0.7437 · α=0.25 → **0.7528 (peak)** · α=0.5 → 0.7415 (gate ∈[0.7395,0.7875]) · α=0.75 → 0.7043 · α=1.0 → 0.6475 (floor). bytes_ratio ≈0.0505 all cells. ~25 min/50 steps (vs ~2 hr locked surface). CROSS-SURFACE: absolute ranking vs B2/dense needs a dense+B2 re-baseline on THIS accel surface (not yet run) — only the RELATIVE curve shape is interpretable; promote nothing to the SOTA card yet.

## Box 41420622 — HELD WARM (operator choice)
- Reused operator 4×H200, team account, `-i ~/.ssh/vast_ai -p 40264 root@84.8.106.109`. ~$12.88/hr, **idle**.
- Ledger row flipped RUNNING→COMPLETE so the teardown Stop hook SKIPS it (hook ignores reused_box; the analyst's verdict would otherwise have reaped it). Heartbeat-stale teardown also disabled by the COMPLETE state. **Will NOT auto-teardown** — operator owns its lifecycle.
- Earmarked for the gated work below. Operator chose "keep warm for gated work" (NOT auto-launch); awaiting explicit go.

## Gated next steps (operator go required; box ready)
1. **Speed-knob bit-identical canary** — `comm_eff.spectral.diagnostics` true-vs-false on one cell, prove ON==OFF trajectory. Branch `exp/spectral-diagnostics-knob` @ 3300cc61 (authored, reviewed, 293 tests pass).
2. **Dense baseline on the accel surface** (comm_eff OFF) — the absolute reference EXP-35's α-curve lacks; also satisfies REVISE next_action #1 (re-baseline) + the analyst's promotion precondition.

## EXP-35 REVISE next_actions (from verdict.md)
- a) box/budget calibration: H200 @ ppo_max_token_len 24576 → ~124GB (outside H100 band [68,76]); either run 4×H100 OR lower budget to ~12288 on H200. (Overlaps the dense baseline / a provisioning fix, not science.)
- b) low-α replication {0, 0.125, 0.25, 0.375} × 2–3 seeds — confirm α=0.25 peak is real + formally re-bin the C1-no-ignition finding. (Optional follow-up; NOT yet spawned as a child issue — deferring to operator.)

## Background tasks
- analyst (EXP-35) — DONE (REVISE).
- All monitors — DONE (sweep complete).
- Speed-investigation workflow + diagnostics-knob authoring — DONE.
