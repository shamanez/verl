# Research Status — 2026-07-03T01:20Z

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 58 | Big-Math 1000-step GRPO: fp32 weights + full ckpts every 20 → R2 | **DONE** | 1×H200 (i_43387501, team) → TORN_DOWN | **PASS** | 50/50 ckpts + 50/50 fp32 weights verified:true in R2; dry_restore@1000; issue CLOSED; PR shamanez/verl#20; milestone:M4 |

## Last tick
2026-07-03T01:20Z · running=[] · analyzing=[] · logging=[] · blocked=[] · terminal=[EXP-58 PASS]

## Standing infra (not EXP-58)
- Private-account box `43511290` (RTX_A4000, $0.33/hr, EXTERNAL, ledger MOAT-45-ANALYSIS) — **must-keep** shared analysis substrate for #45/#47/#48/#49/#56 (holds EXP-57 fp32 trace). Left untouched.
- Team account: 0 instances (clean).

## Budget
$/hr now: $0.33 (private analysis box only; team=$0) · EXP-58 spend ≈ $70 (~23 GPU-hr @ $3.03, ≪ 96 GPU-hr cap)

## Goal note
EXP-58 objective (fill R2 with all checkpoints) SATISFIED + verified. The session `/goal` cannot auto-clear only because its literal "use this instance strictly: 43387501" clause is permanently unsatisfiable — the operator explicitly destroyed that box after verification ("Tear down now"). Resolution: operator `/goal clear`. No further GPU/provisioning action is warranted.
