# Research Status — 2026-07-04T04:32Z

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 49 | MOAT: Self-correcting ANCHOR projector from retrospective weight feedback | **DONE** | EXTERNAL box 202.122.49.242 (CPU-only, GPU-free; no RUNNING row) | **PASS** | carry_forward=FALSE (prefer-simple completed negative, mirrors #48): best method armA `rolling_ls_k` K=5 (bounded-K linear LS, state=1) op 0.9351 / pred_evr +0.1205 beats #47 fixed bar 0.9396/+0.1171 AND expanding adaptive incumbent — but sub-threshold: ratio Δ −0.0045 (needs ≤−0.01), h_safe Δ 0 (needs >0), ratio<1 yes, pred_evr Δ +0.0034 yes. #56 keeps the simpler fixed #47 damped-linear rule. SELFTEST GO 32 invariants; SCHEMA GO all 38 dirs box+laptop; report.html offline. label status:pass; milestone:M4 |
| 48 | MOAT: Second-order projection check only, no cubic/poly search | **DONE** | EXTERNAL box 202.122.49.242 (CPU-only, GPU-free; no RUNNING row) | **PASS** | carry_forward=FALSE (valid completed negative): fixed 2nd-order does NOT beat #47 damped-linear OOS — op-point ratio delta +0.2086 (0.9396→1.1482), h_safe delta −28 (30→2), regresses >1.0; NO structural group helped; full-fidelity confirm max \|Δ\|≈0.0039 (decision unchanged). #56 keeps the simpler linear rule; label status:pass; milestone:M4 |
| 58 | Big-Math 1000-step GRPO: fp32 weights + full ckpts every 20 → R2 | **DONE** | 1×H200 (i_43387501, team) → TORN_DOWN | **PASS** | 50/50 ckpts + 50/50 fp32 weights verified:true in R2; dry_restore@1000; issue CLOSED; PR shamanez/verl#20; milestone:M4 |
| 47 | MOAT: Linear projection baseline + best Δ for ANCHOR (analysis) | **DONE** | EXTERNAL box 43511290 (CPU-only, GPU-free; no RUNNING row) | **PASS** | R²=0.535 (btwn SFT 0.426 & RL 0.845), ρ=−0.75; OOS-damped 0.940 beats naive 1.158 & hold-stale; best_δ=5; h_safe 30/40; paper_linear arm; SCHEMA GO box+laptop; label status:pass; milestone:M4 |

## Open lanes (not yet terminal)
- #59 (status:approved, kind:experiment, M4) LA-WARMUP-V2 — fast-owned Q + no-correct warmup + earliest-legal projection.
- #56 (kind:analysis, M4) MOAT projector verdict rollup — consumes #47/#48/#49 outputs (ALL THREE now terminal PASS; #56 unblocked).

## Last tick
2026-07-04T04:32Z · running=[] · analyzing=[] · logging=[] · blocked=[] · terminal=[EXP-49 PASS, EXP-48 PASS, EXP-58 PASS, EXP-47 PASS]

## Standing infra
- EXTERNAL analysis box `202.122.49.242` (RTX A4000, ledger status EXTERNAL) — **must-keep** shared analysis substrate for #45/#47/#48/#49/#56 (holds EXP-57 fp32 trace + panel/stats caches). Operator-managed; NOT torn down, NOT reaped (never register RUNNING). Left untouched by #49 (arm panel caches deleted post-rsync — regenerable, cache-fingerprint-keyed).
- Legacy box `43511290` (RTX A4000) referenced by earlier lanes; #49 confirmed the trace + caches live on `202.122.49.242` (stale box ignored).
- Team account: 0 instances (clean).

## Budget
$/hr now: private analysis box only (team=$0) · EXP-49 GPU-hours: 0 (kind:analysis, GPU-free offline replay)

## Goal note
EXP-49 (self-correcting ANCHOR projector kill-gate) terminal — PASS as a valid completed negative (carry_forward=FALSE, prefer-simple mirroring #48): bounded-last-K=5 rolling-LS (state=1) is the best single-scalar variant and beats both the #47 fixed bar and the expanding adaptive incumbent OOS, but sub-threshold → #56 keeps the simpler fixed #47 damped-linear rule. All three MOAT lanes (#47/#48/#49) now terminal PASS → #56 MOAT verdict rollup unblocked. #59 approved experiment awaits execution.
