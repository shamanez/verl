# Research Status — 2026-07-04T07:50Z

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 60 | MOAT: Big-Math (EXP-58) cross-dataset projector validation + longer-horizon | **DONE** | EXTERNAL box (CPU-only, GPU-free; no RUNNING row) | **PASS** | carry_forward=FALSE, generalizes=NO (**DATASET-SPECIFIC**): on Big-Math NO projector beats do-nothing — GSM8K-winning fixed #47 damped-linear collapses to λ*=0.0 → ratio 1.0000, pred_evr −0.0012 (rank 18/22); `hold_stale` effectively best; best #49 arm (armD W5 one_over_t) edges do-nothing by only −0.0001 ratio / +0.0002 EVR (< 0.01 kill-rule); margin vs GSM8K winner +0.0604 WORSE (1.0000 vs 0.9396). Mechanism: consec_delta_cos≈0.150 (near-orthogonal updates). All 18 boxes checked; SELFTEST GO; structure_partition 338; full-fidelity fast==full (decision unchanged); report.html offline. Feeds #56; #56 NOT closed. label status:pass; milestone:M4 |
| 49 | MOAT: Self-correcting ANCHOR projector from retrospective weight feedback | **DONE** | EXTERNAL box 202.122.49.242 (CPU-only, GPU-free; no RUNNING row) | **PASS** | carry_forward=FALSE (prefer-simple completed negative, mirrors #48): best method armA `rolling_ls_k` K=5 (bounded-K linear LS, state=1) op 0.9351 / pred_evr +0.1205 beats #47 fixed bar 0.9396/+0.1171 AND expanding adaptive incumbent — but sub-threshold: ratio Δ −0.0045 (needs ≤−0.01), h_safe Δ 0 (needs >0), ratio<1 yes, pred_evr Δ +0.0034 yes. #56 keeps the simpler fixed #47 damped-linear rule. SELFTEST GO 32 invariants; SCHEMA GO all 38 dirs box+laptop; report.html offline. label status:pass; milestone:M4 |
| 48 | MOAT: Second-order projection check only, no cubic/poly search | **DONE** | EXTERNAL box 202.122.49.242 (CPU-only, GPU-free; no RUNNING row) | **PASS** | carry_forward=FALSE (valid completed negative): fixed 2nd-order does NOT beat #47 damped-linear OOS — op-point ratio delta +0.2086 (0.9396→1.1482), h_safe delta −28 (30→2), regresses >1.0; NO structural group helped; full-fidelity confirm max \|Δ\|≈0.0039 (decision unchanged). #56 keeps the simpler linear rule; label status:pass; milestone:M4 |
| 58 | Big-Math 1000-step GRPO: fp32 weights + full ckpts every 20 → R2 | **DONE** | 1×H200 (i_43387501, team) → TORN_DOWN | **PASS** | 50/50 ckpts + 50/50 fp32 weights verified:true in R2; dry_restore@1000; issue CLOSED; PR shamanez/verl#20; milestone:M4 |
| 47 | MOAT: Linear projection baseline + best Δ for ANCHOR (analysis) | **DONE** | EXTERNAL box 43511290 (CPU-only, GPU-free; no RUNNING row) | **PASS** | R²=0.535 (btwn SFT 0.426 & RL 0.845), ρ=−0.75; OOS-damped 0.940 beats naive 1.158 & hold-stale; best_δ=5; h_safe 30/40; paper_linear arm; SCHEMA GO box+laptop; label status:pass; milestone:M4 |

## Open lanes (not yet terminal)
- #59 (status:approved, kind:experiment, M4) LA-WARMUP-V2 — fast-owned Q + no-correct warmup + earliest-legal projection.
- #56 (kind:analysis, M4) MOAT projector verdict rollup — consumes #47/#48/#49/#60 outputs (ALL terminal PASS; #60 adds the cross-dataset generalization negative → #56 fully unblocked).

## Last tick
2026-07-04T07:50Z · running=[] · analyzing=[] · logging=[] · blocked=[] · terminal=[EXP-60 PASS, EXP-49 PASS, EXP-48 PASS, EXP-58 PASS, EXP-47 PASS]

## Standing infra
- EXTERNAL analysis box `202.122.49.242` (RTX A4000, ledger status EXTERNAL) — **must-keep** shared analysis substrate for #45/#47/#48/#49/#56/#60 (holds EXP-57 fp32 trace + EXP-58 Big-Math fp32 trace + panel/stats caches). Operator-managed; NOT torn down, NOT reaped (never register RUNNING). Left untouched by #60 (Big-Math raw EXP-58 tensors staged; MOAT-45/47/48/49 + EXP-57 caches re-confirmed present after the raw-trace delete).
- Legacy box `43511290` (RTX A4000) referenced by earlier lanes; caches confirmed on `202.122.49.242`.
- Team account: 0 instances (clean).

## Budget
$/hr now: private analysis box only (team=$0) · EXP-60 GPU-hours: 0 (kind:analysis, GPU-free offline replay)

## Goal note
EXP-60 (Big-Math cross-dataset projector validation) terminal — PASS as a valid completed result: **weight-projection is DATASET-SPECIFIC** (carry_forward=FALSE, generalizes=NO). On Big-Math no projector beats do-nothing — the GSM8K-winning fixed #47 damped-linear collapses to λ*=0.0 (ratio 1.0000, pred_evr −0.0012, rank 18/22); hold_stale is effectively best; best #49 arm edges do-nothing by <0.01. Mechanism: near-orthogonal consecutive updates (consec_delta_cos≈0.150) vs GSM8K's aligned path. Feeds #56's cross-dataset section (#56 NOT closed). All four MOAT projector lanes (#47/#48/#49/#60) now terminal PASS → #56 verdict rollup fully unblocked. #59 approved experiment awaits execution.
