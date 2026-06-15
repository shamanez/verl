# Research Status — 2026-06-16 (EXP-31 tournament DONE — VERDICT STOP)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 31 | Anchor-signal-usage tournament (L4/L2/L3/L1) | DONE / VERDICT STOP | 1×4H200 (i_41048644, operator box — tournament complete) | STOP | B2_live val@25=0.7202 / val@50=0.7354; dense-this-box=0.7506 (band 0.75–0.78). All 4 levers NULL: L4 σ=0.01→0.7157, L2 μ=0.9→0.5701 (regress) / μ=0.5→0.7089, L3 ratio→0.7119 / cos→0.7134, L1 skipped (gate F1 fails). HEADLINE surpass (≥0.78) FALSIFIED. Mechanistic close: δ reconstructs dense-on-stale-data; no admissible lever provides signal dense lacks. |
| 28 | EXP-28 TRUE error-feedback successor | PLAN_READY? (kind:experiment, no status label) | — | — | not approved; out of scope this drive |

## EXP-31 cell summary (CLOSED — tournament STOP)

- **Cell A (B2_live reproduce)** — DONE. val@25=0.7202 / val@50=0.7354; bytes_ratio=0.0504; no ignition. W&B fy920fty.
- **L4 perturbation σ=0.01** — DONE. val@25=0.7157. NULL (parity; isotropic = regularization control, not anchor-usage). W&B (env-only arm).
- **L2 δ-momentum μ=0.9** — DONE. val@25=0.5701. REGRESS (over-smoothed; −0.15 vs B2_live). W&B ybemd5ux.
- **L2 δ-momentum μ=0.5** — DONE. val@25=0.7089. NULL (parity). W&B knlzxh2x.
- **L3 adaptive dose ratio κ=1.0** — DONE. val@25=0.7119. NULL (parity). W&B kzohyuod.
- **L3 adaptive dose cos κ=1.0** — DONE. val@25=0.7134. NULL (parity). W&B wmpmmdj1.
- **L1 control-variate** — SKIPPED. Gate F1 fails: cov(G_comp,M)≈0 ⇒ no variance to cancel. No L2/L3 surpass signal to gate on.
- **L4 σ=0.03 / κ=0.5** — NOT run (monotone-by-cap; no path to surpass after all levers null).

## Box status

- **i_41048644** (EXP-31 tournament box, 4×H200, operator acct): tournament complete. Operator-managed teardown.

## Last tick

2026-06-16 · running=[] · analyzing=[] · logging=[31 tournament STOP] · blocked=[]

## Budget

Tournament closed. No active billing box (operator acct; teardown operator-managed).
