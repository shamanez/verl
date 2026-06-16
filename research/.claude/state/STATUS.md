# Research Status — 2026-06-17 (EXP-33 DONE · PASS · box torn down)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 33 | β_anc EMA sweep {0,.25,.5,.75,1} on B2 delayed_ef | **DONE** | 1×4H200 (i_41194490, torn down) | **PASS** | All 5 cells complete. β→accuracy curve flat: C0 0.73844 / C1 0.73995 / C2 0.75284 / C3 0.72176 / C4 degenerate (cold-M collapse). Max gap C2 +0.0144 < 0.024 bar — freshness-best hypothesis SUPPORTED. β=0 stays the default. No PR (code_change=false; promote_launcher_as=none). |
| 32 | signed_ema α=0.5 on valid-M | DONE | operator (op-managed) | (closed status:done) | result 0.7271 < B2 0.7528 |
| 31 | anchor-usage 4-lever tournament | DONE | — | STOP | all-null for surpass; B2=SOTA |

## Current state

No running experiments. No provisioned boxes. No pending analyses.

**SOTA** = B2 (`delayed_ef`, λ=1, β_anc=0, PowerSGD r=77, anchor circuit, replay_paired_batch=true).
GSM8K greedy val@50 ≈ 0.74–0.75 = statistical PARITY with dense at ~5% gradient-comm cost.
Ground truth: `runs/EXP-31/B2_baseline/resolved_params_B2.txt`.

## EXP-33 final cell ledger (COMPLETE)

| cell | β_anc | val@25 | val@50 | gap vs C0 | verdict |
|---|---|---|---|---|---|
| C0 b0p00 | 0.00 | 0.71418 | **0.73844** | — (control) | CONTROL PASS, B2 band |
| C1 b0p25 | 0.25 | 0.71418 | **0.73995** | +0.00151 | TIE (within ±0.024) |
| C2 b0p50 | 0.50 | 0.70811 | **0.75284** | +0.01440 | TIE / nominal peak |
| C3 b0p75 | 0.75 | 0.70053 | **0.72176** | −0.01668 | TIE / mild down |
| C4 b1p00 | 1.00 | 0.44807 | — (30-step bracket) | n/a | cold-M collapse confirmed (196/196 fallbacks → plain PowerSGD) |

bytes_ratio all cells: 0.0504–0.0506 (gate [0.0500,0.0510]). recon_rel_error ss≈0.025 (act band). No NaN/OOM/ignition on any cell.

## Box

i_41194490 · **TORN DOWN** (verified 0 live instances via `vastai show instances`). Torn down 2026-06-17T04:45:57+10:00.

## Last tick
2026-06-17 · running=[] · analyzing=[] · logging=[done] · blocked=[]

## Open issues / next steps

No open experiment issues. EXP-33 β sweep closed: β_anc axis exhausted (flat free-averaging region, no improvement). The anchor-usage (EXP-31 tournament) and β_anc (EXP-33 sweep) axes are both NULL. Awaiting new issue from operator for the next research direction.
