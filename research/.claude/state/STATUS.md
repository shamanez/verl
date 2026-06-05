# Research Status — 2026-06-06T09:22Z

## Active
No active experiments. EXP-25 complete (STOP). Box 39613656 TORN_DOWN and verified absent.

## Issue pipeline
| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 25 | Anchor-circuit default | DONE/STOP | 4×H200 i_39613656 (torn down) | STOP | best-α=0.5 val@50=0.7066 ≤ 0.7114 STOP threshold; signed_ema sign-reversal net-harmful; #24 blocked. |
| 24 | Error-feedback + basis-aligned anchor | BLOCKED | — | — | `depends_on:#25 PASS` — #25 returned STOP; correction primitive must be redesigned before #24 starts. |

## EXP-25 summary (STOP)
- id-0 (anchor M / R1): **PASS** · id-1 (anchor-owns-Q R2 + signed_ema R3): **PASS** — both probe gate sets green; implementation correct.
- id-2 α-sweep COMPLETE: α=0.0 val@50=0.354 (catastrophic collapse) · α=0.3 val@50=0.616 (delayed collapse) · α=0.5 val@50=0.7066 (stable, best, still below STOP threshold 0.7114).
- Verdict: FALSIFIED — dose-response monotonic; signed_ema sign-reversal primitive is net-harmful. Root cause documented in `runs/EXP-25/DEEP_FINDINGS.md`.
- Standing entropy-collapse watch: `research/diagnostics/ENTROPY_COLLAPSE_WATCH.md` (T1–T7 triggers).
- No PR (STOP verdict). No launcher promotion (`promote_launcher_as: none`).
- In-container hotfix patch: `runs/EXP-25/hotfix-patches/BACKUP-uncommitted-box-diff.patch` (already on `vast-ai-workload`).
- Next: operator decides new lineage — candidate knobs: α→{0.7,0.85,1.0} sweep, entropy/KL regularizer, or correction_primitive redesign (error-feedback on the PowerSGD residual, see verdict next_actions).

## Last tick
2026-06-06T09:22Z · running=[] · analyzing=[] · logging=[25 DONE] · blocked=[24 dep#25-PASS]

## Budget
EXP-25 final: ~8.5 GPU-hr of 48 max (4 Vast instances provisioned; 2 immediately TORN_DOWN due to Vast SSH key-injection env-failure; active instance 39613656 ran 3×arms + 2×probes ≈ ~8h of training). All instances torn down, verified absent.
