# Research Status — 2026-06-06T04:10+10:00

## Active
**EXP-25** anchor-circuit default α-sweep LIVE on warm box **39613656** (operator-pinned `-p 40872 root@46.243.55.155`). Both probe gates PASS; 4-circuit deep audit CLEAN. **NEW: α=0.0 arm shows an entropy-collapse / length-degeneration spiral** — root-caused to the α=0 signed_ema merger (`|G_noisy|·sign(M)` = sign-SGD); dedicated team `entropy-collapse` documented it (findings + standing watch). DO-NOT-PROVISION.

## Issue pipeline
| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 25 | Anchor-circuit default | RUNNING | 1×4H200 (i_39613656, pinned -p40872) | — | id-0+id-1 PASS; α=0.0 ~step48/50 (collapsed); α=0.3/0.5 pending; winner→100. |
| 24 | Error-feedback + basis-aligned anchor | BLOCKED | — | — | `depends_on:#25` (needs PASS). |

## EXP-25 sequence
- id-0 (anchor M / R1): **PASS** · id-1 (anchor-owns-Q R2 + signed_ema R3): **PASS**
- 4-circuit deep audit (anchor grad/DP/EMA, Q ownership/bcast, merger, fast/parity/config): **CLEAN** (LOW/INFO punch-list only; pt-5b scale by proxy).
- **Unit nuance**: anchor.cadence/delay_K count OPTIMIZER TICKS (2/global-step) ⇒ refresh every ~2.5 global steps + 2.5-step staleness (held fixed → no α-confound; documented for analyst).
- id-2 α-sweep: **α=0.0 ~step 48/50** — val@25=0.718; **ENTROPY COLLAPSE** (entropy 5.69→0.06, resp_len→~8600, reward 0.79@28→0.32@45). α=0.3 + α=0.5 pending. Monitor watching for val@50 + handoff.
- analyst verdict + log-writer + teardown: pending sweep completion.

## Entropy-collapse investigation (team `entropy-collapse`, member `entropy-analyst`)
- Root cause: α=0 merger `|G_noisy|·sign(M)` = magnitude-preserving sign-SGD w/ persistent β=0.95 EMA signs → no cancellation → sharpening; collapse onset pinned to merger_coldM_fallbacks 196→0 @step3; warm rel_change median=√2 (≈50% sign disagreement/step). Isolated by 4 anchor-OFF control runs (no collapse).
- Prediction (falsifiable, to test on the sweep): severity α=0 ≫ α=0.3 > α=0.5, phase transition at α≈0.5.
- Deliverables: `runs/EXP-25/ENTROPY_COLLAPSE_FINDINGS.md` + standing `research/diagnostics/ENTROPY_COLLAPSE_WATCH.md` (T1–T7 triggers, reusable on EVERY run).

## Last tick
2026-06-06T04:10+10:00 · running=[25] · analyzing=[entropy-collapse done] · logging=[] · blocked=[24 dep]

## Budget
1 live box 39613656 (4×H200, $15.23/hr) · max_gpu_hr=48 · started 00:44:45 (~3.4h elapsed). Tear down at verdict / SSH-unreachable. NEVER reprovision (operator mandate).
