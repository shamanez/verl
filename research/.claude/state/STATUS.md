# Research Status — 2026-07-02T20:15+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 47 | MOAT: Linear projection baseline + best Δ for ANCHOR (analysis) | RUNNING (cache builds) | EXTERNAL box 43511290 (no ledger RUNNING row — reaper-proof) | — | harness impl DONE, box SELFTEST GO (18 invariants), regime-T+S cache builds launched under nohup (~4-5h) |

## EXP-47 progress (analysis kind, code_change=true)
- ✅ Implemented #47 additive harness in `research/scripts/moat_scorecard.py` (+753 lines): `damped_linear` (OOS walk-forward λ selector, leakage-guarded), `--cadence per-step/per-tick` reindex + fingerprint (SCHEMA_VERSION), per-scalar linearity R² accumulator+reduction+persist, `paper_linear` direct-scored arm (Wang et al. Eq.4/App.E.1), extended Δ→40, new visuals (λ-selection, R² hist, depth×block R² heatmap, R²-vs-ratio coupling, paper-equivalence panel), 9-key REQUIRED_ROW_KEYS superset, 7 new self-test invariants, verify-schema updates.
- ✅ Local SELFTEST: GO (synthetic manifest) + full end-to-end dry-run on a tiny 338-matrix/160-tick local trace (both regime emits EMIT: GO, cache BUILD→HIT, SCHEMA: GO both).
- ✅ rsync to box + **box SELFTEST: GO** (18 invariants incl real-trace subset: hold_stale identity 0.0, off-path parity 0.0 on real q_proj/down_proj, determinism byte-identical).
- ✅ Cache builds LAUNCHED (nohup, PID 3872 regime-T streaming; driver 3870): regime T (per-tick band-80) → regime S (per-step band-60 + paper_linear), sequential, → `runs/MOAT-47-ANALYSIS/analysis.log`.
- ⏳ IN FLIGHT (~4-5h): regime-T build (chunk 0/25), then regime-S build + paper direct pass.
- ⬜ TODO: author `moat_report.py` (off critical path, do while builds run); schema-verify both (box); rsync to laptop; schema-verify (laptop); render report.html; write verdict.md; LOG.md + SUMMARY.md.

## Last tick
2026-07-02T20:15+10:00 · running=[47 cache-builds] · analyzing=[] · logging=[] · blocked=[]

## Budget
EXP-47 is GPU-free on operator EXTERNAL box 43511290 ($0.11/hr, operator-managed, NEVER torn down by this issue). Zero GPU-hours. No harness provisioning.
