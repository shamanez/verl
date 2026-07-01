# Progress

Durable record: `runs/SUMMARY.md` · `runs/FIXED_CONTROL_SURFACE.md` · `reports/*.html` · W&B · git.
North-star: `.claude/GOAL.md`. Per-run verdicts: `LOG.md`. Tick history pruned (recoverable from
git + LOG.md); the harness appends new ticks below.

**Base:** `signed_ema` (α=0.25, β_anc=0.50), fast 1K surface, HIGH anchor latency
(cadence/delay_K=20/20, the k-collapse regime), locked PowerSGD r=77 anchor. Values:
`runs/FIXED_CONTROL_SURFACE.md`.

**Two priorities:** (1) M4 — solve the k-collapse by projecting WEIGHTS; the dense weight trajectory
is collected → R2; GPU-free analysis spine #44–#56, entry **#44**. (2) M6 — shrink the ~0.04
compression train–inference mismatch.
[2026-07-01T12:58:54+10:00] [research-planner #44] plan written
[2026-07-01T12:59:34+10:00] [triage] dispatched 1 planner (#44), 0 issues already planned; now planned=1 unplanned=0 ALL_PLANNED
[2026-07-01T13:33:47+10:00] [orchestrator] tick: analyzing=[44] (analyst dispatched — build+verify weight_proj_sweep.py engine, GPU-free, streaming EXP-43 R2 trace) running=[] logging=[] blocked=[]; teardown sweep no-op (empty ledger)
[2026-07-01T14:28:29+10:00] [orchestrator] EXP-44 manifest FIX: restored full 160-row full_manifest+r2_manifest from commit 6a2f9255 (prior run streamed a truncated 5/4-row stub); re-dispatched analyst (background) to reuse the already-built engine, run invariants→noise-floor→full-sweep→verdict on the corrected trace. Teardown sweep no-op (empty ledger).
[2026-07-01T15:09:02+10:00] [orchestrator] OPERATOR HALT: 'stop everything' — EXP-44 analyst (a11443b8) terminated mid Step-2 (noise-floor gate); Step-1 invariants had PASSED (0 hard-fails). Streaming procs killed, staging clean, both Vast accounts $0/hr (0 instances, untracked=0), no box teardown needed. Engine files (scripts/weight_proj_sweep.py + weight_proj/) remain on disk uncommitted; manifest fix (160-row) intact. No verdict written. Awaiting operator.
