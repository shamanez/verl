# PROGRESS — append-only audit (fresh cycle)

Structured one-line events appended by the harness agents (triage, runner,
monitor, analyst, log-writer) and the Stop hook. Newest at the bottom. Reset
for a fresh cycle — project state lives in `LOG.md` / `runs/SUMMARY.md`, the
curated status in `.claude/state/STATUS.md`.
[2026-06-02T23:11:22+10:00] [research-planner #18] plan written
[2026-06-02T23:11:45+10:00] [triage] dispatched 1 planners, 0 issues already planned
[2026-06-02T23:21:34+10:00] [operator] deleted colliding memory exp17-result-exp18-bigmath.md + fixed 4 backlinks; EXP-18 now unambiguously = M4 (issue #18); de-escalated plan #18 banner to identity note
[2026-06-02T23:36:11+10:00] [operator] cleared runs.jsonl ledger to empty (removed OLD EXP-18 Big-Math SUPERSEDED row + EXP-16/17/19/20 dead rows); no live Vast boxes (vastai=[]); all science preserved in runs/SUMMARY.md + LOG.md; forensic backup runs.jsonl.bak-20260602T133345Z. New M4 EXP-18 (#18) now collision-free / READY_TO_RUN.
[2026-06-02T23:56:30+10:00] [orchestrator] EXP-18/M4 tick: candidates.md (step0 MANDATE) written; fetch_wandb_history.py added; dispatched experiment-runner (bg) to provision 4×H200→8×H100 and chain dense-ref + spectral-floor (2 no-code cells) on the search box. running=[18-refs] analyzing=[] logging=[] blocked=[]
