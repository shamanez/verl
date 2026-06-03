# PROGRESS — append-only audit (fresh cycle)

Structured one-line events appended by the harness agents (triage, runner,
monitor, analyst, log-writer) and the Stop hook. Newest at the bottom. Reset
for a fresh cycle — project state lives in `LOG.md` / `runs/SUMMARY.md`, the
curated status in `.claude/state/STATUS.md`.
[2026-06-03T01:17:51+10:00] [orchestrator] EXP-18 FLOOR done (rc=0, 50/50, flat reward mean 0.135 range 0.111-0.164 = inert-by-orthogonality CONFIRMED); both reference curves cached (dense TARGET 0.135->0.868, spectral FLOOR ~0.135). OOM fix (18432) validated end-to-end. Dispatched runner-c1 (bg): branch exp/18-anchorinject-c5d5, 3-file inject patch, validate, push, launch C1 on reused box 39132674. Added reusable scripts/cell_watch.sh. running=[18 C1-launching] analyzing=[] blocked=[]
