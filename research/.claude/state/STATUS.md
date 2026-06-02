# Research Status — fresh cycle (2026-06-02)

> The orchestrator rewrites this file each tick. This is the idle stub.

## Compute: no live boxes ($0/hr).

## State
The comm-eff base is **settled** (mask p=0.9 + rescale + `clean_cadence`). Proven:
masked+clean@K is stable and reaches GSM8K dense parity (elicitation), stalls on
Big-Math (gradient-fidelity limit). Anchor+spectral as implemented are inert
(orthogonality). Canonical docs: `.claude/GOAL.md` → `runs/SUMMARY.md` →
`findings/NEXT_RESEARCH.md`.

## Next
Frontier = redesign anchor+spectral as a cheap continuous corrector. Gating
experiments: **p-sweep** + **clean-only ablation** (`findings/NEXT_RESEARCH.md`).

## Notes
Repo de-bloated this cycle (docs + runs artifacts). Working tree has uncommitted
cleanup on `vast-ai-workload`. Kill switch clear.
