# Research Status — 2026-06-01T18:55:00+10:00 (wind-down + theory phase)

## Compute: ALL TORN DOWN — $0/hr
Instance 38877541 (4×H200) destroyed (`vast-teardown`, destroyed=1). No live boxes.

## Runs (all complete; box reused EXP-17→18→19→20 then torn down)
| EXP | dataset | method | result | ledger |
|---|---|---|---|---|
| 17 | GSM8K | mask p=0.9 + clean@20 | val 0.085→**0.735** ≈ dense (0.741) — PARITY @95% comm cut | COMPLETE / PASS |
| 18 | Big-Math | mask clean@20 (math_dapo reward) | SUPERSEDED — confounded reward | SUPERSEDED |
| 19 | Big-Math | mask p=0.9 + clean@20 (fixed reward) | val flat ~0.55 — stalls | COMPLETE |
| 20 | Big-Math | **dense** | val 0.558→~0.59–0.61 — modest real climb | COMPLETE |
| base | — | no RL, \boxed | GSM8K **0.715** vs Big-Math **0.480** | (eval) |

**Headline:** GSM8K is easy for Qwen2.5-1.5B (base 0.715) → RL just elicits → 95%-compressed
masked+clean@20 reaches dense parity. Big-Math is hard (base 0.48) → dense learns, lossy masked
gradient stalls. Mask loss is tolerable for elicitation, fatal for learning.

## Cleanup status (operator-directed)
- ✅ Big-Math in inventory (runs/SUMMARY.md updated; also corrects the old "clean_cadence unsustainable" claim).
- ✅ Git synced: vast-ai-workload @ 1eddb0f61 pushed, remote↔local 0/0. (verl/ math_bigmath route @265fca825 — UNUSED/buggy, kept; bigmath scripts + eval + theory findings committed.)
- ✅ Box torn down (before this: all train.logs + base eval + trajectories captured locally).
- ⏳ **Reports** (must finish BEFORE branch cleanup): theory team REPORT.md pending.
- ⏳ **Branch cleanup** (after reports): origin `exp/16-short-run-stability-matrix`, `exp/17-masked-clean-every20`; local stale `worktree-agent-*`. NOT yet deleted (waiting on reports).

## comm-eff-theory agent team (the active goal)
Lead = me. Teammates (background): `theorist` ✅ wrote findings/theory/theory.md (28KB, rigorous),
`lit-scout` (RLVR noisy/spurious-reward + elicitation literature → literature.md), `empiricist`
(verify theory vs our numbers → empirical_check.md). Next: spawn `synthesizer` → REPORT.md once the
3 inputs land. Brief: findings/theory/CONTEXT.md.

## Next actions (orchestrator)
1. Await lit-scout + empiricist completions (auto-notified).
2. Spawn synthesizer → findings/theory/REPORT.md (theory + literature + our numbers + GSM8K-easy thesis).
3. Commit REPORT.md; THEN clean exp/16, exp/17 + stale worktree branches; shut down + clean up the team.

## Notes
- Kill switch clear. gh default: shamanez/verl-compression-research. Code PRs → shamanez/verl base vast-ai-workload.
- EXP-19 had a step-50 checkpoint (resumable) — discarded with the box per "we have enough training".
