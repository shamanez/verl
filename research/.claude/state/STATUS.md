# Research Status — 2026-05-30T05:05:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 16 | Short-run stability matrix (mask/rescale/clean-cadence/spectral) | READY_TO_RUN — SUPPLY_BLOCKED (poller live) | 1×4H200 (i_38444745) TORN_DOWN | — | status:approved, code_change:true. SSH key rotated to `~/.ssh/vast_ai_name` (id 890294) per operator — fixes the H200 key-injection env-failure. Both tiers still dry this tick (4×H200=0, 8×H100=0, all-H100=0 market-wide). Background poller `.claude/state/supply-poll.sh` watching both tiers @90s; on first offer it exits → orchestrator dispatches experiment-runner. exp branch `exp/16-short-run-stability-matrix`@928bd22a + payload intact. |
| 11 | M3 — 100-step M95+AP GRPO vs dense baseline, K=20 | NOT_CLAIMED | — | — | kind:experiment, milestone:M3; no research:claim/status/plan. Blocked on M2 (EXP-16). Out of orchestrator scope. |
| 10 | M3 — DP gradient compression (PowerSGD-64 + Streaming-DiLoCo) scope | NOT_CLAIMED | — | — | kind:experiment, milestone:M3; gated behind M95+AP smoke. Out of orchestrator scope. |

`baseline` (dense control, `.claude/plans/baseline.md`) is a design template, not a gating EXP-run.

## SSH key rotation (this tick — operator-directed)

Operator created a fresh key `~/.ssh/vast_ai_name` (registered on Vast as **id 890294**, name `vast_ai_name`) and directed the harness to switch to it. The legacy `~/.ssh/vast_ai` (id 835115, `vast-ai-key`) failed container `authorized_keys` injection on H200 instance 38444745 (the EXP-16 env-failure). Verified: the new private key derives exactly the registered pubkey.

Updated all live config (historical records left as-is):
- `project.yaml` `vast_ssh.identity_file` + `pubkey_file` → `vast_ai_name`(.pub)
- `vast-provision/run.sh` → `SSH_IDENTITY="${VAST_SSH_IDENTITY:-~/.ssh/vast_ai_name}"`, used in the handle `ssh_login`
- `vast-provision/SKILL.md`, `experiment-runner.md`, `training-log-monitor.md` docs
- `hooks/sync-metrics.sh` fallback default (it already prefers project.yaml's value)

Both keys remain registered, so `vastai create instance --ssh` injects both and a fresh box accepts either; the harness now OFFERS `vast_ai_name`. Legacy key kept as fallback.

## Untracked / gone instances

- `i_38447289` (1×RTX_2060, operator-owned) — leave alone (per direct instruction).
- `i_38448803` (4×A100, $7.56/hr, ip 23.127.144.217) — the operator's box they SSH'd into with `vast_ai_name`; now **gone** (confirmed absent this tick — `vastai show instances` empty). No harness instances live.

## Last tick
2026-05-30T05:05:00+10:00 · running=[] · analyzing=[] · logging=[] · blocked=[16→SUPPLY_BLOCKED, poller live] · skipped=[11,10 not-claimed]

## Budget
$/hr now: $0 (no instances live) · account credit remaining: ~$1018.54

## Notes
- Kill switch clear (`~/.claude-kill-switch` absent).
- gh default repo: `shamanez/verl-compression-research` (issue queue). Code PRs target `shamanez/verl` base `vast-ai-workload`.
