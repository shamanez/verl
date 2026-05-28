# Research Status — 2026-05-28T16:44:00+10:00 — EXP-9 RUNNING

## Active run

- **EXP-9** (M2 capstone — full M95+AP no-KL 20-step GRPO smoke)
- Instance: `38203582` 4×H200 @ $14.74/hr (tier 0, H200-first chain — no fallback walked)
- SSH: `ssh -i ~/.ssh/vast_ai -p 37334 root@35.130.230.4` · tmux `exp-9-35_130_230_4`
- Branch: `exp/9-m2-final-noKL-maskrecompute-aps` @ `4f76b43b` (pushed to `origin shamanez/verl`)
- Codex-verify: SKIPPED per plan §7 (pre-written `runs/EXP-9/verify/20260528T160000.md` VERIFY: PASS, operator override)
- 56-test CPU suite: 140 PASS / 10 skip / 2 pre-existing skip (unrelated to this branch)

## Monitoring team (3 background watchers, defense-in-depth)

| Member | Role | Cadence | Engine | Status |
|---|---|---|---|---|
| A — training-log-monitor | system health: tmux + SSH log grep + WandB cross-check | 20 s | Opus subagent | RUNNING (background) |
| B — curve-analyst | WandB scalars + 13-criterion checklist + STUCK lines | 5 min | Opus subagent | RUNNING (background) |
| gpu-watchdog (shell loop) | strict per-GPU idle alarm — 60s hard-stall → teardown rec | 15 s | Bash bg | RUNNING (id=bqj7va23j) |

## First-snapshot status (manual SSH probe 06:44:00 UTC)

- Training driver: validate_config PASSED, datasets loaded (7473/1319), `Total training steps: 20`, Qwen2 1.54B FSDP-wrapped × 4 ranks, weights loaded 100%, vLLM workers spawning.
- GPUs: 0% util right now — **expected weight-load + vLLM-init dip**, 8.4 GB memory allocated per GPU, 125 W power draw confirms ranks alive.
- No Traceback / OOM / RayActor / NaN / FSDP error in log.

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 9 | M2 capstone full M95+AP 20-step GRPO | **RUNNING** | 1×4H200 ($14.74/hr) | — | branch pushed, 3 monitors active |
| 12 | REVISE child of EXP-8 anchor isolation | DONE | — | PASS | PR #5 merged |
| 8 | M2 anchor circuit | REVISE → #12 | — | REVISE | lineage closed by #12 |
| 7 | M2 spectral + FSDP grad point | DONE | — | PASS | PR #4 merged |
| 6 | M2 mask contamination guard | DONE | — | PASS | PR #3 merged |
| 5 | M2 actor-only mask | DONE | — | PASS | PR #2 merged |
| 3 | M1 dense GRPO baseline | DONE | — | PASS | milestone:M1 |
| 11 | M3 100-step vs dense | NO_STATUS | — | — | not approved |
| 10 | M3 DP gradient compression | NO_STATUS | — | — | not approved |

## Last tick
2026-05-28T16:44:00+10:00 · verify=[] running=[9 monitors-x3] analyzing=[] logging=[] blocked=[10,11 not-approved]

## Budget
Live instance: i_38203582 $14.74/hr · max_dph=$24 max_gpu_hr=12 · started 06:38 UTC → cap ~$58 if hits 4h wall.
