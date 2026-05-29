# EXP-16 Lead Log (crash-recovery state of record)

**Role:** I am the agent-team **lead** for issue #16, acting as the human-supervised
substitute for the orchestrator (per plan routing exception). Operator authorized
"YOU TAKE THE LEADERSHIP AND MOVE FORWARD" on 2026-05-30.

**Plan (contract):** `research/.claude/plans/16.md` — read in full before any action.
**Team:** `exp-16-stability` (`~/.claude/teams/exp-16-stability/`). Task list `~/.claude/tasks/exp-16-stability/`.
**Issue:** https://github.com/shamanez/verl-compression-research/issues/16

## Durable state lives in (NOT in any teammate's context):
- `runs/EXP-16/` (this dir) — handles, per-cell metrics, verdict, pr_body, this log.
- git branch `exp/16-short-run-stability-matrix` (pushed to origin BEFORE provisioning).
- the shared task list on disk.
- the on-box tmux sessions + `done_<cell>.flag` files.

## Hard constraints (from plan):
- ONE Vast.ai box for ALL 6 cells (warm reuse; NO re-provision between cells). Tier chain: 4×H200 (pref) → 8×H100. NO 4×H100 fallback. max_dph=$24, max_gpu_hr=96, ≤$60 total, ≤12h wall-clock.
- STRICTLY sequential cells (max_parallel:1). Never two training jobs at once.
- EVERY cell sets TOTAL_TRAINING_STEPS explicitly (10/10/20/20/20/25). Launcher default=100 (line 205) = budget trap.
- Cell sequence (EXPERIMENT_NAME):
  - cell0: GPU pre-flight cross-pass mask consistency (BOTH rescale settings) — gate, no training. on_fail=STOP.
  - cell1: grpo_mask_channel_p0p9_no_rescale_10steps (unstable control)
  - cell2: grpo_mask_channel_p0p9_rescale_10steps
  - cell3: grpo_mask_channel_p0p9_no_rescale_clean_every4_20steps
  - cell4: grpo_mask_channel_p0p9_rescale_clean_every4_20steps
  - T5 CODE CHANGE (spectral.cadence gate + COMM_EFF_SPECTRAL_CADENCE/ANCHOR_DELAY_K plumbing + numeric-only metrics + early-stop) — LEAD APPROVES change-plan first.
  - cell5: grpo_mask_channel_p0p9_rescale_anchor2_spectral2_20steps (auditor scrutinizes; needs use_orig_params=true)
  - cell6: dense_grpo_comm_eff_off_25step_reference (COMM_EFF_ENABLED=false; strict-no-op proof; run LAST)
- Cell-by-cell clearance: runner does NOT start cell N+1 until lead posts "cell N cleared".
- TEST_FREQ override per cell (10/10/20/20/20/25) so each cell validates at its final step.
- anchor.delay_K=2 for cell5 (NOT default 20).

## Gates (human-in-loop):
1. Runner plan-approval gate (lead approves runner's execution+code-change plan before spend/code).  [LEAD]
2. T5 code-change-plan approval (lead approves the spectral.cadence design before runner patches).   [LEAD]
3. PR gate — operator re-confirms before opening the DRAFT PR to vast-ai-workload on shamanez/verl.    [OPERATOR]

## End-of-experiment ordering (strict):
cell6 done.flag+rsync → TEARDOWN box (vast-teardown skill) → verdict-synthesizer emits verdict+issue comment → lead opens DRAFT PR (operator re-confirm) → shut down all teammates → TeamDelete.

## Task list (team exp-16-stability):
T0=#1 T1=#2 T2=#3 T3=#4 T4=#5 T5=#6 T6(cell5)=#7 T7(cell6)=#8 M*=#9 A5=#10 V=#11 PR=#12.
blockedBy chain: #2←#1, #3←#2, #4←#3, #5←#4, #6←#5, #7←#6, #8←#7, #10←#7, #11←#8,#9,#10, #12←#8,#11.

## Execution-model decisions (lead's pragmatic realization of the plan's team structure):
- **runner** = persistent background teammate (holds branch/box/tmux context across cells, wakes on lead "advance" messages). Plan-approval gate = explicit propose-then-wait (NOT mode:plan, for deterministic team-messaging behavior). Launches each cell into a DETACHED tmux, confirms start, reports, goes idle — does NOT block in-process for the full step count.
- **wandb-monitor** = a FRESH training-log-monitor dispatched (background) by the lead PER CELL, pointed at the active cell; returns a bounded terminal report (done/dead/stall/env-fail). Lead mediates kill-switch to the runner. (training-log-monitor is a bounded 40-min poller by design; per-cell dispatch fits it better than one eternal teammate.)
- **fsdp-spectral-auditor** = just-in-time read-only background agent dispatched when cell 5 completes (audits rsynced cell-5 logs against the 11-item checklist).
- **verdict-synthesizer** = analyst background agent dispatched at the end over all rsynced metrics.
- Rationale: only the runner needs cross-cell persistence; monitor/auditor/synthesizer are bounded analysis tasks, more robust as fresh just-in-time dispatches. Documented as a sanctioned lead deviation.

## Status timeline:
- 2026-05-30T03:1x  Team created. Task list seeded (12 tasks, deps wired). Spawning runner (background, propose-then-wait gate). Awaiting runner's execution+code-change plan for lead approval BEFORE any spend/code change.
