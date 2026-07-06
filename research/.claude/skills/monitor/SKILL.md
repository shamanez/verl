---
name: monitor
description: "Babysit a running experiment until terminal: bounded background watch cycles, env-failure ladder walk, bounded on-box fix loop for code_change probes, teardown the moment results sync. Stage 5. No adversarial loops — ever."
argument-hint: "<issue-number>"
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# /monitor <N> — watch until terminal, GPU never stale

## Preconditions

```bash
source .claude/skills/_lib.sh
row=$(ledger_row_by_issue <N>) ; [[ -n "$row" ]] || die "no ledger row for #<N> — /launch <N> first"
id=$(jq -r .id <<<"$row")
jq -e '.status=="RUNNING"' <<<"$row" >/dev/null || die "#<N> box is $(jq -r .status <<<"$row") — /analyze <N> if results synced, /launch <N> to relaunch"
[[ -f "$(run_json_path "$id")" ]] || echo "warn: run.json missing — monitor from ledger handles only"
```

## Watch loop (bounded, background, act-on-report)

Repeat up to **12 cycles** (~8 h; for longer runs re-invoke /monitor or run
under `/bg /goal … /go <N>`):

1. Dispatch `training-log-monitor` with `run_in_background: true`, passing
   `run_id=$id` (it reads handles + cells + step target from
   `runs/$id/run.json` / `runs/$id/handles/*.json` — never the plan). It polls
   30 s / ≤ 40 min and returns a terminal report. Never foreground-poll a
   background task with sleep loops.
2. Act on the report — this is a dispatch table, not a judgment call:
   - **done** (flags + steps reached) → confirm metrics rsynced to
     `runs/$id/metrics/`, then teardown NOW (`vast-teardown` skill), then
     `/analyze <N>`. The box never outlives its science.
   - **experiment-failure** (NaN/OOM mid-train, wrong numbers) → the failure
     IS the data. Let remaining cells finish; when done: sync → teardown →
     `/analyze <N>`. Do not "investigate" on the paid box.
   - **probe/env fix needed** (code_change probe crashed on backend
     integration) → `bump_attempt "$id" fix_attempts 3 || {teardown; stop}` —
     ONE focused fix per attempt via the on-box commit-hotfix loop, relaunch
     the cell, next cycle. Exhausted → teardown + `MANUAL_REVIEW_NEEDED`.
   - **env-failure** (docker/CUDA/NCCL/vLLM-init/SSH-dead) → teardown, then
     `bump_attempt "$id" launch_attempts 3 || stop` and re-dispatch the runner
     on the NEXT ladder rung. Attached (`external:true`) box: teardown but
     NEVER auto-provision a replacement — `MANUAL_REVIEW_NEEDED`, stop.
   - **stall** (all GPUs ≤5% for 4 polls, tmux alive) → one `nvidia-smi` +
     log-tail confirmation, then teardown + `MANUAL_REVIEW_NEEDED`. A stalled
     GPU is burning money for nothing.
   - **timeout** (40-min cap, training healthy) → next cycle immediately.
3. Between reports there is nothing to do — do NOT fill the gap with analysis,
   verification, or "deep dives" on the live box.

## Hard rules

- NO adversarial verification, judge panels, or exploratory workflows during a
  run. If something genuinely warrants heavy verification mid-run, append
  `MANUAL_REVIEW_NEEDED: <what/why> — #<N>` to PROGRESS.md and STOP for a
  human go/no-go.
- Teardown always via the `vast-teardown` skill (ledger flip included), the
  instant science is captured. Never `vastai destroy` bare.
- Every loop here is bounded by ledger counters (`fix_attempts`,
  `launch_attempts`, 12 cycles). Exhaustion is a stop, never a spin.
- WandB tail rule: after the final step, backfill the last 1–2 steps from
  train.log at /analyze time (the async uploader drops them).
