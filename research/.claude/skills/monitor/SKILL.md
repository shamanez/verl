---
name: monitor
description: "Babysit a running experiment until terminal: bounded background watch cycles (cheap machine-monitor health poller; Opus training-log-monitor dispatched ONCE per anomaly to classify), env-failure ladder walk, bounded on-box fix loop for code_change probes, teardown the moment results sync. Stage 5. No adversarial loops — ever."
argument-hint: "<issue-number>"
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# /monitor <N> — watch until terminal, GPU never stale

## Preconditions

```bash
source .claude/skills/_lib.sh
row=$(ledger_row_by_issue <N>)
[[ -z "$row" ]] && { slug=$(plan_field <N> slug); [[ -n "$slug" ]] && row=$(ledger_row "<N>-$slug"); }
[[ -n "$row" ]] || die "no ledger row for #<N> — /launch <N> first"
id=$(jq -r .id <<<"$row"); st=$(jq -r .status <<<"$row")
case "$st" in
  RUNNING|EXTERNAL) ;;                       # EXTERNAL = operator-managed box running this issue's work — watch it the same way
  PROVISIONED) ;;                            # runner mid-launch — see below, do NOT bounce to /launch
  *) die "#<N> box is $st — /analyze <N> if results synced, /launch <N> to relaunch" ;;
esac
[[ -f "$(run_json_path "$id")" ]] || echo "warn: run.json missing — monitor from ledger handles only"
```
- **PROVISIONED row:** the runner is (or was) mid-launch. Wait ONE bounded cycle
  (≤ 15 min, checking the row each minute): promoted to RUNNING → proceed;
  still PROVISIONED after 15 min → the reaper will flip it; run
  `flag_human <N> "stuck PROVISIONED"` and stop. Never bounce the
  operator back to /launch (its live-box guard would bounce them here again).

## Watch loop (bounded, background, act-on-report — two tiers)

The default watcher is the CHEAP **`machine-monitor`** (Sonnet — mechanical
health polls only). The Opus **`training-log-monitor`** costs real money per
poll and is dispatched **on demand**: ONCE per anomaly, in classify mode —
never as the every-cycle watcher (project.yaml `verification.monitoring`).

Repeat up to **12 cycles** (~8 h; for longer runs re-invoke /monitor or run
under `/bg /goal … /go <N>`):

1. Dispatch `machine-monitor` with `run_in_background: true`, passing
   `run_id=$id` (it reads handles + cells + step target from
   `runs/$id/run.json` / `runs/$id/handles/*.json` — never the plan). It polls
   30 s / ≤ 40 min and returns ONE terminal report:
   `done | healthy-timeout | anomaly` (+ evidence: log tail, GPU util
   history, step trace). Never foreground-poll a background task with sleep
   loops.
2. `anomaly` → dispatch `training-log-monitor` ONCE (foreground,
   `mode=classify`, passing the anomaly evidence). It classifies into the
   dispatch table below and returns. Do NOT redispatch it for a second
   opinion.
3. Act on the (classified) report — this is a dispatch table, not a judgment
   call:
   - **done** (flags + steps reached) → confirm metrics rsynced to
     `runs/$id/metrics/`, then teardown NOW (`vast-teardown` skill), then
     `/analyze <N>`. The box never outlives its science.
   - **experiment-failure** (NaN/OOM mid-train, wrong numbers) → the failure
     IS the data. Let remaining cells finish; when done: sync → teardown →
     `/analyze <N>`. Do not "investigate" on the paid box.
   - **probe/env fix needed** (code_change probe crashed on backend
     integration) → `bump_attempt "$id" fix_attempts 3 || {teardown; stop}` —
     ONE focused fix per attempt via the on-box commit-hotfix loop, relaunch
     the cell, next cycle. Exhausted → teardown + `flag_human <N> …`.
   - **env-failure** (docker/CUDA/NCCL/vLLM-init/SSH-dead) → teardown, then
     `bump_attempt "$id" launch_attempts 3 || stop` and re-dispatch the runner
     on the NEXT ladder rung. Attached (`external:true`) box: teardown but
     NEVER auto-provision a replacement — `flag_human <N> …`, stop.
   - **stall** (all GPUs ≤5% for 4 polls, tmux alive) → one `nvidia-smi` +
     log-tail confirmation, then teardown + `flag_human <N> "stall"`. A
     stalled GPU is burning money for nothing.
   - **unclear** (the classifier genuinely could not classify) → sync
     whatever evidence exists into `runs/$id/`, then teardown +
     `flag_human <N> "unclassifiable anomaly: <what's missing>"`, stop. An
     unexplained box must never keep billing unwatched — money-safe beats
     diagnosis.
   - **healthy-timeout** (40-min cap, training progressing) → next cycle
     immediately (back to step 1 — the cheap watcher again).
4. Between reports there is nothing to do — do NOT fill the gap with analysis,
   verification, or "deep dives" on the live box.

## Hard rules

- NO adversarial verification, judge panels, or exploratory workflows during a
  run. If something genuinely warrants heavy verification mid-run, run
  `flag_human <N> "<what/why>"` (durable `needs:human` label + issue comment +
  PROGRESS echo) and STOP for a human go/no-go.
- ONE classification per anomaly: machine-monitor detects, training-log-monitor
  classifies once, this skill acts. Classification doubt → the classifier says
  so in its report and the action is `flag_human`, not a re-run.
- Teardown always via the `vast-teardown` skill (ledger flip included), the
  instant science is captured. Never `vastai destroy` bare.
- Every loop here is bounded by ledger counters (`fix_attempts`,
  `launch_attempts`, 12 cycles). Exhaustion is a stop, never a spin.
- WandB tail rule: after the final step, backfill the last 1–2 steps from
  train.log at /analyze time (the async uploader drops them).
