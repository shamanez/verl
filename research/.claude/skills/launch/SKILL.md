---
name: launch
description: "Launch an approved issue: per-issue branch, PREPARE phase on the laptop (implementation + CPU gates), then provision-or-attach compute per gpu_mode (auto = hands-off, ask = pause READY-FOR-GPU), start training, register the ledger row, snapshot run.json, auto-label status:running. Stage 4. Requires status:approved."
argument-hint: "<issue-number> [--gpu auto|ask] [--attach <instance-id>] [--account team|private]"
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# /launch <N> — approved plan → running experiment

## GPU mode (resolution: CLI `--gpu` > plan `gpu_mode:` > project.yaml `default_compute.gpu_mode`)

- **auto** — hands-off: the runner completes PREPARE, then walks the
  `gpu_filter_chain` and provisions by itself.
- **ask** — the runner completes PREPARE only, then /launch pauses the issue:
  `flag_human <N> "READY FOR GPU: #<N> prepared (branch pushed, payload +
  CPU gates green). Resume with /execute <N> --attach <instance-id> (your
  box) or /execute <N> --gpu auto (harness provisions)."` and STOPS. No box,
  no ledger row, no status change. `--attach <id>` on a later invocation is
  the operator handing over login details — vast-attach registers the box.
- `--attach <id>` (or plan `attach_box:`) — skip provisioning entirely; the
  runner attaches the operator's box. Implies the compute phase runs now.

## Preconditions (labels + ledger first, plan second — degrade, never stall)

```bash
source .claude/skills/_lib.sh
st=$(issue_status <N>)
plan_fetch <N> || die "no plan block in issue #<N> — /plan <N> again, then the gate"
slug=$(plan_field <N> slug); names_for <N> "$slug"
mode=<--gpu | plan_field <N> gpu_mode | project.yaml default_compute.gpu_mode>
row=$(ledger_row_by_issue <N>); [[ -z "$row" ]] && row=$(ledger_row "$RUN_ID")   # attach rows may predate the issue stamp
[[ -n "$row" ]] && jq -e '.status=="RUNNING" or .status=="PROVISIONED" or .status=="EXTERNAL"' <<<"$row" >/dev/null \
  && die "#<N> already has a live box ($(jq -r .id <<<"$row")) — /monitor <N> instead"
```
- Gate: `st == approved` is the normal entry. ONE exception — **relaunch**:
  `st == running` AND the last row is `TORN_DOWN` AND no `runs/$RUN_ID/verdict.md`
  (env-failure history) is allowed, guarded by
  `bump_attempt "$RUN_ID" launch_attempts 3 || exit 1`. Anything else → refuse:
  the human gate is sacred.
- **Clear any operator-stop sentinel before starting training** (#63 B10): a
  relaunched run MUST be heartbeat-reapable again, so
  `operator_stop_check "$RUN_ID" && operator_stop_clear "$RUN_ID"` before the
  tmux launch — leaving it set would suppress the reaper's heartbeat triggers
  for the resumed run (only the budget cap would remain).
- `kind: analysis|implementation|brainstorm|literature` → no GPU. Refuse with
  the right next step (`/analyze <N>` for analysis; `/close <N>` for the rest).
- Every `depends_on` issue must be terminal — `status:pass|stop|done` (a closed
  PASSed parent reads `done`) — else refuse, naming the blocker.

## Steps

1. **Names.** Already resolved in preconditions: `RUN_ID`, `RUN_DIR`, `BRANCH`,
   `WANDB_GROUP` from `names_for`.
2. **Dispatch ONE `experiment-runner` subagent** with:
   `issue=<N> run_id=$RUN_ID plan=$(plan_path <N>) branch=$BRANCH
   account=<--account | plan vast_account | team>
   attach=<--attach id | plan attach_box | none>
   compute=<full | prepare-only>`  — `prepare-only` iff mode=ask and no attach.
   (`plan_path` is the fetched cache of the issue-body plan — `plan_fetch`
   above already refreshed it; the runner never talks to GitHub for the plan.)
   The runner owns TWO phases (its contract):
   **PREPARE** (laptop, no spend): per-issue branch from the project.yaml base
   branch, pushed BEFORE anything else (crash survival); code_change patch +
   bundle; launch.sh payload; `runs/$RUN_ID/run.json` snapshot; ONE CPU sanity
   pass. **COMPUTE**: provision-or-attach (bounded ladder walk, ≤ 3 rungs,
   ≤ 1 retry per rung), PROVISIONED row BEFORE any rsync, payload sync, tmux
   launch, promote to RUNNING. PREPARE always completes before COMPUTE starts
   (`verification.gpu_idle_rule` — a box is never up while code is authored).
3. `compute=prepare-only` and the runner returns READY_FOR_GPU → run the
   **ask** pause from the GPU-mode section above and stop. Re-invocations with
   `--attach`/`--gpu auto` find PREPARE artifacts present (branch on origin,
   `runs/$RUN_ID/launch.sh`) — the runner skips straight to COMPUTE.
4. On runner success (COMPUTE ran): `set_status_label <N> running`; print box +
   WandB names (`<N>-<cell>` per cell, group `$RUN_ID`) + `Next: /monitor <N>`.
5. On runner failure: it already flagged `needs:human` (`flag_human`) or
   logged `NO_OFFERS` — surface that verbatim and stop. Do NOT re-dispatch in
   a loop.

## Rules

- Provisioning/attach is a gated single-shot dispatch — never inside a
  workflow, never auto-approved, never parallel-duplicated for one issue.
- The plan is read ONCE here (by the runner) and snapshotted to
  `runs/$RUN_ID/run.json`; monitor/analyze/close read the snapshot only.
- No CPU-verification loops before launch: PREPARE's CPU sanity pass runs
  ONCE with ONE bounded fix attempt; the plan's `hard` invariants get the
  on-box probe cell (1–2 steps); a `hard` probe failure is handled by
  /monitor's bounded fix loop, not here.
- After launch, either run `/monitor <N>` yourself (same session) or hand off:
  the GPU must never sit unwatched while a session idles.
