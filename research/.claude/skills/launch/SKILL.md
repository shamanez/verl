---
name: launch
description: "Launch an approved issue: per-issue branch, provision-or-attach compute, start training, register the ledger row, snapshot run.json, auto-label status:running. Stage 4. Requires status:approved."
argument-hint: "<issue-number> [--attach <instance-id>] [--account team|private]"
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# /launch <N> — approved plan → running experiment

## Preconditions (labels + ledger first, plan second — degrade, never stall)

```bash
source .claude/skills/_lib.sh
[[ "$(issue_status <N>)" == "approved" ]] || die "#<N> is status:$(issue_status <N>), not approved — the human gate is sacred"
plan_exists <N> || die "plan for #<N> was deleted — /plan <N> again, then /approve"
row=$(ledger_row_by_issue <N>)
[[ -n "$row" ]] && jq -e '.status=="RUNNING" or .status=="PROVISIONED"' <<<"$row" >/dev/null \
  && die "#<N> already has a live box ($(jq -r .id <<<"$row")) — /monitor <N> instead"
```
- `kind: analysis|implementation|brainstorm|literature` → no GPU. Refuse with
  the right next step (`/analyze <N>` for analysis; `/close <N>` for the rest).
- Every `depends_on` issue must be `status:pass|stop` — else refuse, naming
  the blocker.
- Prior `TORN_DOWN` row without a verdict (env-failure history): allowed to
  relaunch, but `bump_attempt "$RUN_ID" launch_attempts 3 || exit 1`.

## Steps

1. **Names.** `slug` from the plan (`plan_field <N> slug`);
   `names_for <N> "$slug"` → `RUN_ID`, `RUN_DIR`, `BRANCH`, `WANDB_GROUP`.
2. **Branch — every issue, not just code_change.** If `origin/$BRANCH` doesn't
   exist: create from `origin/vast-ai-workload`, push immediately (survives a
   laptop crash). code_change plans: dispatch happens in a worktree on this
   branch (runner's `isolation: worktree` handles it).
3. **Dispatch ONE `experiment-runner` subagent** with:
   `issue=<N> run_id=$RUN_ID plan=.claude/plans/<N>.md branch=$BRANCH
   account=<--account | plan vast_account | private>
   attach=<--attach id | plan attach_box | none>`.
   The runner owns: provision-or-attach (bounded ladder walk, ≤ 3 rungs, ≤ 1
   retry per rung), PROVISIONED row BEFORE any rsync, payload sync, tmux
   launch, `runs/$RUN_ID/run.json` snapshot, promote to RUNNING.
4. On runner success: `set_status_label <N> running`; print box + WandB names
   (`<N>-<cell>` per cell, group `$RUN_ID`) + `Next: /monitor <N>`.
5. On runner failure: it already logged `MANUAL_REVIEW_NEEDED`/`NO_OFFERS` to
   PROGRESS.md — surface that verbatim and stop. Do NOT re-dispatch in a loop.

## Rules

- Provisioning/attach is a gated single-shot dispatch — never inside a
  workflow, never auto-approved, never parallel-duplicated for one issue.
- The plan is read ONCE here (by the runner) and snapshotted to
  `runs/$RUN_ID/run.json`; monitor/analyze/close read the snapshot only.
- No CPU-verification loops before launch: the plan's `hard` invariants get
  the on-box probe cell (1–2 steps); a `hard` probe failure is handled by
  /monitor's bounded fix loop, not here.
- After launch, either run `/monitor <N>` yourself (same session) or hand off:
  the GPU must never sit unwatched while a session idles.
