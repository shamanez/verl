# Playbook: triage

Watcher for `research:claim` issues. You are executing in the top-level `/loop` session — dispatch `research-planner` subagents in parallel via the `Agent` tool. The planner labels each issue `status:planned`; the human flips to `status:approved` later, so this playbook never crosses the human gate.

## Operating context

Canonical project facts (working dir, gh-default repo, secrets, vast template, branch policy) live in [`.claude/project.yaml`](../project.yaml). Read it once if you need any of them. Your role-specific constraints:

- Write only `PROGRESS.md`. The planner owns plan files; you never edit issues directly.
- Dispatch only `research-planner`. Other subagents belong to the orchestrator playbook.
- Do not read `../major-goal/` — human-only.

### Each iteration

1. List open issues tagged `research:claim` in the configured research repo. The local `gh` default repo must already be set to that repo via `gh repo set-default`. Run:
   ```bash
   gh issue list --label research:claim --state open --json number,title,url
   ```

2. Identify issues that need a plan: any whose `.claude/plans/<NUMBER>.md` does NOT yet exist. The filename is the bare issue number (`7.md`, not `EXP-7.md`).

3. Skip any issue that already carries label `status:planned`, `status:approved`, `status:stop`, or `status:done` (defensive — a missing plan file plus one of those labels means something went wrong and a human should look).

4. **Dispatch one `research-planner` subagent per remaining issue, in parallel** in a single turn using multiple `Agent` tool calls (subagent_type=`research-planner`). Each dispatch prompt:
   ```
   You are research-planner for issue #<NUMBER>.
   Read .claude/agents/research-planner.md for your contract and follow it exactly.
   Read .claude/plans/TEMPLATE.md for the plan structure.
   Run: gh issue view <NUMBER> --json title,body,labels,url
   Write the plan to .claude/plans/<NUMBER>.md, label the issue status:planned,
   and post the plan as an issue comment via gh issue comment <NUMBER> --body-file .claude/plans/<NUMBER>.md.
   Append one line to PROGRESS.md and stop.
   ```

5. After every planner dispatch returns, append a single summary line to `PROGRESS.md`:
   ```bash
   echo "[$(date -Iseconds)] [triage] dispatched <N> planners, <M> issues already planned" >> PROGRESS.md
   ```

6. If `gh` returned an empty list OR every open `research:claim` issue already has a plan file, append:
   ```bash
   echo "[$(date -Iseconds)] [triage] no claimable issues — ALL_PLANNED" >> PROGRESS.md
   ```
   and stop without dispatching anything.

7. Stop. The loop fires you again in 60 min.

### Hard rules

- You may dispatch `research-planner` and nothing else. **Never** dispatch `experiment-runner`, `analyst`, or `log-writer` — those belong to the orchestrator playbook. (`orchestrator` is no longer a subagent — it's its own playbook.)
- **You do not invoke codex.** The plan is written by the planner; the human operator reviews it manually and optionally invokes `bash .claude/skills/codex-verify/run.sh --mode verify --plan .claude/plans/<N>.md --out ...` before flipping `status:planned → status:approved`. See the operator-review section at the bottom of the orchestrator playbook.
- Never call `gh issue create` or `gh pr create`. Read-only on GitHub for issue mutation; `gh issue edit` is the planner's job, not yours.
- Never write any file other than `PROGRESS.md`. The plan file is the planner's responsibility.
- Do not dispatch a planner for an issue whose plan file already exists. The plan file is the claim marker — duplicate dispatches overwrite human edits.
- If `gh` returns an error, log it to `PROGRESS.md` (one line, prefixed `[triage] gh error:`) and stop without dispatching.
- Idempotence: re-firing on the next tick must NOT re-dispatch planners for issues whose plan file now exists, even if that planner failed to label `status:planned`. A missing label with an existing plan file is a signal to the human, not a retry trigger.
