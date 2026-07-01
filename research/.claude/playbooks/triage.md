# Playbook: triage

Watcher for `research:claim` issues. You are executing in the top-level `/loop` session — dispatch `research-planner` subagents in parallel via the `Agent` tool. The planner labels each issue `status:planned`; the human flips to `status:approved` later, so this playbook never crosses the human gate.

## Operating context

Canonical project facts (working dir, gh-default repo, secrets, vast template, branch policy) live in [`.claude/project.yaml`](../project.yaml). Read it once if you need any of them. Your role-specific constraints:

- Write only `PROGRESS.md`. The planner owns plan files; you never edit issues directly.
- Dispatch only `research-planner`. Other subagents belong to the orchestrator playbook.

### Each iteration

1. List open issues tagged `research:claim` in the configured research repo. The local `gh` default repo must already be set to that repo via `gh repo set-default`. Run:
   ```bash
   gh issue list --label research:claim --state open --json number,title,url
   ```

2. Identify issues that need a plan: any whose `.claude/plans/<NUMBER>.md` does NOT yet exist. The filename is the bare issue number (`44.md`, not `EXP-44.md`).

3. Skip any issue that already carries label `status:planned`, `status:approved`, `status:stop`, or `status:done` (defensive — a missing plan file plus one of those labels means something went wrong and a human should look).

4. **Dispatch one `research-planner` subagent per remaining issue, in parallel** in a single turn using multiple `Agent` tool calls (subagent_type=`research-planner`). Each dispatch prompt:
   ```
   You are research-planner for issue #<NUMBER>.
   Read .claude/agents/research-planner.md for your contract and follow it exactly.
   Read .claude/plans/TEMPLATE.md for the plan structure.
   Run: gh issue view <NUMBER> --json title,body,labels,url
   Write the plan to .claude/plans/<NUMBER>.md, label the issue status:planned,
   and post a SHORT STUB comment (link to the plan file + cell list + budget — NEVER
   the full plan body; large comments blow agent context windows on thread re-reads).
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

7. **Print a one-line completion ledger to your reply** (for `/goal`): `TRIAGE LEDGER <ISO>: open research:claim=<N>, planned=<M>, unplanned=<K>`. This loop is launched under `/goal` (researcher_steps.md §3a), whose evaluator is transcript-only — this printed ledger is how it sees "all planned". When every open `research:claim` issue has a `.claude/plans/<N>.md` (K=0), the `/goal` condition is satisfied and the loop ends; otherwise pace ~60 min and re-tick.

8. Stop.

### Hard rules

- You may dispatch `research-planner` and nothing else. **Never** dispatch `experiment-runner`, `analyst`, or `log-writer` — those belong to the orchestrator playbook. (`orchestrator` is no longer a subagent — it's its own playbook.)
- **Hard-plan workflow (allowed).** For a high-uncertainty / wide-design-space issue you MAY launch a dynamic workflow EXPLICITLY (the `ultracode` keyword / "run a workflow") as a judge-panel — draft N plan approaches, score them, synthesize — and hand the winner to a `research-planner` to write up. It is read-only (touches no Vast, no `runs.jsonl`, no label) and still routes through the human gate (planner emits at `status:planned`; the human approves). Do NOT turn the session into `/effort ultracode` (it auto-escalates every tick); invoke per-issue only. See `project.yaml workflows:`.
- **You do not review plans.** The plan is written by the planner; the human operator reviews it manually before flipping `status:planned → status:approved`. See the operator-review section at the bottom of the orchestrator playbook.
- Never call `gh issue create` or `gh pr create`. Read-only on GitHub for issue mutation; `gh issue edit` is the planner's job, not yours.
- Never write any file other than `PROGRESS.md`. The plan file is the planner's responsibility.
- Do not dispatch a planner for an issue whose plan file already exists. The plan file is the claim marker — duplicate dispatches overwrite human edits.
- If `gh` returns an error, log it to `PROGRESS.md` (one line, prefixed `[triage] gh error:`) and stop without dispatching.
- Idempotence: re-firing on the next tick must NOT re-dispatch planners for issues whose plan file now exists, even if that planner failed to label `status:planned`. A missing label with an existing plan file is a signal to the human, not a retry trigger.
