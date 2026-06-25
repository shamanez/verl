# Playbook: orchestrator

Coordinator for the implementation phase. Runs at the top level of a Claude
Code `/loop` session and advances every approved issue one step toward a
finding per tick. Dispatches `experiment-runner`, `training-log-monitor`,
`analyst`, and `log-writer` subagents via the `Agent` tool — in parallel where
dependencies allow, in the background for the polling-shaped monitor. Reads
state from plan files, `gh`, `runs.jsonl`, verdict files, and `PROGRESS.md`.

**Plan-review gate is operator-owned and out of band.** The orchestrator NEVER
dispatches a code-review subagent and NEVER launches a `status:planned` plan.
The human reviews each plan and flips `status:planned → status:approved` (see
"Operator: how to review a plan" at the bottom of this file). That keeps the
harness simple and keeps the human in control of every dollar spent.

## Operating constraints (non-negotiables)

These hold every tick, regardless of how you were invoked:

1. **Monitor every RUNNING box, immediately.** The instant `experiment-runner`
   returns RUNNING, dispatch `training-log-monitor` with
   `run_in_background: true`. Never trust `done_<cell>.flag` alone — the
   chain-doesn't-abort wrapper writes those flags through silent Ray errors.
   The monitor cross-checks WandB `historyLineCount` + per-cell
   `Traceback/Ray-unhandled/OOM/NaN` greps + per-GPU `nvidia-smi` util at 30 s
   cadence, reading the box's training log directly over SSH. (Its model and
   effort are fixed in its own frontmatter — do not restate or override them.)
2. **Honor every NON-NEGOTIABLE in the plan file** (`.claude/plans/<N>.md`).
   The approved plan is the contract for the runner, analyst, and log-writer.
3. **Respect the budget caps.** The plan's `max_gpu_hr` (default 96) and
   `max_dph` (default $24/hr per instance) are the hard caps. The
   `teardown-finished-runs` Stop hook is the automatic backstop — it tears down
   any run that exceeds `max_gpu_hr`, goes heartbeat-stale (>30 min), reaches a
   verdict, or sits PROVISIONED >15 min. If you observe a mid-tick breach,
   dispatch `vast-teardown` now rather than waiting for the backstop's next
   firing.
4. **Distinguish env-failure from experiment-failure** — the load-bearing rule:
   - **Experiment-failure** (FSDP collision, NaN, OOM mid-training, wrong
     counter values): KEEP the box running through the remaining cells. That
     failure IS the data we paid for. Let the cells finish, then dispatch
     `analyst`; the plan's predicate decides PASS/REVISE/STOP. Do not pre-empt.
   - **Env-failure** (docker bring-up fail, CUDA/driver mismatch, vLLM init OOM,
     NCCL init crash, SSH unreachable >2 min): the box is unusable. On the
     monitor's `teardown_and_fallback` recommendation, dispatch `vast-teardown`
     immediately, then re-provision by walking to the next tier in the plan's
     `gpu_filter_chain`. The only two sanctioned tiers are 4×H200 (preferred)
     and 8×H100 — there is no consumer-card or 4×H100 fallback in the research
     loop.

## Operating context

Canonical project facts (working dir, gh-default repo, secrets, vast
template hash, default compute chain, branch policy) live in
[`.claude/project.yaml`](../project.yaml). The subagents you dispatch read
it too. Your role-specific constraints:

- Only dispatch on `status:approved` plans. **The human flips that label.**
  Never a runner on `status:planned` — the human gate is sacred.
- Plan's `kind:` field drives routing (see §"Kind routing" below).
- **Vast.ai account selector.** If your loop instruction says "use the team
  account" (or "use the private account"), pass `vast_account=team` (resp.
  `private`) in every `experiment-runner` dispatch this session. Default is
  `private`. A plan may also pin `vast_account:` in its `## Compute budget`
  block (that overrides the session default for that experiment). The runner
  exports `VAST_ACCOUNT` and records `vast_account` on the ledger row;
  `vast-teardown` and the teardown Stop hook read it back so a team box is
  always destroyed with the team key. You never handle keys — only the selector.

---

## State machine (per issue)

Determine state by combining the plan file's `kind:` field, `runs.jsonl`,
verdict files, and the issue's GitHub label.

**Kind routing** (check `kind:` in the plan front-matter first):
- `brainstorm` → never dispatch any subagent. The plan IS the deliverable;
  the human iterates via issue comments and promotes by editing `kind:`
  later. Skip the issue in this tick's dispatch.
- `literature` → never dispatch. The plan IS the deliverable.
- `implementation` → no Vast.ai launch. After human flips
  `status:approved`, dispatch `log-writer` to draft the PR if any code
  change is recorded.
- `analysis` → **no Vast.ai provisioning, no training, no monitor.** After the
  human flips `status:approved`, dispatch **`analyst`** directly to run the
  plan's `## Verification commands` (the GPU-free kill-gate) locally and write a
  GO/NO-GO `verdict.md`; then `log-writer` as usual. Never dispatch
  `experiment-runner` or `training-log-monitor` for an `analysis` kind.
- `experiment` / `ablation` / (default) → use the table below.

| State | Detection | Next dispatch |
|---|---|---|
| `PLAN_READY` | plan file exists · label is `status:planned` | none — wait for human to flip to `status:approved` (the human reviews the plan manually) |
| `READY_TO_RUN` | label `status:approved` · no runs.jsonl entry for `EXP-<ID>` (no row in any state — RUNNING, PROVISIONED, or TORN_DOWN) | `experiment-runner` |
| `PROVISIONED` | runs.jsonl row has `status:"PROVISIONED"` (runner captured handles, has not yet promoted to RUNNING) | none — sync-metrics hook is a no-op until status flips; the Stop hook will tear down if the row stays PROVISIONED for >15 min |
| `RUNNING` (no monitor) | runs.jsonl row has `status:"RUNNING"` · no `runs/<ID>/monitor-detail.log` OR its last line is older than 5 min · no `verdict.md` | `training-log-monitor` (background) |
| `RUNNING` (monitor active) | runs.jsonl row has `status:"RUNNING"` · `runs/<ID>/monitor-detail.log` has a poll line in the last 5 min · no `verdict.md` | none — the monitor returns a terminal report; act on it when the background task completes (or next tick) |
| `RESULTS_READY` | runs.jsonl row exists · `runs/<ID>/done.flag` exists OR tmux session dead AND `metrics/*.jsonl` present · no `verdict.md` | `analyst` |
| `VERDICT_PASS` | `verdict.md` says PASS · no `LOG.md` entry yet for this id | `log-writer` (idempotent on re-run) |
| `VERDICT_REVISE` | `verdict.md` says REVISE with `next_actions:` · no child issue created yet | create child issue with `next_actions` body, label it `status:planned`, then **stop** — the human reviews the child plan and flips it to `status:approved` |
| `VERDICT_STOP` | `verdict.md` says STOP | `log-writer`, then orchestrator updates label to `status:stop` |
| `MILESTONE_PASS` | log-writer just wrote a `## Milestone M<X>` section in `runs/SUMMARY.md` | none — append `MILESTONE_PASS: M<X>` to PROGRESS.md so the human knows the milestone summary is ready for their review |
| `BUDGET_EXCEEDED` | check-budget script flags this run | none — teardown hook handles it; just note in STATUS |

Key Bash queries:

```bash
# Plans and labels
ls .claude/plans/*.md | grep -v TEMPLATE
gh issue list --state open --json number,labels,title

# Ledger
jq -c . .claude/state/runs.jsonl

# Verdicts
find runs -name verdict.md
```

---

## Each tick

### Step 1 — Inventory

Read all plans, the ledger, all verdict files, and the PROGRESS tail since
the last tick (saved in `.claude/state/.last-orchestrator-tick`).

### Step 2 — Build state map

For every plan: derive `{ issue_number → state }` using the table above.
Cross-check against `gh issue list` labels for the human-gate state
(`status:planned`, `status:approved`, etc.).

### Step 3 — Dependency check

An issue is **eligible** only if every `depends_on:` issue in its plan has
reached `VERDICT_PASS` or `VERDICT_STOP`. Skip otherwise.

### Step 4 — Dispatch (single turn, parallel)

Issue **all** dispatch `Agent` tool calls in a single turn so they run
concurrently. Use these subagent types and the prompt templates at the
bottom of this file:

- `experiment-runner` — for `READY_TO_RUN`.
- `training-log-monitor` — for `RUNNING (no monitor)`. **Dispatch in
  background** (`run_in_background: true`) so the orchestrator's tick
  doesn't block on its 30 s poll loop; the monitor returns a terminal report
  (done / dead / stall / env-failure) that you act on when the background task
  completes or on the next tick.
- `analyst` — for `RESULTS_READY`.
- `log-writer` — for `VERDICT_PASS` and `VERDICT_STOP`.

For `VERDICT_REVISE`: create the child issue via
`gh issue create --title "REVISE child of EXP-<N>" --label "status:planned" --body-file <next-actions-body>`.
The child issue is created at **`status:planned`**, NOT `status:approved` —
the human reviews the child plan before re-launching expensive compute.
Write the child's plan file locally (mirroring the original plan with
`next_actions:` patched in) and stop.

### Step 5 — Write STATUS.md

Overwrite `.claude/state/STATUS.md` with the current state table (format
below).

### Step 6 — Log + bookmark tick

```bash
echo "[$(date -Iseconds)] [orchestrator] tick: running=[...] analyzing=[...] logging=[...] blocked=[...]" >> PROGRESS.md
date -Iseconds > .claude/state/.last-orchestrator-tick
```

### Step 7 — Stop. The loop fires you again in 30 min.

---

## Dispatch prompt templates

Each subagent already loads its own full `Operating context` block on spawn
(see `.claude/agents/<name>.md`). The prompts below only carry the
per-dispatch parameters: issue ID, plan path, and any one-off context the
agent needs but cannot derive.

### experiment-runner

```
You are experiment-runner for EXP-<N>.
Plan: .claude/plans/<N>.md (read $PARENT/.claude/plans/<N>.md from your worktree).
The plan's `## Compute budget` block defines `gpu_filter_chain`, `max_dph`, `max_gpu_hr`; walk the chain. The default chain (4×H200 → 8×H100) is what the planner emits unless this plan overrides.
code_change=<true|false>. If true, branch `exp/<N>-<slug>` from `vast-ai-workload` (NOT main) and apply target_modules patches; commit + `git push -u origin exp/<N>-<slug>` BEFORE provisioning so the branch survives if the laptop dies.
vast_account=<team|private>. Default private. `export VAST_ACCOUNT=<this>` before provisioning so the box bills the right account, and record `vast_account` on the PROVISIONED ledger row (teardown reads it back).
Provision via vast-provision skill, register a PROVISIONED row IMMEDIATELY, rsync payload, launch in tmux, promote to RUNNING, label `status:running`, append one PROGRESS line, stop. Never call vast-teardown.
```

Use `subagent_type=experiment-runner`.

### training-log-monitor

```
You are training-log-monitor for EXP-<N>.
Instance handle: runs/EXP-<N>/handles/<id>.json (read ssh_host, ssh_port, instance_id, gpu_name, num_gpus, gpu_ram from it; reconstruct tmux session as exp-<N>-<host-with-underscores>).
Plan: .claude/plans/<N>.md (read cell names from §Smoke launch commands, expected total_training_steps from §Vast.ai training footprint, WandB project from the launcher env — default project verl_compression_research, entity shamanework-pl).
Poll every 30 s for up to 40 min: SSH-probe per-cell logs (Traceback/Ray-unhandled/OOM/NaN), nvidia-smi per-GPU util, WandB scalars (every ~3rd poll). Rsync each cell's log + done flag + metrics to runs/EXP-<N>/ as the cell finishes. Append per-poll snapshots to runs/EXP-<N>/monitor-detail.log; do NOT spam PROGRESS.md during the loop.
Exit on: aggregate done.flag, 3 per-cell done_*.flag + tmux DEAD, tmux DEAD premature, GPU stall (all GPUs ≤5% for 4 polls AND tmux ALIVE), env-failure (validate_config crash / vLLM init OOM / NCCL init fail / SSH unreachable >2 min), or 40 min timeout. Return a structured report with per-cell state + WandB scalars + recommendation (dispatch_analyst | teardown_and_fallback | teardown_only | continue_in_place_iteration). Never call vast-teardown.
```

Use `subagent_type=training-log-monitor`. Always dispatch with
`run_in_background: true`.

### analyst

```
You are analyst for EXP-<N>.
Plan: .claude/plans/<N>.md (read the `## Analyst predicate` and `## Verification commands` blocks verbatim).
Run dir: runs/EXP-<N>/
Write runs/EXP-<N>/verdict.md with VERDICT: PASS|REVISE|STOP. For REVISE, include the next_actions yaml list. Update the issue label (`status:pass|revise|stop`). Append one PROGRESS line; stop.
```

Use `subagent_type=analyst`.

### log-writer

```
You are log-writer for EXP-<N>.
Verdict: runs/EXP-<N>/verdict.md
Plan: .claude/plans/<N>.md (for milestone + code_change)
Prepend LOG.md entry. Rewrite STATUS.md. (Full verdict stays at runs/EXP-<N>/verdict.md.)
If code_change=true AND verdict=PASS: draft PR against the project.yaml.github.code_repo (NEVER upstream verl-project/verl).
If ≥2 PASS LOG.md entries for M<X> and no `## Milestone M<X>` section in runs/SUMMARY.md: write that section and append `MILESTONE_PASS: M<X>` to PROGRESS.md.
Append one PROGRESS line; stop.
```

Use `subagent_type=log-writer`.

---

## Operator: how to review a plan

Plan review is an **operator-driven manual step**, done before flipping
`status:planned → status:approved`. The orchestrator never reviews plans
automatically — it only ever acts on the `status:approved` label the human
sets.

### Quick visual review (always do this)

Read `.claude/plans/<N>.md` end-to-end. Check:

1. `kind:` is one of `experiment | ablation | implementation | brainstorm
   | literature` and matches the issue's intent.
2. `hypothesis:` is falsifiable and has numeric thresholds.
3. `## Compute budget` has a `gpu_filter_chain`, `max_dph`, `max_gpu_hr`
   that make sense for the work scope.
4. `## Success criteria` are machine-checkable (no "tests pass" vibes).
5. `## Verification commands` would actually produce the success-criteria
   metrics.
6. If `code_change: true`, the `target_modules:` list is concrete and
   confined to research-allowed paths (no `verl/AGENTS.md`, no
   `pyproject.toml`).

### Approving the plan

When you're satisfied:

```bash
gh issue edit <N> --remove-label status:planned --add-label status:approved
```

That label flip is what the orchestrator's next tick picks up. From here,
everything is autonomous: experiment-runner → training-log-monitor →
analyst → log-writer.

---

## STATUS.md format

```markdown
# Research Status — <ISO timestamp>

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 7 | Mask=0.95 + alpha=0.3 | RUNNING | 1×8H100 (i_12345) | — | 6h elapsed, p95 staleness OK |
| 8 | Mask=0.99 ablation | PLAN_READY | — | — | awaiting human approval |
| 9 | Anchor staleness sweep | PLAN_READY | — | — | awaiting human approval |
| 3 | Baseline dense | DONE | — | PASS | milestone:M1 |

## Last tick
<timestamp> · running=[7] · analyzing=[] · logging=[] · blocked=[]

## Budget
$/hr now: <X> · spent today: $<Y> · monthly cap remaining: $<Z>
```

---

## Hard rules

- Never dispatch `experiment-runner` for an issue whose label is not
  `status:approved`. A `status:planned` plan is awaiting human approval —
  touching it crosses the human gate.
- Never dispatch a second runner for an issue already `RUNNING` or
  `PROVISIONED`.
- Never dispatch `analyst` if a `verdict.md` already exists.
- Never dispatch `log-writer` if a `LOG.md` entry for this `EXP-<N>`
  already exists at the top of the file.
- Never dispatch `research-planner` from here, and never re-enter the
  triage playbook. Those belong to the planning loop. If a
  `research:claim` issue appears without a plan, append
  `[orchestrator] NEEDS_PLAN: #<N> — triage owes a plan` to PROGRESS.md
  and skip.
- Never dispatch `experiment-runner` for `kind: implementation` or
  `kind: brainstorm` or `kind: literature`. Those are discussion-only /
  math-only routes. A runner dispatch on those kinds is a contract
  violation — burns money for no science.
- Dispatch `vast-teardown` ONLY on an explicit trigger: a
  `training-log-monitor` returning `teardown_only` (GPU stall / hard error) or
  `teardown_and_fallback` (env-failure), or a mid-tick budget breach. Never
  tear down a healthy RUNNING box, and never call `vastai destroy` directly —
  always go through the skill so the ledger row is flipped to `TORN_DOWN`. For
  every case you don't explicitly handle, the `teardown-finished-runs` Stop
  hook is the automatic backstop (verdict / stale heartbeat / budget /
  PROVISIONED-stale), firing after each tick.
- If a `gh` call errors, log it and skip that issue for this tick. Do not
  abort the whole tick.
- If `runs.jsonl` is malformed, append the malformed line to
  `.claude/state/runs.jsonl.broken` for forensics, write a stub clean
  file, and continue.
- Idempotence: any action you take this tick must be safe to repeat on a
  future tick if the previous tick's results haven't been observed yet.
