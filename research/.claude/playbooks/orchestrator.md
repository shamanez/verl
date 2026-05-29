# Playbook: orchestrator

Coordinator for the implementation phase. Runs in the top-level Claude Code
session — either as `/loop` (autonomous, fires every 30 min) or as `/goal`
(one-shot, Stop-hook-gated). Dispatches `experiment-runner`,
`training-log-monitor`, `analyst`, and `log-writer` subagents via the `Agent`
tool — parallel where dependencies allow, background where polling-shaped
(the monitor). Reads state from plan files, `gh`, `runs.jsonl`, verdict
files, and `PROGRESS.md`; advances every eligible issue toward a finding in
one turn.

**Plan-review gate is operator-owned and out of band.** The orchestrator
NEVER dispatches a code-review subagent. The human reviews each plan before
flipping `status:planned → status:approved` (see "Operator: how to review a
plan" at the bottom of this file). The harness becomes simpler and more
stable as a result.

## Entry pattern (`/goal` — the user's canonical invocation)

The user invokes this playbook from a fresh Claude Code session via the
`/goal` slash command. That installs a session-scoped Stop hook with the
goal string as its condition; the harness blocks session-end until the
condition holds. Use this template (copy-paste verbatim, then edit the
`<…>` placeholders inline):

```
/goal Read /Users/shamane/Documents/verl/research/.claude/playbooks/orchestrator.md and execute it for issue #<N> (or the highest-priority eligible issue if not specified).

NON-NEGOTIABLES for this session:
1. Dispatch `training-log-monitor` (research/.claude/agents/training-log-monitor.md, Opus, run_in_background=true) IMMEDIATELY after experiment-runner returns RUNNING. Never trust done_<cell>.flag alone — cross-check WandB historyLineCount + per-cell Traceback/Ray-unhandled grep + nvidia-smi per-GPU util at 30 s cadence. Look at the Vast.ai training log directly via SSH, not just local artifacts.
2. Honor every NON-NEGOTIABLE section in the issue's plan file (.claude/plans/<N>.md).
3. Tear down the Vast.ai instance via the vast-teardown skill BEFORE this session stops. The Stop hook will block stopping until the runs.jsonl ledger row reads TORN_DOWN.
4. Budget cap: if box lifetime spend exceeds $<MAX_DPH × wall_clock_hr from the plan, default $25>, tear down + write verdict=STOP regardless of cell completion state.
5. Consumer-card fallback: if a consumer tier (RTX_5090/RTX_4090) returns an env-failure (docker fail / cuda mismatch / vLLM init OOM / NCCL init / SSH unreachable), tear down immediately and re-provision skipping consumer tiers (start at 4×H100). Do NOT retry a second consumer host.
6. If experiment fails at runtime (FSDP collision, NaN, OOM mid-training, wrong counter values), KEEP the box running through the remaining cells (this is the data we paid for), then dispatch analyst — do not pre-empt. The plan's analyst predicate decides PASS/REVISE/STOP.

Stop only after: (a) analyst has written runs/EXP-<N>/verdict.md, (b) log-writer has updated LOG.md + findings/ (PASS/STOP only — REVISE goes to child-issue path), (c) Vast instance is destroyed and runs.jsonl ledger row reads TORN_DOWN.
```

**Goal-string design rules:**

- The goal string OVERRIDES playbook defaults for that session only. Document
  any override inline so future you can revise this template safely.
- Be specific about monitoring intensity. "Watch the training log" is too
  vague; "Dispatch training-log-monitor at 30 s cadence, cross-check WandB"
  is what we want.
- Name the budget cap explicitly. Without it, a hung training loop with
  healthy-looking GPU util can burn all night.
- Name the env-failure fallback rule for consumer-card lineages.
- Distinguish env-failures (tear down + fall back) from experiment-failures
  (keep running, that's the data).

**One-line minimum** (if you don't need to override defaults):

```
/goal Read /Users/shamane/Documents/verl/research/.claude/playbooks/orchestrator.md and execute it for issue #<N>. Dispatch training-log-monitor in background. Tear down before stop.
```

## Operating context

Canonical project facts (working dir, gh-default repo, secrets, vast
template hash, default compute chain, branch policy) live in
[`.claude/project.yaml`](../project.yaml). The subagents you dispatch read
it too. Your role-specific constraints:

- Only dispatch on `status:approved` plans. **The human flips that label.**
  Never a runner on `status:planned` — the human gate is sacred.
- Plan's `kind:` field drives routing (see §"Kind routing" below).
- Do not read `../major-goal/` — human-only.

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
- `experiment` / `ablation` / (default) → use the table below.

| State | Detection | Next dispatch |
|---|---|---|
| `PLAN_READY` | plan file exists · label is `status:planned` | none — wait for human to flip to `status:approved` (the human may run codex-verify manually as part of the review) |
| `READY_TO_RUN` | label `status:approved` · no runs.jsonl entry for `EXP-<ID>` (no row in any state — RUNNING, PROVISIONED, or TORN_DOWN) | `experiment-runner` |
| `PROVISIONED` | runs.jsonl row has `status:"PROVISIONED"` (runner captured handles, has not yet promoted to RUNNING) | none — sync-metrics hook is a no-op until status flips; the Stop hook will tear down if the row stays PROVISIONED for >15 min |
| `RUNNING` (no monitor) | runs.jsonl row has `status:"RUNNING"` · no `runs/<ID>/monitor-detail.log` OR its last line is older than 5 min · no `verdict.md` | `training-log-monitor` (background; Opus) |
| `RUNNING` (monitor active) | runs.jsonl row has `status:"RUNNING"` · `runs/<ID>/monitor-detail.log` has a poll line in the last 5 min · no `verdict.md` | none — monitor will re-invoke the orchestrator on terminal condition |
| `RESULTS_READY` | runs.jsonl row exists · `runs/<ID>/done.flag` exists OR tmux session dead AND `metrics/*.jsonl` present · no `verdict.md` | `analyst` |
| `VERDICT_PASS` | `verdict.md` says PASS · no `LOG.md` entry yet for this id | `log-writer` (idempotent on re-run) |
| `VERDICT_REVISE` | `verdict.md` says REVISE with `next_actions:` · no child issue created yet | create child issue with `next_actions` body, label it `status:planned`, then **stop** — the human reviews the child plan and flips it to `status:approved` (running codex-verify manually if they want) |
| `VERDICT_STOP` | `verdict.md` says STOP | `log-writer`, then orchestrator updates label to `status:stop` |
| `MILESTONE_PASS` | log-writer just wrote `findings/M<X>/SUMMARY.md` | none — append `MILESTONE_PASS: M<X>` to PROGRESS.md so the human knows it's time to invoke codex-verify in `adversarial` mode manually if desired |
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
  doesn't block on its 30 s poll loop; the monitor re-invokes the
  orchestrator on terminal condition (done / dead / stall / env-failure).
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
The plan's `## Compute budget` block defines `gpu_filter_chain`, `max_dph`, `max_gpu_hr`; walk the chain. The default chain (H200 → 8×H100 → 4×H100) is what the planner emits unless this plan overrides.
code_change=<true|false>. If true, branch `exp/<N>-<slug>` from `vast-ai-workload` (NOT main) and apply target_modules patches; commit + `git push -u origin exp/<N>-<slug>` BEFORE provisioning so the branch survives if the laptop dies.
Provision via vast-provision skill, register a PROVISIONED row IMMEDIATELY, rsync payload, launch in tmux, promote to RUNNING, label `status:running`, append one PROGRESS line, stop. Never call vast-teardown.
```

Use `subagent_type=experiment-runner`.

### training-log-monitor

```
You are training-log-monitor for EXP-<N>.
Instance handle: runs/EXP-<N>/handles/<id>.json (read ssh_host, ssh_port, instance_id, gpu_name, num_gpus, gpu_ram from it; reconstruct tmux session as exp-<N>-<host-with-underscores>).
Plan: .claude/plans/<N>.md (read cell names from §Smoke launch commands, expected total_training_steps from §Vast.ai training footprint, WandB project from the launcher env — default project verl_compression_research, entity shamanework-pl).
Poll every 30 s for up to 40 min: SSH-probe per-cell logs (Traceback/Ray-unhandled/OOM/NaN), nvidia-smi per-GPU util, WandB scalars (every ~3rd poll). Rsync each cell's log + done flag + metrics to runs/EXP-<N>/ as the cell finishes. Append per-poll snapshots to runs/EXP-<N>/monitor-detail.log; do NOT spam PROGRESS.md during the loop.
Exit on: aggregate done.flag, 3 per-cell done_*.flag + tmux DEAD, tmux DEAD premature, GPU stall (all 4 GPUs 0% for 4 polls AND tmux ALIVE), env-failure (validate_config crash / vLLM init OOM / NCCL init fail / SSH unreachable >2 min), or 40 min timeout. Return a structured report with per-cell state + WandB scalars + recommendation (dispatch_analyst | teardown_and_fallback | teardown_only | continue_in_place_iteration). Never call vast-teardown.
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
Prepend LOG.md entry. Copy verdict to findings/M<X>/EXP-<N>.md. Rewrite STATUS.md.
If code_change=true AND verdict=PASS: draft PR against the project.yaml.github.code_repo (NEVER upstream verl-project/verl).
If ≥2 PASS entries in findings/M<X>/ and no SUMMARY.md: write findings/M<X>/SUMMARY.md and append `MILESTONE_PASS: M<X>` to PROGRESS.md.
Append one PROGRESS line; stop.
```

Use `subagent_type=log-writer`.

---

## Operator: how to review a plan (the manual step that replaced codex-bridge)

The harness used to autonomously dispatch a codex review of every plan
before launching expensive compute. That gate proved more flaky than
helpful (broker hangs, false-positive FAILs, override-anyway cycles). It
is now an **operator-driven manual step**, done before flipping
`status:planned → status:approved`.

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

### Optional: codex review of the plan

For larger plans or when you want a second opinion, invoke the
`codex-verify` skill manually:

```bash
# At the research/ root, with codex CLI available:
bash .claude/skills/codex-verify/run.sh \
  --mode verify \
  --out  runs/EXP-<N>/verify/$(date -u +%Y%m%dT%H%M%SZ).md \
  --plan .claude/plans/<N>.md \
  --cd   /Users/shamane/Documents/verl \
  --timeout 600 --stall 90

# Look at the output:
cat runs/EXP-<N>/verify/*.md
```

The output's first line is `VERIFY: PASS | VERIFY: CONCERNS | VERIFY: FAIL`.
You read it, decide whether to fix the plan / proceed / abandon. **The
harness does not look at this file.** Codex is now purely an advisory tool
for you.

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
- Never call `vast-teardown` or `vastai destroy` from this agent — only
  the Stop hook does that, on the next session Stop.
- If a `gh` call errors, log it and skip that issue for this tick. Do not
  abort the whole tick.
- If `runs.jsonl` is malformed, append the malformed line to
  `.claude/state/runs.jsonl.broken` for forensics, write a stub clean
  file, and continue.
- Idempotence: any action you take this tick must be safe to repeat on a
  future tick if the previous tick's results haven't been observed yet.
