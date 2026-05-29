# researcher_steps.md — operator guide for the research harness

Single human-facing doc. Drives the
`shamanez/verl-compression-research` issue queue end-to-end against the
harness at `/Users/shamane/Documents/verl/research/`.

If you're an agent reading this — wrong layer. Agents read
`.claude/playbooks/{triage,orchestrator}.md` and the leaf subagents in
`.claude/agents/`.

---

## The loop, in one picture

```
file GitHub issue (label: research:claim)
        │
        ▼
Session A — triage /loop 60m
   spawns research-planner → writes .claude/plans/<N>.md + posts comment
        │
        ▼
[HUMAN GATE]
   read the plan, optionally run codex-verify manually as a 2nd opinion,
   then flip status:planned → status:approved
        │
        ▼
Session B — orchestrator /loop 30m
   experiment-runner (provision Vast → train → done.flag)
     → analyst (writes verdict.md)
       → log-writer (LOG.md + findings/ + draft PR on PASS)
   REVISE creates a child issue at status:planned → back through the human gate
```

The only mandatory human action on the happy path is the label flip.

---

## 0. One-time prerequisites

Always `cd` to the harness root first:

```bash
cd /Users/shamane/Documents/verl/research
```

Check:

```bash
ls -l ~/.config/verl-research/secrets.env       # -rw------- (HF + WandB + VAST keys)
gh repo set-default --view                       # shamanez/verl-compression-research
which claude gh vastai codex uv                  # all on PATH
codex doctor && codex login status               # green (optional codex review needs this)
git rev-parse --abbrev-ref HEAD                  # vast-ai-workload
```

### Branch / PR policy

| Branch (`shamanez/verl`) | Role | Write here? |
|---|---|---|
| `main` | tracks upstream `verl-project/verl` | NO. The protect-upstream hook blocks edits unless you're on `exp/*` or `vast-ai-workload`. |
| `vast-ai-workload` | harness + launcher edits | YES. Default working branch. |
| `exp/<N>-<slug>` | per-experiment, auto-created by experiment-runner for `code_change: true` plans, PR'd back to `vast-ai-workload` on PASS | runner-managed. |

Two repos, separate purposes:
- **`shamanez/verl`** (origin) — code. PRs land here, base = `vast-ai-workload`. Never `main`.
- **`shamanez/verl-compression-research`** (research, `gh repo set-default`) — issue queue + verdict comments. No PRs.

---

## 1. File a hypothesis issue

On `shamanez/verl-compression-research`:

**Required**: label `research:claim`. Body contains `hypothesis: <falsifiable, numeric>` (n/a for brainstorm/literature).

**Pick a `kind:`** (see `.claude/plans/TEMPLATE.md` §Kind):

| kind | runs on Vast | output |
|---|---|---|
| `experiment` (default) | yes | verdict + LOG entry |
| `ablation` (requires `depends_on:`) | yes | same; gates on parent PASS |
| `implementation` (`code_change: true`, no Vast launch) | NO | plan is the deliverable; draft PR after approval |
| `brainstorm` | NO | plan is the deliverable; iterate as comments |
| `literature` | NO | plan/issue is the deliverable |

Common body fields (planner reads them; defaults from `.claude/project.yaml`):

```
kind: experiment
milestone: M3
baseline_run: baseline                 # the dense control (= comm-eff OFF); or 'none'
depends_on: [EXP-N]                    # required for ablation
budget_gpu_hr: 96
budget_dph_max: 24.0
code_change: false                     # auto-true for kind:implementation
target_modules: []
seed_replicates: 1
```

---

## 2. Phase 1 — Planning

```bash
cd /Users/shamane/Documents/verl/research
claude
```

Inside the claude session:

```
/bg /loop 60m Read .claude/playbooks/triage.md and execute it.
```

Triage polls open `research:claim` issues every 60 min and spawns a
`research-planner` per unplanned issue. Each planner:

1. Writes `.claude/plans/<N>.md`.
2. Labels the issue `status:planned`.
3. Posts the plan as a comment ending with an **Operator review** footer
   that includes the exact `codex-verify` invocation.

Triage closes itself between ticks; spin up Session B in parallel (see §3).

### The human gate (the one mandatory step)

```bash
# Read the plan
cat .claude/plans/<N>.md

# Optional second opinion from codex (the harness does NOT read this file —
# it's purely advisory for you):
mkdir -p runs/EXP-<N>/verify
bash .claude/skills/codex-verify/run.sh \
    --mode verify \
    --out  runs/EXP-<N>/verify/$(date -u +%Y%m%dT%H%M%SZ).md \
    --plan .claude/plans/<N>.md \
    --cd   /Users/shamane/Documents/verl
cat runs/EXP-<N>/verify/*.md       # first line: VERIFY: PASS | CONCERNS | FAIL
```

| Decision | Action |
|---|---|
| Plan good | `gh issue edit <N> --add-label status:approved --remove-label status:planned` |
| Small fix | edit `.claude/plans/<N>.md`, then approve |
| Wrong scope | `gh issue close <N>`, `rm .claude/plans/<N>.md`, refile |

---

## 3. Phase 2 — Implementation

Open a second bash terminal (so Session A keeps watching for new issues):

```bash
cd /Users/shamane/Documents/verl/research
claude
```

Inside:

```
/bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it.
```

Each tick, the orchestrator advances every approved plan:

| State | Auto-dispatch |
|---|---|
| `status:approved`, no `runs.jsonl` row | `experiment-runner` |
| `runs.jsonl` row is `RUNNING`, no monitor active | `training-log-monitor` (background) |
| `done.flag` exists OR tmux dead with metrics | `analyst` |
| `verdict.md` says PASS | `log-writer` (appends LOG, copies to findings/, drafts PR on `code_change=true`) |
| `verdict.md` says REVISE | opens child issue at `status:planned`, **stops** — you review the child plan |
| `verdict.md` says STOP | `log-writer`, label `status:stop` |

`experiment-runner` defaults: **single Vast.ai node, $24/hr per-instance cap,
96 GPU-hr total**, walking the 4×H200 → 8×H100 fallback chain unless
the plan overrides it.

### When to invoke codex manually (outside the autonomous loop)

The orchestrator never dispatches codex. The operator runs
`codex-verify` manually for:

| Use | Mode |
|---|---|
| Plan review before approval | `--mode verify` |
| Diagnose a STUCK run | `--mode code-rescue` |
| Walk a derivation | `--mode math-rescue` |
| Adversarial milestone review | `--mode adversarial` |

Recipes in `.claude/skills/codex-verify/SKILL.md`. The harness does not
look at the output files; you read them and decide.

---

## 4. Monitor

```bash
cat .claude/state/STATUS.md                     # orchestrator rewrites every tick
tail -30 PROGRESS.md                            # append-only audit
gh issue list --state open --label research:claim
jq -c . .claude/state/runs.jsonl
python scripts/check_budget.py --month
```

### Passive hooks (in every session)

| Hook | Trigger | Purpose |
|---|---|---|
| `kill-switch.sh` | PreToolUse | `touch ~/.claude-kill-switch` halts every agent tool call |
| `protect-upstream.sh` | PreToolUse | blocks verl/ writes unless on `exp/*` or `vast-ai-workload` |
| `sync-metrics.sh` | PostToolUse Bash (5-min debounce) | pulls remote `train.log` for RUNNING experiments |
| `teardown-finished-runs.sh` | Stop | destroys Vast instances whose run is verdicted / dead / over-budget |
| `commit-on-stop.sh` | Stop | autosaves uncommitted research/ changes |

---

## 5. Every human-intervention point

| Trigger | Action |
|---|---|
| Issue planned | Read plan, optionally codex-verify, flip `status:planned` → `status:approved` |
| REVISE child issue appears | Same as above (review child plan, approve when ready) |
| `MANUAL_REVIEW_NEEDED:` in PROGRESS.md | Read the line, fix the root cause, re-approve |
| Budget cap exceeded | Edit `budget.json` or pause |
| Vast instance not torn down | `bash .claude/skills/vast-teardown/run.sh <instance_id>` |
| Emergency stop | `touch ~/.claude-kill-switch` (resume: `rm ~/.claude-kill-switch`) |

---

## 6. Common failure modes

| Symptom | Fix |
|---|---|
| Triage fires, no plan appears | check label is `research:claim`; grep PROGRESS.md for planner failure |
| Orchestrator picks up unapproved plan | label was set to `status:approved` by mistake → demote |
| Codex hangs during your manual review | the wrapper has 600s hard + 90s stall timeouts; output starts with `TIMEOUT:` / `BROKER_DIED:` |
| Codex CLI itself broken | `codex doctor && codex login status`; `!codex login` to re-auth |
| Vast instance not torn down | `bash .claude/skills/vast-teardown/run.sh <instance_id>` |
| Loop stopped | session ended → re-run `/bg /loop` in a new session |
| `/bg` says "Nothing to background yet" | must prefix `/loop`: `/bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it.` |
| protect-upstream refusing an edit | not on `exp/*` or `vast-ai-workload` — `git rev-parse --abbrev-ref HEAD` |

---

## 7. File layout

```
research/
├── researcher_steps.md     this file (operator guide)
├── README.md               one-paragraph orientation
├── LOG.md                  newest-first PASS/STOP entries
├── PROGRESS.md             append-only structured audit
├── runs/
│   ├── EXP-<N>/                    per-experiment dirs (runtime artifacts)
│   └── SUMMARY.md                  durable record: baseline, method, knobs, tried-so-far
├── findings/
│   ├── M<N>/EXP-<N>.md             per-experiment PASS findings
│   └── derivations/                math-rescue outputs (operator-invoked)
├── notes/                          conceptual docs (memory, masking semantics, ...)
├── scripts/                        analyze.py, check_budget.py, diff_against_baseline.py
└── .claude/
    ├── project.yaml                single source of truth (repo, secrets, vast template, defaults, branch policy)
    ├── playbooks/                  triage.md, orchestrator.md
    ├── agents/                     research-planner, experiment-runner, analyst, log-writer, training-log-monitor
    ├── plans/                      TEMPLATE.md, baseline.md, <N>.md per issue
    ├── hooks/                      kill-switch, protect-upstream, sync-metrics, teardown-finished-runs, commit-on-stop
    ├── skills/                     vast-provision, vast-teardown, codex-verify (operator-invoked)
    └── state/                      STATUS.md, runs.jsonl, .last-orchestrator-tick

verl/examples/grpo_trainer/
├── vast_baseline_qwen25_1p5b_grpo_gsm8k.sh             dense reference launcher
└── vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh    comm-eff reference launcher

verl/CLAUDE.md                      fork-specific agent instructions
```

The harness is **transferable**: edit `.claude/project.yaml`, swap the
GitHub repo, rewrite `.claude/GOAL.md` for the new project, and the
agents/playbooks/hooks come along unchanged.

---

## 8. Quick reference

```bash
# Start the loop
cd /Users/shamane/Documents/verl/research && claude
#   Session A:  /bg /loop 60m Read .claude/playbooks/triage.md and execute it.
#   Session B:  /bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it.

# Approve a plan
gh issue edit <N> --add-label status:approved --remove-label status:planned

# Manually review a plan with codex (optional; advisory only)
bash .claude/skills/codex-verify/run.sh \
    --mode verify --out runs/EXP-<N>/verify/$(date -u +%Y%m%dT%H%M%SZ).md \
    --plan .claude/plans/<N>.md --cd /Users/shamane/Documents/verl

# Status
cat .claude/state/STATUS.md
tail -30 PROGRESS.md
jq -c . .claude/state/runs.jsonl

# Cost
python scripts/check_budget.py --month

# Manual teardown
bash .claude/skills/vast-teardown/run.sh <instance_id>

# Kill switch
touch ~/.claude-kill-switch       # pause
rm    ~/.claude-kill-switch       # resume
```
