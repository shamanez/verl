# researcher_steps.md — Running the automated R&D loop end-to-end

The single operator guide for driving the
**`shamanez/verl-compression-research`** harness autonomously. Happy-path
checklist with every human-intervention point called out, and enough
reference material (compute profile, codex bridge, troubleshooting) that
you do not need any other harness doc to operate.

If you are a Claude/agent reading this file, you are wrong layer — the
agents read `.claude/playbooks/triage.md` and `.claude/playbooks/orchestrator.md`
(top-level loop) and the leaf subagent definitions under `.claude/agents/`.

---

## What it does

A two-phase, GitHub-issue-driven research orchestration system:

```
GitHub issue (label: research:claim)
        │
        ▼
Session A — triage /loop 60m
   polls issues, spawns research-planner subagents → writes .claude/plans/<N>.md
        │
        ▼
[HUMAN GATE]  read plan → flip status:planned → status:approved
        │
        ▼
Session B — orchestrator /loop 30m
   for each approved plan:
     1. codex-bridge --mode=verify     (only if code_change: true)
     2. experiment-runner               (provision Vast.ai → train → done.flag)
     3. analyst                          (write verdict.md PASS/REVISE/STOP)
     4. log-writer                       (append LOG.md, copy to findings/, draft PR)
   REVISE iterations auto-promote once codex-verify passes (no human).
```

The research repo is **`shamanez/verl-compression-research`** (private).
The harness lives at **`/Users/shamane/Documents/verl/research/`**.

---

## 0. One-time prerequisites (on the laptop)

> **The operating directory is `/Users/shamane/Documents/verl/research/`.**
> Every `claude` session, every `gh` command, every `cat .claude/plans/*.md`
> in this doc assumes you have `cd`'d there first. Open one terminal,
> `cd` once, and stay.

```bash
cd /Users/shamane/Documents/verl/research        # ← do this once at the top of the session

# Secrets file — chmod 600, contains HF_TOKEN, WANDB_API_KEY, VAST_API_KEY
ls -l ~/.config/verl-research/secrets.env       # expect -rw-------

# gh default points at the research repo
gh repo set-default --view                       # shamanez/verl-compression-research

# Tooling
which claude gh vastai codex uv                  # all on PATH
codex doctor && codex login status               # green

# On the working branch — must be `vast-ai-workload` (or an `exp/*` branch
# inside an experiment-runner worktree). `main` tracks upstream `verl-project/verl`
# and is read-only — never edit on main, never PR to main.
git rev-parse --abbrev-ref HEAD                  # expect: vast-ai-workload
```

If any step fails, fix before starting the loop — the autonomous agents
will not bootstrap missing credentials.

### Branch + PR policy (read once, then forget — the harness enforces this)

| Branch (`shamanez/verl`) | Role | You write here? |
|---|---|---|
| `main` | tracks upstream `verl-project/verl` | **NO**. Read-only. The protect-upstream hook will block edits unless you're on an `exp/*` branch or `vast-ai-workload`. |
| `vast-ai-workload` | primary working branch | **YES**. Harness changes (research/ tree, examples/grpo_trainer/vast_*.sh) live here. |
| `exp/<N>-<slug>` | per-experiment, auto-created by `experiment-runner` for `code_change: true` plans | created from `vast-ai-workload`, pushed to origin BEFORE training launches (so the branch survives even if your laptop crashes), then PR'd back to `vast-ai-workload` on PASS. |

Two GitHub repos, completely separate roles:
- **`shamanez/verl`** (origin) — code. PRs land here. Base: `vast-ai-workload`.
- **`shamanez/verl-compression-research`** (research, `gh repo set-default`) — issue queue + verdict comments. **No PRs**.

### Vast.ai volatility safety

Vast.ai instances die without warning. The harness assumes this:

1. `experiment-runner` pushes `exp/<N>-<slug>` to origin BEFORE provisioning Vast — so the branch survives a laptop crash mid-train.
2. The locked Vast template (`verl-research-vllm020`, see `.claude/skills/vast-provision/templates.json`) clones `shamanez/verl @ vast-ai-workload` onto every fresh box. For `code_change: true` runs, the runner replaces this with the `exp/<N>-<slug>` branch via `exp.bundle`.
3. Every run dir on the box has a `commit-hotfix.sh` helper. Any in-container edit to `/workspace/verl` → run `bash /workspace/runs/EXP-<N>/commit-hotfix.sh "<msg>"`. This commits the change and drops a `.patch` file under `hotfix-patches/`. The `sync-metrics` hook rsyncs the patch back to the laptop every 5 min, so even if the instance dies the next minute, the hotfix lives on at `$PARENT/runs/EXP-<N>/hotfix-patches/`. (If `GH_PUSH_TOKEN` is set in the container env, the helper also pushes to origin directly — best case.)
4. `log-writer` surfaces any hotfix-patches in the PR body so you can `git am` them onto the merged branch.

---

## 1. File one or more hypothesis issues

On `shamanez/verl-compression-research`, open one issue per hypothesis.

**Required:**
- label: `research:claim`
- body contains: `hypothesis: <one paragraph, falsifiable, numeric>` (n/a for `kind: brainstorm` / `literature`)

**Pick a `kind:`** (drives orchestrator routing — see `.claude/plans/TEMPLATE.md` §Kind):

| kind | when | runs on Vast | output |
|---|---|---|---|
| `experiment` (default) | new hypothesis test | yes | verdict + LOG entry |
| `ablation` | knob sweep tied to a parent EXP — requires `depends_on: [EXP-N]` | yes | same; gates on parent PASS |
| `implementation` | code change to verify but NOT run — requires `code_change: true` + `target_modules:` | NO | codex-verify + draft PR on PASS |
| `brainstorm` | idea / proposal / discussion | NO | plan is the deliverable; iterate via comments; promote later |
| `literature` | math / derivation review | NO | codex math-rescue writes `findings/derivations/<topic>.md` |

This lets you run implementations, ablations, brainstorms, and full experiments **in parallel** from the same issue queue.

**Other body fields** (planner parses them; defaults from `.claude/project.yaml` fill absent values):

```
kind: experiment              # one of the kinds above; default `experiment` if missing
milestone: M3
baseline_run: EXP-07           # for ablation, the parent EXP (must have PASSED)
depends_on: [EXP-7]            # required for kind:ablation; optional for kind:experiment
budget_gpu_hr: 96
budget_dph_max: 24.0
gpu_filter: "num_gpus=8 gpu_name=H100 gpu_ram>=80 reliability>=0.95"   # overrides the default chain
code_change: false             # auto-true for kind:implementation
target_modules: []             # required when code_change: true
seed_replicates: 3
escalate_to_codex_if:
  - <pattern>
```

Issues missing `research:claim` are ignored. Issues missing `hypothesis:` (when applicable) produce a plan whose first acceptance criterion is `clarification_needed:`.

---

## 2. Phase 1 — Planning (Session A)

### Step 2a. Open a bash terminal and launch claude

```bash
cd /Users/shamane/Documents/verl/research        # ← always start here
claude                                            # opens the claude TUI
```

### Step 2b. Inside the claude session, type the loop command

```
/bg /loop 60m Read .claude/playbooks/triage.md and execute it.
```

The `/bg` prefix runs `/loop` in the background so the session stays
interactive. The triage playbook will now fire every 60 min.

### Step 2c. What happens automatically

Triage queries `gh issue list --label research:claim` and, for each issue
that does not yet have a plan file, spawns a `research-planner` subagent.
Each planner:

1. Writes `.claude/plans/<NUMBER>.md` (bare issue number, e.g. `1.md`).
2. Labels the issue `status:planned`.
3. Posts the plan as a comment on the issue.

You can close Session A after every open issue has a plan; triage will
re-spawn the next time you open a session in this directory.

### Step 2d. **[HUMAN GATE]** Read the plan and approve it

In a **separate bash terminal** (or with the `!` prefix inside the claude
session to shell out):

```bash
cd /Users/shamane/Documents/verl/research        # ← same dir

ls .claude/plans/                                # list plans triage produced
cat .claude/plans/1.md                           # read the plan for issue #1
```

Then decide:

| Decision | Action (run in bash) |
|---|---|
| Plan good | `gh issue edit <N> --add-label status:approved --remove-label status:planned` |
| Plan needs small fix | edit `.claude/plans/<N>.md` in your editor, then run the approve command above |
| Plan scope is wrong | `gh issue close <N>`, `rm .claude/plans/<N>.md`, refile the issue with a corrected body |

Approval (flipping the label to `status:approved`) is the **only mandatory
human action** on the happy path. Once at least one plan is approved,
move to Phase 2.

---

## 3. Phase 2 — Implementation (Session B)

### Step 3a. Decide: same terminal or new terminal?

You have two options:

- **Option A — parallel (recommended for first run):** open a **second** bash terminal, leave Session A running so triage keeps watching for new issues. Both loops run independently.
- **Option B — sequential:** in your existing terminal, `/exit` Session A, then immediately start Session B in the same shell. Simpler but triage stops while you do this.

Either way:

```bash
cd /Users/shamane/Documents/verl/research        # ← same dir, always
claude
```

### Step 3b. Inside the claude session, type

```
/bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it.
```

### Step 3c. What happens automatically

Every 30 min the orchestrator walks each plan through its state machine:

| State | Auto-dispatch |
|---|---|
| `status:approved` + `code_change: true` + no verify yet | `codex-bridge --mode=verify` |
| latest verify is `VERIFY: PASS` or `CONCERNS` | `experiment-runner` |
| latest verify is `VERIFY: FAIL` | demote label to `status:planned`, post critique, stop |
| `status:approved` + `code_change: false` + no runs.jsonl row | `experiment-runner` |
| `runs/<ID>/done.flag` OR remote tmux dead + metrics present | `analyst` |
| verdict `REVISE` | open child issue with `next_actions:`, route through codex-verify, **auto-approve on PASS (no human)** |
| verdict `PASS` | `log-writer` (appends LOG.md, copies to `findings/M<N>/`, opens draft PR if `code_change: true`) |
| ≥2 PASS verdicts in a milestone | `codex-bridge --mode=adversarial` |

`experiment-runner` defaults to **a single Vast.ai node, $24/hr ceiling,
96 GPU-hr total budget**, walking a GPU-filter fallback chain. The runner
picks the first tier with ≥1 offer ≤ `max_dph`:

```yaml
gpu_count: 1
gpu_filter_chain:
  - "num_gpus=4 gpu_name=H200 gpu_ram>=140 reliability>=0.95 rentable=true verified=true"   # preferred — half the $/hr of 8×H100 for ≤4B models
  - "num_gpus=8 gpu_name=H100 gpu_ram>=80 reliability>=0.95 rentable=true verified=true"
  - "num_gpus=4 gpu_name=H100 gpu_ram>=80 reliability>=0.95 rentable=true verified=true"
max_dph: 24.0
max_gpu_hr: 96
```

The chosen tier is recorded in the `PROVISIONED` row of `runs.jsonl`. The
runner reads `.gpu_count` from each handle JSON to set `NGPUS_PER_NODE`
for the training launch. M0 smoke (`milestone:M0`) deliberately overrides
this with a cheap CPU or RTX-3090 filter — see §6.

### Step 3d. What to expect after approval — the happy-path timeline

This is a typical M0 run from the moment you flip `status:approved`. All
events happen automatically; "your action" entries are the only places
you intervene.

| Time after approval | Event | Your action |
|---|---|---|
| 0–30 min | orchestrator's next tick sees `status:approved` + no `runs.jsonl` row → dispatches `experiment-runner` | none |
| +5 min | `experiment-runner` calls `vast-provision`, writes `runs/EXP-<N>/handles/<id>.json`, opens SSH, rsyncs the verl repo, runs `docker-bootstrap.sh`, launches training under remote tmux | `cat STATUS.md` to confirm `PROVISIONED → RUNNING` |
| +5 min onward | `sync-metrics.sh` hook tails the remote `train.log` into `runs/EXP-<N>/metrics/` every 5 min | optionally `tail -f runs/EXP-<N>/metrics/train.log` |
| training ends | wrapper touches `runs/EXP-<N>/done.flag` → orchestrator dispatches `analyst` | none |
| +1 min | analyst writes `runs/EXP-<N>/verdict.md` with **PASS** / **REVISE** / **STOP** | none |
| if PASS | orchestrator dispatches `log-writer` → appends `LOG.md`, copies the verdict into `findings/M<N>/`, opens a **draft PR** on the research repo (only when `code_change: true`) | optional: `gh pr view` to review the draft |
| if REVISE | orchestrator opens a child issue with the analyst's `next_actions:`, routes it through `codex-bridge --mode=verify`, auto-approves on PASS, and loops back to runner | **none** (REVISE is fully automated) |
| if STOP | the recipe itself is broken on this tier; orchestrator stops dispatch and emits `MANUAL_REVIEW_NEEDED:` to `PROGRESS.md` | read `verdict.md`, fix the plan or escalate to codex `code-rescue` |
| verdict in place | `teardown-finished-runs.sh` (Stop hook) destroys the Vast.ai instance on the next session Stop | none — instance is gone, money stops |

If you only filed one issue, you're done. If you filed several, the
orchestrator processes them in parallel as their plans get approved.

### Passive hooks running in every session

| Hook | Trigger | Purpose |
|---|---|---|
| `kill-switch.sh` | PreToolUse | `~/.claude-kill-switch` halts every agent |
| `protect-upstream.sh` | PreToolUse | refuses writes under verl/ unless on `exp/*` or `vast-ai-workload` branch |
| `sync-metrics.sh` | PostToolUse Bash (5-min debounce) | pulls remote `train.log` for RUNNING experiments |
| `teardown-finished-runs.sh` | Stop | destroys Vast.ai instances whose run is verdict'd / dead / over-budget |
| `commit-on-stop.sh` | Stop | autosaves uncommitted research/ changes |
| `on-session-start.sh` | SessionStart | heartbeat + $/hr burn rate into `~/.claude-events.log` |

### Codex bridge

Four modes, all orchestrator-driven, never user-invoked:

| Mode | When | Output |
|---|---|---|
| `verify` | structural gate before any `code_change: true` launch or REVISE auto-promotion | `runs/EXP-<N>/verify/<ts>.md` with `VERIFY: PASS|FAIL|CONCERNS` |
| `code-rescue` | runner posted `STUCK: <ctx>` to PROGRESS.md | `runs/EXP-<N>/rescue/<ts>.md` |
| `math-rescue` | planner/analyst posted `RESCUE_REQUEST: math <ctx>` | `findings/derivations/<topic>.md` |
| `adversarial` | log-writer posted `MILESTONE_PASS: M<X>` | `findings/M<X>/codex-review.md` |

All invocations go through `.claude/skills/codex-verify/run.sh`, which
wraps `codex exec` directly (bypassing the `codex-companion` plugin —
its v1.0.4 broker dies mid-task on this machine, two-for-two failures
2026-05-24) with two safety layers:

- **Hard wall-clock timeout** (default 600 s) → SIGKILL + `TIMEOUT: hard wall-clock 600s exceeded`.
- **Stall watchdog** (default 90 s) → if the output file stops growing for 90 s while the child is alive, kill + `TIMEOUT: stalled 90s with no output growth`.

A timeout NEVER auto-approves a run. It demotes the plan back to
`status:planned` and emits `MANUAL_REVIEW_NEEDED: EXP-<N>` to PROGRESS.md
so you can intervene. If the codex CLI itself starts misbehaving (rare —
`codex doctor` is the source of truth):

```bash
codex doctor && codex login status        # check
!codex login                                # interactive re-auth if dropped
# the orchestrator's next tick re-dispatches verify on its own
```

---

## 4. Monitor (no action required if everything is PASS)

```bash
cat STATUS.md                                    # orchestrator rewrites every 30 min
tail -30 PROGRESS.md                             # append-only audit
gh issue list --state open --label research:claim
jq -c . .claude/state/runs.jsonl
python scripts/check_budget.py --month
```

---

## 5. Every human-intervention point (the only times you act)

| When | Why | Action |
|---|---|---|
| Before starting | bootstrap | secrets in place, gh default set, CLIs authed |
| Issue planned | hybrid gate | flip `status:planned` → `status:approved` |
| `verdict.md` = `REVISE` | **auto-handled** — child issue + codex-verify | none |
| `MANUAL_REVIEW_NEEDED:` in PROGRESS.md | codex timeout, no GPU offers, fatal error | read line, fix root cause, re-approve |
| `VERIFY: FAIL` posted on issue | structural critique from codex | edit plan or diff target, re-approve |
| Budget cap exceeded | `check_budget.py` fails | edit `budget.json` or pause experiments |
| Vast instance not torn down | Stop hook didn't fire | `bash .claude/skills/vast-teardown/run.sh <instance_id>` |
| Emergency stop | pause every agent tool call | `touch ~/.claude-kill-switch` (resume: `rm ~/.claude-kill-switch`) |
| Want a milestone goal predicate | optional 3rd session | `/goal milestone M<N> has >=2 PASS experiments AND research/findings/M<N>/SUMMARY.md exists` |

---

## 6. M0 smoke test — validate the harness for ≤ $0.20

File one issue on the research repo:

```
title: M0 SMOKE — wire end-to-end research loop with no-op experiment
labels: research:claim, milestone:M0, kind:experiment
body:
  hypothesis: The pipeline (triage → planner → codex-verify → runner → analyst → log-writer) processes a no-op experiment end-to-end, terminating in a PASS verdict and a LOG.md entry, within 30 min wall-clock, using a CPU-only Vast.ai instance with hard dph <= 0.10.
  milestone: M0
  budget_gpu_hr: 0.5
  budget_dph_max: 0.10
  gpu_filter: "rentable=true verified=true cpu_cores>=2 disk_space>=8"
  code_change: false
  seed_replicates: 1
```

Run Phase 1 + Phase 2 with a faster orchestrator tick:

```
/bg /loop 5m Read .claude/playbooks/orchestrator.md and execute it.
```

Expected inside 30 min: provisioned → no-op `done.flag` → `verdict.md` PASS
→ LOG.md entry → Vast instance torn down → `runs.jsonl` shows `TORN_DOWN`
→ spend < $0.10.

After M0 passes, file a second smoke with `code_change: true` and
`target_modules: []` to exercise the `codex-bridge --mode=verify` path.

---

## 7. Common failure modes

| Symptom | Fix |
|---|---|
| Triage fires, no plan appears | check `research:claim` label present, grep `PROGRESS.md` for planner failure |
| Orchestrator picks up unapproved plan | label was set to `status:approved` by mistake → demote |
| Worker stuck | open the issue, read latest comment + `runs/EXP-<N>/` |
| `VERIFY: FAIL` | read critique in `runs/EXP-<N>/verify/<ts>.md`; fix plan or diff; re-approve |
| Codex hangs | hard 600 s + stall 90 s watchdogs kick in → `TIMEOUT:` / `BROKER_DIED:` lands in PROGRESS.md as `MANUAL_REVIEW_NEEDED:` |
| Vast instance not torn down | run `bash .claude/skills/vast-teardown/run.sh <instance_id>` |
| Loop stopped | session ended → re-run `/bg /loop` in a new session |
| `/bg` says "Nothing to background yet" | must prefix `/loop`: `/bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it.` |
| Triage/orchestrator logs `Agent tool unavailable in this session` | Loop prompt is the old `Use the X subagent.` form — subagents can't dispatch in CC 2.1.x. Update the loop prompt to the `Read .claude/playbooks/<name>.md and execute it.` form. |
| protect-upstream refusing edit | not on an `exp/*` branch — `git rev-parse --abbrev-ref HEAD` inside the worktree |

---

## 8. File layout (where things live)

```
verl/research/
├── researcher_steps.md     this file — the only human-facing harness doc
├── README.md               one-paragraph orientation
├── STATUS.md               orchestrator dashboard (rewritten every 30 min by orchestrator playbook)
├── PROGRESS.md             append-only structured audit (only meaningful markers — see file header)
├── LOG.md                  newest-first research log (one entry per PASS / STOP verdict)
├── budget.json             $ caps
├── scripts/                check_budget.py, analyze.py, diff_against_baseline.py, prepare_dataset.py
├── runs/EXP-<N>/           per-experiment config, handles, metrics, verdict, verify, rescue
├── findings/M<N>/          PASS verdicts, SUMMARY.md, codex-review.md, derivations/
└── .claude/
    ├── project.yaml        ★ single source of truth — repo, secrets, vast template, default compute, branch policy
    ├── playbooks/          triage, orchestrator (top-level workflows — read by the /loop session, NOT subagents)
    ├── agents/             5 leaf subagents: research-planner, codex-bridge, experiment-runner, analyst, log-writer
    ├── plans/              TEMPLATE.md + <N>.md per issue
    ├── hooks/              6 hook scripts (see §3)
    ├── skills/             vast-provision, vast-teardown, codex-verify
    └── state/              runs.jsonl, vast-handles/, .last-orchestrator-tick

verl/major-goal/                           ← HUMAN-ONLY reference. Agents must not read this.
├── core-task.md                          research goal & method (compression two-circuit design)
└── LLM_adaptation_neurips.pdf            paper draft

verl/examples/grpo_trainer/
└── vast_baseline_qwen25_1p5b_grpo_gsm8k.sh   real dense GRPO baseline launcher
                                              (on branch vast-ai-workload)

verl/CLAUDE.md              fork-specific agent instructions
```

**Why this is the harness**: agents read only their own role contract + `.claude/project.yaml` + the per-experiment plan file. They don't need to grep around for the secrets path, the vast template hash, or the gh-default repo — those are project.yaml. They don't need to read the research goal — that's `major-goal/` (human-only). This is what makes the harness transferable to a new research project: edit `project.yaml` + the GitHub repo + `major-goal/` and the agent machinery comes along unchanged.

---

## 9. Porting to a new research project

This harness is **transferable**. To run a different major research goal on the same machinery:

1. **Edit `.claude/project.yaml`** — the canonical truth for repo name, secrets path, vast template, default compute chain, branch policy. All agents/playbooks read from it; the inline mirrors in their Operating-context blocks track this file.
2. **Update the GitHub repo** — create a new private repo, set `gh repo set-default <org>/<new-repo>`, mirror the label scheme (`research:claim`, `status:planned/approved/running/pass/revise/stop/done`, `kind:*`, `milestone:M*`).
3. **Update the Vast.ai template** — either reuse `verl-research-vllm020` (if you're still on the same verl fork) or `vastai create template ...` for a new one. Record its hash in `templates.json` and `project.yaml`.
4. **Update `major-goal/`** — replace `core-task.md` and the paper PDF with the new goal's content. The harness itself ignores this directory.
5. **Edit `verl/CLAUDE.md`** — model choice, hardware mandate, references. Most other agent files are project-agnostic.

What survives the port unchanged: the 5 agents, 2 playbooks, 7 hooks (one was removed for noise), 3 skills, TEMPLATE.md, and the issue-first state machine. The harness logic is decoupled from the research goal.

## 10. Quick reference

```bash
# Start the loop
cd /Users/shamane/Documents/verl/research && claude
#   Session A:  /bg /loop 60m Read .claude/playbooks/triage.md and execute it.
#   Session B:  /bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it.

# Approve a plan
gh issue edit <N> --add-label status:approved --remove-label status:planned

# State
cat STATUS.md
tail -30 PROGRESS.md
jq -c . .claude/state/runs.jsonl

# Cost
python scripts/check_budget.py --month
python scripts/check_budget.py runs/EXP-7

# Manual teardown
bash .claude/skills/vast-teardown/run.sh <instance_id>

# Kill switch
touch ~/.claude-kill-switch       # pause
rm    ~/.claude-kill-switch       # resume
```
