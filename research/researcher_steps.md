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
   read the plan, then flip status:planned → status:approved
        │
        ▼
Session B — orchestrator /loop 30m
   experiment-runner (provision Vast → train → done.flag)
     → analyst (writes verdict.md)
       → log-writer (LOG.md + runs/SUMMARY.md + draft PR on PASS)
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
which claude gh vastai uv                        # all on PATH
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
| `analysis` (offline kill-gate) | NO (local, GPU-free) | GO/NO-GO verdict + LOG entry; analyst runs the kill-gate locally |

Common body fields (planner reads them; defaults from `.claude/project.yaml`):

```
kind: experiment
milestone: M3
baseline_run: baseline                 # the dense control (= comm-eff OFF); or 'none'
depends_on: [run-N]                    # required for ablation
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
3. Posts the plan as a comment ending with an **Operator review** footer.

Triage closes itself between ticks; spin up Session B in parallel (see §3).

### The human gate (the one mandatory step)

```bash
# Read the plan
cat .claude/plans/<N>.md
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

### Choosing the Vast.ai account (team vs private)

By default every box provisions + bills the **private** account (`VAST_API_KEY`).
To run a session on the shared **team** account (`VAST_API_KEY_TEAM`, the
"Pluralis Research" team) instead, just say so in the loop instruction:

```
/bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it. Use the team account.
```

…or `Use the private account.` to be explicit. The orchestrator passes
`vast_account=team|private` to every `experiment-runner` dispatch; the runner
`export`s `VAST_ACCOUNT` for `vast-provision` and records `vast_account` on the
ledger row, so `vast-teardown` **and** the automatic teardown Stop hook destroy
the box with the *same* account's key (a team box is never orphaned under the
private key, and vice-versa). Nothing else changes — same template, same chain,
same caps.

Deterministic alternative (e.g. a manual `/vast-provision` outside the loop, or
to force a whole shell): `export VAST_ACCOUNT=team` before launching `claude`,
and the skills pick it up directly. A plan may also pin `vast_account: team` in
its `## Compute budget` block to override the session default for that one
experiment. Both keys live in `secrets.env` (`VAST_API_KEY` + `VAST_API_KEY_TEAM`);
the harness never echoes either.

Each tick, the orchestrator advances every approved plan:

| State | Auto-dispatch |
|---|---|
| `status:approved`, no `runs.jsonl` row | `experiment-runner` |
| `runs.jsonl` row is `RUNNING`, no monitor active | `training-log-monitor` (background) |
| `done.flag` exists OR tmux dead with metrics | `analyst` |
| `verdict.md` says PASS | `log-writer` (appends LOG, updates runs/SUMMARY.md, drafts PR on `code_change=true`) |
| `verdict.md` says REVISE | opens child issue at `status:planned`, **stops** — you review the child plan |
| `verdict.md` says STOP | `log-writer`, label `status:stop` |

`experiment-runner` defaults: **single Vast.ai node, $24/hr per-instance cap,
96 GPU-hr total**, walking the 4×H200 → 8×H100 fallback chain unless
the plan overrides it.

---

## 3b. Fast path — attach an already-running box (skip provisioning)

Got a box already up (a warm box from a prior run, or one you provisioned by hand)?
Skip the ~1–3 min provision + ~5–8 min warm-up and start immediately. The
`vast-attach` skill registers it as an **EXTERNAL** handle (provenance: the harness
didn't provision it). It is **still torn down** after its run completes or on request —
teardown is a must; `external` is not an exemption.

```bash
cd /Users/shamane/Documents/verl/research

# Real Vast box — the API fills in ssh/gpu details from just the instance id:
bash .claude/skills/vast-attach/run.sh --instance-id 41680420 --account team

# …or give them explicitly (any box, Vast or not):
bash .claude/skills/vast-attach/run.sh \
  --instance-id 41680420 --ssh-host 84.8.106.109 --ssh-port 40206 --num-gpus 4 --account team
```

This writes `runs/ATTACH-<id>/handles/<id>.json` plus a `RUNNING`+`external` ledger row.

### Mode 1 — drive it by hand (fastest; interactive `claude`)

```bash
claude
```
Then hand the box to the session:
```
A running Vast box is attached at runs/ATTACH-<id>/handles/<id>.json. SSH in using its ssh_login;
make sure /workspace/verl is on vast-ai-workload and current (git fetch && git checkout
vast-ai-workload && git pull --ff-only), then launch the baseline in a tmux and tail the first 50
lines of train.log to confirm it's stepping:
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
Report once steps are flowing. When the run is done — or when I say stop — TEAR THE BOX DOWN:
  bash .claude/skills/vast-teardown/run.sh <instance_id>
```
> Purest manual run: add `--no-register` to `vast-attach` (or skip the skill entirely and just
> SSH in). With no ledger row the harness never tracks it — so **you** must tear it down when done
> (`bash .claude/skills/vast-teardown/run.sh <id>`, or destroy it on vast.ai).

### Mode 2 — run the autonomous orchestrator loop against your box

Two ways. Either way the box is **EXTERNAL** (provenance only) and is **torn down after its run**, like any box:

**(a) Session directive — name the box in the loop instruction** (simplest; mirrors the
account selector). The orchestrator attaches your box for the next eligible experiment
instead of provisioning:
```
/bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it. Use the team account.
Use box instance_id=41680420 ssh_host=84.8.106.109 ssh_port=40206 num_gpus=4 this session
instead of provisioning — it is EXTERNAL (provenance only; still torn down after its run), one
experiment at a time on it.
```
The loop runs an approved experiment on that box (provision → SKIPPED; attach → train → monitor →
analyze → log), then **tears the box down at the run's verdict** (teardown is a must). One box runs
one experiment at a time; it does NOT persist for a later run — re-attach a box for another.

**(b) Per-plan — pin the box in a specific plan's `## Compute budget`** (precise; overrides (a)):
```yaml
attach_box: { instance_id: 41680420, ssh_host: 84.8.106.109, ssh_port: 40206, num_gpus: 4, account: team }
```
Approve the plan as usual; the runner attaches that box for that experiment.

Either way the box is torn down after its experiment completes (or on request) — teardown is a
must; `external` is provenance, not protection.

### Tearing down an attached box

It happens **automatically** after the run (verdict written / heartbeat stale), exactly like a
provisioned box. To tear it down sooner — or for a `--no-register` box the harness isn't tracking —
do it yourself (no `--force` needed):
```bash
bash .claude/skills/vast-teardown/run.sh <instance_id>   # or just destroy it on vast.ai
```

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
| Issue planned | Read plan, flip `status:planned` → `status:approved` |
| REVISE child issue appears | Same as above (review child plan, approve when ready) |
| `MANUAL_REVIEW_NEEDED:` in PROGRESS.md | Read the line, fix the root cause, re-approve |
| Budget cap exceeded | Edit `budget.json` or pause |
| Vast instance not torn down | `bash .claude/skills/vast-teardown/run.sh <instance_id>` |
| Emergency stop | `touch ~/.claude-kill-switch` (resume: `rm ~/.claude-kill-switch`) |
| Use an already-running box (skip provisioning) | `bash .claude/skills/vast-attach/run.sh --instance-id <id> [...]` — see §3b. EXTERNAL (provenance); torn down after its run like any box |
| Attached box still up after work | tear it down — `bash .claude/skills/vast-teardown/run.sh <id>` (also auto-torn-down on verdict/stale) |

---

## 6. Common failure modes

| Symptom | Fix |
|---|---|
| Triage fires, no plan appears | check label is `research:claim`; grep PROGRESS.md for planner failure |
| Orchestrator picks up unapproved plan | label was set to `status:approved` by mistake → demote |
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
│   ├── <run-id>/                   runtime artifact dirs
│   └── SUMMARY.md                  durable record: baseline, method, knobs, tried-so-far
├── scripts/                        analyze.py, check_budget.py, diff_against_baseline.py
└── .claude/
    ├── project.yaml                single source of truth (repo, secrets, vast template, defaults, branch policy)
    ├── playbooks/                  triage.md, orchestrator.md
    ├── agents/                     research-planner, experiment-runner, analyst, log-writer, training-log-monitor
    ├── plans/                      TEMPLATE.md, <N>.md per issue
    ├── hooks/                      kill-switch, protect-upstream, sync-metrics, teardown-finished-runs, commit-on-stop
    ├── skills/                     vast-provision, vast-attach, vast-teardown, de-bloat
    └── state/                      STATUS.md, runs.jsonl, .last-orchestrator-tick

verl/examples/grpo_trainer/
├── vast_baseline_qwen25_1p5b_grpo_gsm8k.sh             dense reference launcher
├── vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh  THE comm-eff baseline (signed_ema, 20/20 collapse regime, resp 1024)
└── vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh    generic comm-eff engine

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

# Status
cat .claude/state/STATUS.md
tail -30 PROGRESS.md
jq -c . .claude/state/runs.jsonl

# Cost
python scripts/check_budget.py --month

# Attach an already-running box (skip provisioning; EXTERNAL provenance, still torn down after its run)
bash .claude/skills/vast-attach/run.sh --instance-id <id> --account team

# Manual teardown (any box, including attached/external — teardown is a must)
bash .claude/skills/vast-teardown/run.sh <instance_id>

# Kill switch
touch ~/.claude-kill-switch       # pause
rm    ~/.claude-kill-switch       # resume
```
