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
Session A — triage  (/goal-driven · ~60m cadence)
   spawns research-planner → writes .claude/plans/<N>.md + posts comment
        │
        ▼
[HUMAN GATE]
   read the plan, then flip status:planned → status:approved
        │
        ▼
Session B — orchestrator  (/goal-driven · ~30m cadence)
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
/bg /goal Every open research:claim issue has a .claude/plans/<N>.md (my printed triage ledger shows unplanned=0), OR I have logged a triage error. Until then, read .claude/playbooks/triage.md and execute one tick, pacing ~60m between checks. Stop after 100 turns.
```

> **Why `/goal`, not `/loop`?** `/goal` keeps the session working until the condition
> holds (then auto-stops), instead of firing blindly on a clock — see §3a.

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
/bg /goal Every status:approved plan has reached a terminal verdict (PASS/STOP) with its box TORN_DOWN and LOG.md updated — confirmed by the plan-completion ledger I print each tick from runs.jsonl + verdict.md + WandB + gh labels — OR I have logged a STUCK / MANUAL_REVIEW_NEEDED line. Until then, read .claude/playbooks/orchestrator.md and execute one tick, pacing ~30m between active checks. Stop after 200 turns.
```

> This loop is `/goal`-driven — it runs to plan completion, then auto-stops. The
> team-account / attach-box variants below take the SAME `/goal` wrapper: just append
> their extra directive to the `/goal` prompt. See §3a.

### Choosing the Vast.ai account (team vs private)

By default every box provisions + bills the **private** account (`VAST_API_KEY`).
To run a session on the shared **team** account (`VAST_API_KEY_TEAM`, the
"Pluralis Research" team) instead, just say so in the loop instruction:

```
/bg /goal <the orchestrator completion condition from §3 above> Use the team account.
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

## 3a. The `/goal`-driven loop + workflow lanes + teams (new features, 2026-06-29)

The harness uses three Claude Code features. Full design + rollout:
`.claude/HARNESS_FEATURE_INTEGRATION.md`; the rules' single source of truth is
`project.yaml` (`goal_command:` / `workflows:` / `agent_teams:`).

**`/goal` drives the loop to completion.** Both loop commands above START with `/goal`
(not `/loop`) so a session does not stop until the plan is actually done (then it
auto-stops), instead of firing blindly on a clock. The `/goal` evaluator is a cheap
yes/no judge that is **transcript-only** — it can't read `runs.jsonl`, WandB, or labels —
so each tick the playbook **prints a completion ledger** (the evidence) the judge reads.
Every `/goal` condition carries an escape (`… OR log STUCK …` + a turn bound) so an
impossible criterion can't spin forever.
- Evaluator model: **Opus 4.8 — no Haiku anywhere.** Claude Code's small-fast slot (env
  `ANTHROPIC_DEFAULT_HAIKU_MODEL`, whose default is Haiku) is overridden to `claude-opus-4-8`
  in `settings.json`. The var's name says "HAIKU" but the override ELIMINATES it — do not
  delete it (that lets Haiku return). It also routes other small-fast/background calls to
  Opus — an accepted cost trade-off under the best-model policy.
- Safety: `/goal` blocks the Stop event, so the orchestrator runs the teardown sweep
  **in-foreground every tick** — that's what reaps idle/over-budget boxes while a goal
  holds the session open. One-time before your first long unattended run: confirm on a
  throwaway session that a `/goal` block doesn't suppress the teardown Stop hook.

**Workflows (`ultracode`) for the hard lanes.** The driver stays at `effort: max` and
launches a dynamic workflow EXPLICITLY (the `ultracode` keyword / "run a workflow") for:
moment-of-truth analysis (fan-out adversarial verdict), live in-training diagnostics, hard
`code_change` patches, hard planning (judge-panel), and parallel runs
(`.claude/workflows/parallel-runs.md`). Do NOT set `/effort ultracode` on the unattended
loops (it auto-escalates every tick and moves state out of the crash-durable ledger) — that
mode is for YOUR interactive sessions. **Workflow workers auto-approve edits, so they stay
READ-ONLY** (analysis/reports); provisioning, git/PR, and ledger writes never run inside a
workflow — they stay gated single-shot dispatches, preserving the `status:approved` gate.

**Agent teams (opt-in, GPU-free).** `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set. The two
background loops stay the crash-durable spine — do NOT replace them with a team (teammates
don't survive `/resume`; background bash has no egress). Sanctioned operator-run uses: (1)
parallel runs — one teammate per box, typed from `experiment-runner` (provisioning still via
the gated runner); (2) adversarial verdict review — spawn analyst-typed teammates that defend
vs. challenge a finished verdict and reconcile into one `verdict.md` (post-teardown, GPU-free,
NEVER flips `status:approved`). A teammate does NOT inherit a definition's `skills`/`mcpServers`
frontmatter — keep team uses skill-free.

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
/bg /goal <the orchestrator completion condition from §3 above> Use the team account.
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

---

## 3c. Hand-drive an experiment yourself (off the auto-loop) — fresh-session hand-off

Use this **instead of** the `/goal` auto-loop (§2/§3) when: a `code_change` experiment needs
hands-on authoring, it's an off-queue re-run with **no issue** (e.g. EXP-42), or you simply want
to watch each step and gate the GPU spend. Because the **plan file is the durable hand-off**, you
can run across SEVERAL fresh sessions to keep each context window small (cheaper, faster) — a new
session resumes exactly where the last left off.

**Each session — start:**
1. Open Claude Code in this dir (`cd /Users/shamane/Documents/verl/research && claude`, or the desktop app).
2. `/effort ultracode` — hard coding/training; interactive sessions only (NOT the auto-loop). Model is already Opus 4.8.
3. Paste a kickoff prompt that points at the plan and says where to start:
   ```
   Read .claude/plans/<N>.md including its "## Progress / session hand-off" section, and
   CODE_WALKTHROUGH.md. I'm driving this directly (not the orchestrator loop). Do the NEXT
   unchecked phase in the Progress section. Keep every GPU step gated — ask me before the box.
   ```

**The plan is the hand-off (the load-bearing part):**
- A fresh session has **no memory** of prior sessions — the only thing it inherits is the plan file.
  So the session keeps the plan's **`## Progress / session hand-off`** section current: tick finished
  phases, record the branch, data paths, and the single next action. (Same discipline as a run
  close-out: done / data-paths / next.)
- When the context window gets large, **stop and open a fresh session.** It reads the Progress
  section and continues — no re-explaining.

**GPU stays OFF** until you explicitly say go (box off by default). Write the code and pass the
CPU-testable correctness gates locally first; provision only when those pass.

> Why not the `/goal` auto-loop here? It keys plans by *issue number* and auto-provisions; for a
> delicate `code_change` or an off-queue re-run (no issue) you want to watch the patch and gate the
> spend. Hand-driving gives that control; the plan's Progress section gives the durability the loop
> would otherwise get from the ledger.

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
| `on-session-start.sh` | SessionStart | logs RUNNING count + $/hr burn to `~/.claude-events.log` |

> **`/goal` is also a session Stop hook.** When a loop runs under `/goal` (§3a), the
> evaluator intercepts Stop each turn to judge done-ness. It is transcript-only and
> **additive** — it does NOT replace the wired Stop hooks above. Because `/goal` blocks
> Stop, the orchestrator runs `teardown-finished-runs.sh` in-foreground each tick so budget
> safety never depends on Stop firing. The evaluator runs on **Opus 4.8 (no Haiku)** — CC's
> small-fast slot (`ANTHROPIC_DEFAULT_HAIKU_MODEL`, default Haiku) is overridden to
> `claude-opus-4-8` in settings.json; do not delete that override.

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
| Loop stopped | session ended → re-run the `/bg /goal …` command in a new session (§3) |
| `/bg` says "Nothing to background yet" | must prefix a command: `/bg /goal … Read .claude/playbooks/orchestrator.md …` (full command in §3) |
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
    ├── project.yaml                single source of truth (repos, secrets, vast template, defaults, branch policy, model/goal/workflow/team policy)
    ├── GOAL.md                     project north-star (what "done" means)
    ├── HARNESS_FEATURE_INTEGRATION.md   /goal + workflows + agent-teams design
    ├── playbooks/                  triage.md, orchestrator.md
    ├── agents/                     research-planner, experiment-runner, analyst, log-writer, training-log-monitor
    ├── plans/                      TEMPLATE.md, <N>.md per issue
    ├── hooks/                      kill-switch, protect-upstream, sync-metrics, teardown-finished-runs, commit-on-stop, on-session-start
    ├── skills/                     vast-provision, vast-attach, vast-teardown, vast-cost, de-bloat, codex-verify
    ├── workflows/                  parallel-runs.md (opt-in saved workflows)
    └── state/                      STATUS.md, runs.jsonl, .last-orchestrator-tick

verl/examples/grpo_trainer/
├── vast_baseline_qwen25_1p5b_grpo_gsm8k.sh             dense reference launcher (comm-eff OFF)
├── vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh  THE canonical comm-eff baseline (values live in the launcher; see project.yaml fixed_control_surface)
└── vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh    generic comm-eff engine (all knobs exposed)

verl/CLAUDE.md                      fork-specific agent instructions
```

The harness is **transferable**: edit `.claude/project.yaml`, swap the
GitHub repo, rewrite `.claude/GOAL.md` for the new project, and the
agents/playbooks/hooks come along unchanged.

---

## 8. Quick reference

```bash
# Start the loop (now /goal-driven — runs to plan completion, then auto-stops; see §3a)
cd /Users/shamane/Documents/verl/research && claude
#   Session A:  /bg /goal All open research:claim issues planned (triage ledger unplanned=0) … or log a triage error. Read .claude/playbooks/triage.md, one tick, pace ~60m. Stop after 100 turns.
#   Session B:  /bg /goal All status:approved plans terminal (PASS/STOP, box TORN_DOWN, LOG.md written) per my plan-completion ledger … or log STUCK/MANUAL_REVIEW_NEEDED. Read .claude/playbooks/orchestrator.md, one tick, pace ~30m. Stop after 200 turns.

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
