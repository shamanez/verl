# researcher_steps.md — operator guide for the research harness

Single human-facing doc for driving one experiment end-to-end, in **three prompts**:

```
research:claim issue
  │  ① TRIAGE  (playbook: triage.md) ─────────────▶ plan written
  ▼                                                  │ [HUMAN GATE: review + approve]
  │  ② IMPLEMENT on the MacBook (local, NO GPU) ─────▶ code_change written, CPU-gated, exp/<N> pushed
  ▼                                                  │
  │  ③ /goal  (playbook: orchestrator.md) ───────────▶ provision → run → tear down → analyse → HTML report
  ▼
done: verdict + LOG + report
```

- **Model is Opus 4.8 everywhere** (incl. the `/goal` judge); the only knob is reasoning **effort**, floor `high`. Per-agent tiers + the `/goal`/workflows/teams rules live in `.claude/project.yaml` (design: `.claude/HARNESS_FEATURE_INTEGRATION.md`).
- **Every prompt is `/goal`-driven** — it runs to completion, then auto-stops.
- **② is skipped for a non-`code_change` experiment** → then it's just ① → ③.

> If you're an **agent** reading this — wrong layer. Agents read `.claude/playbooks/{triage,orchestrator}.md` + the leaf subagents in `.claude/agents/`.

---

## 0. One-time prerequisites

```bash
cd /Users/shamane/Documents/verl/research
ls -l ~/.config/verl-research/secrets.env   # -rw------- (HF + WandB + VAST keys)
gh repo set-default --view                    # shamanez/verl-compression-research
which claude gh vastai uv                     # all on PATH
git rev-parse --abbrev-ref HEAD               # vast-ai-workload
```

**Branch / PR policy** (`shamanez/verl`):
- `main` — tracks upstream; **read-only** (the protect-upstream hook blocks edits unless you're on `exp/*` or `vast-ai-workload`).
- `vast-ai-workload` — harness + launcher edits; the default working branch.
- `exp/<N>-<slug>` — per-experiment, created in ②; code PRs land on `shamanez/verl` with base `vast-ai-workload` (never `main`). The research repo is the issue queue only — no PRs.

---

## 1. ① Triage — issue → plan

File a `research:claim` issue with `hypothesis:` (falsifiable, numeric) and a `kind:`:

| kind | GPU | output |
|---|---|---|
| `experiment` (default) / `ablation` (needs `depends_on:`) | yes | verdict + LOG |
| `implementation` (`code_change`, no launch) | no | plan + draft PR |
| `brainstorm` / `literature` | no | plan is the deliverable |
| `analysis` (offline kill-gate) | no | GO/NO-GO verdict, run locally |

Then run **Prompt ①** (writes the plan):
```
/bg /goal Every open research:claim issue has a .claude/plans/<N>.md (my triage ledger shows unplanned=0), OR I logged a triage error. Read .claude/playbooks/triage.md, execute one tick, pace ~60m. Stop after 100 turns.
```
Review the plan (`cat .claude/plans/<N>.md`), then **the one mandatory human action — approve**:
```
gh issue edit <N> --add-label status:approved --remove-label status:planned
```
Wrong scope → `gh issue close <N>` + `rm .claude/plans/<N>.md`. Small fix → edit the plan, then approve.

---

## 2. ② Implement on the MacBook — local, NO GPU

*Only for `code_change` plans; skip otherwise.* Open a fresh session, `/effort ultracode`, run **Prompt ②**:
```
/goal The local implementation of .claude/plans/<N>.md is COMPLETE: branch exp/<N>-<slug> created off vast-ai-workload, the code_change implemented per the plan, every CPU-testable correctness gate printed PASS in this conversation, the branch committed + pushed, and the plan's "## Progress / session hand-off" section ticked — as shown by the outputs I print. Do NOT touch any GPU. If a gate can't pass, log STUCK and stop. Stop after 50 turns.
```

**Why a separate step (not the orchestrator):** the code is written + verified **cheaply on the MacBook**, so the GPU box in ③ only ever *runs* proven code — it never burns rental writing or debugging it. The plan's `## Progress / session hand-off` section is the durable hand-off: a fresh ③ session reads it to learn the branch is ready. (Long ②? Split across several fresh sessions — each reads the Progress section and continues.)

---

## 3. ③ `/goal` — GPU run + analysis + report (the orchestrator)

This prompt **is the orchestrator loop** (`.claude/playbooks/orchestrator.md`), driven by `/goal` to completion: it provisions, runs (`experiment-runner`), monitors (`training-log-monitor`), tears the box down, analyses (`analyst`), logs (`log-writer`), and delivers the report. Because ② already implemented + pushed the branch, **the runner only provisions + launches it — it does not re-implement.**

Fresh session, `/effort ultracode`, run **Prompt ③**:
```
/bg /goal Plan <N> is COMPLETE per .claude/playbooks/orchestrator.md: the experiment ran to target steps, the box is TORN_DOWN (runs.jsonl shows it), verdict.md is written, LOG.md + runs/SUMMARY.md are updated, and a solid HTML report is delivered — as shown by the plan-completion ledger I print each tick. The implementation is already done + pushed (exp/<N>-<slug>), so the runner only provisions + launches it (no re-patch). Read orchestrator.md and execute one tick, pacing ~30m. Stop after 150 turns.
```

- **Analysis is part of ③, not a 4th prompt.** The box is torn down the *moment* results sync; the analysis + HTML report then run GPU-free on the MacBook. The `/goal` is not satisfied until the report exists **and** the box is `TORN_DOWN`.
- **Vast account:** append `Use the team account.` (or `Use the private account.`) to the `/goal` — the orchestrator passes it to every dispatch, and teardown uses the same key. A plan may pin `vast_account:` in its `## Compute budget`.
- **Attach a box you already have** (skip provisioning): `bash .claude/skills/vast-attach/run.sh --instance-id <id> --account team`, then add to the `/goal`: *"Use box instance_id=<id> ssh_host=… ssh_port=… num_gpus=… this session instead of provisioning — EXTERNAL, torn down after its run."*
- **Off-queue plan (no GitHub issue)?** The orchestrator's state machine is keyed on a `status:approved` issue, so it won't auto-pick a plan that has no issue. Either file + approve an issue first, or drive ③ directly on that plan (point the `/goal` at the plan and the orchestrator's *procedure*).

---

## 4. Monitor / control

```bash
cat .claude/state/STATUS.md       # orchestrator rewrites every tick
tail -30 PROGRESS.md              # append-only audit + flags (MANUAL_REVIEW_NEEDED / STUCK / MILESTONE_PASS)
jq -c . .claude/state/runs.jsonl  # the lifecycle ledger
python scripts/check_budget.py --month
```

**Passive hooks (every session):** `kill-switch` (PreToolUse — `touch ~/.claude-kill-switch` halts every tool call), `protect-upstream` (PreToolUse — blocks verl/ writes off `exp/*`/`vast-ai-workload`), `sync-metrics` (PostToolUse — pulls remote `train.log`), `teardown-finished-runs` + `commit-on-stop` (Stop), `on-session-start` (SessionStart — logs $/hr burn).

> `/goal` is itself a transcript-only Stop-judge — **additive** to the hooks above. Because it blocks Stop, the orchestrator runs the teardown sweep **in-foreground every tick**, so budget safety never depends on Stop firing. The `/goal` evaluator runs on **Opus 4.8 (no Haiku)** — CC's small-fast slot (`ANTHROPIC_DEFAULT_HAIKU_MODEL`, default Haiku) is overridden to `claude-opus-4-8` in `settings.json`; do not delete that override.

**Human-intervention points:** approve a planned plan (the gate); review a REVISE child issue; act on `MANUAL_REVIEW_NEEDED` / `STUCK` in PROGRESS.md (fix the cause, re-approve); a box still up → `bash .claude/skills/vast-teardown/run.sh <id>`; emergency stop → `touch ~/.claude-kill-switch` (resume: `rm`).

---

## 5. Common failure modes

| Symptom | Fix |
|---|---|
| Triage fires, no plan | check the label is `research:claim`; grep PROGRESS.md for the planner failure |
| Orchestrator picks up an unapproved plan | label was set to `status:approved` by mistake → demote |
| Box not torn down | `bash .claude/skills/vast-teardown/run.sh <instance_id>` |
| Loop stopped early | session ended → re-run the `/bg /goal …` command (it resumes from the plan + ledger) |
| `/bg` says "Nothing to background yet" | must prefix a command: `/bg /goal … Read .claude/playbooks/orchestrator.md …` |
| protect-upstream refused an edit | not on `exp/*` / `vast-ai-workload` — check `git rev-parse --abbrev-ref HEAD` |

---

## 6. File layout

```
research/
├── researcher_steps.md     this guide
├── LOG.md / PROGRESS.md    newest-first PASS/STOP log · append-only audit
├── runs/<id>/ , runs/SUMMARY.md
├── scripts/                analyze.py, check_budget.py, weight_proj_sweep.py, …
└── .claude/
    ├── project.yaml        single source of truth (repos, secrets, vast template, defaults, branch + model/goal/workflow/team policy)
    ├── GOAL.md             project north-star (what "done" means)
    ├── HARNESS_FEATURE_INTEGRATION.md   /goal + workflows + agent-teams design
    ├── playbooks/          triage.md, orchestrator.md
    ├── agents/             research-planner, experiment-runner, analyst, log-writer, training-log-monitor
    ├── plans/              TEMPLATE.md, <N>.md  (each carries a "## Progress / session hand-off")
    ├── hooks/              kill-switch, protect-upstream, sync-metrics, teardown-finished-runs, commit-on-stop, on-session-start
    ├── skills/             vast-provision, vast-attach, vast-teardown, vast-cost, de-bloat, codex-verify
    ├── workflows/          parallel-runs.md (opt-in saved workflows)
    └── state/              STATUS.md, runs.jsonl, .last-orchestrator-tick
```

The harness is **transferable**: edit `.claude/project.yaml`, swap the GitHub repo, rewrite `.claude/GOAL.md`, and the agents/playbooks/hooks come along unchanged.
