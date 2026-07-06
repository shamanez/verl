# Agent Instructions for this verl fork

> This is **shamanez/verl** — a private fork of `verl-project/verl` used as
> the substrate for building a **communication-efficient, pipeline-parallel
> GRPO trainer**. It is **not** for upstream contributions. PRs back to
> `verl-project/verl` are out of scope here; if you want those, read
> upstream's [`AGENTS.md`](AGENTS.md) (preserved unmodified as the original).

## 1. What this fork is for

This is **not** vanilla verl. It's a private research fork building a
**communication-efficient, pipeline-parallel verl GRPO trainer**: the *training*
path (activation + gradient traffic across pipeline-stage boundaries) runs under
a communication-efficient method — per-element (per-token, per-dimension)
activation masking at the boundaries, with optional anchor + spectral correction. Rollouts may come from ordinary,
non-pipeline-parallel verl + vLLM. Trained on **Qwen2.5-1.5B-Instruct + GSM8K**.
With the method disabled, training is byte-identical to upstream verl. The
project's goal is captured in
[`research/.claude/GOAL.md`](research/.claude/GOAL.md); the engineering map is
[`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md).

**Fixed control variables — do not change without a separate justification:**
- **Model**: Qwen2.5-1.5B-Instruct (the dense control is anchored to it).
- **RL loss**: vanilla GRPO (not DAPO / GSPO), no-KL no-entropy.
- **Hardware**: default **1×H200** on Vast.ai via the fixed
  `verl-research-vllm020` template; provisioning ladder 1×H200 → 1×B200 →
  2×H200, machine reliability strictly >0.99 on every rung (see
  `research/.claude/project.yaml` `default_compute`). 1–8 GPUs supported;
  the legacy 4×H200 / 8×H100 shapes are retained for explicit operator
  request only.
- **Datasets**: EASY = GSM8K (the default); HARD = Big-Math
  (`gshasiri/Big-Math-RL-Verified-filtered`) at `MAX_RESPONSE_LENGTH=4096`.
  Registry: `research/.claude/project.yaml` `datasets:`.

## 2. Where to look

This is **issue-first** development. Everything an agent needs to do its
job is reachable from the GitHub issue queue plus the harness files
under `research/`. The project's north-star — what "done" means — is
[`research/.claude/GOAL.md`](research/.claude/GOAL.md) (agents may read it freely).

| If you need… | Read |
|---|---|
| **Single source of truth for all project-level config** | `research/.claude/project.yaml` |
| **How the comm-eff method is actually implemented in this fork** | [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md) — per-component map (mask / anchor / spectral / FSDP integration), end-to-end data flow, and an explicit "what's NOT yet implemented" gap list |
| **The stage commands** (one per lifecycle stage, `/<cmd> <issue>`) | `research/researcher_steps.md` (index) → `research/.claude/skills/*/SKILL.md` |
| Harness architecture + audit rationale | `research/.claude/HARNESS_DESIGN.md` |
| Leaf subagent definitions | `research/.claude/agents/*.md` |
| **Dense control launcher (= comm-eff OFF)** | `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` (branch `vast-ai-workload`) |
| **Comm-eff method launcher** (baseline = run it with `COMM_EFF_ENABLED=false`) | `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh` |
| Vast-ai launcher conventions + launch-script stability contract | `examples/grpo_trainer/VAST_README.md` |
| Vast template registry (FIXED, one entry) | `research/.claude/skills/vast-provision/templates.json` |
| Credentials (path only — never echo values) | `~/.config/verl-research/secrets.env` (`chmod 600`) |
| **Project north-star (what "done" means)** | `research/.claude/GOAL.md` |

The GitHub repo the harness watches is
**`shamanez/verl-compression-research`** (private). It is set as the local
`gh` default via the `research` git remote on this checkout. Confirm with
`gh repo set-default --view`.

## 3. Hard rules

### Do not edit upstream verl code outside `exp/*` branches

Everything under `/Users/shamane/Documents/verl/` **except** `research/`,
this `CLAUDE.md`, and `.gitignore` is considered
upstream-owned. Writes are blocked by
`research/.claude/hooks/protect-upstream.sh` (a PreToolUse hook) for any
session opened in `research/`. The only legitimate way to patch verl code
is from inside an `experiment-runner` worktree on an `exp/<ID>-<slug>`
branch, as defined in `research/researcher_steps.md`.

If you find yourself wanting to edit `verl/`, `pyproject.toml`,
`setup.py`, `examples/`, etc. outside a branch: stop. Either you are
solving the wrong problem, or you need to open an `exp/*` branch first.

`AGENTS.md` stays exactly as upstream wrote it — that file is the
canonical merge point with upstream, so don't touch it.

### Do not echo secrets

`HF_TOKEN`, `WANDB_API_KEY`, `VAST_API_KEY` live in
`~/.config/verl-research/secrets.env`. Source them via shell; never print,
log, or write the values to memory files, commit messages, or PR bodies.
Memory holds the file path only — see `research_repo`, `secrets_location`,
and `setup_flow` entries in your memory index.

### Branch + PR policy (read this carefully — it's the load-bearing contract)

Three branches on the fork (`shamanez/verl`) matter:

| Branch | Role | Write policy |
|---|---|---|
| `main` | tracks upstream `verl-project/verl` | **READ-ONLY.** Never commit. Never PR to it. Never branch off of it for work. |
| `vast-ai-workload` | primary working branch on the fork | All harness work + research/ + vast-ai launchers commit here. **All experiment PRs target this branch as the base.** |
| `exp/<N>-<slug>` | per-experiment branches | Created by `experiment-runner` from `vast-ai-workload`, pushed to origin BEFORE training launches so the branch survives a laptop crash. PRs land back on `vast-ai-workload`. |

**Two repos, two purposes, no overlap:**
- **`shamanez/verl` (origin)** receives code PRs. Head=`exp/<N>-<slug>`, base=`vast-ai-workload`. Set in `research/.claude/project.yaml.github.code_repo` + `.code_pr_base_branch`.
- **`shamanez/verl-compression-research` (research remote, `gh repo set-default`)** is the issue queue ONLY. Triage polls it; analyst/log-writer post verdict comments. **No PRs go here.**

The `log-writer` agent reads these from `project.yaml` rather than hardcoding them, so porting to a new project is one-file.

### Authorisation is scoped

A user approving an action once does not approve it in all contexts.
Re-confirm before pushing, opening/commenting on PRs or issues outside
the harness's autonomous loop, destroying Vast.ai instances manually, or
any destructive git operation.

## 4. Quick start

### One-time setup (laptop)

```bash
# Secrets file present, owner-read-only
ls -l ~/.config/verl-research/secrets.env   # expect -rw-------

# gh default points at the research repo
gh repo set-default --view                   # expect shamanez/verl-compression-research
```

### Run an experiment (per-issue stage commands — harness-v1)

```bash
cd /Users/shamane/Documents/verl/research
claude
```

One issue = one lifecycle = one command per stage, each `/<command> <issue>`;
labels are applied automatically (you never hand-flip). Full index:
`research/researcher_steps.md`.

```
/new-issue "does signed_ema α=0.4 hold parity at cadence 10/10?"   → files #N
/go <N>          # plan → YOUR /approve → launch → monitor → analyze → close
```

`/go <N>` is resumable (detects the stage from labels + ledger) and is the
unattended entry point — for a days-long run, wrap it:
`/bg /goal Issue <N> is terminal (status:done, box TORN_DOWN, LOG entry) or
PROGRESS.md flags it … run /go <N> each turn. Stop after 120 turns.`
Parallel issues run in parallel sessions, each in its own worktree
(`claude --worktree <N>-<slug>`).

**Model & feature policy.** Every agent — and the `/goal` done-judge — runs on
**Opus 4.8** (no Sonnet, no Haiku anywhere); the only knob is reasoning **effort** with a
floor of `high`; per-agent tiers live in `research/.claude/project.yaml`. Heavy
deliberation (judge-panel workflows, adversarial review, agent teams) is
**planning-time only** (`/plan deep`, `/approve`); during execution the stages run
single-shot subagents with bounded retries and pause for a human go/no-go instead of
looping (project.yaml `verification:` is the single source of truth; design in
`research/.claude/HARNESS_DESIGN.md`).

Kill switch (instant pause of all agent tool calls):

```bash
touch ~/.claude-kill-switch
# resume
rm ~/.claude-kill-switch
```

### Manual baseline launch (bypass the harness)

Provision directly with `research/.claude/skills/vast-provision/run.sh` (locked
template `verl-research-vllm020`, hash `3b0f8b726ac3036d6c007bfa13b6d75f` — its
onstart script handles docker + verl install + HF/WandB auth), then ssh in and run
`examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`. Full procedure:
`research/researcher_steps.md`.

## 5. Commits

```text
<short subject>

Co-authored-by: Claude
Signed-off-by: Shamane Siri <shamane@pluralis.ai>
```

Skip the upstream-PR attribution conventions (`Generated-by:`,
`gemini-code-assist`, the duplicate-PR check, etc.) — they don't apply to
this fork.

## Acknowledgements

Adapts the orientation pattern of upstream verl's [`AGENTS.md`](AGENTS.md);
contribution-policy machinery removed — this fork is research-only.
