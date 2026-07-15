# Agent Instructions for this verl fork

> This is **shamanez/verl** — a private fork of `verl-project/verl` used as
> the substrate for building a **communication-efficient, pipeline-parallel
> GRPO trainer**. It is **not** for upstream contributions. PRs back to
> `verl-project/verl` are out of scope here; if you want those, read
> upstream's [`AGENTS.md`](AGENTS.md) (preserved unmodified as the original).

## 1. What this fork is for

This private fork builds a **communication-efficient, pipeline-parallel verl GRPO
trainer**. PowerSGD projects boundary activations, a delayed dense anchor supplies
`Q` and `M`, and RELEX projects anchor weights forward. Rollouts may use ordinary
verl + vLLM; disabling the method leaves the dense upstream path. See
[`research/.claude/GOAL.md`](research/.claude/GOAL.md) and [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md).

**Current default answer/reference surface (report `df10b9be`):**
- **Model/data**: `Qwen/Qwen2.5-Math-1.5B`; MATH train/test from
  `EleutherAI/hendrycks_math`; last-`\boxed{}` + `is_equiv` reward.
- **GRPO**: batch 512, mini-batch 256, rollout `n=8`, prompt/response 1024/3072,
  AdamW `1e-6`, `low_var_kl=0.001`, reward-side KL off, entropy 0, 100 steps,
  evaluation every 25; the reference run used **2×H200 NVL**.
- **Activation projection**: PowerSGD rank 77; synchronized, warm-started,
  activation-derived `Q`, owned by the anchor and read by the fast path.
- **Anchor signal**: paired dense CPU replay, cadence/delay 20/20 optimizer ticks,
  256-prompt PPO-mini-batch scope, and all-floating signed-EMA `M` (338 tensors,
  `alpha=0.25`, `beta_anc=0.50`).
- **Latest completed reference**: qboot-v2 composite — fast-Q bootstrap,
  `stale_correct`, RELEX W2→W3→W4 (`window=4`, min 2, strength 1), current trajectories.
- **Scientific status**: this is the default surface/reference, **not a promoted
  compression champion**. The completed run predates anchor-KL parity; its corrected
  rerun stopped before the first dense anchor backward. Generic Hydra stays all-off.

Exact values: `research/.claude/project.yaml` `compression_defaults.math_qwen25_math_1p5b`.

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
| Harness operator manual (human-only, hosted HTML; design rationale lives there) | link at the top of `research/researcher_steps.md` |
| **Current experiment report** | `docs/experiments/relex_rank1_report.html` |
| **MATH dense control** | `bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_comparison_fsdp.sh dense` |
| **Latest completed comm-eff reference** | `bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh composite` (explicit arm required; bare invocation runs a matrix) |
| **Where finished-run verdicts + reports live** (published, never local) | issue close comment (SSOT) → https://com-eff-rlvr.pages.dev/runs/ (auto-published by `/close`) → R2 bucket `shamane-pluralis`, everything for a run under prefix `autonomous-harness-rlvr-compression/<run_id>/…` — reports at `…/<run_id>/…` and training checkpoints at `…/<run_id>/<cell>/checkpoints/…` (the old `verl-research/` prefix is retired); config: `project.yaml` `reports.r2` |
| Vast-ai launcher conventions + launch-script stability contract | `examples/grpo_trainer/VAST_README.md` |
| Vast template registry (FIXED, one entry) | `research/.claude/skills/vast-provision/templates.json` |
| Credentials (path only — never echo values) | `~/.config/verl-research/secrets.env` (`chmod 600`) |

The GitHub repo the harness watches is
**`shamanez/verl-compression-research`** (private) — the issue queue only:
`/new-issue` files there, the stage commands set its labels, `/plan` writes
the plan INTO the issue body (between plan markers — the plan's single source
of truth), `/close` posts the closing verdict comment. No PRs. It is set as
the local `gh` default via the `research` git remote on this checkout. Confirm
with `gh repo set-default --view`.

## 3. Hard rules

### Do not edit upstream verl code outside `exp/*` branches

Everything under the checkout root (wherever this clone lives — the hook
self-locates from its own path; never assume a fixed location) **except**
`research/`, this `CLAUDE.md`, and `.gitignore` is considered
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

Three branches on the fork (`shamanez/verl`) matter. **This harness line is
self-hosting**: on `autonomous-harness-v1`, every operative branch reference
resolves from `research/.claude/project.yaml` (`source_tree.base_branch` /
`github.code_pr_base_branch`) — never a hardcoded name.

| Branch | Role | Write policy |
|---|---|---|
| `main` | tracks upstream `verl-project/verl` | **READ-ONLY.** Never commit. Never PR to it. Never branch off of it for work. |
| `autonomous-harness-v1` | **THE base branch of this harness line** | All harness work + research/ + launcher promotions commit here. **All experiment PRs target this branch as the base.** |
| `exp/<N>-<slug>` | per-experiment branches | Created by `experiment-runner` from the project.yaml base branch, pushed to origin BEFORE training launches so the branch survives a laptop crash. PRs land back on the base branch. |

**Two repos, two purposes, no overlap:**
- **`shamanez/verl` (origin)** receives code PRs. Head=`exp/<N>-<slug>`, base=`project.yaml github.code_pr_base_branch` (= `autonomous-harness-v1` on this line).
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
cd <your-checkout>/research   # ALWAYS research/ of the CURRENT checkout — custom
claude                        # agents, protect-upstream, and the Stop-hook reaper
                              # register from here (verify: bash .claude/hooks/check-workspace.sh)
```

One issue = one lifecycle, driven as **three phases, one fresh window each**;
labels are applied automatically (you never hand-flip). Full index:
`research/researcher_steps.md`.

```
/build "does signed_ema α=0.4 hold parity at cadence 10/10?"   → files #N
/plan <N>        # fresh window: plan → written into the issue body → YOUR approve
/execute <N>     # fresh window: launch → monitor → analyze → close
```

`/go <N>` is the resume-from-anywhere fallback (detects the stage from labels +
ledger); the optional unattended `/bg /goal … /go <N>` wrapper is an appendix in
the operator manual, not the default path. Parallel issues run in parallel
windows, each in its own worktree (`claude --worktree <N>-<slug>`).

**Model & feature policy.** Every agent — and the `/goal` done-judge — runs on
**Opus 4.8** (no Sonnet, no Haiku anywhere); the only knob is reasoning **effort** with a
floor of `high`; per-agent tiers live in `research/.claude/project.yaml`. Heavy
deliberation (judge-panel workflows, adversarial review, agent teams) is
**planning-time only** (`/plan deep`, `/approve`); during execution the stages run
single-shot subagents with bounded retries and pause for a human go/no-go instead of
looping (project.yaml `verification:` is the single source of truth; design rationale
lives in the hosted operator manual linked from `research/researcher_steps.md`).

Kill switch (instant pause of all agent tool calls):

```bash
touch ~/.claude-kill-switch
# resume
rm ~/.claude-kill-switch
```

### Manual reference launch (bypass the harness)

Provision directly with `research/.claude/skills/vast-provision/run.sh` (locked
template `verl-research-vllm020`, hash `3b0f8b726ac3036d6c007bfa13b6d75f` — its
onstart script handles docker + verl install (editable, `--no-deps`); the
HF/WandB/R2 secrets are seeded onto the box by the provisioning tools
(`_seed_secrets.sh`, called from vast-provision/vast-attach), NOT the onstart),
then ssh in and run
`bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh composite`.
This reproduces the latest completed diagnostic reference, not a promoted arm. Full procedure:
operator-manual appendix (link at the top of `research/researcher_steps.md`).

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
