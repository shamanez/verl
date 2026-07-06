# Harness redesign — autonomous-harness-v1

> Phase-0 analysis + the implemented architecture. Replaces
> `HARNESS_FEATURE_INTEGRATION.md` (deleted — it was a landed proposal whose
> content lives in `project.yaml`). This doc is the durable "why"; the "how to
> use it" is `researcher_steps.md`; per-value config is `project.yaml`.

## 1. What was wrong (audit findings, 2026-07-06)

**Architecture.** Two fleet-scan `/goal` loop sessions (triage + orchestrator)
re-read *everything* each tick — orchestrator ticks cost ~60–65k tokens
(playbook 6k + project.yaml 5.5k + ALL plans ≈50k) every ~30 min, and
terminated only when a hand-printed transcript ledger convinced a
transcript-only judge. Neither loop could be pointed at one issue.

**Plans.** 254-line/18.2KB template, 17 mandatory sections, real plans
17–57 KB of wall-to-wall bold prose. All five live plans were `kind:analysis`,
so the ~90 template lines of Vast machinery were `(n/a)` ceremony in every
one. Ghost fields referenced downstream but absent from the template
(`## Progress`, `§Smoke launch commands`, `vast_account:`,
`total_training_steps` vs `total_train_steps`).

**Hang / stall / money-leak modes** (each now has a specific fix, §3):
1. Runner mandated an *unbounded* fix-rerun loop on code_change errors.
2. Heartbeat = `incoming.log` mtime, refreshed by any successful ssh tail —
   a dead-but-reachable box never reaped until the budget cap (observed).
3. `vastai`/ssh calls in hooks had no timeout — a hung CLI hung session Stop
   and the orchestrator's mandated foreground sweep.
4. `vast-attach` wrote `status:RUNNING` (never `EXTERNAL`) → the 60-min
   never-heartbeat trigger killed an attached analysis box mid-download
   (instance 43495538).
5. Mid-run adversarial-verify workflow lanes were sanctioned doctrine
   (orchestrator lanes 1–2, parallel-runs.md) — the main observed hang cause.
6. Triage re-dispatched a crashing planner forever (idempotence keyed only on
   plan-file existence); a deleted plan file made an approved issue invisible
   forever; `TORN_DOWN`-without-verdict was unrepresentable in the state table.
7. No session open ⇒ no teardown at all (Stop-hook-only reaper).
8. `runs.jsonl` had three unlocked read-modify-write writers (lost-row risk
   under parallel sessions) and was git-tracked + autosave-committed, so every
   worktree would fork a divergent ledger.
9. Bugs: `vast-cost` PROJECT_DIR off-by-one (flags every box as a LEAK);
   de-bloat `grep -q "$ID"` substring match (EXP-4 matches EXP-44 → deletes a
   pending experiment); de-bloat numeric-id parser can't process slug run dirs;
   42 git-tracked ~352MB `.npz` + one 416MB jsonl pushed `.git` to 7 GB.

**Naming.** Contract said `runs/EXP-<N>/` but reality was slug dirs
(`MOAT-58-ANALYSIS` holds EXP-60…) mapped only in SUMMARY prose; cells were
`c1/c2/armA-rlsK10-pertick`, needing hand-written decoder tables; WandB names
collided across runs.

## 2. The architecture: per-issue stage commands

The fleet-scan loops are **deleted**. The unit of execution is one issue moving
through stages, each stage a self-contained skill invoked as `/<cmd> <N>`:

| stage | command | dispatches | labels set (automatic) |
|---|---|---|---|
| file | `/new-issue "…"` | — | `research:claim`, `kind:*` |
| plan | `/plan <N> [deep]` | research-planner | `status:planned` |
| gate | `/approve <N>` | — (human digest + confirm) | `status:approved` |
| launch | `/launch <N> [--attach <id>]` | experiment-runner | `status:running` |
| watch | `/monitor <N>` | training-log-monitor (bg, bounded) | — |
| judge | `/analyze <N>` | analyst | `status:pass\|revise\|stop` |
| finish | `/close <N>` | log-writer | `status:done` + issue closed |
| resume/drive | `/go <N>` | whichever stage is next | (per stage) |
| overview | `/status` | — | — |

Rules that make this never-hang:
- **Stage preconditions come from labels + ledger first, plan file second.**
  A deleted plan or run dir degrades to a named refusal or a terminal
  derivation, never a retry loop (§5).
- **Every retry is bounded and counted in the ledger** (provision attempts per
  rung, monitor re-dispatches, REVISE depth ≤ `iterations`). Exhaustion ⇒
  `MANUAL_REVIEW_NEEDED` in PROGRESS.md + stop, never spin.
- **`/go <N>`** is the resumable driver for unattended multi-day execution:
  detect stage from labels/ledger, run remaining stages, pause only at the
  approval gate or a `MANUAL_REVIEW_NEEDED`. Long-run form:
  `/bg /goal Issue <N> is terminal (status:done, box TORN_DOWN, LOG entry) or
  PROGRESS.md has STUCK/MANUAL_REVIEW_NEEDED for it. Run /go <N>. Stop after
  120 turns.`
- The human's *decision* stays (approve gate); the human's *mechanics*
  (label flips, `gh` invocations) are gone — `/approve` does them.

**Where humans belong.** All questions, brainstorming, judge-panels, and
adversarial review live in `/plan` (deep tier) and `/approve`. During
`/launch → /close` the harness is maximally autonomous: no adversarial loops, no
self-review workflows. If something mid-run genuinely warrants heavy
verification, the stage appends `MANUAL_REVIEW_NEEDED: <why> — <N>` to
PROGRESS.md and stops for an explicit human go/no-go. This is enforced by
`project.yaml verification:` and by the removal of the mid-run workflow lanes.

## 3. Never-hang / GPU-never-stale mechanics

- **Heartbeat = progress, not mtime.** `sync-metrics.sh` now appends to
  `runs/<id>/metrics/incoming.log` only when the tail *content advanced*
  (last-line hash comparison). Same reaper triggers, but mtime now means
  "training moved", so a dead-but-reachable box is reaped at 30 min.
- **Timeouts everywhere.** Every `vastai` call in the reaper/skills and every
  ssh in hooks is wrapped in `timeout`. The Stop hook can no longer hang a
  session; the foreground sweep is bounded.
- **Reaper triggers unchanged** (verdict / heartbeat-30min /
  never-heartbeat-60min / budget / PROVISIONED>15min) — the taxonomy was
  right; the signal sources were wrong.
- **EXTERNAL vs RUNNING is now explicit at attach time**: `vast-attach`
  defaults to `RUNNING` (reaped like any box) for training runs and takes
  `--manual` for operator-managed boxes → `status:EXTERNAL` (tracked by
  vast-cost, never auto-reaped, torn down by explicit `/close` or
  vast-teardown). vast-teardown + vast-cost now both understand EXTERNAL.
- **Ledger locking**: all `runs.jsonl` writers go through `flock` (shared
  helper). The ledger is un-git-tracked (machine state, not history) so
  worktrees share exactly one ledger.
- **Session-independent backstop**: optional launchd/cron line (documented in
  vast-teardown SKILL.md) runs the reaper hourly even with zero sessions open.
- **commit-on-stop never blocks Stop** (no more exit 2) and is worktree-aware.

## 4. Two-tier plans

`plans/TEMPLATE-fast.md` (~45 lines) and `plans/TEMPLATE-deep.md` (~95 lines).
One YAML frontmatter block carries every machine-read field; prose is for
humans. The analyst predicate is now a **harness default** (PASS iff all
success boxes ✓; STOP on falsification/budget/depth; REVISE ≤ `iterations`
with `next_actions`) — plans only override it. Kind-routing doctrine and Vast
utilization doctrine moved out of plans into the run/monitor skills (stated
once). Fast tier is the default; `deep` is chosen by the operator
(`/plan N deep`) or by the planner when the issue is genuinely multi-stage.
Deep tier adds staged sequences with gates, correctness invariants, and a
`## Progress` hand-off section (the former ghost section, now real).

Target sizes: fast plan ≤ 4 KB, deep plan ≤ 15 KB (vs 17–57 KB before).

## 5. One name, everywhere + snapshot isolation

Canonical id: **`<N>-<slug>`** (e.g. `63-anchor-ema-sweep`), chosen at plan
time. It is *simultaneously* the runs/ dir name, the ledger `id`, the Vast
instance label, the WandB group, and the branch suffix (`exp/63-anchor-ema-sweep`).
Cells are self-describing kebab slugs (`adaptive-ls-k10`, `dense-control`) —
`c1`/`armA…` patterns are banned; WandB run name = `<N>-<cell>`
(`63-adaptive-ls-k10`), group = the canonical id. Names derive from one
helper (`skills/_lib.sh: names_for`), not convention.

**Snapshot (`runs/<id>/run.json`)**: `/launch` materializes everything
downstream stages need (cells, wandb names, step target, milestone,
promote_launcher_as, branch, account) so monitor/analyst/log-writer never
depend on the plan file — deleting any plan or run dir mid-flight degrades
gracefully: stages fall back label+ledger-first and refuse with a named
reason instead of stalling.

## 6. Branch + PR discipline (every issue)

- `/launch` creates `exp/<id>` from `vast-ai-workload` (worktree for
  code_change; branch-only otherwise) and pushes before launch.
- `/close` commits the issue's durable deliverables (plan, verdict,
  LOG/SUMMARY delta, code, promoted launcher) to `exp/<id>`, opens a PR
  (base `vast-ai-workload`) whose body carries the run results table, merges
  it if there is anything to land, and deletes the branch. Nothing to land ⇒
  logged skip, no empty PR.
- Bulk artifacts (npz caches, big jsonl, metrics) are gitignored — PRs carry
  results, not gigabytes.
- Parallel issues: separate sessions each in their own worktree
  (`EnterWorktree` / `claude --worktree`), state paths always resolved to the
  primary checkout via `_lib.sh`, ledger writes flock'd. The autosave hook
  commits only that session's branch, so no cross-contamination.

## 7. Testing philosophy (CPU vs GPU) and adversarial gating

- **CPU checks are allowed exactly once per gate**: import/syntax, shape/dtype
  probe, off-path parity when cheap. ONE bounded fix attempt on failure, then
  either proceed to GPU (if the plan's invariants gate is `soft`) or escalate
  `STUCK` (if `hard`). No CPU verification loops.
- **Research validation happens on the GPU box** — the probe cell (1–2 steps)
  is the real correctness gate; the commit-hotfix loop on the box is bounded
  by `max_fix_iterations` (default 3) per run, then `MANUAL_REVIEW_NEEDED`.
- **Adversarial review / judge panels: planning-time only** (`/plan deep`,
  `/approve`, operator-invoked `codex-verify`). During execution they are
  forbidden; a stage that believes one is warranted pauses and asks.

## 8. Deletions (this branch)

- `research/reports/` — HTMLs migrated into their run dirs; dir deleted.
- `research/tmp/`, `research/wandb/` — debris, deleted.
- `research/STATUS.md` (stale duplicate), `research/budget.json` (folded into
  `project.yaml budget:`), `.claude/plans/SUMMARY.md` (zero consumers),
  `.claude/HARNESS_FEATURE_INTEGRATION.md` (landed proposal),
  `.claude/playbooks/` (both loops), `.claude/state/vast-handles/*` (stale),
  `.claude/state/supply-poll.sh` (retired tiers), empty `smoke/`.
- `.gitignore` gains `research/runs/**` bulk patterns + state files. The
  ~7 GB of already-committed blobs need a history rewrite — **operator
  decision, not done autonomously** (flagged in the hand-off summary).
- 21.5 GB of terminal `runs/` dirs are foldable via the fixed, now
  **human-only** de-bloat — not auto-deleted (deleting science is a human act).

## 9. Primitive choices (Claude Code, mid-2026)

Verified against current docs: skills are the unified command primitive
(frontmatter: `disable-model-invocation`, `allowed-tools`, arguments);
subagents don't survive restarts (durable state lives in labels + ledger +
files, never in a session); agent teams stay opt-in/GPU-free (in-process
teammates don't survive /resume); worktrees are the zero-contamination
primitive for parallel sessions; there is no timer hook — Stop hook + optional
cron are the reaper triggers; `/goal` + `/bg` remain the unattended driver
wrapper around `/go <N>`.
