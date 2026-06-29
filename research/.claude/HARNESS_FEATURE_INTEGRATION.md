# Harness Feature Integration Plan — v2

**Status:** proposed · **Date:** 2026-06-29 · **Supersedes** the earlier "fence-first" draft.
**Validated against the live checkout:** CC `2.1.177`; `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set; no workflow/ultracode disable anywhere.

Folds three Claude Code features — `/goal`, dynamic workflows (`ultracode`), agent teams — into the
autonomous research harness, **enabling them in the GPU/training path** (the moment of truth), under a
strict model policy. Nothing load-bearing is retired: the `status:approved` human gate, `runs.jsonl`
ledger, GitHub labels, `verdict.md`, the teardown backstop, and `GOAL.md` all stay.

---

## 0. Operator rulings locked this session

1. **Use these features in the GPU/training path.** The moment of truth is *during* training; rich
   orchestration belongs there, not fenced out. Future = many sequential/parallel runs.
2. **The autonomous loop is `/goal`-driven** — it must not stop until the plan's success-criteria are
   met (or it escalates).
3. **Model policy (strict):** always **Opus 4.8** (`claude-opus-4-8`) for every working agent/session.
   The *only* tunable is reasoning **effort**, with a **floor of `high`** (never `low`/`medium`).
   **Harder coding + training/analysis tasks use the `ultracode` workflow mechanism.**
4. **Human `status:approved` gate stays** as the pre-run condition; `/goal` drives the *approved* plan
   to completion (not a substitute for approval).

---

## 1. Model & effort policy (the only knob is effort)

Model is fixed at `claude-opus-4-8` for all of the below. Effort ladder in use: `high` (floor) →
`xhigh` → `max`; plus **`ultracode`** = `xhigh` + dynamic-workflow orchestration, used as a *mechanism*
for the hard lanes (launched explicitly, see §3). No agent runs below `high`; no agent runs on Sonnet/Haiku.

| Agent / session | Model | Effort | Uses the ultracode/workflow mechanism? | Change vs today |
|---|---|---|---|---|
| **orchestrator** (driver session) | Opus 4.8 | `xhigh` | **Explicit per-lane** workflows (NOT session-wide `/effort ultracode`) | effort xhigh; add explicit-workflow rule |
| **triage** (driver session) | Opus 4.8 | `xhigh` | Explicit workflow for hard/parallel planning | effort xhigh |
| **research-planner** | Opus 4.8 | `max` | Hard plans → triage launches a judge-panel **workflow** | unchanged (already Opus@max) |
| **experiment-runner** | Opus 4.8 | `max` | Hard `code_change` → a coding **workflow** produces the patch; runner executes it | unchanged (already Opus@max) |
| **analyst** | Opus 4.8 | `xhigh` | **Yes** — moment-of-truth fan-out (multi-dimension adversarial verdict) | unchanged (already Opus@xhigh) |
| **training-log-monitor** | Opus 4.8 | `high` (floor) | Selective deep live diagnostics workflow; keep the 30s SSH poll as heartbeat | **model ↑ Sonnet → Opus 4.8** |
| **log-writer** | Opus 4.8 | `high` (floor) | No (durable git/GitHub mutations stay single-shot) | **model ↑ Sonnet → Opus 4.8** |
| **`/goal` evaluator** | Opus 4.8 | n/a (yes/no judge) | n/a | **model ↑ Haiku → Opus 4.8** (set `ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-opus-4-8`) — strict "best model" directive; higher per-turn cost noted in §7 |

**Why `ultracode` is a *mechanism*, not a per-agent setting:** workflows launch at the **session** level
and spawn their own worker agents; CC disallows nested orchestration, so a dispatched leaf subagent
cannot host a workflow. "experiment-runner uses ultracode for hard coding" means *the orchestrator/triage
session launches a coding workflow whose workers do the hard patch*, and the runner executes the result.

---

## 2. The `/goal`-driven persistent loop

`/goal`'s evaluator is **transcript-only** (no tools, can't read WandB/`runs.jsonl`/labels). So "don't
stop until the plan is done" works **iff the loop surfaces completion evidence into the transcript.**

**Launch command** (replaces the `/loop 30m` form in `researcher_steps.md`, both sessions):

```
/bg /goal Every status:approved plan has reached a terminal verdict (PASS/STOP) with its box
TORN_DOWN and LOG.md updated — confirmed by the plan-completion ledger I print each tick from
runs.jsonl + verdict.md + WandB + gh labels — OR I have logged a STUCK / MANUAL_REVIEW_NEEDED line.
Until then, read .claude/playbooks/orchestrator.md and execute one tick, pacing ~30m between active
checks. Stop after 200 turns as a safety bound.
```

**Three wiring requirements that make this correct (engineering, not caution):**
1. **Evidence → transcript.** Each tick the orchestrator prints a **plan-completion ledger**: every
   success-criterion + its evidence (`runs.jsonl` / `verdict.md` / WandB scalars surfaced by the
   monitor / gh label), marked DONE/PENDING. This is what the evaluator reads.
2. **Foreground teardown sweep every tick stays.** `/goal` blocks the *Stop* event, so budget safety
   must not depend on the Stop hook firing — the orchestrator already runs
   `teardown-finished-runs.sh` in-foreground each tick; keep that.
3. **Escape hatch in the condition.** `… OR log STUCK/MANUAL_REVIEW_NEEDED` + a turn bound, so an
   impossible criterion can't burn money forever.

---

## 3. Ultracode / workflow placement

**Session decision:**
- **Interactive (operator) session** → `/effort ultracode` ✅ (big analysis, design, hard coding).
- **Unattended `/goal` driver** → **NOT** session-wide `/effort ultracode`. Keep it `xhigh` and invoke
  workflows **explicitly per lane.** Reasons: session-wide auto-escalates every tick (cost), workflow
  state is ephemeral/in-process (the driver's value is crash-durable file state), and explicit
  invocation fires fan-out only where it pays.

**Lane → workflow map** (the session launches these; subagents are the workers inside them):

| Lane | Workflow? | What it does |
|---|---|---|
| Analysis (moment of truth) | ✅ Yes | Fan out N analysts (reward · length · entropy · grad-cosine · train-infer gap) → adversarial verify → one verdict |
| Live in-training diagnostics | ✅ Selective | Deep probe fan-out on the live box; the 30s SSH poll stays the always-on heartbeat |
| Planning (hard plans) | ✅ Yes | Judge-panel: draft N approaches → score → synthesize |
| Parallel / sequential runs | ✅ Yes | Pipeline/fan-out across multiple approved plans & boxes |
| Hard `code_change` patches | ✅ Yes | Implement → adversarial review → test, in a worktree; runner executes the result |
| experiment-runner (provision/launch) | ❌ No | Money-spending + durable ledger/git = gated single-shot subagent |
| log-writer (LOG/PR/promotion) | ❌ No | Durable git+GitHub mutations = single-shot |

**The safety line that keeps this safe unattended:** workflow worker agents auto-approve edits
(`acceptEdits`, no prompt when headless). That is **fine for the read-only lanes** (analysis, planning,
diagnostics — they touch only `runs/` + reports). **Anything that spends money or mutates durable state**
(`vast-provision`, git branches, PRs, `runs.jsonl`) **stays a gated single-shot subagent — never a worker
inside an auto-approving workflow.** This preserves the `status:approved` gate + budget caps while letting
moment-of-truth analysis fan out freely.

**Agent teams:** reuse the existing subagent definitions as teammate *types* for parallel runs (one
teammate per concurrent box) and for operator-run adversarial verdict review. Keep the file/issue/ledger
spine as the unattended substrate (teammates don't survive `/resume`; background bash has no egress).

---

## 4. Changes by component

| File | Type | Change | Why |
|---|---|---|---|
| `researcher_steps.md` | runbook | Replace both loop-launch commands with the **`/goal`-wrapped** form (§2). Add a `/goal` operator section + an agent-teams parallel/adversarial-review section. | The persistent-loop ask. |
| `.claude/playbooks/orchestrator.md` | playbook | Each tick **print the plan-completion ledger**; keep the foreground teardown sweep; sanction **explicit** per-lane workflow launches (analysis/parallel/coding) — NOT session-wide ultracode. | Makes `/goal` terminate correctly + enables the moment-of-truth fan-out. |
| `.claude/playbooks/triage.md` | playbook | `/goal`-wrap; may launch a judge-panel planning workflow for hard plans. | Persistence + hard-plan quality. |
| `.claude/agents/training-log-monitor.md` | agent | **`model: claude-opus-4-8`** (↑ from Sonnet); **print key in-training scalars into its report**; may be a worker in a diagnostics workflow. | Strict model policy + evidence-to-transcript for the goal-ledger. |
| `.claude/agents/log-writer.md` | agent | **`model: claude-opus-4-8`** (↑ from Sonnet), `effort: high`. | Strict model policy. |
| `.claude/agents/analyst.md` | agent | Note it may be a worker inside an analysis/diagnostics workflow; surface verdict evidence to transcript. Frontmatter stays Opus@xhigh. | Moment-of-truth fan-out feeds the ledger. |
| `.claude/agents/research-planner.md` | agent | Note hard plans may be produced via a judge-panel workflow at the triage session. Frontmatter stays Opus@max. | Hard-plan quality. |
| `.claude/agents/experiment-runner.md` | agent | Note hard `code_change` patches come from a coding workflow (runner executes); support N concurrent boxes. Frontmatter stays Opus@max. | Parallel runs + hard coding. |
| `.claude/project.yaml` | config | Rewrite `agent_models:` to **Opus-4.8-always + effort floor `high`** per §1; add `goal_command:` (transcript-evidence rule + escape hatch + `ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-opus-4-8`), `workflows:` (enabled; read-only-auto-approve safety line; explicit-not-session on the driver), `agent_teams:` (parallel/adversarial use). | Single source of truth for the new policy. |
| `.claude/settings.json` | config | Set `"env": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-opus-4-8" }`; **do NOT** disable workflows; keep all hooks on (never `disableAllHooks` — `/goal` rides hooks). Global model stays `claude-opus-4-8`. | Enable workflows + Opus goal-evaluator. |
| `../CLAUDE.md` | config (root) | Update Quick-start to the `/goal` launch form; state Opus-always + effort-floor-high + ultracode-for-hard-lanes; workflows/teams sanctioned for orchestration/analysis. | Always-loaded; supersede stale framing. |
| `.claude/workflows/parallel-runs.md` *(NEW, optional)* | skill (saved workflow) | Saved `/command` fanning out across multiple approved plans / concurrent boxes; read-only orchestration of analysis (provisioning still via the gated runner). | The "parallel things" future. |
| `vast-*`, `deep-research`, `codex-verify` skills | skill | **No change** — reused as-is. | Already cover provisioning + fan-out analysis. |

**No hook changes.** (The earlier draft's `protect-upstream` teammate-hardening and `TaskCompleted` gate
were dropped: the first would break the live `log-writer`/`vast-ai-workload` write path and keyed on an
unverified payload field; the second guarded a state its sanctioned use already excludes.)

---

## 5. Invariants preserved

- **`status:approved` human gate** — pre-run; `/goal` drives only the approved plan.
- **Budget caps + teardown backstop** — `teardown-finished-runs.sh` stays; orchestrator's foreground
  sweep every tick is what keeps it working under `/goal`'s Stop-block.
- **Read-only-auto-approve line** — money/durable-state roles never run inside an auto-approving workflow.
- **Crash-durable spine** — issues + `runs.jsonl` + files remain the unattended substrate (not workflow
  variables, not a teammate mailbox).
- **`kill-switch.sh` + `protect-upstream.sh`** — still PreToolUse-gate every spawned worker's tool calls.
- **Two-repo/three-branch PR policy + secrets handling** — untouched.

---

## 6. Phased rollout (verify-first)

> **Invariant check after each phase:** kill-switch refuses tool calls · orchestrator acts only on
> `status:approved` · `teardown-finished-runs.sh` runs clean · `protect-upstream` blocks off-branch
> `verl/` writes · no hook errors in `~/.claude-events.log`.

- **Phase A — Model/effort policy.** Upgrade `log-writer` + `training-log-monitor` to Opus 4.8; set the
  effort floor; set `ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-opus-4-8`; rewrite `project.yaml agent_models:`.
  *Verify:* a dispatched monitor/log-writer reports `claude-opus-4-8`; no agent below `high`.
- **Phase B — `/goal`-driven loop.** Add the ledger step to `orchestrator.md`/`triage.md`, rewrite the
  launch commands. *Verify first:* the `/bg /goal` invocation + the `/goal`-blocks-Stop-vs-foreground-sweep
  interaction on a throwaway session (both unconfirmed at `2.1.177`); confirm the foreground teardown still
  reaps a live box while a goal is active.
- **Phase C — Workflow lanes (read-only).** Wire the analysis/diagnostics/planning/parallel workflow
  launches into the playbooks; add `project.yaml workflows:`. *Verify:* an analysis workflow runs GPU-free,
  touches only `runs/`, and its result lands in the transcript ledger; confirm no workflow worker can call
  `vast-provision`.
- **Phase D — Parallel runs + teams (opt-in).** Add `parallel-runs.md`; document the teams adversarial
  review. *Verify:* a 2-box parallel pipeline provisions only via the gated runner.

**Rollback:** every edit is additive/reversible; deleting added lines/blocks returns today's behavior.

---

## 7. Open questions

1. **`/goal` evaluator = Opus 4.8?** Honors "strictly the best model," but it runs **every turn** of a
   long loop → materially higher cost than the default Haiku judge. Confirm, or allow this one slot to
   stay cheap. *(Plan assumes Opus per your directive.)*
2. **Driver effort `xhigh` vs `max`?** xhigh keeps per-tick cost sane on a frequent poller; max if you
   want the deepest single-context reasoning on every tick regardless of cost.
3. **`/bg /goal` + Stop-hook composition unverified at `2.1.177`** — Phase B gates on confirming a
   `/goal` block doesn't suppress the teardown sweep. Acceptable to verify on a throwaway session first?
4. **Relax the `status:approved` gate?** Currently kept. Say so explicitly if you want `/goal` to also
   auto-approve plans (a money-spending change I won't make unilaterally).
