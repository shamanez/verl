---
name: codex-verify
description: Hang-protected wrapper around `codex exec` (hard wall-clock timeout + stall watchdog) for OPERATOR-INVOKED plan review, code-rescue diagnosis, math-rescue derivation walks, and milestone adversarial review. NOT auto-dispatched by the orchestrator — the human operator runs it manually.
allowed-tools: Bash
---

# codex-verify

Operator-invoked Codex tool with two layers of hang protection. The
`codex-companion` plugin (v1.0.4) had a broker lifecycle bug — direct
`codex exec` works, but a runaway call can still hang forever. This skill
wraps `codex exec` with a hard wall-clock timeout and a stall watchdog so a
hung Codex call can never block the operator's terminal.

## Why this is operator-invoked (not auto-dispatched)

The autonomous orchestrator used to dispatch this skill before launching
Vast.ai runs. In practice the autonomous flow produced:

- broker hangs (codex CLI bug) → harness blocked
- false-positive `VERIFY: FAIL` verdicts → operator override → relaunch
- `VERIFY: CONCERNS` (already advisory) → operator decided anyway
- pre-skipped `VERIFY: PASS` files → operator decided anyway

The operator was always the final authority. So the skill is now what it
should have been: a tool the operator runs **between `status:planned` and
`status:approved`** if they want a second opinion on a plan, OR after a
runtime failure if they want a diagnosis, OR before promoting a milestone
if they want an adversarial review of the summary.

**The harness does not look at the skill's output file.** Read it
yourself; decide what to do.

## Usage (always operator-invoked)

```bash
# At the research/ root, with the codex CLI installed and authed:
bash .claude/skills/codex-verify/run.sh \
    --mode  verify|code-rescue|math-rescue|adversarial \
    --out   <output-path> \
    [--plan <plan-path>] [--diff <diff-or-next-actions-path>] [--ctx <free text>] \
    [--cd /Users/shamane/Documents/verl] \
    [--timeout 600] [--stall 90] [--model gpt-5.5]
```

### Per-mode output prefix

The first content line of the output file is always one of these markers:

| Mode | Output prefix |
|---|---|
| `verify` | `VERIFY: PASS` / `VERIFY: CONCERNS` / `VERIFY: FAIL` |
| `code-rescue` | `RESCUE: DIAGNOSED` / `RESCUE: PATCH_SUGGESTED` / `RESCUE: UNCLEAR` |
| `math-rescue` | `RESCUE: DIAGNOSED` |
| `adversarial` | `ADVERSARIAL: CLEAN` / `ADVERSARIAL: CONCERNS` / `ADVERSARIAL: CONTESTED` |

## Recipes — when to invoke each mode

### `verify` — review a plan before approving it

```bash
# After triage's research-planner writes .claude/plans/<N>.md, BEFORE flipping
# status:planned → status:approved, optionally run this:
mkdir -p runs/EXP-<N>/verify
bash .claude/skills/codex-verify/run.sh \
    --mode verify \
    --out  runs/EXP-<N>/verify/$(date -u +%Y%m%dT%H%M%SZ).md \
    --plan .claude/plans/<N>.md \
    --cd   /Users/shamane/Documents/verl
cat runs/EXP-<N>/verify/*.md
```

Read the verdict. If `VERIFY: PASS` or `VERIFY: CONCERNS` (and the concerns
are non-blocking), approve. If `VERIFY: FAIL`, fix the plan and re-run, or
override with intent (the harness does not check this file).

### `code-rescue` — diagnose a STUCK / failed experiment

```bash
# Pass the failing log + the affected module slice:
bash .claude/skills/codex-verify/run.sh \
    --mode code-rescue \
    --out  /tmp/rescue.md \
    --ctx  "STUCK: EXP-<N> grad_norm exploded at step 17; train.log attached" \
    --diff runs/EXP-<N>/train.log \
    --cd   /Users/shamane/Documents/verl
cat /tmp/rescue.md
```

### `math-rescue` — walk a derivation

```bash
bash .claude/skills/codex-verify/run.sh \
    --mode math-rescue \
    --out  findings/derivations/<topic>.md \
    --ctx  "<question or hypothesis to check>" \
    --plan <optional path to the issue body or paper section> \
    --cd   /Users/shamane/Documents/verl
```

### `adversarial` — milestone summary review

```bash
# After log-writer writes findings/M<X>/SUMMARY.md, optionally:
bash .claude/skills/codex-verify/run.sh \
    --mode adversarial \
    --out  findings/M<X>/codex-review.md \
    --plan findings/M<X>/SUMMARY.md \
    --ctx  "Verdicts: runs/EXP-<a>/verdict.md, runs/EXP-<b>/verdict.md, ..." \
    --cd   /Users/shamane/Documents/verl
```

## Hang protection layers

1. **Hard wall-clock timeout** (default 600 s). If `codex exec` doesn't
   return in 10 min, the child is SIGTERM'd then SIGKILL'd. Output starts
   with `TIMEOUT: hard wall-clock 600s exceeded`.
2. **Stall watchdog** (default 90 s). If the output file stops growing for
   90 s while the child is still alive, kill the child. Output starts with
   `TIMEOUT: stalled 90s with no output growth`.

## Exit codes

| Code | Meaning | Output file marker |
|---|---|---|
| 0 | success | normal Codex response (session header stripped) |
| 64 | bad usage (missing `--mode` or `--out`) | none |
| 124 | hard wall-clock exceeded | `TIMEOUT: hard wall-clock ...` |
| 125 | stall watchdog fired | `TIMEOUT: stalled ...` |
| 126 | `codex exec` returned non-zero (auth broken, CLI error, etc.) | `BROKER_DIED: codex exit=<N>` |

A timeout / error is **not** a PASS verdict — it just means Codex didn't
finish. Read the partial output (the script captures the last 50 lines of
what Codex managed to emit before the kill) and decide what to do.

## Tiny end-to-end smoke

```bash
bash .claude/skills/codex-verify/run.sh \
    --mode verify --out /tmp/v.md \
    --ctx "Output exactly: VERIFY: PASS. Nothing else." \
    --timeout 60 --stall 30
head -3 /tmp/v.md  # should show VERIFY: PASS
echo "exit=$?"
```

## What the skill will NOT do

- Edit any source code (it's a `--sandbox read-only --ephemeral` Codex call).
- Touch `runs.jsonl`, GitHub labels, PR state, or any harness ledger.
- Spawn subprocesses other than `codex exec` (no `vastai`, no `gh`, no
  `ssh`).
- Auto-fire on any harness event — the operator runs it.
