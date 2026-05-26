---
name: codex-verify
description: Run `codex exec` non-interactively with hang protection (hard wall-clock timeout + stall watchdog). Used by the codex-bridge agent for verify / code-rescue / math-rescue / adversarial modes. Bypasses the codex-companion plugin entirely because v1.0.4 of that plugin has a broker lifecycle bug that hangs mid-task on this machine — `codex exec` is robust and `codex doctor` is the source of truth.
allowed-tools: Bash
---

# codex-verify

Companion to the `codex-bridge` subagent. Wraps `codex exec` so a runaway Codex call can never block the harness indefinitely.

## Why this exists

The `codex-plugin-cc` plugin (v1.0.4) advertises a `task` subcommand routed through a shared broker process. On this machine, the broker dies mid-task — two-for-two failures observed 2026-05-24 at ~17:46 and ~18:18. Direct `codex exec` does not use the broker at all and works reliably.

This skill is the only path through which agents may call Codex. Calling `codex` raw from an agent prompt is **forbidden** — without the watchdog, a hung Codex call would freeze the agent and block experiment dispatch.

## Usage

```bash
$CLAUDE_PROJECT_DIR/.claude/skills/codex-verify/run.sh \
    --mode  verify|code-rescue|math-rescue|adversarial \
    --out   <output-path> \
    [--plan <plan-path>] [--diff <diff-or-next-actions-path>] [--ctx <free text>] \
    [--cd /Users/shamane/Documents/verl] \
    [--timeout 600] [--stall 90] [--model gpt-5.5]
```

### Per-mode output prefix

The first content line of the output file is always one of these markers (the orchestrator greps for them):

| Mode | Output prefix |
|---|---|
| `verify` | `VERIFY: PASS` / `VERIFY: CONCERNS` / `VERIFY: FAIL` |
| `code-rescue` | `RESCUE: DIAGNOSED` / `RESCUE: PATCH_SUGGESTED` / `RESCUE: UNCLEAR` |
| `math-rescue` | `RESCUE: DIAGNOSED` |
| `adversarial` | `ADVERSARIAL: CLEAN` / `ADVERSARIAL: CONCERNS` / `ADVERSARIAL: CONTESTED` |

## Hang protection layers

1. **Hard wall-clock timeout** (default 600 s). If `codex exec` doesn't return in 10 min, the child is SIGTERM'd then SIGKILL'd. Output starts with `TIMEOUT: hard wall-clock 600s exceeded`.
2. **Stall watchdog** (default 90 s). If the output file stops growing for 90 s while the child is still alive, kill the child. Output starts with `TIMEOUT: stalled 90s with no output growth`.

## Exit codes

| Code | Meaning | Output file marker |
|---|---|---|
| 0 | success | normal Codex response (header stripped) |
| 64 | bad usage (missing `--mode` or `--out`) | none |
| 124 | hard wall-clock exceeded | `TIMEOUT: hard wall-clock ...` |
| 125 | stall watchdog fired | `TIMEOUT: stalled ...` |
| 126 | `codex exec` returned non-zero (auth broken, CLI error, etc.) | `BROKER_DIED: codex exit=<N>` |

**Non-zero exit codes are NEVER treated as PASS by the orchestrator.** The codex-bridge agent maps any timeout/error to `VERIFY: FAIL` semantics, which routes the plan to `VERIFY_TIMEOUT` for human review. Silent fail-open on Codex unavailability defeats the entire verify gate.

## How the agent invokes this skill

The codex-bridge agent contract (`.claude/agents/codex-bridge.md`) calls this skill exactly once per dispatch. The wrapper composes the per-mode prompt prefix internally — the agent just supplies `--plan`, optional `--diff`, and optional `--ctx`. The agent never invokes `codex` directly and never uses the `codex-companion` plugin.

## Verification

```bash
# Tiny end-to-end smoke (CPU-cheap, no Vast.ai cost):
$CLAUDE_PROJECT_DIR/.claude/skills/codex-verify/run.sh \
    --mode verify --out /tmp/v.md --ctx "Output exactly: VERIFY: PASS. Nothing else." \
    --timeout 60 --stall 30
head -3 /tmp/v.md  # should show VERIFY: PASS
echo "exit=$?"
```
