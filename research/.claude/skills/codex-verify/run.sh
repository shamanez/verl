#!/usr/bin/env bash
# codex-verify — operator-invoked wrapper around `codex exec` with hang protection.
#
# Invoke this skill MANUALLY between status:planned and status:approved when you
# want a second opinion on a plan, OR after a STUCK runtime failure for diagnosis,
# OR before a milestone promotion for an adversarial summary review. The
# autonomous harness (orchestrator playbook) does NOT call this skill — it stays
# operator-driven because the codex CLI's broker is occasionally flaky and the
# autonomous-loop "verify before launch" gate was empirically more friction than
# value (see SKILL.md "Why this is operator-invoked").
#
# Two layers of hang protection:
#   - Layer 1: hard wall-clock timeout (default 600 s).
#   - Layer 2: stall watchdog (default 90 s with no stdout growth -> SIGKILL).
#
# Exit codes:
#   0    success — output file holds the Codex response (session header stripped)
#   64   bad usage
#   124  hard wall-clock exceeded (output starts with TIMEOUT: hard wall-clock)
#   125  stall (output starts with TIMEOUT: stalled)
#   126  codex exec returned non-zero (output starts with BROKER_DIED:)
#
# Usage:
#   codex-verify --mode verify|code-rescue|math-rescue|adversarial \
#       --out  <output-path> \
#       [--plan <plan-path>] [--diff <diff>] [--ctx <free text>] \
#       [--cd <workdir>] [--timeout 600] [--stall 90] [--model gpt-5.5]
#
# See .claude/skills/codex-verify/SKILL.md for per-mode recipes.
set -euo pipefail

MODE=""
OUT=""
PLAN=""
DIFF=""
CTX=""
CD_DIR=""
TIMEOUT_SEC=600
STALL_SEC=90
MODEL=""

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  if [[ -f "$SKILL_DIR/SKILL.md" ]]; then
    cat "$SKILL_DIR/SKILL.md"
  else
    cat <<'EOF'
codex-verify — invoke `codex exec` with hang protection.
Usage:
  codex-verify --mode verify|code-rescue|math-rescue|adversarial \
               --out <path> [--plan <path>] [--diff <path>] [--ctx <text>] \
               [--cd <dir>] [--timeout 600] [--stall 90] [--model gpt-5.5]
Exit codes: 0=ok 64=usage 124=hard-timeout 125=stall 126=codex-error
EOF
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --mode)    MODE="$2";        shift 2 ;;
    --out)     OUT="$2";         shift 2 ;;
    --plan)    PLAN="$2";        shift 2 ;;
    --diff)    DIFF="$2";        shift 2 ;;
    --ctx)     CTX="$2";         shift 2 ;;
    --cd)      CD_DIR="$2";      shift 2 ;;
    --timeout) TIMEOUT_SEC="$2"; shift 2 ;;
    --stall)   STALL_SEC="$2";   shift 2 ;;
    --model)   MODEL="$2";       shift 2 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 64 ;;
  esac
done

if [[ -z "$MODE" || -z "$OUT" ]]; then
  echo "codex_with_watchdog: --mode and --out are required" >&2
  exit 64
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex_with_watchdog: codex CLI not in PATH" >&2
  echo "BROKER_DIED: codex CLI missing" > "$OUT"
  exit 126
fi

mkdir -p "$(dirname "$OUT")"

# Compose the prompt by mode. Sent on stdin so the prompt can be long.
TMP_PROMPT=$(mktemp -t codex-prompt.XXXXXX)
PARTIAL="$OUT.partial"
trap 'rm -f "$TMP_PROMPT"' EXIT
: > "$PARTIAL"

{
  case "$MODE" in
    verify)
      echo "You are a code-review verifier for a research-orchestration harness."
      echo "Output the first content line as one of: VERIFY: PASS | VERIFY: CONCERNS | VERIFY: FAIL"
      echo "Then a one-paragraph critique. FAIL only on a concrete defect that would invalidate"
      echo "the experiment or risk wasted compute. CONCERNS for non-blocking smells. PASS for clean."
      ;;
    code-rescue)
      echo "An agent is stuck on a code issue. Diagnose and propose a minimal patch as a unified"
      echo "diff. Open the first line with: RESCUE: DIAGNOSED | PATCH_SUGGESTED | UNCLEAR."
      ;;
    math-rescue)
      echo "Walk this derivation step-by-step. Identify which assumptions hold, which need empirical"
      echo "check, and where the argument can fail. Open with: RESCUE: DIAGNOSED."
      ;;
    adversarial)
      echo "Adversarially review the following milestone summary. Identify methodological holes,"
      echo "baseline weaknesses, p-hacking risks, and unsupported claims. End with:"
      echo "ADVERSARIAL: CLEAN | CONCERNS | CONTESTED."
      ;;
    *) echo "Unknown mode: $MODE" >&2 ; exit 64 ;;
  esac
  echo
  [[ -n "$CTX"  ]] && { echo "## Context"; echo "$CTX"; echo; }
  if [[ -n "$PLAN" && -f "$PLAN" ]]; then
    echo "## Plan / target ($(basename "$PLAN"))"
    echo '```markdown'
    cat "$PLAN"
    echo '```'
    echo
  fi
  if [[ -n "$DIFF" && -f "$DIFF" ]]; then
    echo "## Diff / next_actions"
    echo '```'
    cat "$DIFF"
    echo '```'
    echo
  fi
} > "$TMP_PROMPT"

# Build codex exec invocation.
CODEX_ARGS=(exec --skip-git-repo-check --sandbox read-only --ephemeral)
[[ -n "$CD_DIR" ]] && CODEX_ARGS+=(--cd "$CD_DIR")
[[ -n "$MODEL"  ]] && CODEX_ARGS+=(--model "$MODEL")
CODEX_ARGS+=(-)  # read prompt from stdin

# Layer 1: hard wall-clock. macOS doesn't ship GNU timeout; use perl alarm.
START=$(date +%s)

# Launch codex in the background with prompt on stdin -> redirected from TMP_PROMPT.
( exec codex "${CODEX_ARGS[@]}" < "$TMP_PROMPT" >"$PARTIAL" 2>&1 ) &
CODEX_PID=$!

# Layer 2: poll for stall every 5s; also enforce wall-clock.
last_size=0
last_change=$START
while kill -0 "$CODEX_PID" 2>/dev/null; do
  sleep 5
  now=$(date +%s)
  cur_size=$(wc -c < "$PARTIAL" 2>/dev/null || echo 0)
  if (( cur_size > last_size )); then
    last_size=$cur_size
    last_change=$now
  fi

  if (( now - START > TIMEOUT_SEC )); then
    kill -TERM "$CODEX_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$CODEX_PID" 2>/dev/null || true
    {
      echo "TIMEOUT: hard wall-clock ${TIMEOUT_SEC}s exceeded"
      echo
      echo "---last partial output (last 50 lines):"
      tail -50 "$PARTIAL" 2>/dev/null
    } > "$OUT"
    exit 124
  fi

  if (( now - last_change > STALL_SEC )); then
    kill -TERM "$CODEX_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$CODEX_PID" 2>/dev/null || true
    {
      echo "TIMEOUT: stalled ${STALL_SEC}s with no output growth"
      echo
      echo "---last partial output (last 50 lines):"
      tail -50 "$PARTIAL" 2>/dev/null
    } > "$OUT"
    exit 125
  fi
done

# Codex finished on its own. Collect exit code.
exit_code=0
wait "$CODEX_PID" || exit_code=$?

if (( exit_code != 0 )); then
  {
    echo "BROKER_DIED: codex exit=$exit_code"
    echo
    echo "---last partial output:"
    tail -50 "$PARTIAL" 2>/dev/null
  } > "$OUT"
  exit 126
fi

# Strip the codex header (lines from "OpenAI Codex" through the second "--------" rule
# inclusive) and the trailing "tokens used <N>" line so the output is just the
# assistant turn. If parsing fails for any reason, fall back to the raw output.
if grep -qE '^OpenAI Codex' "$PARTIAL"; then
  python3 - "$PARTIAL" <<'PY' > "$OUT" || cp "$PARTIAL" "$OUT"
import re, sys
src = open(sys.argv[1]).read().splitlines()
# Drop the leading session header: everything from "OpenAI Codex" through the
# line that follows the second "--------" rule. Then drop the literal "user"
# line, then the literal "codex" line. Then drop trailing "tokens used N".
out = []
i = 0
# find start of post-header content
while i < len(src) and not src[i].startswith("OpenAI Codex"):
    i += 1
seen_rules = 0
while i < len(src) and seen_rules < 2:
    if src[i].startswith("--------"):
        seen_rules += 1
    i += 1
# now i points after the second rule. Skip the echoed user prompt (lines until
# we hit a line that's exactly "codex").
while i < len(src) and src[i].strip() != "codex":
    i += 1
i += 1  # skip the "codex" marker line
# collect until we hit "tokens used"
while i < len(src):
    if src[i].startswith("tokens used"):
        break
    out.append(src[i])
    i += 1
# strip a single trailing blank
while out and not out[-1].strip():
    out.pop()
print("\n".join(out))
PY
else
  cp "$PARTIAL" "$OUT"
fi

rm -f "$PARTIAL"
exit 0
