#!/usr/bin/env bash
# PreToolUse (matcher: *) — refuse Edit/Write/NotebookEdit/MultiEdit on verl upstream paths.
# The research/ subtree is always writable. Upstream verl paths are read-only EXCEPT when
# the current branch starts with `exp/` (experiment-runner doing a code_change=true patch
# inside its worktree).
#
# This hook is the harness-level enforcement of verl/CLAUDE.md's "no pure code-agent PRs"
# rule. Without it, an over-eager agent could silently edit upstream files.
set -euo pipefail

PAYLOAD="$(cat)"
TOOL=$(echo "$PAYLOAD" | jq -r '.tool_name // ""')

# Only gate write tools.
case "$TOOL" in
  Edit|Write|NotebookEdit|MultiEdit) ;;
  *) exit 0 ;;
esac

PATH_ARG=$(echo "$PAYLOAD" | jq -r '.tool_input.file_path // .tool_input.notebook_path // ""')
[[ -z "$PATH_ARG" ]] && exit 0

# Resolve to absolute path.
case "$PATH_ARG" in
  /*) RAW_ABS="$PATH_ARG" ;;
  *)  RAW_ABS="$(cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null && pwd)/$PATH_ARG" ;;
esac

# Canonicalize: resolve symlinks AND `..` components so we cannot be tricked by
# a research/-relative path like `../verl/AGENTS.md` or a symlink under research/
# that points into upstream. python3 is always present on macOS and handles
# nonexistent leaves gracefully (Edit on a new file).
ABS=$(python3 -c '
import os, sys
p = sys.argv[1]
# os.path.realpath follows symlinks; abspath also normalises "..".
print(os.path.realpath(os.path.abspath(p)))
' "$RAW_ABS" 2>/dev/null || echo "$RAW_ABS")

# Canonicalize the anchor the same way so prefix comparison is symmetric.
VERL_ROOT_RAW="${VERL_ROOT:-/Users/shamane/Documents/verl}"
VERL_ROOT=$(python3 -c '
import os, sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))
' "$VERL_ROOT_RAW" 2>/dev/null || echo "$VERL_ROOT_RAW")
RESEARCH_ROOT="$VERL_ROOT/research"

# Always allow paths under the canonical research/ subtree.
case "$ABS" in
  "$RESEARCH_ROOT"|"$RESEARCH_ROOT/"*) exit 0 ;;
esac

# Refuse paths under canonical verl root (anywhere outside research/).
case "$ABS" in
  "$VERL_ROOT"|"$VERL_ROOT/"*)
    # Exceptions (branch is read from the canonical path so a symlink can't lie):
    #   exp/*             — experiment-runner agent patching upstream files inside its
    #                       worktree on an ephemeral, per-experiment branch.
    #   vast-ai-workload  — the stable home for vast.ai-specific launchers under
    #                       examples/grpo_trainer/vast_*.sh on the shamanez/verl fork.
    #                       Edits flow git-tracked (laptop → push → box `git pull`).
    BRANCH="$(git -C "$(dirname "$ABS")" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
    case "$BRANCH" in
      exp/*|vast-ai-workload|autonomous-harness-*) exit 0 ;;
    esac
    cat >&2 <<EOF
protect-upstream: refused write to upstream verl path.

  raw path:   $RAW_ABS
  canonical:  $ABS
  branch:     ${BRANCH:-<none>}

Only the experiment-runner agent on an exp/* branch may write under the verl tree.
If you need to change verl source for an experiment, do it inside an isolated worktree
on an exp/<ID>-<slug> branch — never on main or in the parent checkout.

If the canonical path resolved OUTSIDE the raw path, you may be writing through
a symlink — check whether the target is intended.

If you are intending to change harness state, write under /Users/shamane/Documents/verl/research/ instead.
EOF
    exit 2
    ;;
esac

exit 0
