#!/usr/bin/env bash
# _lib.sh — shared deterministic core for the per-issue stage skills.
# Source me:  source "$(dirname "$0")/../_lib.sh"   (or via absolute path)
# Everything here must be: bounded (no unbounded waits), locked (ledger),
# and main-checkout-anchored (worktree sessions share ONE state dir).

# macOS ships NO timeout(1) (and this laptop has no coreutils). Shim it with
# the perl alarm+exec idiom — alarm(2) survives execve, so the target dies on
# SIGALRM after N seconds. Every hook/skill that can't source _lib.sh carries
# the same 2-line shim.
command -v timeout >/dev/null 2>&1 || timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }

# ---------- paths (always the PRIMARY checkout, even from a worktree) ----------
lib_main_checkout() {
  # First line of `git worktree list` is the main working tree.
  git worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2; exit}'
}
lib_research_dir() {
  local root; root="$(lib_main_checkout)"
  [[ -n "$root" ]] && echo "$root/research" || echo "${CLAUDE_PROJECT_DIR:-$PWD}"
}
RESEARCH_DIR="${RESEARCH_DIR:-$(lib_research_dir)}"
STATE_DIR="$RESEARCH_DIR/.claude/state"
LEDGER="$STATE_DIR/runs.jsonl"
PLANS_DIR="$RESEARCH_DIR/.claude/plans"
RUNS_DIR="$RESEARCH_DIR/runs"
PROGRESS="$RESEARCH_DIR/PROGRESS.md"

progress() { echo "[$(date -Iseconds)] $*" >> "$PROGRESS"; }
die() { echo "REFUSED: $*" >&2; exit 3; }   # named refusal — never a retry loop

# ---------- ledger (flock'd; the ONLY sanctioned way to touch runs.jsonl) ----------
# macOS has no flock(1); use a mkdir spinlock with a hard 30s bound.
# A lockdir older than 5 min is STALE (its holder crashed) — steal it, else one
# crashed process would wedge every future writer.
_ledger_lock() {
  local d="$STATE_DIR/.runs.jsonl.lock" n=0 age
  until mkdir "$d" 2>/dev/null; do
    age=$(( $(date +%s) - $(stat -f %m "$d" 2>/dev/null || stat -c %Y "$d" 2>/dev/null || date +%s) ))
    (( age > 300 )) && { rmdir "$d" 2>/dev/null || true; continue; }
    n=$((n+1)); (( n > 300 )) && { echo "ledger lock timeout" >&2; return 1; }
    sleep 0.1
  done
}
_ledger_unlock() { rmdir "$STATE_DIR/.runs.jsonl.lock" 2>/dev/null || true; }

ledger_append() {  # ledger_append '<one-line json row>'
  _ledger_lock || return 1
  echo "$1" >> "$LEDGER"
  _ledger_unlock
}
ledger_update() {  # ledger_update <id> '<jq row filter, e.g. .status="RUNNING">'
  local id="$1" filter="$2" tmp rc=0
  _ledger_lock || return 1
  tmp=$(mktemp)
  if jq -c --arg id "$id" "if .id == \$id then ${filter} else . end" "$LEDGER" > "$tmp" 2>/dev/null; then
    mv "$tmp" "$LEDGER"
  else
    rc=1; rm -f "$tmp"   # bad filter / jq error — caller MUST see the failure
  fi
  _ledger_unlock
  return $rc
}
ledger_row() {  # ledger_row <id>  -> last row for id (or empty)
  [[ -f "$LEDGER" ]] || return 0
  jq -c --arg id "$1" 'select(.id == $id)' "$LEDGER" 2>/dev/null | tail -1
}
ledger_row_by_issue() {  # ledger_row_by_issue <N>
  [[ -f "$LEDGER" ]] || return 0
  jq -c --argjson n "$1" 'select(.issue == $n)' "$LEDGER" 2>/dev/null | tail -1
}

# ---------- naming: ONE derivation for every surface (goal: readable at a glance) ----------
# canonical id: <N>-<slug>   e.g. 63-anchor-ema-sweep
# cells: self-describing kebab slugs (adaptive-ls-k10, dense-control).
# BANNED cell patterns: ^c[0-9]+$, ^arm[A-Za-z0-9]+ — enforced by lint_cell_name.
names_for() {  # names_for <N> <slug>  -> exports RUN_ID RUN_DIR BRANCH WANDB_GROUP VAST_LABEL TMUX_SESSION
  local n="$1" slug="$2"
  [[ "$slug" =~ ^[a-z0-9][a-z0-9-]{2,39}$ ]] || die "slug '$slug' must be kebab-case, 3-40 chars"
  export RUN_ID="${n}-${slug}"
  export RUN_DIR="$RUNS_DIR/$RUN_ID"
  export BRANCH="exp/$RUN_ID"
  export WANDB_GROUP="$RUN_ID"
  export VAST_LABEL="$RUN_ID"
  export TMUX_SESSION="run-$n"
}
wandb_run_name() { echo "$1-$2"; }   # <N> <cell>  e.g. 63-adaptive-ls-k10
lint_cell_name() {
  local c="$1"
  [[ "$c" =~ ^c[0-9]+$ || "$c" =~ ^arm ]] && die "cell name '$c' is opaque — name the method+knob (e.g. adaptive-ls-k10)"
  [[ "$c" =~ ^[a-z0-9][a-z0-9.-]{1,39}$ ]] || die "cell name '$c' must be kebab-case"
  return 0
}

# ---------- plans: flat-frontmatter parsing + graceful absence ----------
plan_path() { echo "$PLANS_DIR/$1.md"; }
plan_exists() { [[ -f "$PLANS_DIR/$1.md" ]]; }
plan_field() {  # plan_field <N> <key> [default] — reads the first ```yaml fence, flat keys only
  local f="$PLANS_DIR/$1.md" key="$2" dflt="${3:-}" v
  [[ -f "$f" ]] || { echo "$dflt"; return 0; }
  v=$(awk '/^```yaml/{f=1;next} /^```/{if(f)exit} f' "$f" \
      | grep -m1 -E "^${key}:" | sed -E "s/^${key}:[[:space:]]*//; s/[[:space:]]*(#.*)?$//")
  echo "${v:-$dflt}"
}

# ---------- run snapshot: downstream stages read THIS, never the plan ----------
run_json_path() { echo "$RUNS_DIR/$1/run.json"; }   # <id>
snapshot_get() {  # snapshot_get <id> <jq path> [default] — default applies for missing FILE or missing KEY
  local f="$RUNS_DIR/$1/run.json" v=""
  [[ -f "$f" ]] && v=$(jq -r "$2 // empty" "$f" 2>/dev/null)
  echo "${v:-${3:-}}"
}

# ---------- github: labels are the durable state machine; ALWAYS set by commands ----------
ALL_STATUS_LABELS="status:planned status:approved status:running status:pass status:revise status:stop status:done"
ensure_labels() {  # idempotent; run by /new-issue on first use
  local l
  for l in research:claim $ALL_STATUS_LABELS \
           kind:experiment kind:ablation kind:implementation kind:brainstorm kind:literature kind:analysis; do
    timeout 30 gh label create "$l" --force >/dev/null 2>&1 || true
  done
}
issue_labels() { timeout 60 gh issue view "$1" --json labels -q '.labels[].name' 2>/dev/null; }
set_status_label() {  # set_status_label <N> <planned|approved|running|pass|revise|stop|done>
  local n="$1" new="status:$2" cur rm=""
  cur=$(issue_labels "$n" | grep '^status:' || true)
  local l; for l in $cur; do [[ "$l" != "$new" ]] && rm="$rm --remove-label $l"; done
  # shellcheck disable=SC2086
  timeout 60 gh issue edit "$n" --add-label "$new" $rm >/dev/null
  echo "label: #$n -> $new"
}
issue_status() { issue_labels "$1" | grep -m1 '^status:' | sed 's/^status://'; }

# ---------- bounded external calls ----------
vast() { timeout "${VAST_TIMEOUT:-90}" vastai "$@"; }   # NEVER call vastai bare in a skill
sshb() {  # bounded ssh: sshb <port> <host> <cmd...>
  local port="$1" host="$2"; shift 2
  timeout "${SSH_TIMEOUT:-45}" ssh -i "${VAST_SSH_IDENTITY:-$HOME/.ssh/vast_ai_name}" \
    -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -p "$port" "root@$host" "$@"
}

# ---------- attempt counters (bounded retries live in the ledger, not in prose) ----------
bump_attempt() {  # bump_attempt <id> <field> <max> -> nonzero (and logs) when exhausted
  # Read the MAX across ALL rows for this id — relaunches append fresh rows, so
  # tail-1 alone would reset the counter and defeat the bound. The update writes
  # the new value onto every row with the id, keeping future reads consistent.
  local id="$1" field="$2" max="$3" cur
  cur=$( [[ -f "$LEDGER" ]] && jq -r --arg id "$id" "select(.id == \$id) | .${field} // 0" "$LEDGER" 2>/dev/null | sort -n | tail -1 )
  cur="${cur:-0}"
  if (( cur >= max )); then
    progress "MANUAL_REVIEW_NEEDED: $id exhausted ${field} (${cur}/${max})"
    return 1
  fi
  ledger_update "$id" ".${field} = $((cur + 1))"
}
