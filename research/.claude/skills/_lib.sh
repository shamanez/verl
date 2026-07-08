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
PLANS_DIR="$RESEARCH_DIR/.claude/plans"          # templates + legacy pre-P2 plan files only
PLAN_CACHE_DIR="$STATE_DIR/plan-cache"           # gitignored cache of the GitHub-resident plans
RUNS_DIR="$RESEARCH_DIR/runs"
PROGRESS="$RESEARCH_DIR/PROGRESS.md"

# PROGRESS.md is THE one local mutable file (operator directive 2026-07-08):
# a CAPPED tick echo + the end-of-session checklist an agent reads before
# ending its window. Durable signals live in labels + issue comments + the
# published report page; a closed issue's ticks are swept by progress_sweep
# (called from /close's cleanup) — full history survives in git anyway.
PROGRESS_CAP_LINES=400
progress() {
  echo "[$(date -Iseconds)] $*" >> "$PROGRESS"
  local n; n=$(wc -l < "$PROGRESS" 2>/dev/null | tr -d '[:space:]') || return 0
  if [[ -n "$n" && "$n" -gt $((PROGRESS_CAP_LINES + 40)) ]]; then
    { head -3 "$PROGRESS"; echo "…(older ticks pruned — full history in git)…"
      tail -n "$PROGRESS_CAP_LINES" "$PROGRESS"; } > "$PROGRESS.tmp" \
      && mv "$PROGRESS.tmp" "$PROGRESS"
  fi
  return 0
}
progress_sweep() {  # progress_sweep <N> <id> — drop a CLOSED issue's ticks from PROGRESS.md.
  # Word-bounded so #4 never sweeps #44 and 4-foo never sweeps 44-foo. Standing
  # operator notes (no issue reference) survive; the durable record is the close
  # comment + report page, and git keeps the full tick history regardless.
  local n="$1" id="$2" tmp
  [[ -f "$PROGRESS" ]] || return 0
  tmp=$(mktemp)
  grep -Ev "(^|[^0-9A-Za-z])#${n}([^0-9]|$)|(^|[^0-9A-Za-z-])${id}([^0-9A-Za-z-]|$)" \
    "$PROGRESS" > "$tmp" || true
  mv "$tmp" "$PROGRESS"
}
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
ledger_compact() {  # ledger_compact <id> — collapse a TERMINAL id's history to its last row.
  # Relaunch attempts append rows; once the issue is done only the final row
  # matters. Refuses (rc 1) while ANY row for the id is live (not just the
  # last — a failed destroy can leave an earlier RUNNING row hidden behind a
  # later TORN_DOWN one, and compacting it away would blind the reaper to a
  # live billing box). Refuses on a corrupt ledger — never rewrite from a bad
  # parse (a mid-file bad line would silently drop every row after it).
  local id="$1" tmp last
  [[ -f "$LEDGER" ]] || return 0
  last=$(ledger_row "$id"); [[ -n "$last" ]] || return 0
  jq -e --arg id "$id" \
    'select(.id==$id and (.status=="RUNNING" or .status=="PROVISIONED" or .status=="EXTERNAL"))' \
    "$LEDGER" >/dev/null 2>&1 && return 1
  _ledger_lock || return 1
  tmp=$(mktemp)
  if ! jq -c --arg id "$id" 'select(.id != $id)' "$LEDGER" > "$tmp" 2>/dev/null; then
    rm -f "$tmp"; _ledger_unlock; return 1
  fi
  printf '%s\n' "$last" >> "$tmp"
  mv "$tmp" "$LEDGER"
  _ledger_unlock
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

# ---------- plans: SSOT is the GITHUB ISSUE BODY ----------
# The plan lives INSIDE the issue body between the two marker lines below; its
# machine fields are the first ```yaml fence in that block. Local copies under
# $PLAN_CACHE_DIR are a derived, gitignored cache (offline reads + worktrees);
# legacy .claude/plans/<N>.md files remain a read-only fallback for pre-P2
# issues. Freshness rule: stages that SPEND or GATE (/plan /approve /launch)
# call plan_fetch first; everything else may read the cache.
PLAN_MARK_START='<!-- plan:start -->'
PLAN_MARK_END='<!-- plan:end -->'
plan_path() { echo "$PLAN_CACHE_DIR/$1.md"; }
plan_fetch() {  # plan_fetch <N> — refresh cache from GitHub; stale cache/legacy file survive a network failure
  local n="$1" body rc=0 f    # NB: $n referenced on its own line — same-line local n="$1" f="…$n…" is unbound under set -u
  f="$PLAN_CACHE_DIR/$n.md"
  mkdir -p "$PLAN_CACHE_DIR"
  body=$(timeout 60 gh issue view "$n" --json body -q .body 2>/dev/null) || rc=$?
  if (( rc == 0 )); then
    if grep -qF "$PLAN_MARK_START" <<<"$body"; then
      awk -v s="$PLAN_MARK_START" -v e="$PLAN_MARK_END" \
          'index($0,e){f=0} f{print} index($0,s){f=1}' <<<"$body" > "$f"
    else
      rm -f "$f"   # GitHub is SSOT: plan block gone ⇒ a stale cache must not resurrect it
    fi
  fi
  [[ -s "$f" ]] && return 0
  [[ -f "$PLANS_DIR/$n.md" ]] && { cp "$PLANS_DIR/$n.md" "$f"; return 0; }
  return 1
}
plan_exists() {  # cheap: cache/legacy first; one bounded fetch only when neither is present
  [[ -s "$PLAN_CACHE_DIR/$1.md" || -f "$PLANS_DIR/$1.md" ]] || plan_fetch "$1"
}
plan_field() {  # plan_field <N> <key> [default] — reads the first ```yaml fence, flat keys only
  local n="$1" key="$2" dflt="${3:-}" f v
  f="$PLAN_CACHE_DIR/$n.md"
  [[ -s "$f" ]] || plan_fetch "$n" >/dev/null 2>&1 || true
  [[ -s "$f" ]] || f="$PLANS_DIR/$n.md"
  [[ -f "$f" ]] || { echo "$dflt"; return 0; }
  v=$(awk '/^```yaml/{f=1;next} /^```/{if(f)exit} f' "$f" \
      | grep -m1 -E "^${key}:" | sed -E "s/^${key}:[[:space:]]*//; s/[[:space:]]*(#.*)?$//")
  echo "${v:-$dflt}"
}
plan_publish() {  # plan_publish <N> <plan-file> — install/replace the plan block in the issue body.
  # Text OUTSIDE the markers (the claim + any human notes) is preserved verbatim;
  # re-publishing replaces ONLY the marked block. Also refreshes the local cache.
  local n="$1" src="$2" body tmp
  [[ -s "$src" ]] || { echo "plan_publish: $src missing/empty" >&2; return 1; }
  body=$(timeout 60 gh issue view "$n" --json body -q .body 2>/dev/null) || return 1
  tmp=$(mktemp)
  { awk -v s="$PLAN_MARK_START" -v e="$PLAN_MARK_END" \
        'index($0,s){f=1} !f{print} index($0,e){f=0}' <<<"$body"
    echo ""; echo "$PLAN_MARK_START"; cat "$src"; echo "$PLAN_MARK_END"
  } > "$tmp"
  if ! timeout 60 gh issue edit "$n" --body-file "$tmp" >/dev/null; then rm -f "$tmp"; return 1; fi
  rm -f "$tmp"
  mkdir -p "$PLAN_CACHE_DIR" && cp "$src" "$PLAN_CACHE_DIR/$n.md"
}
plan_tick() {  # plan_tick <N> [<literal substring>] — flip '- [ ]' → '- [x]' on matching lines
  # INSIDE the plan block of the issue body (all unticked boxes when no pattern).
  # The pattern is a LITERAL substring, matched with index() — never a regex
  # (callers pass raw checkbox text; an unbalanced bracket must not break awk)
  # — and it travels via the environment so backslashes survive (-v mangles
  # them). The rewritten body is pushed ONLY if awk succeeded AND the plan:end
  # marker survived — a truncated body must never reach the plan SSOT.
  local n="$1" pat="${2:-}" body tmp
  body=$(timeout 60 gh issue view "$n" --json body -q .body 2>/dev/null) || return 1
  grep -qF "$PLAN_MARK_START" <<<"$body" || return 1
  tmp=$(mktemp)
  if ! PLAN_TICK_PAT="$pat" awk -v s="$PLAN_MARK_START" -v e="$PLAN_MARK_END" '
      BEGIN{pat=ENVIRON["PLAN_TICK_PAT"]}
      index($0,s){f=1} index($0,e){f=0}
      f && /^[[:space:]]*- \[ \]/ && (pat=="" || index($0,pat)) { sub(/\[ \]/,"[x]") }
      { print }' <<<"$body" > "$tmp"; then
    rm -f "$tmp"; return 1
  fi
  grep -qF "$PLAN_MARK_END" "$tmp" || { rm -f "$tmp"; return 1; }
  if ! timeout 60 gh issue edit "$n" --body-file "$tmp" >/dev/null; then rm -f "$tmp"; return 1; fi
  rm -f "$tmp"
  plan_fetch "$n" >/dev/null 2>&1 || true
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
PAUSE_LABELS="needs:human awaiting:approval"   # the durable pause signals (P4: labels, not PROGRESS prose)
ensure_labels() {  # idempotent; run by /new-issue on first use
  local l
  for l in research:claim $ALL_STATUS_LABELS $PAUSE_LABELS \
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

# ---------- human-pause signals: a LABEL is the durable flag; PROGRESS is the local echo ----------
# flag_human replaces bare "MANUAL_REVIEW_NEEDED → PROGRESS.md" prose: the label
# survives laptop loss and is visible from any window/gh query; the reason lands
# as an issue comment; the PROGRESS line remains as a local, capped echo.
flag_human() {  # flag_human <N> <reason…>
  local n="$1"; shift
  timeout 60 gh issue edit "$n" --add-label needs:human >/dev/null 2>&1 || true
  timeout 60 gh issue comment "$n" --body "**needs:human** — $*" >/dev/null 2>&1 || true
  progress "MANUAL_REVIEW_NEEDED: #$n $*"
}
flag_awaiting_approval() {  # unattended /approve parks the issue here
  timeout 60 gh issue edit "$1" --add-label awaiting:approval >/dev/null 2>&1 || true
  progress "AWAITING_APPROVAL: #$1"
}
clear_human_flags() {  # the operator decided — the resuming stage clears both pause labels
  timeout 60 gh issue edit "$1" --remove-label needs:human --remove-label awaiting:approval >/dev/null 2>&1 || true
}
has_human_flag() { issue_labels "$1" | grep -qE '^(needs:human|awaiting:approval)$'; }

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
