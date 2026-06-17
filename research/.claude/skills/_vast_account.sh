#!/usr/bin/env bash
# _vast_account.sh — shared Vast.ai account/key resolver.
# Sourced by: vast-provision/run.sh, vast-teardown/run.sh, and the
# teardown-finished-runs.sh Stop hook. ONE definition so every provision and
# teardown path resolves the SAME key for a given box — a team-account instance
# is always torn down with the team key (no orphaned, un-destroyable boxes).
#
# Convention:
#   VAST_ACCOUNT = team | private   (default: private)
#     team    -> VAST_API_KEY_TEAM  (shared "Pluralis Research" team account)
#     private -> VAST_API_KEY       (personal account)
#   Both keys live in ~/.config/verl-research/secrets.env (chmod 600, never committed).
#   Provision records the chosen account as `vast_account` on the handle + ledger
#   row; teardown reads it back so it auths against the right account.
#
# This file contains ONLY function defs (no side effects at source time).

# Load both keys from the secrets file if either is missing (idempotent).
vast_load_secrets() {
  local f="${VERL_SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
  if [ -r "$f" ] && { [ -z "${VAST_API_KEY:-}" ] || [ -z "${VAST_API_KEY_TEAM:-}" ]; }; then
    set +u
    # shellcheck disable=SC1090
    . "$f"
    set -u
  fi
}

# Echo the API key for an account name ("team"|"private"; default private).
# Falls back to the private key if the team key is unset, so default behaviour
# is unchanged when VAST_API_KEY_TEAM is absent.
vast_key_for() {
  case "${1:-private}" in
    team) printf '%s' "${VAST_API_KEY_TEAM:-${VAST_API_KEY:-}}" ;;
    *)    printf '%s' "${VAST_API_KEY:-}" ;;
  esac
}

# Normalise an account label to team|private (anything not "team" -> private).
vast_account_norm() {
  case "${1:-private}" in team) printf 'team' ;; *) printf 'private' ;; esac
}
