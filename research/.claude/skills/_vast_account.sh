#!/usr/bin/env bash
# _vast_account.sh — shared Vast.ai account/key resolver.
# Sourced by: vast-provision/run.sh, vast-teardown/run.sh, and the
# teardown-finished-runs.sh Stop hook. ONE definition so every provision and
# teardown path resolves the SAME key for a given box — a team-account instance
# is always torn down with the team key (no orphaned, un-destroyable boxes).
#
# Convention:
#   VAST_ACCOUNT = team | private   (default: team — the private account is out of credits;
#                                    all call sites default to team, 2026-07-07)
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
# NO team→private fallback: destroying a team box with the private key 404s,
# the verify-authoritative classifier would count it "already gone", and the
# still-billing box would vanish from the ledger. An EMPTY key makes every
# caller's empty-key guard fire (skip + log) — the safe failure.
vast_key_for() {
  case "${1:-team}" in
    team) printf '%s' "${VAST_API_KEY_TEAM:-}" ;;
    *)    printf '%s' "${VAST_API_KEY:-}" ;;
  esac
}

# Normalise an account label to team|private (anything not "team" -> private).
vast_account_norm() {
  case "${1:-team}" in team) printf 'team' ;; *) printf 'private' ;; esac
}
