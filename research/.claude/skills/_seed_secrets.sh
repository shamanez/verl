#!/usr/bin/env bash
# _seed_secrets.sh — push a STRIPPED copy of the laptop's secrets to a box so
# HF + WandB + R2 "just work" the instant the harness owns the box, with NO
# manual scp and NO agent step. Sourced by vast-provision (post SSH-verify) and
# vast-attach (post reachability-probe); both are deterministic scripts, so
# every box — provisioned OR bring-your-own — is seeded on the same code path
# that makes it ready. Closes the launcher's `FATAL: secrets.env not found`
# (examples/grpo_trainer/vast_*.sh) at its source.
#
# HARD SECURITY RULE — the box NEVER receives the Vast API keys. The on-box
# launcher aborts if VAST_API_KEY is present, and a rented box is shared/
# ephemeral hardware, so we ALLOWLIST only the box-relevant keys (deny-by-
# default: any future secret added to the laptop file is withheld unless
# explicitly listed here) and then assert no VAST_* survived the strip.
#
# Disable with VERL_SEED_SECRETS=0. Source location is fixed
# ($SKILL_DIR/../_seed_secrets.sh) alongside _lib.sh / _vast_account.sh.

# macOS ships no timeout(1); perl alarm survives execve (same shim as _lib.sh).
command -v timeout >/dev/null 2>&1 || timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }

# Exactly the keys a training/checkpoint box legitimately needs. HF aliases are
# derived on-box by the launcher, but seeding all three is harmless and covers
# clients that read a specific name. NEVER add VAST_API_KEY* here.
VERL_SEED_ALLOWLIST_RE='^(export )?(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_HUB_TOKEN|WANDB_API_KEY|R2_ACCESS_KEY_ID|R2_SECRET_ACCESS_KEY|R2_ENDPOINT|R2_BUCKET|R2_ACCOUNT_ID)='

# seed_secrets_to_box <host> <port> <ssh-identity-file>
# Best-effort: prints a one-line result to stderr and returns non-zero on any
# failure so the caller can WARN — it must NEVER abort provisioning (the
# launcher's own FATAL is the backstop if seeding silently no-ops).
seed_secrets_to_box() {
  local host="$1" port="${2:-22}" ident="$3"
  [[ "${VERL_SEED_SECRETS:-1}" == "0" ]] && { echo "seed-secrets: disabled (VERL_SEED_SECRETS=0) — skipped" >&2; return 0; }
  [[ -n "$host" && -n "$ident" ]] || { echo "seed-secrets: missing host/identity — skipped" >&2; return 1; }

  local src="${VERL_SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
  [[ -r "$src" ]] || { echo "seed-secrets: no laptop secrets at $src — skipped" >&2; return 1; }

  ident="${ident/#\~/$HOME}"
  local tmp; tmp="$(mktemp -t secrets-box.XXXXXX)" || return 1
  chmod 600 "$tmp"
  # shred is Linux-only; fall back to rm on macOS.
  local _wipe='rm -f'; command -v shred >/dev/null 2>&1 && _wipe='shred -u'
  # Wipe the plaintext secrets tmpfile on ANY exit from this function — including
  # a SIGINT/SIGTERM during the up-to-40s ssh/scp (the explicit wipes below only
  # cover the return paths). RETURN fires when the function unwinds.
  trap '$_wipe "$tmp" 2>/dev/null || rm -f "$tmp" 2>/dev/null' RETURN INT TERM

  # Allowlist strip — box-relevant keys ONLY.
  grep -E "$VERL_SEED_ALLOWLIST_RE" "$src" > "$tmp" 2>/dev/null || true
  # Defence in depth: a Vast key must never be on the box (launcher hard-fails).
  if grep -qiE '^(export )?VAST_API_KEY' "$tmp"; then
    echo "seed-secrets: ABORT — a VAST key matched the allowlist (bug) — NOT pushing" >&2
    $_wipe "$tmp" 2>/dev/null || rm -f "$tmp"; return 1
  fi
  if ! [[ -s "$tmp" ]]; then
    echo "seed-secrets: stripped file empty (no box-relevant keys in $src) — skipped" >&2
    $_wipe "$tmp" 2>/dev/null || rm -f "$tmp"; return 1
  fi
  if ! grep -qE '^(export )?HF_TOKEN=' "$tmp" || ! grep -qE '^(export )?WANDB_API_KEY=' "$tmp"; then
    echo "seed-secrets: WARN — HF_TOKEN and/or WANDB_API_KEY absent from stripped file (launcher will FATAL)" >&2
  fi

  local ssh_opts=(-i "$ident" -o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
  local dst='/root/.config/verl-research/secrets.env'
  if ! timeout 40 ssh -n "${ssh_opts[@]}" -p "$port" "root@$host" \
        'mkdir -p /root/.config/verl-research && chmod 700 /root/.config/verl-research' >/dev/null 2>&1; then
    echo "seed-secrets: mkdir on root@$host:$port failed — skipped (launcher will FATAL if unseeded)" >&2
    $_wipe "$tmp" 2>/dev/null || rm -f "$tmp"; return 1
  fi
  # scp port flag is -P (uppercase); ssh is -p.
  if ! timeout 40 scp "${ssh_opts[@]}" -P "$port" "$tmp" "root@$host:$dst" >/dev/null 2>&1; then
    echo "seed-secrets: scp to root@$host:$port failed — skipped" >&2
    $_wipe "$tmp" 2>/dev/null || rm -f "$tmp"; return 1
  fi
  if ! timeout 40 ssh -n "${ssh_opts[@]}" -p "$port" "root@$host" \
        "chmod 600 '$dst' && test -s '$dst'" >/dev/null 2>&1; then
    echo "seed-secrets: post-push verify on root@$host:$port failed" >&2
    $_wipe "$tmp" 2>/dev/null || rm -f "$tmp"; return 1
  fi
  $_wipe "$tmp" 2>/dev/null || rm -f "$tmp"
  echo "seed-secrets: pushed HF+WandB+R2 to root@$host:$port ($dst, chmod 600; VAST keys withheld)" >&2
  return 0
}
