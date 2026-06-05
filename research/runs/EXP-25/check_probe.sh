#!/usr/bin/env bash
# check_probe.sh <logfile> <probe-id 0|1>  — grep the EXP-25 invariant falsifiers
# from a probe's training log and print PASS/FAIL per checklist item.
set -uo pipefail
LOG="${1:?usage: check_probe.sh <log> <0|1>}"
PID="${2:?usage: check_probe.sh <log> <0|1>}"
fail=0
ok()   { echo "  [PASS] $1"; }
bad()  { echo "  [FAIL] $1"; fail=1; }
note() { echo "  [info] $1"; }

echo "=== EXP-25 id-$PID probe invariant check: $LOG ==="
[[ -f "$LOG" ]] || { echo "NO LOG FILE"; exit 2; }

# --- universal: no crash ---
grep -qE "Traceback|CUDA out of memory|RuntimeError|AssertionError" "$LOG" && bad "found Traceback/OOM/RuntimeError/AssertionError" || ok "no Traceback/OOM/RuntimeError/AssertionError"
grep -qE "EARLY_STOP_SIGNAL" "$LOG" && bad "early-stop watcher fired (corrupting failure)" || ok "no early-stop signal"
# NaN/Inf in a loss/grad field (word-bounded, not 'infer')
grep -qE "(loss|grad_norm|pg_loss|reward)[^A-Za-z].{0,80}\b([Nn]a[Nn]|[Ii]nf)\b" "$LOG" && bad "NaN/Inf in a loss/grad field" || ok "no NaN/Inf in loss/grad fields"
# single-GPU fallback guard (must be 4..8)
grep -qiE "world_size.{0,4}[=:].{0,4}1\b|single.gpu" "$LOG" && note "check world_size (grep hit — verify it's >=4)" || ok "no single-GPU fallback marker"

# --- codec confirmation (both probes use powersgd r=77, mask OFF) ---
grep -qE "powersgd codec: rank=77" "$LOG" && ok "codec = powersgd rank=77" || bad "codec NOT powersgd r=77"

# --- R1 anchor M invariants (both probes) ---
grep -qE "\[comm_eff\]\[EXP-25\]\[coverage\].*set_equal=True" "$LOG" && ok "coverage set-equal (M == merger set)" || bad "coverage NOT set-equal"
grep -oE "anchor_targets=[0-9]+ merger_expected=[0-9]+" "$LOG" | tail -1 | grep -qE "anchor_targets=196 merger_expected=196" && ok "coverage = 196/196" || { note "coverage count line:"; grep -oE "anchor_targets=[0-9]+ merger_expected=[0-9]+" "$LOG" | tail -1; bad "coverage != 196/196"; }
grep -qE "\[M-dp-identical\].*cross_rank_max_rel_dev=0\.000e\+00|\[M-dp-identical\].*cross_rank_max_rel_dev=0\.0+e" "$LOG" && ok "M bit-identical across DP ranks" || { note "M-dp line:"; grep -oE "\[M-dp-identical\].*" "$LOG" | tail -1; note "(rel_dev must be ~0)"; }
grep -qE "\[dp-reduce\].*all-reduced\(MEAN\)" "$LOG" && ok "G_anchor DP all-reduced(MEAN)" || bad "no DP-reduce(MEAN) line"
# scale proxy: post/pre ratio mean — a SUM bug shows ~dp_world (=4); MEAN ~O(1)
note "DP-reduce scale proxy (MEAN => far below dp_world=4):"; grep -oE "\|\|G\|\|_post/\|\|G\|\|_pre_mean=[0-9.]+" "$LOG" | tail -1
grep -qE "anchor-load.*loaded ([0-9]+)/\1 " "$LOG" && ok "anchor clone loaded REAL stale weights (X==Y)" || { note "anchor-load line:"; grep -oE "anchor-load.*loaded [0-9]+/[0-9]+" "$LOG" | tail -1; bad "anchor-load X!=Y (random init?)"; }
# M evolves
grep -oE "\|\|dM_anchor\|\|_mean=[0-9.e+-]+" "$LOG" | tail -2
grep -qE "\|\|dM_anchor\|\|_mean=[1-9]|\|\|dM_anchor\|\|_mean=[0-9]+\.[0-9]*[1-9]" "$LOG" && ok "M evolves (||dM||>0)" || bad "M frozen (||dM||==0)"
# anchor stays clean
grep -qE "anchor_ratio=1.0" "$LOG" && ok "anchor_ratio=1.0 (clean PG, no clip)" || bad "anchor_ratio != 1.0"
grep -qE "anchor_optimizer_steps=0" "$LOG" && ok "anchor_optimizer_steps=0" || bad "anchor took an optimizer step"
grep -qE "anchor_mask_applications=0" "$LOG" && ok "anchor_mask_applications=0 (unmasked)" || bad "anchor mask leaked"
grep -qE "anchor_grad_corrected=0" "$LOG" && ok "anchor_grad_corrected=0 (raw into EMA)" || bad "anchor grad was corrected"
grep -oE "anchor_backwards=[0-9]+" "$LOG" | tail -1 | grep -qvE "anchor_backwards=0" && ok "anchor fired (anchor_backwards>=1)" || bad "anchor never fired (anchor_backwards=0)"

if [[ "$PID" == "1" ]]; then
  echo "--- id-1 R2/R3 invariants ---"
  grep -qE "anchor_owns_q=True" "$LOG" && ok "anchor_owns_q=True (R2 on)" || bad "anchor_owns_q not True"
  grep -qE "correction_mode=signed_ema" "$LOG" && ok "correction_mode=signed_ema (R3 on)" || bad "correction_mode not signed_ema"
  # Q broadcast lands + cross-rank guard didn't raise
  grep -qE "\[comm_eff\]\[bcast\].*Q updated=True" "$LOG" && ok "anchor Q updated + broadcast" || bad "no Q broadcast line"
  grep -qE "\[comm_eff\]\[bcast\].*cross_rank_max_rel_dev=(0\.|0e|n/a)" "$LOG" && ok "Q cross-rank consensus OK (no raise)" || note "check bcast cross_rank line"
  grep -qE "basis Q DIVERGED across DP ranks" "$LOG" && bad "verify_basis_agreement RAISED" || ok "verify_basis_agreement did NOT raise"
  grep -qE "\[comm_eff\]\[bcast\].*M broadcast targets=" "$LOG" && ok "M broadcast landed" || bad "no M broadcast line"
  # R2 "the anchor is the SOLE Q writer" — HARD CHECK (not just a print):
  #   anchor_q_updates MUST be > 0 (the anchor actually refreshed Q each cadence), and
  #   the FAST-net powersgd_basis_updates MUST be 0 (fast maybe_update_basis gated OFF).
  # Both counters are emitted on the [comm_eff][bcast] "Q updated" line (fires only when
  # anchor_owns_q=true), so anchor them to that line.
  aqu=$(grep -E "\[comm_eff\]\[bcast\].*anchor_q_updates=" "$LOG" | tail -1 | grep -oE "anchor_q_updates=[0-9]+" | grep -oE "[0-9]+")
  fbu=$(grep -E "\[comm_eff\]\[bcast\].*powersgd_basis_updates=" "$LOG" | tail -1 | grep -oE "powersgd_basis_updates=[0-9]+" | grep -oE "[0-9]+")
  note "Q-writer counters: anchor_q_updates=${aqu:-<missing>} fast_powersgd_basis_updates=${fbu:-<missing>}"
  if [[ -n "$aqu" && "$aqu" -gt 0 ]]; then
    ok "anchor updated Q (anchor_q_updates=$aqu > 0) — anchor is a Q writer"
  else
    bad "anchor never updated Q (anchor_q_updates=${aqu:-missing}) — R2 anchor-as-Q-writer did NOT fire"
  fi
  if [[ -n "$fbu" && "$fbu" -eq 0 ]]; then
    ok "fast net issued NO local Q-update (powersgd_basis_updates=0) — fast maybe_update_basis gated off"
  else
    bad "fast net updated Q (powersgd_basis_updates=${fbu:-missing}) — fast maybe_update_basis NOT gated off (R2 SOLE-writer violated)"
  fi
  # COLD-M fallback: step-1 cold==corrected then warms
  echo "--- merger cold-M fallback trace (step1 cold==corrected, then ->0) ---"
  grep -oE "\[comm_eff\]\[merger\].*corrected=[0-9]+ merger_coldM_fallbacks=[0-9]+" "$LOG"
  grep -qE "\[comm_eff\]\[merger\].*merger_coldM_fallbacks=[1-9]" "$LOG" && ok "cold-M fallback FIRED (step 1, grads preserved not zeroed)" || bad "cold-M fallback never fired (silent-zero risk)"
fi

echo ""
if [[ $fail -eq 0 ]]; then echo "=== id-$PID PROBE: ALL CHECKED INVARIANTS GREEN ==="; else echo "=== id-$PID PROBE: ONE OR MORE FAILURES (see [FAIL] above) ==="; fi
exit $fail
