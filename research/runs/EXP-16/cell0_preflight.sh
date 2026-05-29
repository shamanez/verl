#!/usr/bin/env bash
# EXP-16 Cell 0 — GPU pre-flight: cross-pass mask-consistency GATE (on_fail:stop).
# Runs on the Vast.ai box AFTER the exp branch is applied. NOT a training cell.
# Proves the EXISTING cross-pass mask-consistency guarantee on the provisioned
# GPU for BOTH rescale settings before any training step is spent.
#
# Asserts (per plan ## Cell 0):
#   - the {0,1} MASK pattern is bit-identical across the two forwards (exact
#     integer PRF; zero mismatch) and per-boundary independent-but-consistent;
#   - the existing activation-mask test-lock passes ON THE BOX;
#   - h*mask compared only within FP tolerance (the rescale test-lock covers this);
#   - dynamic-sampling / filter_groups OFF in the launcher (the note's residual).
#
# Writes PASS/FAIL lines to /workspace/runs/EXP-16/cell0_preflight.log and a
# /workspace/runs/EXP-16/cell0.PASS sentinel on success (empty/absent on fail).
set -uo pipefail   # NOT -e: capture the test exit code, don't abort mid-script

RUN_ROOT=/workspace/runs/EXP-16
LOG="$RUN_ROOT/cell0_preflight.log"
PASS_SENTINEL="$RUN_ROOT/cell0.PASS"
LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
mkdir -p "$RUN_ROOT"
rm -f "$PASS_SENTINEL"

cd /workspace/verl
{
  echo "==== EXP-16 cell 0 pre-flight @ $(date -u +%FT%TZ) ===="
  echo "-- GPU inventory --"
  nvidia-smi -L || true
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  echo "nvidia_smi_gpu_count=$NGPU"
  if (( NGPU < 4 || NGPU > 8 )); then
    echo "CELL0 FAIL: GPU count $NGPU not in {4..8} (launcher hard-fails single-GPU)"; exit 1
  fi

  echo "-- torch CUDA sanity (kernel launch on the host driver) --"
  python3 -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'dev0', torch.cuda.get_device_name(0))" \
    || { echo "CELL0 FAIL: torch.cuda kernel/device probe failed (driver mismatch?)"; exit 1; }

  echo "-- launcher dynamic-sampling / filter_groups must be OFF (cross-pass sample_id residual) --"
  if grep -nE 'filter_groups[[:space:]]*=[[:space:]]*[Tt]rue|over_sample_rate[[:space:]]*=[[:space:]]*[^0]|dynamic_sampling[[:space:]]*=[[:space:]]*[Tt]rue' "$LAUNCHER"; then
    echo "CELL0 FAIL: launcher appears to enable dynamic sampling / filter_groups — cross-pass sample_id no longer stable"; exit 1
  fi
  echo "OK: no filter_groups / over_sample_rate / dynamic_sampling enabled in launcher"

  RC=0
  echo "==== both rescale settings (false/true) are parametrized by the test-locks below ===="
  echo "-- test-lock 1/2: tests/workers/comm_eff/test_activation_mask.py --"
  python3 -m pytest -v tests/workers/comm_eff/test_activation_mask.py
  A=$?; echo "activation_mask pytest rc=$A"; (( A != 0 )) && RC=1

  echo "-- test-lock 2/2: tests/workers/comm_eff/test_mask_rescale.py (h*mask/(1-p) within FP tol; rescale false vs true) --"
  python3 -m pytest -v tests/workers/comm_eff/test_mask_rescale.py
  B=$?; echo "mask_rescale pytest rc=$B"; (( B != 0 )) && RC=1

  echo "-- cross-pass MASK equality + per-boundary independence (explicit, on the box GPU) --"
  python3 - <<'PYEOF'
import sys, inspect, torch
# prf_token_mask(sample_ids, position_ids, *, layer_idx, global_step, base_seed,
#                hidden_size, p, device, dtype) -> (N, hidden_size) {0,1} keep-mask.
try:
    from verl.workers.comm_eff.activation_mask import prf_token_mask
except Exception as e:
    print("CELL0 FAIL: cannot import prf_token_mask:", e); sys.exit(1)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N, H, p = 64, 1536, 0.9
sample_ids   = torch.arange(N, device=dev, dtype=torch.int64)
position_ids = torch.arange(N, device=dev, dtype=torch.int64)
def draw(layer_idx, step):
    return prf_token_mask(sample_ids, position_ids, layer_idx=layer_idx,
                          global_step=step, base_seed=0, hidden_size=H, p=p,
                          device=dev, dtype=torch.float32)
print("prf_token_mask signature:", inspect.signature(prf_token_mask))
print("device:", dev)
m1 = draw(3, 7); m2 = draw(3, 7)
exact = bool(torch.equal(m1.to(torch.int64), m2.to(torch.int64)))
print("cross_pass_mask_bit_identical(layer=3,step=7):", exact)
if not exact:
    print("CELL0 FAIL: cross-pass mask pattern differs at fixed (layer,step) — IS ratio corrupt"); sys.exit(1)
m3 = draw(5, 7)
indep = not bool(torch.equal(m1.to(torch.int64), m3.to(torch.int64)))
print("per_boundary_independent(layer3 vs layer5):", indep)
m4 = draw(3, 8)
stepdiff = not bool(torch.equal(m1.to(torch.int64), m4.to(torch.int64)))
print("different_step_different_mask(step7 vs step8):", stepdiff)
frac = 1.0 - (m1.float().mean().item())
print("masked_fraction~p: %.3f (target %s)" % (frac, p))
if not (indep and stepdiff):
    print("CELL0 FAIL: PRF not keying on layer_idx/global_step as required"); sys.exit(1)
print("CELL0 cross-pass PRF assertions: PASS")
PYEOF
  C=$?; echo "cross_pass_prf rc=$C"; (( C != 0 )) && RC=1

  echo "==== cell0 aggregate rc=$RC (0=PASS) ===="
  if (( RC == 0 )); then
    echo "CELL0 PASS: mask {0,1} pattern bit-identical across forwards (both rescale settings via test-lock), test-locks green, per-boundary independent, filter_groups OFF."
    echo "NOTE: pre-update IS-ratio exp(log_prob-old_log_prob)~=1 is validated end-to-end by cell 1's first logged step (mask_recompute=true => same masked subnetwork/weights pre-update); the structural cross-pass mask equality proven here is its source."
    : > "$PASS_SENTINEL"
  else
    echo "CELL0 FAIL: one or more mask-consistency checks failed — STOP before any training cell."
  fi
  exit "$RC"
} 2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"
