#!/usr/bin/env bash
# ===========================================================================
# install_verl_foreign_image.sh
#
# Install this verl fork on a Vast.ai (or any) box whose Docker image is NOT our
# custom `verl-research-vllm020` template — e.g. the vast.ai recommended images:
#   * vastai/vllm      (ships torch + vllm; we add ONLY verl + deps + flash)
#   * vast PyTorch base (ships torch only; we add vllm + verl + deps + flash)
#
# Proven 2026-07-21 on cu13 / py3.12 boxes:
#   H200 vastai/vllm  : torch 2.11.0+cu130, vllm 0.25.0  -> steps 3..6 only
#   H100 PyTorch base : torch 2.12.0+cu130 (-> 2.11 via vllm), no vllm -> all steps
#
# GOLDEN RULES (why this is fast and does not break the image):
#   1. NEVER install cuda / cudnn / nccl. They arrive as torch's own
#      nvidia-*-cu13 dependencies. `nvcc`/driver are the host's, not pip's.
#   2. NEVER let pip move torch / vllm / numpy / transformers. We snapshot them
#      to a constraints file and install verl with --no-deps.
#   3. flash-attn is the ONLY thing that needs care: a matching PREBUILT wheel
#      installs in ~15s (no compiler). Source build is the last resort.
#
# Usage:  bash install_verl_foreign_image.sh
# Env overrides: VERL_BRANCH, VERL_DIR, VLLM_VERSION, FLASH_WHEEL_URL
# ===========================================================================
set -euo pipefail

VERL_BRANCH="${VERL_BRANCH:-autonomous-harness-v1}"
VERL_DIR="${VERL_DIR:-/workspace/verl}"
VERL_REPO="${VERL_REPO:-https://github.com/shamanez/verl.git}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"     # pins the coherent torch/transformers set
CONSTRAINTS="${CONSTRAINTS:-/workspace/constraints.txt}"

# --------------------------------------------------------------------------
# 0. Find the interpreter that actually owns torch. On vast images the stack
#    lives in /venv/main, NOT in system python3 — target it explicitly.
# --------------------------------------------------------------------------
if [[ -x /venv/main/bin/python ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
  PY=/venv/main/bin/python
else
  PY="$(command -v python3)"
fi
command -v uv >/dev/null || python3 -m pip install -q uv || true
PIP=(uv pip install --python "$PY")
echo "[install] interpreter: $PY"
"$PY" -c 'import torch,sys; print("[install] torch", torch.__version__, "cuda", torch.version.cuda, "py", "%d.%d"%sys.version_info[:2])' \
  || { echo "[install] FATAL: no torch in $PY — this is not a PyTorch/vLLM image" >&2; exit 1; }

# --------------------------------------------------------------------------
# 1. vLLM — install ONLY if the image lacks it. Installing it pins a coherent
#    (torch, transformers, flashinfer) set; on cu13/py312 it lands on
#    torch 2.11.0+cu130 / vllm 0.25.1 (keeps CUDA 13).
# --------------------------------------------------------------------------
if "$PY" -c 'import vllm' 2>/dev/null; then
  echo "[install] vllm present ($("$PY" -c 'import vllm;print(vllm.__version__)')) — preserving image stack"
else
  echo "[install] vllm absent — installing vllm==$VLLM_VERSION (+ ray) — pulls the coherent torch stack"
  "${PIP[@]}" "vllm==$VLLM_VERSION" "ray[default]"
fi

# --------------------------------------------------------------------------
# 2. Snapshot the good stack as constraints so nothing below can move it.
# --------------------------------------------------------------------------
"$PY" -m pip freeze --exclude-editable | grep '==' | grep -v ' @ ' > "$CONSTRAINTS"
echo "[install] constraints -> $CONSTRAINTS ($(wc -l < "$CONSTRAINTS") pins)"
grep -iE '^(torch|vllm|transformers|numpy|ray|flashinfer)' "$CONSTRAINTS" | sed 's/^/[install]   /'

# --------------------------------------------------------------------------
# 3. verl itself: editable, NO deps (its stale requirements.txt pins numpy<2 and
#    vllm==0.8.4, which would wreck the working stack).
# --------------------------------------------------------------------------
if [[ ! -d "$VERL_DIR/.git" ]]; then
  git clone --depth 1 --branch "$VERL_BRANCH" "$VERL_REPO" "$VERL_DIR"
fi
"${PIP[@]}" --no-deps -e "$VERL_DIR"

# --------------------------------------------------------------------------
# 4. The missing light pure-python deps, guarded by the constraints file.
#    TransferQueue is EXCLUDED on purpose: it hard-pins numpy<2.0 (incompatible
#    with this numpy-2.x stack) and is unused on the FSDP training path.
# --------------------------------------------------------------------------
"${PIP[@]}" -c "$CONSTRAINTS" \
  tensordict codetiming hydra-core omegaconf torchdata peft pylatexenc \
  latex2sympy2_extended math_verify tensorboard wandb pybind11 datasets \
  pandas pyarrow accelerate

# --------------------------------------------------------------------------
# 5. flash-attention-2 — THE FAST WAY (prebuilt wheel, ~15s, no compiler).
#    The Dao wheel built against torch2.10+cu13 is ABI-compatible with torch2.11
#    (the torch2.9 wheel is NOT). We try the prebuilt; only if its kernel fails
#    to import do we fall back to a Hopper-only (sm_90) source build.
# --------------------------------------------------------------------------
FLASH_WHEEL_URL="${FLASH_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
if "$PY" -c 'import flash_attn' 2>/dev/null; then
  echo "[install] flash_attn already present"
else
  echo "[install] installing prebuilt flash-attn wheel (no build): $FLASH_WHEEL_URL"
  "${PIP[@]}" --no-deps "$FLASH_WHEEL_URL" || true
  if ! "$PY" -c 'import flash_attn' 2>/dev/null; then
    echo "[install] prebuilt wheel did not import (ABI mismatch) — Hopper-only source build fallback"
    "${PIP[@]}" uninstall flash-attn 2>/dev/null || true
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS="${MAX_JOBS:-$(nproc)}" FLASH_ATTENTION_FORCE_BUILD=TRUE \
      "${PIP[@]}" --no-build-isolation --no-deps flash-attn
  fi
fi

# --------------------------------------------------------------------------
# 6. Verify: heavy stack intact + verl imports + flash usable for training.
# --------------------------------------------------------------------------
"$PY" - <<'PY'
import torch, vllm, transformers, numpy, tensordict, flash_attn, verl
from verl.trainer import main_ppo
from transformers.utils.import_utils import is_flash_attn_2_available
print("[verify] torch", torch.__version__, "| vllm", vllm.__version__,
      "| transformers", transformers.__version__, "| numpy", numpy.__version__,
      "| tensordict", tensordict.__version__, "| flash_attn", flash_attn.__version__)
print("[verify] is_flash_attn_2_available:", is_flash_attn_2_available(), "| verl + main_ppo import OK")
PY

echo "[install] DONE. Next: prepare data + run the comm-eff launcher, e.g."
echo "    $PY research/scripts/prepare_rlvr_math.py --dataset math --local_save_dir \$HOME/data/math"
echo "    bash examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
