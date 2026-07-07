#!/usr/bin/env bash
# =============================================================================
# setup_run_verl_env.sh
#
# Build the `run-verl` virtualenv that REPRODUCES our verl docker image
# (verlai/verl:vllm020.dev1  ==  docker/Dockerfile.stable.vllm, verl v0.7.1),
# FSDP+vLLM subset, on a BARE Vast box that does NOT ship our image.
#
# EXACT core versions (identical to the Dockerfile):
#     torch        2.11.0+cu130   (torchvision 0.26.0, torchaudio 2.11.0)
#     vllm         0.20.2
#     transformers 5.3.0
#     flash-attn   2.8.3          (BUILT FROM SOURCE — no torch-2.11 wheel exists;
#                                  the docker image compiles it too, line 86)
# CUDA: we do NOT install a CUDA toolkit. torch's cu130 wheels bundle the CUDA
# runtime and match this instance's driver (CUDA 13); the flash-attn build uses
# the instance's existing /usr/local/cuda nvcc. Nothing CUDA is installed here.
#
# We SKIP the Dockerfile's Megatron/TransformerEngine/apex/DeepEP/nsight/trl/
# mbridge/torchcodec/multimodal layers — none are used by FSDP+vLLM GRPO.
#
# Idempotent + reusable: writes a marker; re-running is an instant no-op once
# built, so FUTURE SESSIONS pay ZERO setup time (the ~15-min flash-attn compile
# happens once). Use FORCE=1 to rebuild from scratch.
#
# Usage:
#   bash research/scripts/setup_run_verl_env.sh              # build (once)
#   FORCE=1 bash research/scripts/setup_run_verl_env.sh      # rebuild clean
#   source /workspace/venvs/run-verl/bin/activate            # use it
#
# NOTE: this SUPERSEDES setup_vast_verl_grpo_fsdp_vllm.sh (that one targeted the
# older torch-2.8/vllm-0.11 wheel stack, which does NOT match our docker image).
# =============================================================================
set -Eeuo pipefail

VENV="${VENV:-/workspace/venvs/run-verl}"
VERL_DIR="${VERL_DIR:-/workspace/verl}"
PYBIN="${PYBIN:-python3}"   # box default python3 is 3.12.x
MAX_JOBS="${MAX_JOBS:-96}"
# H200 is sm_90. Building only sm_90 is ~4x faster than the Dockerfile's all-arch
# build and is FUNCTIONALLY IDENTICAL on this GPU. Set FLASH_ATTN_ARCHS="" to build
# every arch exactly like the Dockerfile (slow: ~1-3h).
FLASH_ATTN_ARCHS="${FLASH_ATTN_ARCHS:-9.0}"
MARKER="$VENV/.run_verl_ready"
LOG="${LOG:-/workspace/setup-logs/setup_run_verl_$(date -u +%Y%m%d_%H%M%S).log}"

# --- exact docker image pins ---
TORCH_VERSION=2.11.0
TORCHVISION_VERSION=0.26.0
TORCHAUDIO_VERSION=2.11.0
VLLM_VERSION=0.20.2
TRANSFORMERS_VERSION=5.3.0
FLASH_ATTENTION_VERSION=2.8.3
CU_INDEX=https://download.pytorch.org/whl/cu130

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== setup_run_verl_env @ $(date -u +%FT%TZ)  ->  $VENV ==="

# 0. fast path — already built
if [[ -f "$MARKER" && "${FORCE:-0}" != "1" ]]; then
  echo "run-verl already built ($MARKER). Nothing to do."
  echo "activate:  source $VENV/bin/activate"
  exit 0
fi

# 1. system build tools (NO cuda toolkit — driver + existing nvcc are used as-is).
#    Guarded: only if root + apt present.
if [[ "$(id -u)" == "0" ]] && command -v apt-get >/dev/null 2>&1; then
  echo "=== apt: build tools + venv module (no cuda) ==="
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends build-essential ninja-build git wget \
    "${PYBIN}-venv" "${PYBIN}-dev" || echo "WARN: some apt pkgs missing (continuing)"
fi

# 2. remove any prior run-verl venv and make a FRESH, isolated one (no old pip
#    installs inherited). The base-image venv /venv/main is intentionally left
#    untouched (deleting it can break Vast's managed services); run-verl does not
#    use it and does not see its packages.
echo "=== creating fresh venv (removing any prior run-verl) ==="
deactivate 2>/dev/null || true
rm -rf "$VENV"
mkdir -p "$(dirname "$VENV")"
"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V
python -m pip install -U pip setuptools wheel packaging pybind11 ninja

# 3. torch stack — cu130 wheels match the instance driver (CUDA 13); no toolkit.
echo "=== torch $TORCH_VERSION (+cu130) / torchvision $TORCHVISION_VERSION / torchaudio $TORCHAUDIO_VERSION ==="
pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" --index-url "$CU_INDEX"
python -c "import torch; assert torch.__version__.startswith('$TORCH_VERSION'), torch.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# 4. flash-attn FROM SOURCE (exactly as the docker image; no torch-2.11 wheel).
echo "=== flash-attn $FLASH_ATTENTION_VERSION from source (arch=${FLASH_ATTN_ARCHS:-ALL}, MAX_JOBS=$MAX_JOBS) ==="
FA_ENV=(FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="$MAX_JOBS")
if [[ -n "$FLASH_ATTN_ARCHS" ]]; then
  FA_ENV+=("TORCH_CUDA_ARCH_LIST=$FLASH_ATTN_ARCHS" "FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_ARCHS/./}")
fi
env "${FA_ENV[@]}" pip install -v --no-build-isolation "flash_attn==$FLASH_ATTENTION_VERSION"

# 5. vllm + transformers (EXACT template pairing; torch already satisfied -> kept).
echo "=== vllm $VLLM_VERSION ==="
pip install "vllm==$VLLM_VERSION"
echo "=== transformers $TRANSFORMERS_VERSION ==="
pip install "transformers==$TRANSFORMERS_VERSION"
python -c "import torch; assert torch.__version__.startswith('$TORCH_VERSION'), 'torch was clobbered: '+torch.__version__; print('torch preserved:', torch.__version__)"

# 6. verl runtime deps (from requirements.txt, FSDP+vLLM subset) + editable verl.
#    numpy is intentionally UNPINNED (torch 2.11 pulls numpy 2.x, matching the image;
#    verl's stale numpy<2 pin is not applied). TransferQueue is installed --no-deps
#    (its numpy<2 pin would wrongly downgrade numpy).
echo "=== verl deps + editable install ==="
pip install \
  accelerate codetiming datasets dill hydra-core liger-kernel mathruler \
  pandas peft "pyarrow>=19.0.0" pylatexenc torchdata wandb \
  "ray[default]" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
  latex2sympy2_extended math-verify \
  hf-transfer hf-xet "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" \
  "optree>=0.13.0" "pydantic>=2.9" uvicorn tensorboard qwen-vl-utils==0.0.14
pip install --no-deps -e "$VERL_DIR"
pip install --no-deps "TransferQueue==0.1.7"

# 7. verify the full path imports, then stamp the marker.
echo "=== verify ==="
python - <<'PY'
import importlib.metadata as m, torch
for p in ["torch","torchvision","torchaudio","vllm","transformers","flash_attn","numpy","tensordict","ray"]:
    print(f"  {p:14s} {m.version(p)}")
assert torch.cuda.is_available(), "CUDA not available"
import flash_attn; from flash_attn import flash_attn_varlen_func   # noqa
import vllm, verl, transfer_queue                                   # noqa
import verl.trainer.main_ppo                                        # noqa
import verl.workers.comm_eff.spectral_filter                        # noqa
print("  device:", torch.cuda.get_device_name(0), "| torch cuda:", torch.version.cuda)
print("RUN_VERL_IMPORTS_OK")
PY
python -m pip check || echo "WARN: pip check reported issues (review above)"
touch "$MARKER"
echo ""
echo "=== run-verl READY ==="
echo "activate:  source $VENV/bin/activate"
echo "log:       $LOG"
