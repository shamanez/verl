#!/usr/bin/env bash
# EXP-16 B200 env bring-up (bare CUDA container, NOT the docker template).
# Operator directive: install latest vllm + verl as pip packages; tear the box
# down on any CUDA/Blackwell issue. Logs to /workspace/install.log and writes
# /workspace/install.DONE when finished (success or not — read the log).
set +e
export PIP_BREAK_SYSTEM_PACKAGES=1      # Ubuntu 24.04 PEP 668
export PIP_ROOT_USER_ACTION=ignore
export DEBIAN_FRONTEND=noninteractive
LOG=/workspace/install.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "==== EXP-16 install start $(date -u +%FT%TZ) ===="
nvidia-smi --query-gpu=index,name,driver_version --format=csv | head

python3 -m pip install -U pip setuptools wheel

echo "==== install latest vllm (pulls a Blackwell-capable torch) ===="
python3 -m pip install -U vllm
echo "vllm_install_rc=$?"

echo "==== torch / vllm / B200 capability after vllm ===="
python3 - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("device_count", torch.cuda.device_count())
    print("arch_list", torch.cuda.get_arch_list())
    print("cap0", torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
except Exception as e:
    print("TORCH_PROBE_FAIL", repr(e))
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("VLLM_IMPORT_FAIL", repr(e))
PY

echo "==== verl editable (no-deps; vllm owns torch/transformers) ===="
cd /workspace/verl
python3 -m pip install --no-deps -e .
echo "verl_editable_rc=$?"

echo "==== verl runtime deps (pip resolves vs installed torch/numpy) ===="
python3 -m pip install \
  hydra-core codetiming "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
  pylatexenc pybind11 dill peft accelerate datasets wandb tensorboard \
  liger-kernel mathruler math-verify latex2sympy2_extended \
  "pyarrow>=19.0.0" "ray[default]>=2.41.0"
echo "verl_deps_rc=$?"

echo "==== flash-attn (use_remove_padding=True needs it) ===="
python3 -m pip install flash-attn --no-build-isolation
echo "flash_attn_install_rc=$?"

echo "==== final import check ===="
python3 - <<'PY'
import importlib
ok = True
for m in ["torch","vllm","ray","tensordict","transformers","verl","datasets","hydra","wandb","pyarrow"]:
    try:
        mod = importlib.import_module(m)
        print("OK", m, getattr(mod, "__version__", ""))
    except Exception as e:
        ok = False; print("FAIL", m, repr(e))
try:
    import flash_attn; print("OK flash_attn", flash_attn.__version__)
except Exception as e:
    print("WARN flash_attn", repr(e))
print("ALL_CORE_OK" if ok else "CORE_IMPORT_FAILED")
PY

echo "==== B200 CUDA kernel smoke (matmul on each GPU) ===="
python3 - <<'PY'
import torch
fails = 0
print("cuda_is_available", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    try:
        x = torch.randn(2048, 2048, device=f"cuda:{i}", dtype=torch.bfloat16)
        y = (x @ x).float().sum().item()
        print(f"CUDA_MATMUL_OK gpu{i}", round(y, 1))
    except Exception as e:
        fails += 1; print(f"CUDA_MATMUL_FAIL gpu{i}", repr(e))
print("CUDA_SMOKE_PASS" if fails == 0 and torch.cuda.is_available() else "CUDA_SMOKE_FAIL")
PY

echo "==== EXP-16 install done $(date -u +%FT%TZ) ===="
touch /workspace/install.DONE
