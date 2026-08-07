#!/usr/bin/env bash
# Evaluate ONE model on ONE in-domain/OOD benchmark via verl val-only (no training).
# Reuses the GRPO training launcher so the prompt template and the boxed scorer are
# identical to training-time validation.
#
# Usage: ood_eval.sh <MODEL_PATH> <BENCH> <TAG> <GPUS> [N] [TEMP] [TOPP]
#   MODEL_PATH  merged HF checkpoint (or a HF hub id for the base model)
#   BENCH       benchmark name; must match a $OOD_DATA_ROOT/<BENCH>/test.parquet built by ood_prep.py
#   TAG         output group (results land under $OOD_EVAL_ROOT/<TAG>/<BENCH>/)
#   GPUS        CUDA_VISIBLE_DEVICES value, e.g. "0,1"
#   N/TEMP/TOPP sampling: N=1 temp=0 -> greedy mean@1; N>1 -> avg@N with do_sample
#
# Portability knobs (override via env; defaults match the reference single-box layout):
#   VERL_DIR       verl checkout                         (default /workspace/verl)
#   OOD_EVAL_ROOT  where per-tag/per-bench logs are kept (default /workspace/runs/ood-eval)
#   OOD_DATA_ROOT  where ood_prep.py wrote the parquets  (default /root/data/ood)
#   LAUNCHER       training launcher reused for val-only (default examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh)
#   SHIM_DIR       optional nvidia-smi -L shim on PATH   (default /workspace/shim, skipped if absent)
#   OOD_EXTRA_HYDRA  extra Hydra overrides, space separated (default empty)
#
# OOD_EXTRA_HYDRA exists because this script emits no rollout.max_model_len, so
# vLLM falls back to the checkpoint's max_position_embeddings. That is 4096 for
# Qwen2.5-Math-1.5B, which is why the reference study never noticed, but 32768
# for Qwen3, and with enable_chunked_prefill=False hardcoded in the engine vLLM
# REFUSES TO BOOT whenever max_num_batched_tokens (8192 by default) is below
# max_model_len. Evaluating any Qwen3 checkpoint therefore requires:
#   OOD_EXTRA_HYDRA="actor_rollout_ref.rollout.max_model_len=4096"
# Empty by default, so every existing caller is byte-identical.
set -uo pipefail
VERL_DIR="${VERL_DIR:-/workspace/verl}"
OOD_EVAL_ROOT="${OOD_EVAL_ROOT:-/workspace/runs/ood-eval}"
OOD_DATA_ROOT="${OOD_DATA_ROOT:-/root/data/ood}"
LAUNCHER="${LAUNCHER:-examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh}"
SHIM_DIR="${SHIM_DIR:-/workspace/shim}"
read -r -a EXTRA_HYDRA <<< "${OOD_EXTRA_HYDRA:-}"
cd "$VERL_DIR"
MODEL="$1"; BENCH="$2"; TAG="$3"; GPUS="$4"
N="${5:-1}"; TEMP="${6:-0}"; TOPP="${7:-1.0}"
DO_SAMPLE=False; [ "${N}" -gt 1 ] && DO_SAMPLE=True
DIR="$OOD_EVAL_ROOT/$TAG/$BENCH"
mkdir -p "$DIR"
export CUDA_VISIBLE_DEVICES=$GPUS
[ -d "$SHIM_DIR" ] && export PATH="$SHIM_DIR:$PATH"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_MAX_THREADS=8 \
       TORCHINDUCTOR_COMPILE_THREADS=16 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
echo "=== EVAL $TAG/$BENCH model=$MODEL n=$N temp=$TEMP $(date -Iseconds) ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
env MODEL_PATH="$MODEL" DATA_DIR="$OOD_DATA_ROOT/$BENCH" \
  PROJECT_NAME=quick-test WANDB_RUN_GROUP=quick-test EXPERIMENT_NAME=ood-$TAG-$BENCH \
  COMM_EFF_ENABLED=false \
  TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1 VAL_BEFORE_TRAIN=True TEST_FREQ=1 SAVE_FREQ=-1 \
  LOG="$DIR/train.log" \
  bash "$LAUNCHER" \
    trainer.val_only=True \
    actor_rollout_ref.rollout.val_kwargs.n=$N \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMP \
    actor_rollout_ref.rollout.val_kwargs.top_p=$TOPP \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$DO_SAMPLE \
    ray_kwargs.ray_init.num_cpus=48 \
    ${EXTRA_HYDRA[@]+"${EXTRA_HYDRA[@]}"} \
    >> "$DIR/driver.log" 2>&1
rc=$?
acc=$(grep -oE "val-core/[^ ]*acc/mean@[0-9]+['\"]?[: ]+[0-9.]+" "$DIR/train.log" 2>/dev/null | tail -1)
echo "=== DONE $TAG/$BENCH rc=$rc  $acc  $(date -Iseconds) ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
