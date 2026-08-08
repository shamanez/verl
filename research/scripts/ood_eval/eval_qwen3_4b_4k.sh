#!/usr/bin/env bash
# eval_qwen3_4b_4k.sh
#
# In-domain + OOD capability audit for the Qwen3-4B-Base 4096-context 500-step
# pair: PRF exact-k against the dense control, at every saved checkpoint.
#
#   bash research/scripts/ood_eval/eval_qwen3_4b_4k.sh
#
# Pipeline, per tag: (1) merge the FSDP checkpoint (local first, R2 pull only if
# absent) into a clean HF model, (2) evaluate it on 10 benchmarks via ood_eval.sh
# (val-only, no training) fanned over GPU pairs, (3) tabulate.
#
# The roster is PRIORITY ORDERED and the whole script is resumable at
# per-benchmark granularity (a bench that already logged acc/mean@ is skipped),
# so an interrupted run resumes without repeating work and a run that is cut
# short still holds the headline comparison. base / commeff500 / dense500 come
# first for exactly that reason.
#
# Portability knobs (override via env):
#   VERL_DIR        verl checkout                  (default /workspace/verl)
#   OOD_EVAL_ROOT   eval output root               (default /workspace/runs/ood-eval-4b)
#   OOD_DATA_ROOT   prepared benchmark parquets    (default /root/data/ood)
#   CKPT_ROOT       local FSDP checkpoint root     (default $VERL_DIR/checkpoints/qwen3-4b-4k-500)
#   BASE_MODEL      untrained reference            (default Qwen/Qwen3-4B-Base)
#   PAIRS_CSV       GPU-pair pool                  (default "0,1|2,3" for a 4-GPU box)
#   R2_PREFIX       key prefix under the bucket    (default autonomous-harness-rlvr-compression/qwen3-4b-4k-500)
#   R2_CKPT_BUCKET  bucket holding the checkpoints (default shamane-pluralis)
#   STEPS           checkpoint steps to evaluate   (default "500 400 300 200 100")
# R2 credentials come from ~/.config/verl-research/secrets.env and are never
# stored here. If every checkpoint is present locally, no R2 access happens.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="${VERL_DIR:-/workspace/verl}"
OOD_EVAL_ROOT="${OOD_EVAL_ROOT:-/workspace/runs/ood-eval-4b}"
OOD_DATA_ROOT="${OOD_DATA_ROOT:-/root/data/ood}"
RUN_ID="${RUN_ID:-qwen3-4b-4k-500}"
CKPT_ROOT="${CKPT_ROOT:-$VERL_DIR/checkpoints/$RUN_ID}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Base}"
R2_PREFIX="${R2_PREFIX:-autonomous-harness-rlvr-compression/$RUN_ID}"
export R2_CKPT_BUCKET="${R2_CKPT_BUCKET:-shamane-pluralis}"
export OOD_DATA_ROOT
COMMEFF_RUN="${COMMEFF_RUN:-qwen3-4b-4k-commeff-500}"
DENSE_RUN="${DENSE_RUN:-qwen3-4b-4k-dense-500}"
# REQUIRED for any Qwen3 checkpoint. ood_eval.sh emits no rollout.max_model_len,
# so vLLM falls back to max_position_embeddings, which is 32768 here against the
# 4096 we actually trained at. With enable_chunked_prefill=False hardcoded in the
# engine, vLLM then refuses to boot because max_num_batched_tokens (8192) is
# below max_model_len, and every one of the 110 evaluations dies at startup.
export OOD_EXTRA_HYDRA="${OOD_EXTRA_HYDRA:-actor_rollout_ref.rollout.max_model_len=4096 actor_rollout_ref.rollout.max_num_batched_tokens=8192}"
cd "$VERL_DIR" || { echo "FATAL: cannot cd $VERL_DIR" >&2; exit 1; }
MERGED="$OOD_EVAL_ROOT/merged"
mkdir -p "$MERGED" "$OOD_EVAL_ROOT"

# --- 0. benchmark parquets. ood_prep.py is idempotent and cheap when present ---
if [[ ! -f "$OOD_DATA_ROOT/math500/test.parquet" ]]; then
  echo "=== preparing benchmark parquets in $OOD_DATA_ROOT ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
  OOD_ROOT="$OOD_DATA_ROOT" MATH_TRAIN="${MATH_TRAIN:-$HOME/data/math/train.parquet}" \
    python3 "$SCRIPT_DIR/ood_prep.py" 2>&1 | tee -a "$OOD_EVAL_ROOT/eval.log"
fi

# --- 1. merge <run_name> <step> <tag>: local checkpoint or R2 pull (actor only, no optim) ---
# A checkpoint PULLED from R2 is deleted again once the merged HF model exists:
# on a 200G box the alternative (10 pulled FSDP trees + 10 merged models) does
# not fit. A checkpoint that was already local is never touched.
merge() {
  local run="$1" step="$2" tag="$3"
  local ck="$CKPT_ROOT/$run/global_step_$step/actor" out="$MERGED/$tag" pulled=0
  [[ -f "$out/config.json" ]] && { echo "$out"; return; }
  if [[ ! -f "$ck/fsdp_config.json" ]]; then
    set -a; source ~/.config/verl-research/secrets.env; set +a
    export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    local r2="s3://$R2_CKPT_BUCKET/$R2_PREFIX/$run/checkpoints/global_step_$step/actor/"
    echo "=== R2 pull $run s$step -> $ck ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
    mkdir -p "$ck"
    aws s3 cp "$r2" "$ck/" --recursive --exclude "optim*" --only-show-errors \
      --endpoint-url "$R2_ENDPOINT" >> "$OOD_EVAL_ROOT/merge.log" 2>&1 \
      || { echo "R2 PULL FAILED $run s$step" | tee -a "$OOD_EVAL_ROOT/eval.log"; echo ""; return; }
    pulled=1
  fi
  if [[ -f "$ck/fsdp_config.json" ]]; then
    echo "=== merge $run s$step -> $out ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
    OMP_NUM_THREADS=8 python3 -m verl.model_merger merge --backend fsdp \
      --local_dir "$ck" --target_dir "$out" >> "$OOD_EVAL_ROOT/merge.log" 2>&1
  fi
  if (( pulled )) && [[ -f "$out/config.json" ]]; then
    rm -rf "$CKPT_ROOT/$run/global_step_$step"
    echo "=== cleaned pulled ckpt $run s$step (merged model kept) ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
  fi
  echo "$out"
}

# --- 2. eval matrix. greedy mean@1 for the large sets, avg@8 for competition sets ---
#     bench  n  temp  top_p
BENCHES=(
  "math500   1 0   1.0"
  "gsm8k     1 0   1.0"
  "minerva   1 0   1.0"
  "olympiad  1 0   1.0"
  "amc23     8 0.7 0.8"
  "aime24    8 0.7 0.8"
  "aime25    8 0.7 0.8"
  "aime26    8 0.7 0.8"
  "hmmt25    8 0.7 0.8"
  "mmlu_stem 1 0   1.0"
)
IFS='|' read -r -a PAIRS <<< "${PAIRS_CSV:-0,1|2,3}"

run_tag() {  # run_tag <tag> <model_path>
  local tag="$1" model="$2" i=0 pid_list=()
  [[ "$model" == "$BASE_MODEL" || -f "$model/config.json" ]] || { echo "SKIP $tag: no model at $model" | tee -a "$OOD_EVAL_ROOT/eval.log"; return; }
  [[ -f "$OOD_EVAL_ROOT/$tag/.done" ]] && { echo "SKIP $tag: already evaluated"; return; }
  for spec in "${BENCHES[@]}"; do
    read -r b n t p <<<"$spec"
    # Per-bench success skip: never re-run or overwrite a bench that already has a result.
    if [[ -f "$OOD_EVAL_ROOT/$tag/$b/train.log" ]] && grep -q "acc/mean@" "$OOD_EVAL_ROOT/$tag/$b/train.log" 2>/dev/null; then
      echo "SKIP $tag/$b: already has acc/mean@" | tee -a "$OOD_EVAL_ROOT/eval.log"
      continue
    fi
    local gpus="${PAIRS[$(( i % ${#PAIRS[@]} ))]}"
    VERL_DIR="$VERL_DIR" OOD_EVAL_ROOT="$OOD_EVAL_ROOT" OOD_DATA_ROOT="$OOD_DATA_ROOT" \
      bash "$SCRIPT_DIR/ood_eval.sh" "$model" "$b" "$tag" "$gpus" "$n" "$t" "$p" &
    pid_list+=($!); i=$((i+1))
    (( i % ${#PAIRS[@]} == 0 )) && wait "${pid_list[@]}" && pid_list=()
  done
  wait "${pid_list[@]:-}" 2>/dev/null || true
  # Success-aware completion: only mark the tag done when ALL benches produced a result.
  local ok=0
  for spec in "${BENCHES[@]}"; do
    read -r b n t p <<<"$spec"
    [[ -f "$OOD_EVAL_ROOT/$tag/$b/train.log" ]] && grep -q "acc/mean@" "$OOD_EVAL_ROOT/$tag/$b/train.log" 2>/dev/null && ok=$((ok+1))
  done
  if (( ok == ${#BENCHES[@]} )); then
    touch "$OOD_EVAL_ROOT/$tag/.done" 2>/dev/null || true
    echo "DONE $tag: $ok/${#BENCHES[@]} benches" | tee -a "$OOD_EVAL_ROOT/eval.log"
  else
    echo "PARTIAL $tag: $ok/${#BENCHES[@]} benches (no .done; will resume)" | tee -a "$OOD_EVAL_ROOT/eval.log"
  fi
}

# ---- model roster (tag  run  step), PRIORITY ORDERED. The headline three come
#      first so a truncated run still answers the question; the dose axis fills
#      in behind them. ----
# Ordered so the HEADLINE comparison evaluates first. The compressed arm was
# stopped at step 200 after collapsing (val 0.437 against an untrained 0.645),
# so its only checkpoints are 100 (the healthy peak) and 200 (post-collapse).
# Leading with 200 then 100 pairs both against dense before the dense-only dose
# points; the missing commeff 300/400/500 tags skip cheaply.
STEPS="${STEPS:-200 100 500 400 300}"
ROSTER=("base - -")
read -r HEAD_STEP _ <<<"$STEPS"
ROSTER+=("commeff${HEAD_STEP} $COMMEFF_RUN ${HEAD_STEP}")
ROSTER+=("dense${HEAD_STEP} $DENSE_RUN ${HEAD_STEP}")
for s in $STEPS; do
  [[ "$s" == "$HEAD_STEP" ]] && continue
  ROSTER+=("commeff${s} $COMMEFF_RUN ${s}")
  ROSTER+=("dense${s} $DENSE_RUN ${s}")
done

for entry in "${ROSTER[@]}"; do
  read -r tag run step <<<"$entry"
  if [[ "$tag" == "base" ]]; then model="$BASE_MODEL"; else model=$(merge "$run" "$step" "$tag" | tail -1); fi
  echo "=== EVAL MODEL $tag $(date -Iseconds) ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
  run_tag "$tag" "$model"
done

# --- 3. tabulate + emit the machine-readable results the report reads ---
OOD_EVAL_ROOT="$OOD_EVAL_ROOT" STEPS="$STEPS" \
  python3 "$SCRIPT_DIR/collect_qwen3_4b_4k.py" | tee "$OOD_EVAL_ROOT/RESULTS.txt"
echo "=== ALL EVAL DONE $(date -Iseconds) ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
touch "$OOD_EVAL_ROOT/OOD_DONE"
