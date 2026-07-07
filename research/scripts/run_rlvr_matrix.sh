#!/usr/bin/env bash
# Issue #62 stage 3-4 driver — run the 25-cell smoke matrix + dense-control parity
# on ONE box, sequentially, reusing the canonical comm-eff accel launcher (override
# ONLY MODEL_PATH + DATA_DIR + smoke knobs; NEVER fork the algorithm path).
#
# Modes:
#   prep            prep all 5 datasets to $HOME/data/<dir> (idempotent)
#   canary          prep math + run the single canary cell (smoke-math-qwen25-math-1p5b)
#   rest            run every smoke cell EXCEPT the canary
#   parity          run the dense-control-parity cell
#   all             prep + every smoke cell + parity
#
# Per-cell status is appended to $RUNROOT/matrix_status.tsv:  cell<TAB>rc<TAB>verdict_json
set -uo pipefail
cd /workspace/verl
source /root/.verl_secrets.env 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=0

MODE="${1:-all}"
RUNROOT="runs/62-rlvr-models-datasets"
STATUS="$RUNROOT/matrix_status.tsv"
LAUNCHER="examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh"
mkdir -p "$RUNROOT"

# --- registries -------------------------------------------------------------
declare -A DSDIR=( [math]=math [numina-cot]=numina_cot [deepscaler]=deepscaler [skywork-or1]=skywork_or1 [dapo-math]=dapo_math )
declare -A MODEL=(
  [qwen25-math-1p5b]="Qwen/Qwen2.5-Math-1.5B"
  [qwen3-1p7b-base]="Qwen/Qwen3-1.7B-Base"
  [r1-distill-qwen-1p5b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  [open-nemotron-1p5b]="nvidia/OpenReasoning-Nemotron-1.5B"
  [qwen3-4b-base]="Qwen/Qwen3-4B-Base"
)
DATASETS=(math numina-cot deepscaler skywork-or1 dapo-math)
# canary first, then the rest of the models
MODELS_ORDER=(qwen25-math-1p5b qwen3-1p7b-base r1-distill-qwen-1p5b open-nemotron-1p5b qwen3-4b-base)
CANARY="smoke-math-qwen25-math-1p5b"

prep_one() { # slug
  local slug="$1" dir="${DSDIR[$1]}"
  if [[ -f "$HOME/data/$dir/train.parquet" && -f "$HOME/data/$dir/test.parquet" ]]; then
    echo "[prep] $slug present ($HOME/data/$dir)"; return 0; fi
  echo "[prep] $slug -> $HOME/data/$dir"
  python research/scripts/prepare_rlvr_math.py --dataset "$slug" \
    --local_save_dir "$HOME/data/$dir" --train-cap 2000 --val-size 200 --seed 42
}

prep_all() { for d in "${DATASETS[@]}"; do prep_one "$d" || echo "[prep] FAILED $d"; done; }

judge_cell() { # cell
  # The launcher may exit non-zero on a benign wandb-teardown error AFTER training
  # succeeds, so judge on the training exit code (train_rc from setup.log) + the log,
  # NOT on the launcher's shell exit code.
  local cell="$1" S="$RUNROOT/$cell"
  local trc; trc="$(grep -oE 'train_rc=[0-9]+' "$S/setup.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')"
  [[ -z "$trc" ]] && trc="?"
  local v; v="$(python research/scripts/assert_cell_reward.py "$S/train.log" 2>/dev/null)"
  local es="ok"; [[ -f "$S/EARLY_STOP_SIGNAL" ]] && es="EARLY_STOP_SIGNAL"
  printf '%s\ttrain_rc=%s\t%s\t%s\n' "$cell" "$trc" "$es" "$v" >> "$STATUS"
  echo "[judge] $cell train_rc=$trc $es -> $v"
}

run_cell() { # cell model_hf ds_slug|"" extra_hydra...
  local cell="$1" mp="$2" ds="$3"; shift 3
  local dir_arg=() ; local extra=("$@")
  mkdir -p "$RUNROOT/$cell"
  echo "======== [cell] $cell  model=$mp  ds=${ds:-gsm8k-auto}  extra=${extra[*]:-none} ========"
  # SAVE_FREQ=-1 disables checkpoint writes (smoke needs no weights); we also
  # rm -rf checkpoints/ after every cell as a hard disk-safety backstop — an
  # unattended 26-cell matrix filled a 200G disk with FSDP checkpoints once.
  local env_common=( MODEL_PATH="$mp" EXPERIMENT_NAME="62-rlvr-models-datasets/$cell"
                     TOTAL_TRAINING_STEPS=5 TEST_FREQ=1 VAL_BEFORE_TRAIN=True SAVE_FREQ=-1 )
  if [[ -n "$ds" ]]; then env_common+=( DATA_DIR="$HOME/data/${DSDIR[$ds]}" ); fi
  env "${env_common[@]}" bash "$LAUNCHER" "${extra[@]}" > "$RUNROOT/$cell/setup.log" 2>&1 || true
  judge_cell "$cell"
  rm -rf checkpoints 2>/dev/null || true
  echo "[disk] after $cell: $(df -h / | awk 'NR==2{print $4" free"}')"
}

smoke_cell() { # ds model
  local ds="$1" m="$2" cell="smoke-$1-$2" extra=()
  # R1-Distill emits long <think>; give it room (resp 4096). accel-base hardcodes 1024,
  # so override as a trailing Hydra arg (last-wins).
  [[ "$m" == "r1-distill-qwen-1p5b" ]] && extra=( data.max_response_length=4096 )
  run_cell "$cell" "${MODEL[$m]}" "$ds" "${extra[@]}"
}

already_passed() { # cell -> 0 if a pass:true row exists for it
  grep -F "$1"$'\t' "$STATUS" 2>/dev/null | grep -q '"pass": true'
}

run_all_smoke() { # optional: skip_canary | resume
  local skip="${1:-}"
  for ds in "${DATASETS[@]}"; do
    for m in "${MODELS_ORDER[@]}"; do
      local cell="smoke-$ds-$m"
      [[ "$skip" == "skip_canary" && "$cell" == "$CANARY" ]] && continue
      [[ "$skip" == "resume" ]] && already_passed "$cell" && { echo "[skip] $cell already passed"; continue; }
      smoke_cell "$ds" "$m"
    done
  done
}

run_parity() { # optional: resume
  [[ "${1:-}" == "resume" ]] && already_passed "dense-control-parity" && { echo "[skip] parity already passed"; return 0; }
  # dense control = comm-eff OFF on the gsm8k path (launcher auto-preps gsm8k when DATA_DIR unset)
  run_cell "dense-control-parity" "Qwen/Qwen2.5-1.5B-Instruct" "" \
    actor_rollout_ref.actor.comm_eff.enabled=false
}

case "$MODE" in
  prep)   prep_all ;;
  canary) prep_one math; smoke_cell math qwen25-math-1p5b ;;
  rest)   prep_all; run_all_smoke skip_canary ;;
  parity) run_parity ;;
  resume) prep_all; run_all_smoke resume; run_parity resume ;;   # run only cells not already pass:true
  all)    prep_all; run_all_smoke; run_parity ;;
  *) echo "unknown mode: $MODE"; exit 2 ;;
esac
echo "=== run_rlvr_matrix ($MODE) DONE — status: $STATUS ==="
