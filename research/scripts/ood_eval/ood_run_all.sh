#!/usr/bin/env bash
# In-domain + OOD capability audit as a KL DOSE-RESPONSE matrix.
# Research question: does the two-circuit compression KL drift damage OOD / base
# capability while in-domain val still looks healthy?
#   dose axis:    commeff@50 (~1 nat) -> commeff@100 (~3-4) -> commeff@150 (~5-7)
#   control:      dense@50/100/150 (KL ~0.005, same training recipe)
#   causal probe: tis@150 (same codec, mismatch-corrected, ~half KL)
#   anchor:       base (untrained)
#
# Pipeline: (1) merge each FSDP checkpoint (local, or pulled from R2) to a clean HF
# model, (2) eval each model x 10 benchmarks via ood_eval.sh (val-only), fanning
# benches over a GPU-pair pool, (3) tabulate val-core acc/mean@N with delta columns.
#
# Portability knobs (override via env; defaults match the reference single-box layout):
#   VERL_DIR       verl checkout                (default /workspace/verl)
#   OOD_EVAL_ROOT  eval output root             (default /workspace/runs/ood-eval)
#   CKPT_ROOT      local FSDP checkpoint root   (default $VERL_DIR/checkpoints/quick-test)
#   BASE_MODEL     untrained reference model    (default Qwen/Qwen2.5-Math-1.5B)
#   PAIRS_CSV      GPU-pair pool                (default "0,1|2,3|4,5|6,7")
#   R2_PREFIX      key prefix under the bucket  (default autonomous-harness-rlvr-compression/quick-test)
#   R2_CKPT_BUCKET bucket holding the checkpoints (required for R2 pulls; no default)
# R2 pull (only for checkpoints not present locally) reads credentials from the
# off-repo secrets file ~/.config/verl-research/secrets.env, which must define
# R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, and R2_CKPT_BUCKET (the
# bucket your training checkpoints are synced to). No credential is ever stored in
# this script. If all checkpoints are already merged locally, no R2 access is used.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="${VERL_DIR:-/workspace/verl}"
OOD_EVAL_ROOT="${OOD_EVAL_ROOT:-/workspace/runs/ood-eval}"
CKPT_ROOT="${CKPT_ROOT:-$VERL_DIR/checkpoints/quick-test}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Math-1.5B}"
R2_PREFIX="${R2_PREFIX:-autonomous-harness-rlvr-compression/quick-test}"
cd "$VERL_DIR"
MERGED="$OOD_EVAL_ROOT/merged"
mkdir -p "$MERGED" "$OOD_EVAL_ROOT"

# --- 1. merge <run_name> <step> <tag>: local checkpoint or R2 pull (actor only, no optim) ---
merge() {
  local run="$1" step="$2" tag="$3"
  local ck="$CKPT_ROOT/$run/global_step_$step/actor" out="$MERGED/$tag"
  [[ -f "$out/config.json" ]] && { echo "$out"; return; }
  if [[ ! -f "$ck/fsdp_config.json" ]]; then
    set -a; source ~/.config/verl-research/secrets.env; set +a
    export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    : "${R2_CKPT_BUCKET:?set R2_CKPT_BUCKET (in secrets.env) to the bucket holding your checkpoints}"
    local r2="s3://$R2_CKPT_BUCKET/$R2_PREFIX/$run/checkpoints/global_step_$step/actor/"
    echo "=== R2 pull $run s$step -> $ck ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
    mkdir -p "$ck"
    aws s3 cp "$r2" "$ck/" --recursive --exclude "optim*" --only-show-errors \
      --endpoint-url "$R2_ENDPOINT" >> "$OOD_EVAL_ROOT/merge.log" 2>&1 \
      || { echo "R2 PULL FAILED $run s$step" | tee -a "$OOD_EVAL_ROOT/eval.log"; echo ""; return; }
  fi
  if [[ -f "$ck/fsdp_config.json" ]]; then
    echo "=== merge $run s$step -> $out ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
    OMP_NUM_THREADS=8 python -m verl.model_merger merge --backend fsdp \
      --local_dir "$ck" --target_dir "$out" >> "$OOD_EVAL_ROOT/merge.log" 2>&1
  fi
  echo "$out"
}

# --- 2. eval matrix. greedy mean@1 for large sets, avg@8 for competition sets ---
BENCHES=(
  "gsm8k     1 0   1.0"
  "math500   1 0   1.0"
  "minerva   1 0   1.0"
  "olympiad  1 0   1.0"
  "amc23     8 0.7 0.8"
  "mmlu_stem 1 0   1.0"
  "aime24    8 0.7 0.8"
  "aime25    8 0.7 0.8"
  "aime26    8 0.7 0.8"
  "hmmt25    8 0.7 0.8"
)
IFS='|' read -r -a PAIRS <<< "${PAIRS_CSV:-0,1|2,3|4,5|6,7}"

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

# ---- model roster (tag  run  step), priority-ordered. EXAMPLE from the reference
# study; edit for your own runs. `run` is the WandB/run-dir name, `step` the saved
# global_step; `base` is the untrained reference. ----
ROSTER=(
  "base       -                          -"
  "dense150   quicktest-ood-dense-150    150"
  "commeff150 quicktest-ood-commeff-150  150"
  "tis150     quicktest-math-tis-token-200 150"
  "commeff100 quicktest-ood-commeff-150  100"
  "commeff50  quicktest-ood-commeff-150  50"
  "dense100   quicktest-ood-dense-150    100"
  "dense50    quicktest-ood-dense-150    50"
)

for entry in "${ROSTER[@]}"; do
  read -r tag run step <<<"$entry"
  if [[ "$tag" == "base" ]]; then model="$BASE_MODEL"; else model=$(merge "$run" "$step" "$tag" | tail -1); fi
  echo "=== EVAL MODEL $tag $(date -Iseconds) ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
  run_tag "$tag" "$model"
done

# --- 3. tabulate (edit `tags` to match your roster) ---
OOD_EVAL_ROOT="$OOD_EVAL_ROOT" python3 - <<'PY' | tee "$OOD_EVAL_ROOT/RESULTS.txt"
import os, re
root=os.environ["OOD_EVAL_ROOT"]
benches=["math500","gsm8k","minerva","olympiad","amc23","mmlu_stem","aime24","aime25","aime26","hmmt25"]
tags=["base","dense50","dense100","dense150","commeff50","commeff100","commeff150","tis150"]
tbl={}
for tag in tags:
    for b in benches:
        f=f"{root}/{tag}/{b}/train.log"; acc=None
        if os.path.exists(f):
            m=re.findall(r"acc/mean@[0-9]+['\"]?[: ]+([0-9.]+)", open(f).read())
            if m: acc=float(m[-1])
        tbl[(tag,b)]=acc
hdr=f"{'bench':10s} "+" ".join(f"{t:>10s}" for t in tags)+"  ce150-d150  tis150-ce150"
print(hdr); print("-"*len(hdr))
for b in benches:
    row=[tbl[(t,b)] for t in tags]
    def fmt(v): return f"{v:10.4f}" if v is not None else f"{'.':>10s}"
    d1 = (tbl[("commeff150",b)]-tbl[("dense150",b)]) if (tbl[("commeff150",b)] is not None and tbl[("dense150",b)] is not None) else None
    d2 = (tbl[("tis150",b)]-tbl[("commeff150",b)]) if (tbl[("tis150",b)] is not None and tbl[("commeff150",b)] is not None) else None
    print(f"{b:10s} "+" ".join(fmt(v) for v in row)
          +f"  {('%+.4f'%d1) if d1 is not None else 'n/a':>9s}  {('%+.4f'%d2) if d2 is not None else 'n/a':>11s}")
PY
echo "=== ALL OOD EVAL DONE $(date -Iseconds) ===" | tee -a "$OOD_EVAL_ROOT/eval.log"
touch "$OOD_EVAL_ROOT/OOD_DONE"
