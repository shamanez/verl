#!/usr/bin/env bash
# run_qwen25_math_1p5b_aqsgd_fsdp.sh
#
# The CODEC ablation the COMPASS RLVR section is missing. Table 6 selects the
# boundary codec by comparing an anchor-learned basis (low rank) against a
# keyed random mask (sparsification) and nothing else, so the taxonomy has no
# QUANTIZATION arm. AQ-SGD (Wang et al., NeurIPS 2022) is the named baseline
# from the pipeline-parallel activation-compression literature that fills it.
#
# EVERYTHING IS PINNED TO PUBLISHED PAIR 1. Same model, data, data order,
# recipe and single-GPU shape as the 600-step runs whose numbers are already in
# the paper, so a new arm is a new ROW in the existing table rather than a new
# table. The controls are NOT re-run: research/runs/90-{dense,prf-exactk}-600
# already hold them and they reproduce Table 3 exactly (dense MATH500 0.6700,
# compressed 0.7060, ten-benchmark means 0.314 and 0.322).
#
#   Qwen2.5-Math-1.5B on MATH, 1024 prompt / 2048 response, 128 prompts per
#   step and G=8, mini-batch 128 so there is ONE on-policy optimizer tick per
#   step, AdamW 1e-6, low_var_kl to the reference at 0.001, anchor cadence and
#   delay 20/20 on the full rollout batch, rank-1 RELEX W2 secant at full
#   strength, signed EMA alpha=0.25 beta_anc=0.25, 600 steps, 20 epochs.
#
# ONE GPU PER ARM, which is not a throughput choice. The published Pair 1 rows
# come from a single-GPU run, and at DP>1 the anchor's dense replay gradient is
# averaged across ranks by FSDP's own gradient reduction (the two-circuit write
# up states this outright), so G_anchor is a lower-variance estimate and the
# arm stops being comparable to the rows it is being read against. On a 4x H200
# box run four arms side by side, one per GPU, with GPU=<id>.
#
# ARMS. Arm 1 is the deliverable; the rest are companions that cost no extra
# wall clock on a 4-GPU box and pre-empt the three obvious objections to it.
#
#   aqsgd          PRIMARY. Byte-matched AQ-SGD: bits=2, subset_k=493,
#                  block=32, stochastic rounding, scope=prompt. At H=1536 this
#                  costs 493*2 + 493*16/32 = 1232.5 bits per token per
#                  boundary against PRF exact-k's 77*16 = 1232, so the wire is
#                  matched to 1.0004x. scope=prompt confines the buffer to the
#                  prompt prefix, which is the ONLY part of an RLVR sequence
#                  that recurs across epochs with byte-identical tokens, so
#                  this is AQ-SGD's best case in this regime rather than a
#                  handicapped port.
#
#   srquant        The delta-coding isolator. sr_quant at the IDENTICAL
#                  (bits, subset_k, block_size), so it is byte-identical to
#                  arm 1 on the wire and differs in exactly one thing: whether
#                  a buffered history is differenced against. Any gap between
#                  these two arms is attributable to delta coding and nothing
#                  else. It also closes an open question of its own: this is
#                  #93 cell a3-srq-parity-k493, which ranked STABILITY 1 of 12
#                  (gap slope +0.000101 against the incumbent's +0.000838) but
#                  ran only 120 steps with val off, and COMM_EFF_CONFIG.md
#                  still lists it as "unproven, not beaten".
#
#   aqsgd-all      The literal port. scope=all buffers every position,
#                  including resampled response tokens whose buffered history
#                  belongs to a DIFFERENT token. For roughly independent
#                  activations E||a-m||^2 = ||a||^2 + ||m||^2, so the delta
#                  carries about sqrt(2) times the norm of the value and a
#                  fixed bit budget buys a coarser grid than quantizing the
#                  value would. This arm measures that penalty instead of
#                  asserting it.
#
#   aqsgd-rn       The faithfulness control. Round-to-nearest instead of
#                  stochastic. AQ-SGD's Theorem 3.1 reads "consider an
#                  unbiased quantization function Q(x)" and its worked bound
#                  rounds "stochastically", so sr is the faithful setting and
#                  this arm exists to show the choice was not what decided the
#                  result. Expect it to be worse: #93 killed a 1-bit RN arm at
#                  step 60 with a run-MINIMUM grad_norm 6.9x its stochastic
#                  twin's whole-run maximum.
#
#   aqsgd-payload  PAYLOAD ACCOUNTING, not accuracy. first_visit=dense sends
#                  the first message for each example uncompressed, faithful to
#                  the paper and OFF the byte budget by construction. Short by
#                  design (SMOKE_STEPS): it exists to measure what AQ-SGD
#                  actually costs on the wire in a regime where most tokens are
#                  first visits. Run this one FIRST; it is under an hour.
#
# Usage, inside tmux, from /workspace:
#   ARM=aqsgd-payload GPU=0 bash examples/grpo_trainer/run_qwen25_math_1p5b_aqsgd_fsdp.sh
#   ARM=aqsgd GPU=0 bash examples/grpo_trainer/run_qwen25_math_1p5b_aqsgd_fsdp.sh
#   N_ARMS_ON_BOX=4 ARM=srquant GPU=1 bash examples/grpo_trainer/run_qwen25_math_1p5b_aqsgd_fsdp.sh
set -uo pipefail

# -------- knobs you may edit --------
ARM="${ARM:?set ARM=aqsgd|srquant|aqsgd-all|aqsgd-rn|aqsgd-payload}"
GPU="${GPU:-0}"                          # which physical GPU this arm owns
N_ARMS_ON_BOX="${N_ARMS_ON_BOX:-1}"      # how many arms share this host's RAM
WORK="${WORK:-/workspace}"
TOTAL_STEPS="${TOTAL_STEPS:-600}"
SMOKE_STEPS="${SMOKE_STEPS:-25}"
TEST_FREQ="${TEST_FREQ:-50}"             # val is the only instrument that catches quiet drift
SAVE_FREQ="${SAVE_FREQ:-200}"            # ckpts at 200/400/600, so a late collapse does not cost the arm
GPU_MEM="${GPU_MEM:-0.72}"
# ------------------------------------

# 1. Arm table. Every arm is the published Pair 1 recipe with the codec swapped
#    and nothing else touched.
CODEC="aq_sgd"; AQ_BITS=2; AQ_K=493; AQ_BLOCK=32; AQ_ROUNDING="sr"
AQ_SCOPE="prompt"; AQ_FIRST_VISIT="rescaled"; AQ_MAX_POSITIONS=0
STEPS="$TOTAL_STEPS"; ARM_SAVE_FREQ="$SAVE_FREQ"; ARM_TEST_FREQ="$TEST_FREQ"
case "$ARM" in
  aqsgd)         ;;
  aqsgd-all)     AQ_SCOPE="all" ;;
  aqsgd-rn)      AQ_ROUNDING="rn" ;;
  aqsgd-payload) AQ_FIRST_VISIT="dense"; STEPS="$SMOKE_STEPS"
                 ARM_SAVE_FREQ=-1; ARM_TEST_FREQ=-1 ;;
  srquant)       CODEC="sr_quant" ;;
  *) echo "FATAL: unknown ARM='$ARM' (aqsgd|srquant|aqsgd-all|aqsgd-rn|aqsgd-payload)" >&2; exit 1 ;;
esac

RUN_ID="${RUN_ID:-compass-codec-$ARM-600}"
RUN_DIR="$WORK/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$WORK/verl" 2>/dev/null || { echo "FATAL: expected a verl checkout at $WORK/verl (the shared bring-up prepares it)" >&2; exit 1; }
echo "=== verl HEAD: $(git rev-parse HEAD) ==="

# 2. Money gate: prove THIS checkout carries the codec, before any GPU spend.
#    A stale tree would otherwise run the default PRF codec under an arm name
#    that claims otherwise, which is the failure mode that is hardest to catch
#    afterwards because the run looks healthy.
python3 - <<'PY' || { echo "FATAL: aq_sgd codec absent from this checkout" >&2; exit 1; }
from verl.workers.config.comm_eff import COMPRESSION_TYPES
from verl.workers.comm_eff.activation_aqsgd import ActivationAQSGD, aqsgd_example_ids  # noqa: F401
from verl.workers.comm_eff.state import CommEffState
assert "aq_sgd" in COMPRESSION_TYPES, COMPRESSION_TYPES
assert hasattr(CommEffState, "per_token_codec"), "engine cannot route the codec"
codec = ActivationAQSGD(bits=2, block_size=32, subset_k=493)
codec._record_bits(1536)
assert abs(codec.logical_pp_bits_aq_sgd - 1232.5) < 1e-6, codec.logical_pp_bits_aq_sgd
print("OK: aq_sgd present, wire ledger 1232.5 bits/token/boundary at H=1536")
PY

# 3. Data. Shared across arms on one box; prepared once by whoever gets there first.
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] \
  || { echo "FATAL: prepared MATH parquet required in $DATA_DIR (run research/scripts/prepare_rlvr_math.py --dataset math)" >&2; exit 1; }

# 4. Buffer sizing, MEASURED rather than assumed. AQ-SGD's store has to span
#    the reuse distance to score a hit, and in RLVR that distance is one full
#    epoch of prompts. The need is therefore
#      n_examples * mean_prompt_tokens * n_boundaries * H * 2 bytes
#    which nothing in the config subtree knows, so it is computed here from the
#    real dataset and the real tokenizer. Under-capacity is not a correctness
#    problem (the LRU simply lowers the hit rate, which is logged), but a
#    silently tiny buffer would make the arm meaningless, so this both sizes
#    the cap and reports the coverage it buys. The cap itself is a fixed
#    AQ_CAPACITY_GB (default 16), NOT a fraction of free RAM, so the fanout's
#    host-memory gate has a number it can rely on.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}"
AQ_CAPACITY_GB="${AQ_CAPACITY_GB:-16}"          # DETERMINISTIC, so a fanout RAM gate can trust it
AQ_CAPACITY=$((AQ_CAPACITY_GB * 1073741824))
if [[ "$CODEC" == "aq_sgd" ]]; then
  python3 - <<PY || { echo "FATAL: AQ-SGD buffer coverage check failed" >&2; exit 1; }
import os, sys
import pandas as pd
from transformers import AutoTokenizer

H, BOUNDARIES, ITEMSIZE = 1536, 7, 2
scope, cap = "${AQ_SCOPE}", ${AQ_CAPACITY}

df = pd.read_parquet(os.path.join("${DATA_DIR}", "train.parquet"))
tok = AutoTokenizer.from_pretrained("${MODEL_PATH}", trust_remote_code=True)
sample = df["prompt"].head(512)
lens = [len(tok.apply_chat_template(p, tokenize=True, add_generation_prompt=True))
        if isinstance(p, (list, tuple)) else len(tok(str(p)).input_ids)
        for p in sample]
mean_prompt, n = sum(lens) / len(lens), len(df)

per_position = n * BOUNDARIES * H * ITEMSIZE
need = int(per_position * (mean_prompt + 2048 if scope == "all" else mean_prompt))
gib = lambda b: b / 1024**3
print(f"# {n} examples, mean prompt {mean_prompt:.1f} tok, "
      f"{gib(per_position):.3f} GiB per buffered position", file=sys.stderr)
print(f"# scope={scope}: one epoch of reuse distance needs {gib(need):.1f} GiB, "
      f"cap is {gib(cap):.1f} GiB -> {min(100.0, 100*cap/need):.0f}% coverage", file=sys.stderr)
if cap < need:
    print(f"# NOTE: the cap is below the reuse distance, so the LRU will evict within an "
          f"epoch and aq_sgd/hit_rate will report the consequence. This is a MEASUREMENT, "
          f"not a fault: the storage requirement is part of the result.", file=sys.stderr)
PY
fi
echo "=== aq_sgd buffer cap for arm $ARM: ${AQ_CAPACITY_GB} GiB (deterministic) ==="

# 5. Patched launcher copy: the Pair 1 scalars, exactly as run_prf_exactk_600.sh
#    sets them, with the same fail-loud checks in case the base shape drifts.
BASE="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
PATCHED="examples/grpo_trainer/run_aqsgd_${ARM}.gen.sh"
sed -e 's/^export MAX_RESPONSE_LENGTH=3072$/export MAX_RESPONSE_LENGTH=2048/' \
    -e 's/^export TRAIN_BATCH_SIZE=512$/export TRAIN_BATCH_SIZE=128/' \
    -e 's/^export PPO_MINI_BATCH_SIZE=256$/export PPO_MINI_BATCH_SIZE=128/' \
    "$BASE" > "$PATCHED"
chmod +x "$PATCHED"
grep -q '^export TRAIN_BATCH_SIZE=128$'     "$PATCHED" || { echo "FATAL: batch patch missed"      >&2; exit 1; }
grep -q '^export PPO_MINI_BATCH_SIZE=128$'  "$PATCHED" || { echo "FATAL: mini-batch patch missed" >&2; exit 1; }
grep -q '^export MAX_RESPONSE_LENGTH=2048$' "$PATCHED" || { echo "FATAL: response patch missed"   >&2; exit 1; }

# 6. The codec. Everything below the codec block is the published recipe.
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE="$CODEC"
export COMM_EFF_MASK_ENABLED=false          # no PRF mask under either quantizing codec
export COMM_EFF_MASK_RECOMPUTE=true         # REQUIRED by aq_sgd; the buffer is read every eligible pass
export COMM_EFF_MASK_REFERENCE=true
export COMM_EFF_MASK_PP_SIZE=8              # 7 compressed boundaries over 28 decoder layers
export COMM_EFF_MASK_SEED=0
export COMM_EFF_ANCHOR_OWNS_Q=false         # neither codec carries a basis Q
export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP=false
if [[ "$CODEC" == "aq_sgd" ]]; then
  export COMM_EFF_AQ_SGD_BITS="$AQ_BITS"
  export COMM_EFF_AQ_SGD_SUBSET_K="$AQ_K"
  export COMM_EFF_AQ_SGD_BLOCK_SIZE="$AQ_BLOCK"
  export COMM_EFF_AQ_SGD_ROUNDING="$AQ_ROUNDING"
  export COMM_EFF_AQ_SGD_SCOPE="$AQ_SCOPE"
  export COMM_EFF_AQ_SGD_FIRST_VISIT="$AQ_FIRST_VISIT"
  export COMM_EFF_AQ_SGD_CAPACITY_BYTES="$AQ_CAPACITY"
  export COMM_EFF_AQ_SGD_BUFFER_DEVICE=cpu
  export COMM_EFF_AQ_SGD_MAX_POSITIONS="$AQ_MAX_POSITIONS"
else
  export COMM_EFF_QUANT_BITS="$AQ_BITS"
  export COMM_EFF_QUANT_SUBSET_K="$AQ_K"
  export COMM_EFF_QUANT_BLOCK_SIZE="$AQ_BLOCK"
  export COMM_EFF_QUANT_ROUNDING="$AQ_ROUNDING"
fi

# 7. Run controls, matched to the published arms.
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM"
export TOTAL_TRAINING_STEPS="$STEPS"
export TOTAL_EPOCHS=20                      # >=11 needed for 600 steps to be the stop at batch 128
export TEST_FREQ="$ARM_TEST_FREQ"
export SAVE_FREQ="$ARM_SAVE_FREQ"
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME="$RUN_ID"
export PROJECT_NAME="compass-codec-ablation"
export WANDB_RUN_GROUP="compass-codec-ablation"
export LOG="$RUN_DIR/train.log"
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE="${WANDB_MODE:-offline}"

cat <<EOF
=== launching $RUN_ID on GPU $GPU ===
  codec          $CODEC  bits=$AQ_BITS subset_k=$AQ_K block=$AQ_BLOCK rounding=$AQ_ROUNDING
  aq_sgd         scope=$AQ_SCOPE first_visit=$AQ_FIRST_VISIT capacity=$((AQ_CAPACITY/1073741824))GiB max_positions=$AQ_MAX_POSITIONS
  wire           493*2 + 493*16/32 = 1232.5 bits/token/boundary  (PRF exact-k: 1232)
  shape          128 prompts x G=8, mini 128 (one on-policy tick/step), 1024/2048
  horizon        $STEPS steps, val every $ARM_TEST_FREQ, ckpt every $ARM_SAVE_FREQ
  controls       NOT re-run; read against research/runs/90-{dense,prf-exactk}-600
EOF
exec bash "$PATCHED"
