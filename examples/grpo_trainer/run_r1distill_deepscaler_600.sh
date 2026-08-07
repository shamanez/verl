#!/usr/bin/env bash
# run_r1distill_deepscaler_600.sh
#
# ONE-COMMAND bring-up and launch of ONE arm of the 600-step communication-
# efficient GRPO run on DeepSeek-R1-Distill-Qwen-1.5B / DeepScaleR at 4096 total
# context. Two arms share this file and differ in exactly one variable:
#
#   ARM=prf     PRF exact-k, p=0.95, constant rescale  (the method)
#   ARM=dense   COMM_EFF_ENABLED=false                 (the control)
#
#   ARM=dense bash examples/grpo_trainer/run_r1distill_deepscaler_600.sh
#   ARM=prf   bash examples/grpo_trainer/run_r1distill_deepscaler_600.sh
#
# Use run_99_both_arms.sh to drive both in one sitting on a 4-GPU box.
#
# Nothing is authored on the box: this script obtains the checkout, installs it,
# prepares the data, gates the science, gates the hardware, patches the base
# launcher, and execs it. Every knob is env-overridable and trailing arguments
# are forwarded verbatim to Hydra (this launcher's overrides come first, so a
# caller's duplicate key wins -- Hydra applies CLI overrides in order).
#
# SURFACE, and the deltas from the run-90 reference (90-prf-exactk-600):
#   model        Qwen2.5-Math-1.5B -> DeepSeek-R1-Distill-Qwen-1.5B
#   data         MATH              -> DeepScaleR (qingy2024/DeepScaleR-40k)
#   context      1024/2048 (3072)  -> 1024/3072 (4096, the project protocol)
#   chat prompt  RELEX ChatML      -> the model's OWN template (see gate E)
#   steps        600               -> 600           UNCHANGED
#   batch/mini   128/128           -> 128/128       UNCHANGED
#   rollout n    8                 -> 8             UNCHANGED
#   codec        prf_mask p=0.95 exact-k constant   UNCHANGED
#   anchor       20/20 rollout_batch rank1_relex W2 UNCHANGED
#   val/save     150/100           -> 100/100
#
# The codec's wire budget is IDENTICAL to run 90 and to run 96's 1.5B reference:
# R1-Distill-Qwen-1.5B has hidden 1536, so exact-k keeps exactly 77 of 1536
# coordinates per token per boundary (1232 bits/token/boundary), and its 28
# decoder layers over 8 pipeline stages cut at [3, 7, 11, 15, 18, 21, 24]. Both
# facts are asserted by the money gate below before a single GPU is touched.
#
# KNOWN RISK, stated once and measured every step. R1-Distill is a long-CoT
# model: its own template opens the assistant turn inside <think>, and issue #63
# ran it at 16384 response tokens. At a 3072-token response cap a completion that
# never closes </think> emits no \boxed{} and scores 0, so part of the reward
# signal is "ran out of tokens" rather than "got it wrong". That is the
# truncation-feedback mechanism that sparked the run-96 collapse. It is not a
# reason not to run -- it is the reason to WATCH response_length/clip_ratio,
# which verl already logs every step. The dense arm prices it first.
#
# Run inside tmux. The engine redirects training to $LOG, which is the heartbeat
# log the harness registers as run.json's remote_log.
set -uo pipefail

# ---------------------------- knobs ----------------------------------------
ARM="${ARM:-prf}"                      # prf | dense
RUN_ID="${RUN_ID:-99-r1distill-deepscaler-600}"
BRANCH="${BRANCH:-exp/99-r1distill-deepscaler-prf-exactk-600}"
REPO="${REPO:-https://github.com/shamanez/verl.git}"
WORK="${WORK:-/workspace}"

# Money gate. There are TWO independent `${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}`
# defaults on the path to Hydra (run_qwen25_math_1p5b_rank1_relex_fsdp.sh and
# vast_comm_eff_engine_grpo.sh), so an unset MODEL_PATH does not fail -- it
# silently trains Qwen2.5-Math-1.5B for six hundred steps. Set it explicitly.
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1536}"     # R1-Distill-Qwen-1.5B
NUM_LAYERS="${NUM_LAYERS:-28}"

# Context and batch. MAX_PROMPT_LENGTH/MAX_RESPONSE_LENGTH already hold the
# values this run wants in the base launcher, so they are ASSERTED, not patched;
# the two batch scalars are bare `export NAME=value` lines that no env var can
# override, so they have to be patched into a generated copy. See section 6.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Data. The canonical prep call, with its canonical defaults, so the parquet is
# reproducible from the command line alone.
DATA_DIR="${DATA_DIR:-$HOME/data/deepscaler}"
TRAIN_CAP="${TRAIN_CAP:-20000}"
VAL_SIZE="${VAL_SIZE:-500}"
DATA_SEED="${DATA_SEED:-42}"

# Box gates. Defaults describe ONE arm on a 4-GPU H200 box.
EXPECT_GPUS="${EXPECT_GPUS:-4}"
MIN_RAM_GIB="${MIN_RAM_GIB:-200}"      # anchor keeps a CPU clone + CPU snapshots + CPU M
MIN_DISK_GIB="${MIN_DISK_GIB:-150}"    # 1.5B checkpoints stage locally before the R2 mirror
SKIP_BOX_GATES="${SKIP_BOX_GATES:-0}"  # 1 downgrades the RAM and disk gates to loud warnings
# ---------------------------------------------------------------------------

case "$ARM" in
  prf|dense) ;;
  *) echo "FATAL: ARM='$ARM' must be 'prf' or 'dense'" >&2; exit 1 ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-$RUN_ID-$ARM}"
RUN_DIR="$WORK/runs/$RUN_ID/$ARM"
mkdir -p "$RUN_DIR" || { echo "FATAL: cannot create $RUN_DIR" >&2; exit 1; }
cd "$WORK" || { echo "FATAL: cannot cd $WORK" >&2; exit 1; }

echo "=== $EXPERIMENT_NAME: R1-Distill-Qwen-1.5B / DeepScaleR / 4096 ctx / 600 steps / ARM=$ARM ==="

# ---------------------------------------------------------------------------
# 1. Box preflight. Everything that makes this run impossible, checked before
#    the checkout so a bad box fails in seconds instead of after a model pull.
# ---------------------------------------------------------------------------
box_gate_fail() {
  if [[ "$SKIP_BOX_GATES" == "1" && "${2:-hard}" == "soft" ]]; then
    echo "WARN: $1" >&2
    echo "WARN: continuing anyway because SKIP_BOX_GATES=1." >&2
    return 0
  fi
  echo "FATAL: $1" >&2
  [[ "${2:-hard}" == "soft" ]] && echo "       (set SKIP_BOX_GATES=1 to downgrade this to a warning)" >&2
  exit 1
}

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
echo "--- preflight GPUs:  detected=${GPU_COUNT:-0} required=$EXPECT_GPUS"
[[ "$GPU_COUNT" == "$EXPECT_GPUS" ]] \
  || box_gate_fail "expected $EXPECT_GPUS GPUs, detected ${GPU_COUNT:-0} (set EXPECT_GPUS to override)"

RAM_GIB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
echo "--- preflight RAM:   ${RAM_GIB:-unknown} GiB required=$MIN_RAM_GIB GiB"
if [[ "$RAM_GIB" =~ ^[0-9]+$ ]]; then
  (( RAM_GIB >= MIN_RAM_GIB )) \
    || box_gate_fail "host RAM ${RAM_GIB} GiB is below the ${MIN_RAM_GIB} GiB this run needs" soft
else
  box_gate_fail "could not read host RAM from 'free -g'" soft
fi

DISK_GIB="$(df -Pk "$WORK" 2>/dev/null | awk 'NR==2{print int($4/1048576)}')"
echo "--- preflight disk:  ${DISK_GIB:-unknown} GiB free on $WORK required=$MIN_DISK_GIB GiB"
if [[ "$DISK_GIB" =~ ^[0-9]+$ ]]; then
  (( DISK_GIB >= MIN_DISK_GIB )) \
    || box_gate_fail "only ${DISK_GIB} GiB free on $WORK, need ${MIN_DISK_GIB} GiB for checkpoint staging" soft
else
  box_gate_fail "could not read free disk for $WORK" soft
fi

CKPT_R2_ENABLED="${CKPT_R2_ENABLED:-true}"
AWS_BIN="$(command -v aws 2>/dev/null)"
echo "--- preflight aws:   ${AWS_BIN:-MISSING}"
if [[ "$CKPT_R2_ENABLED" == "true" ]]; then
  [[ -n "$AWS_BIN" ]] \
    || box_gate_fail "'aws' is not on PATH but CKPT_R2_ENABLED is on -- the R2 checkpoint mirror cannot run. The step-600 capability eval reads its checkpoints from R2, so this is fatal, not cosmetic."
fi

# HF token, for the tokenizer the chat-template gate loads in section 5. The
# engine re-sources and re-validates this file itself. We only need HF_TOKEN
# early, and nothing secret is printed.
SECRETS_FILE="${SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
if [[ -r "$SECRETS_FILE" ]]; then
  # shellcheck disable=SC1090
  { set -a; source "$SECRETS_FILE"; set +a; } || true
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
fi

# ---------------------------------------------------------------------------
# 2. Obtain the verl checkout (fast path reuse, else shallow clone).
#    ORDERING IS LOAD-BEARING: every reset/checkout happens HERE, and the money
#    gate in section 5 runs on the FINAL tree. run_layer_rotation_300.sh gated
#    first and then hard-reset to a different branch, and burned 16 GPU-hours
#    running code the gate never saw. Nothing below this section touches git.
# ---------------------------------------------------------------------------
if [[ -d verl/.git ]] \
   && (cd verl && git remote set-url origin "$REPO" \
       && git fetch --depth 1 origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD \
       && git reset --hard FETCH_HEAD); then
  echo "=== reused checkout, reset to origin/$BRANCH ==="
else
  { [[ -e verl ]] && mv verl "verl.stale.$(date +%s)"; true; }
  git clone --depth 1 --single-branch -b "$BRANCH" "$REPO" verl \
    || { echo "FATAL: could not clone $REPO @ $BRANCH" >&2; exit 1; }
fi
cd "$WORK/verl" || { echo "FATAL: cannot cd $WORK/verl" >&2; exit 1; }
echo "=== verl HEAD: $(git rev-parse HEAD) ($BRANCH) ==="

# ---------------------------------------------------------------------------
# 3. Editable install (no deps) if verl is not importable yet.
# ---------------------------------------------------------------------------
python3 -c "import verl" 2>/dev/null || {
  if command -v uv >/dev/null 2>&1; then uv pip install --no-deps -e . ; else python3 -m pip install --no-deps -e . ; fi
}
python3 -c "import verl" || { echo "FATAL: verl import failed after install" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 4. NO SELECTIVE COMPRESSION. This study is the uniform 95 percent budget on
#    every boundary. p_by_boundary is the run-97 dense-middle lever and setting
#    it here would silently change the science while every other gate passed.
# ---------------------------------------------------------------------------
if [[ -n "${COMM_EFF_MASK_P_BY_BOUNDARY:-}" ]]; then
  echo "FATAL: COMM_EFF_MASK_P_BY_BOUNDARY='${COMM_EFF_MASK_P_BY_BOUNDARY}' is set." >&2
  echo "       This run is UNIFORM 95 percent compression on all boundaries." >&2
  echo "       Selective / dense-middle compression belongs to run 97, not here." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 5. Money gate, on the FINAL checkout. All CPU, all cheap. Five claims:
#      A) the exact-k lever exists at all;
#      B) at hidden 1536 exact-k keeps exactly 77 coordinates per token, so the
#         wire budget is 1232 bits/token/boundary -- the SAME budget as runs 90
#         and 96, which is what makes this a like-for-like replication;
#      C) 28 layers over 8 stages cut at [3, 7, 11, 15, 18, 21, 24];
#      D) the anchor clone inherits gradient checkpointing (without that fix the
#         anchor's dense replay stores every activation: measured +62.9 GiB at
#         the first fire on a 1.5B model in run 90 at step 20);
#      E) THE CHAT TEMPLATE. This is the new one and it is the whole reason this
#         is not just run 90 with a different --model flag. The base launcher
#         pins RELEX's Qwen ChatML template (<|im_start|>/<|im_end|>), which is
#         the correct prompt for Qwen2.5-Math-1.5B and the WRONG prompt for
#         R1-Distill, whose own template is <|begin_of_sentence|> + <|User|> +
#         <|Assistant|><think>. Feeding ChatML to R1-Distill would tokenize the
#         markers as ordinary text, drop the <think> opener the model was
#         distilled to continue from, and quietly evaluate a differently-
#         prompted model -- with no error anywhere. Section 6 patches the
#         override out of a generated copy, and this gate proves the model's own
#         template is the one that will be used, and that it is not ChatML.
# ---------------------------------------------------------------------------
MODEL_PATH="$MODEL_PATH" HIDDEN_SIZE="$HIDDEN_SIZE" NUM_LAYERS="$NUM_LAYERS" \
COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}" \
python3 - <<'PY' || { echo "FATAL: money gate FAILED -- this checkout must not be launched" >&2; exit 1; }
import inspect
import os

import torch

from verl.workers.comm_eff.activation_mask import prf_token_mask
from verl.workers.comm_eff.anchor import build_anchor_module
from verl.workers.comm_eff.boundary import decoder_boundary_indices

H = int(os.environ["HIDDEN_SIZE"])
L = int(os.environ["NUM_LAYERS"])
PP = int(os.environ["COMM_EFF_MASK_PP_SIZE"])
MODEL = os.environ["MODEL_PATH"]

# A) the lever exists.
assert "exact_k" in inspect.signature(prf_token_mask).parameters, "prf_token_mask is missing exact_k"

# B) the wire budget is the one this study claims.
mask = prf_token_mask(
    sample_ids=torch.zeros(1, dtype=torch.int64),
    position_ids=torch.zeros(1, dtype=torch.int64),
    layer_idx=0,
    global_step=1,
    base_seed=0,
    hidden_size=H,
    p=0.95,
    device=torch.device("cpu"),
    dtype=torch.float32,
    exact_k=True,
)
kept = int((mask != 0).sum().item())
assert kept == 77, f"exact-k at hidden {H} kept {kept} coordinates, expected 77"
print(f"OK: exact-k keeps {kept}/{H} coords per token per boundary ({kept * 16} bits/token/boundary)")
print("    identical to the run-90 and run-96 1.5B wire budget")

# C) the boundary set.
boundaries = decoder_boundary_indices(L, PP)
assert boundaries == [3, 7, 11, 15, 18, 21, 24], f"unexpected boundary set {boundaries}"
print(f"OK: {L} layers over {PP} stages -> boundaries {boundaries}")

# D) the anchor clone is checkpointed.
anchor_src = inspect.getsource(build_anchor_module)
assert "gradient_checkpointing_enable" in anchor_src, (
    "build_anchor_module does NOT enable gradient checkpointing on the anchor clone. "
    "This checkout predates that fix; the first anchor fire will balloon host and device "
    "memory. Refusing to launch."
)
print("OK: anchor clone inherits gradient checkpointing")

# E) the chat template. Render the ACTUAL prompt this run will send.
from transformers import AutoConfig, AutoTokenizer

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
assert cfg.hidden_size == H, f"{MODEL} has hidden_size {cfg.hidden_size}, gate assumed {H}"
assert cfg.num_hidden_layers == L, f"{MODEL} has {cfg.num_hidden_layers} layers, gate assumed {L}"
print(f"OK: {MODEL} config matches the gate (hidden {H}, {L} layers)")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
assert tok.chat_template, f"{MODEL} ships no chat_template; the run has no defined prompt"
msgs = [{"role": "user", "content": "What is 2+2? Let's think step by step and output the final answer within \\boxed{}."}]
rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

assert "<|im_start|>" not in rendered and "<|im_end|>" not in rendered, (
    "The rendered prompt contains ChatML markers. The RELEX Qwen template leaked into an "
    "R1-Distill run: <|im_start|>/<|im_end|> are not this model's turn markers, so they would "
    "be tokenized as ordinary text and the model would be prompted in a format it was never "
    "distilled on. Refusing to launch.\n--- rendered ---\n" + rendered
)
for marker in ("<｜User｜>", "<｜Assistant｜>"):
    assert marker in rendered, (
        f"expected {marker!r} in the rendered prompt; got:\n{rendered}"
    )
assert rendered.rstrip().endswith("<think>") or "<think>" in rendered.split("<｜Assistant｜>")[-1], (
    "the generation prompt does not open the assistant turn inside <think>; R1-Distill is "
    "distilled to continue from that opener:\n" + rendered
)
print("OK: prompt uses the model's OWN template, not RELEX ChatML")
print("    rendered generation prompt:")
for line in rendered.splitlines() or [rendered]:
    print(f"      | {line}")
PY

# ---------------------------------------------------------------------------
# 6. DeepScaleR parquet. qingy2024/DeepScaleR-40k is train-only, so the prep
#    carves a held-out test split from the front of a seeded shuffle. That split
#    is BOTH the in-training validation set and the in-domain benchmark of the
#    step-600 capability audit, which is what makes the two numbers comparable.
# ---------------------------------------------------------------------------
export DATA_DIR
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preparing DeepScaleR parquet in $DATA_DIR (cap $TRAIN_CAP, val $VAL_SIZE, seed $DATA_SEED) ==="
  python3 research/scripts/prepare_rlvr_math.py --dataset deepscaler \
    --local_save_dir "$DATA_DIR" --train-cap "$TRAIN_CAP" --val-size "$VAL_SIZE" --seed "$DATA_SEED" 2>&1 | tail -8
fi
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] \
  || { echo "FATAL: DeepScaleR parquet unavailable in $DATA_DIR" >&2; exit 1; }
# A QUOTED heredoc with values passed through the environment, not an
# interpolated one: an interpolated heredoc expands every $ in the Python body,
# so a path with a space or a quote in it, or an innocent f-string, becomes a
# syntax error at the worst possible moment.
DATA_DIR="$DATA_DIR" STEPS="${TOTAL_TRAINING_STEPS:-600}" BATCH="$TRAIN_BATCH_SIZE" \
python3 - <<'PY' || { echo "FATAL: prepared DeepScaleR parquet failed its shape check" >&2; exit 1; }
import os

import pyarrow.parquet as pq

d = os.environ["DATA_DIR"]
steps = int(os.environ["STEPS"])
batch = int(os.environ["BATCH"])

tr = pq.read_table(os.path.join(d, "train.parquet"))
te = pq.read_table(os.path.join(d, "test.parquet"))
print(f"OK: DeepScaleR train={tr.num_rows} rows, test={te.num_rows} rows")
assert tr.num_rows >= 1000, f"train split has only {tr.num_rows} rows"
assert te.num_rows >= 100, f"test split has only {te.num_rows} rows"

# The step cap has to be the stopping condition, so TOTAL_EPOCHS must exceed the
# passes this many draws implies. Printed rather than asserted: the engine's own
# epoch count is the thing that would end the run early, and it is set below.
draws = steps * batch
print(f"    {steps} steps at batch {batch} = {draws} draws = {draws / tr.num_rows:.1f} epochs over train")

# The reward verifier extracts the last \boxed{} span, and the prep appends the
# boxed instruction to every prompt. If that ever stops being true the run would
# train against an all-zero reward and look merely bad rather than broken.
row = tr.slice(0, 1).to_pylist()[0]
content = row["prompt"][0]["content"]
assert "\\boxed{}" in content, f"prompt does not ask for a boxed answer:\n{content[:300]}"
assert str(row["reward_model"]["ground_truth"]).strip(), "first train row has an empty ground truth"
print(f"    reward routing: data_source={row['data_source']}, boxed instruction present")
PY

# ---------------------------------------------------------------------------
# 7. Patched launcher copy. Two batch scalars are rewritten, two context scalars
#    are only ASSERTED (the base already holds the values this run wants), and
#    the RELEX ChatML override is replaced with an explicit null so the model's
#    own template applies. Each rewrite is checked against its EXPECTED CURRENT
#    value in the base first (a drifted base fails loud instead of being
#    silently un-patched) and asserted again in the generated copy afterwards.
# ---------------------------------------------------------------------------
BASE="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
PATCHED="examples/grpo_trainer/run_r1distill_deepscaler_600.gen.sh"
[[ -f "$BASE" ]] || { echo "FATAL: base launcher $BASE not found" >&2; exit 1; }
cp "$BASE" "$PATCHED" || { echo "FATAL: could not copy $BASE" >&2; exit 1; }

# The two context scalars this run INHERITS. Asserted, never patched: if the
# base ever moves off the 4096-token protocol we want a loud stop, not a run at
# a context nobody chose.
for spec in "MAX_PROMPT_LENGTH|$MAX_PROMPT_LENGTH" "MAX_RESPONSE_LENGTH|$MAX_RESPONSE_LENGTH"; do
  IFS='|' read -r pname pval <<< "$spec"
  grep -q "^export ${pname}=${pval}\$" "$BASE" || {
    echo "FATAL: base launcher no longer holds 'export ${pname}=${pval}'." >&2
    echo "       This run inherits the 4096-token protocol from the base rather than patching it." >&2
    exit 1
  }
  echo "--- inherited ${pname}=${pval}"
done

# name | value the base is expected to hold today | value this run needs
PATCH_SPEC=(
  "TRAIN_BATCH_SIZE|512|$TRAIN_BATCH_SIZE"
  "PPO_MINI_BATCH_SIZE|256|$PPO_MINI_BATCH_SIZE"
)
for spec in "${PATCH_SPEC[@]}"; do
  IFS='|' read -r pname pfrom pto <<< "$spec"
  grep -q "^export ${pname}=${pfrom}\$" "$BASE" || {
    echo "FATAL: base launcher shape drifted -- expected 'export ${pname}=${pfrom}' in $BASE" >&2
    echo "       Re-derive this launcher's patch table before running." >&2
    exit 1
  }
  sed -i.bak -e "s/^export ${pname}=${pfrom}\$/export ${pname}=${pto}/" "$PATCHED" \
    || { echo "FATAL: sed failed for ${pname}" >&2; exit 1; }
  rm -f "$PATCHED.bak"
  grep -q "^export ${pname}=${pto}\$" "$PATCHED" \
    || { echo "FATAL: ${pname} patch missed (wanted ${pto})" >&2; exit 1; }
  echo "--- patched ${pname}: ${pfrom} -> ${pto}"
done

# The chat template. Done as a line REWRITE in the generated copy rather than as
# a trailing Hydra override, because a trailing override would make
# actor_rollout_ref.model.custom_chat_template appear TWICE on the same command
# line, and this repo does not rely on Hydra's duplicate-key behaviour for a key
# whose wrong value is silent. Python, not sed: the line carries ${oc.env:...},
# which is a minefield of shell and regex metacharacters.
BASE="$BASE" PATCHED="$PATCHED" python3 - <<'PY' || { echo "FATAL: chat-template patch failed" >&2; exit 1; }
import os
import sys

base_p, patched_p = os.environ["BASE"], os.environ["PATCHED"]
OLD = "'actor_rollout_ref.model.custom_chat_template=${oc.env:RELEX_QWEN_CHAT_TEMPLATE}'"
NEW = "'actor_rollout_ref.model.custom_chat_template=null'"

src = open(base_p, encoding="utf-8").read()
if src.count(OLD) != 1:
    sys.exit(
        f"base launcher shape drifted: expected exactly one occurrence of\n  {OLD}\n"
        f"in {base_p}, found {src.count(OLD)}. Re-derive this launcher's template patch."
    )

out = open(patched_p, encoding="utf-8").read().replace(OLD, NEW)
if OLD in out or NEW not in out:
    sys.exit("template rewrite did not take")
if "oc.env:RELEX_QWEN_CHAT_TEMPLATE" in out:
    sys.exit("RELEX ChatML template still referenced in the generated launcher")
open(patched_p, "w", encoding="utf-8").write(out)
print("--- patched custom_chat_template: RELEX ChatML -> null (the model's own template)")
PY
chmod +x "$PATCHED"

TOTAL_CTX=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
(( TOTAL_CTX == MAX_MODEL_LEN )) \
  || { echo "FATAL: prompt+response ($TOTAL_CTX) != MAX_MODEL_LEN ($MAX_MODEL_LEN)" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 8. Model + codec + run controls.
#
#    ARM=prf   the project default codec, byte-identical to run 90 and run 96:
#              prf_mask, p=0.95, exact-k, constant rescale, masking the train
#              forward + the old-logprob recompute + the reference forward, 8
#              pipeline stages, no anchor-owned Q.
#    ARM=dense the SAME surface with the master switch off. That single variable
#              is the only science delta between the two arms.
# ---------------------------------------------------------------------------
export MODEL_PATH
if [[ "$ARM" == "prf" ]]; then
  export COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"
  export COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-prf_mask}"
  export COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-true}"
  export COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.95}"
  export COMM_EFF_MASK_RESCALE_MODE="${COMM_EFF_MASK_RESCALE_MODE:-constant}"
  export COMM_EFF_MASK_EXACT_K="${COMM_EFF_MASK_EXACT_K:-true}"
  export COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-true}"
  export COMM_EFF_MASK_REFERENCE="${COMM_EFF_MASK_REFERENCE:-true}"
  export COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"
  export COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-false}"   # prf_mask has no basis Q to own
  export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP="${COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP:-false}"
else
  export COMM_EFF_ENABLED=false                                      # the ONLY science delta
fi

# Run schedule. 600 steps at batch 128; the step cap is the stopping condition
# and TOTAL_EPOCHS only has to be large enough not to end the run first.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-600}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export TEST_FREQ="${TEST_FREQ:-100}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export SAVE_FREQ="${SAVE_FREQ:-100}"

# Checkpoints -> R2, deleted locally after a verified upload. The step-600
# capability audit reads them from there, on whatever box it runs on.
export CKPT_R2_ENABLED
export CKPT_R2_MAX_STAGED_GB="${CKPT_R2_MAX_STAGED_GB:-60}"
export CKPT_R2_WORKERS="${CKPT_R2_WORKERS:-8}"
# Key layout: autonomous-harness-rlvr-compression/<experiment>/<regime>/checkpoints/
# One prefix per run, one regime per arm, so the eval driver discovers both arms
# from a single listing.
export R2_EXPERIMENT="${R2_EXPERIMENT:-$RUN_ID}"
export R2_REGIME="${R2_REGIME:-$ARM}"

export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.72}"
export ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"

export EXPERIMENT_NAME
export PROJECT_NAME="${PROJECT_NAME:-$RUN_ID}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RUN_ID}"
export LOG="${LOG:-$RUN_DIR/train.log}"
# WandB: use it if a key is present, else fall back to offline (no crash).
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE="${WANDB_MODE:-offline}"

# ---------------------------------------------------------------------------
# 9. Hydra overrides this launcher owns.
#
#    max_model_len / max_num_batched_tokens are REQUIRED, not cosmetic. Neither
#    key is emitted by vast_comm_eff_engine_grpo.sh, so both fall back to their
#    rollout.yaml defaults: max_model_len=null (vLLM then uses the checkpoint's
#    max_position_embeddings, which is 131072 for R1-Distill-Qwen-1.5B and would
#    size the KV cache for a context 32x larger than this run uses) and
#    max_num_batched_tokens=8192. Both keys already exist in the schema, so they
#    take no `+` prefix, and passing them once here creates no duplicate.
#
#    "$@" goes LAST so a caller's override of the same key wins.
# ---------------------------------------------------------------------------
HYDRA_OVERRIDES=(
  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE"
)

cat <<EOF
=== launching $EXPERIMENT_NAME ===
    arm              $ARM
    model            $MODEL_PATH
    data             $DATA_DIR (DeepScaleR, held-out test = in-domain val)
    prompt template  the model's own (RELEX ChatML patched out)
    context          $MAX_PROMPT_LENGTH prompt + $MAX_RESPONSE_LENGTH response = $TOTAL_CTX
    batch            train $TRAIN_BATCH_SIZE / mini $PPO_MINI_BATCH_SIZE (one on-policy tick per step)
    codec            ${COMM_EFF_COMPRESSION_TYPE:-<dense: comm_eff off>} p=${COMM_EFF_MASK_P:-n/a} exact_k=${COMM_EFF_MASK_EXACT_K:-n/a} rescale=${COMM_EFF_MASK_RESCALE_MODE:-n/a} pp=${COMM_EFF_MASK_PP_SIZE:-n/a}
    schedule         $TOTAL_TRAINING_STEPS steps, test_freq $TEST_FREQ, save_freq $SAVE_FREQ
    checkpoints      R2=$CKPT_R2_ENABLED -> $R2_EXPERIMENT/$R2_REGIME/checkpoints/
    rollout          gpu_mem $ROLLOUT_GPU_MEM_UTIL, max_model_len $MAX_MODEL_LEN
    log              $LOG
    hydra            ${HYDRA_OVERRIDES[*]} $*

    WATCH response_length/clip_ratio. At a 3072-token cap on a long-CoT model it
    is the truncation rate, and it is the one number that decides whether this
    surface is measuring reasoning or measuring the cap.
EOF

exec bash "$PATCHED" "${HYDRA_OVERRIDES[@]}" "$@"
