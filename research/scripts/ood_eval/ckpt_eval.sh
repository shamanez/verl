#!/usr/bin/env bash
# ckpt_eval.sh
#
# In-domain + OOD capability audit for a MULTI-ARM run. Built for run 99
# (99-r1distill-deepscaler-600: DeepSeek-R1-Distill-Qwen-1.5B on DeepScaleR,
# 1024 prompt + 3072 response), and parameterised by env so a differently-named
# run with the same R2 layout needs no code edit.
#
#   bash research/scripts/ood_eval/ckpt_eval.sh
#   EVAL_STEPS="200 600" bash research/scripts/ood_eval/ckpt_eval.sh
#   DRY_RUN=1 bash research/scripts/ood_eval/ckpt_eval.sh     # gates only, no downloads
#
# The roster is ARMS x EVAL_STEPS plus the untrained base model. With the
# defaults that is three columns -- base, dense600, prf600 -- which is exactly
# the "dense against 95 percent compressed" comparison, with the base as the
# anchor that says how much either arm moved at all.
#
# THIS SCRIPT IS WRITTEN TO RUN ON A DIFFERENT BOX THAN THE TRAINER. It never
# assumes a local checkpoint tree: every checkpoint is pulled from R2, merged to
# a clean HF model, and the pulled FSDP shards are deleted as soon as the merge
# succeeds.
#
# WHAT IT REUSES, AND WHAT IT DOES NOT
#   ood_prep.py     builds every OOD benchmark parquet. Never reimplemented here.
#   ood_eval.sh     runs ONE model on ONE benchmark. Called directly, per pair.
#   ood_run_all.sh  its BENCHES table (bench name + sampling protocol) is PARSED
#                   out of the file so the sampling protocol has exactly one
#                   definition in the repo. Its merge() and ROSTER are hardcoded
#                   to an older matrix, so the orchestration is re-done here.
#
# THE PROMPT MUST MATCH TRAINING. The base launcher pins RELEX's Qwen ChatML
# template, which is the WRONG prompt for R1-Distill and is patched out of the
# training launcher by run_r1distill_deepscaler_600.sh. Section 6 performs the
# SAME rewrite on the generated eval launcher and gates it. Without that, every
# number below would be measured through a prompt the model never trained on --
# and it would look like a capability result, not a bug.
#
# R2 LAYOUT IS DISCOVERED, NOT ASSUMED. secrets.env on these boxes ships
# R2_BUCKET set to the PREFIX string by mistake, so R2_BUCKET is deliberately
# ignored and R2_CKPT_BUCKET is pinned below. The key layout under each arm's
# prefix is read from a real listing (the sink writes .../<exp>/<regime>/
# checkpoints/global_step_N/actor/ in some runs and .../<exp>/global_step_N/
# actor/ in others), so both shapes work and a wrong guess cannot silently
# produce an empty download.
#
# Knobs (env):
#   RUN_ID          run whose checkpoints to audit (default 99-r1distill-deepscaler-600)
#   ARMS            regimes under the run prefix   (default "dense prf")
#   EVAL_STEPS      steps to evaluate per arm      (default "600")
#   VERL_DIR        verl checkout                  (default /workspace/verl)
#   OOD_EVAL_ROOT   eval output root               (default /workspace/runs/<RUN_ID>-eval)
#   OOD_DATA_ROOT   benchmark parquets             (default /root/data/ood)
#   INDOMAIN_DATA_DIR  the training data dir       (default $HOME/data/deepscaler)
#   BASE_MODEL      untrained anchor               (default deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
#   PAIRS_CSV       GPU-pair pool                  (default "0,1|2,3", a 4-GPU box)
#   MAX_RESPONSE_LENGTH  generation cap            (default 3072, matches training)
#   R2_CKPT_BUCKET  bucket                         (pinned to shamane-pluralis)
#   R2_MIN_MB_S     slowest download rate, sizes the aws timeout (default 15)
#   DRY_RUN=1       run every preflight gate and stop before the first download
#
# Credentials come from ~/.config/verl-research/secrets.env (off-repo, chmod 600).
# The engine this driver ends up calling reads the SAME file and hard-requires
# HF_TOKEN and WANDB_API_KEY, and refuses to start if VAST_API_KEY is present, so
# all three are checked here rather than once per job at the far end of a
# download. Nothing secret is printed or written by this script.
#
# Run inside tmux. Every log is appended, never truncated, so an interrupted run
# resumes into the same files.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ID="${RUN_ID:-99-r1distill-deepscaler-600}"
ARMS="${ARMS:-dense prf}"
EVAL_STEPS="${EVAL_STEPS:-600}"

VERL_DIR="${VERL_DIR:-/workspace/verl}"
OOD_EVAL_ROOT="${OOD_EVAL_ROOT:-/workspace/runs/$RUN_ID-eval}"
OOD_DATA_ROOT="${OOD_DATA_ROOT:-/root/data/ood}"
INDOMAIN_DATA_DIR="${INDOMAIN_DATA_DIR:-$HOME/data/deepscaler}"
CKPT_ROOT="${CKPT_ROOT:-$OOD_EVAL_ROOT/pulled}"
MERGED_ROOT="${MERGED_ROOT:-$OOD_EVAL_ROOT/merged}"
BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
PAIRS_CSV="${PAIRS_CSV:-0,1|2,3}"

# R2. R2_BUCKET is NOT read: on these boxes it holds the prefix, not a bucket.
R2_CKPT_BUCKET="shamane-pluralis"
R2_ROOT="${R2_ROOT:-autonomous-harness-rlvr-compression}"
R2_MIN_MB_S="${R2_MIN_MB_S:-15}"
[[ "$R2_MIN_MB_S" =~ ^[0-9]+$ ]] && (( R2_MIN_MB_S > 0 )) || {
  echo "FATAL: R2_MIN_MB_S must be a positive integer (it divides the download timeout)" >&2; exit 1; }
SECRETS_FILE="${SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"

# Generation surface. Identical to training, so the in-domain number is a
# CROSS-CHECK of the in-training val rather than merely a second opinion.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.72}"

# Size model for the disk gate, printed with its derivation. Only the per-pull
# number is measured from R2. These are the 1.5B constants, env-overridable so a
# different model size needs no code edit. verl keeps the FSDP actor master copy
# in fp32 (~6 GiB of weights for 1.5B) plus two fp32 AdamW moments (~12 GiB);
# the optimizer shards are EXCLUDED on download. The merger casts to bf16.
MERGE_GIB="${MERGE_GIB:-5}"                  # one merged bf16 HF model
MERGE_MIN_GIB="${MERGE_MIN_GIB:-2}"          # a merge holding less than this is truncated, not finished
BASE_CACHE_GIB="${BASE_CACHE_GIB:-5}"        # the HF download of the untrained anchor
DISK_SLACK_GIB="${DISK_SLACK_GIB:-20}"       # logs, vLLM scratch, tokenizer caches
MERGE_RAM_GIB="${MERGE_RAM_GIB:-32}"         # the merge loads every fp32 rank shard at once
MIN_GPU_GIB="${MIN_GPU_GIB:-24}"             # hard floor per GPU
WARN_GPU_GIB="${WARN_GPU_GIB:-40}"

IN_DOMAIN_BENCH="${IN_DOMAIN_BENCH:-deepscaler_indomain}"

mkdir -p "$OOD_EVAL_ROOT" "$MERGED_ROOT" "$CKPT_ROOT" || {
  echo "FATAL: cannot create $OOD_EVAL_ROOT" >&2; exit 1; }
DRIVER_LOG="$OOD_EVAL_ROOT/ckpt_eval.log"

say() { echo "$*" | tee -a "$DRIVER_LOG"; }
die() { echo "FATAL: $*" | tee -a "$DRIVER_LOG" >&2; exit 1; }

file_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null; }

# One driver at a time per eval root. A second copy would put a second pair of
# vLLM stacks on the same GPUs and would rewrite train.log files the first copy
# is still filling, which reads afterwards as a mysteriously failed benchmark.
# The lock is an atomic mkdir plus a `kill -0` liveness test on the recorded
# pid: NEVER pgrep/pkill on a pattern, which in this project has repeatedly
# matched the checking command itself (and once killed the caller).
LOCK_DIR="$OOD_EVAL_ROOT/.ckpt_eval.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  OTHER="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ "$OTHER" =~ ^[0-9]+$ ]] && kill -0 "$OTHER" 2>/dev/null; then
    echo "FATAL: another ckpt_eval.sh (pid $OTHER) is already driving $OOD_EVAL_ROOT." >&2
    echo "       Attach to its tmux window, or wait for it. Delete $LOCK_DIR only if you are sure it is dead." >&2
    exit 1
  fi
  echo "WARN: clearing a stale lock at $LOCK_DIR (recorded pid '${OTHER:-none}' is not running)" >&2
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" || { echo "FATAL: could not take the lock at $LOCK_DIR" >&2; exit 1; }
fi
echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

say ""
say "=== capability audit  $(date -Iseconds) ==="
say "    run            $RUN_ID"
say "    arms           $ARMS"
say "    steps          $EVAL_STEPS"
say "    base model     $BASE_MODEL"
say "    context        $MAX_PROMPT_LENGTH prompt + $MAX_RESPONSE_LENGTH response = $MAX_MODEL_LEN"
say "    eval root      $OOD_EVAL_ROOT"
say "    r2             s3://$R2_CKPT_BUCKET/$R2_ROOT/$RUN_ID/<arm>/"

# ===========================================================================
# 1. Local preflight. Nothing here touches the network.
# ===========================================================================
say ""
say "--- preflight: local"

[[ -d "$VERL_DIR/.git" || -d "$VERL_DIR/verl" ]] || die "no verl checkout at VERL_DIR=$VERL_DIR"
for f in ood_eval.sh ood_prep.py ood_run_all.sh; do
  [[ -f "$SCRIPT_DIR/$f" ]] || die "missing sibling harness file $SCRIPT_DIR/$f"
done
BASE_LAUNCHER="$VERL_DIR/examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
[[ -f "$BASE_LAUNCHER" ]] || die "base launcher not found: $BASE_LAUNCHER"
command -v python3 >/dev/null 2>&1 || die "python3 is not on PATH"
( cd "$VERL_DIR" && python3 -c "import verl" ) >/dev/null 2>&1 \
  || die "verl is not importable from $VERL_DIR (install it: pip install --no-deps -e .)"

# The bare scalars this driver has to patch into a copy of the base launcher.
# The SHAPE CHECK lives here, not next to the sed, so DRY_RUN=1 exercises the
# one thing most likely to have drifted: the base launcher's own defaults.
# MAX_PROMPT_LENGTH / MAX_RESPONSE_LENGTH are the values this surface already
# wants, so they appear with from == to and the rewrite is a no-op that still
# proves the base holds them.
PATCH_SPEC=(
  "MAX_PROMPT_LENGTH|1024|$MAX_PROMPT_LENGTH"
  "MAX_RESPONSE_LENGTH|3072|$MAX_RESPONSE_LENGTH"
  "TRAIN_BATCH_SIZE|512|$TRAIN_BATCH_SIZE"
  "PPO_MINI_BATCH_SIZE|256|$PPO_MINI_BATCH_SIZE"
)
for spec in "${PATCH_SPEC[@]}"; do
  IFS='|' read -r pname pfrom _pto <<< "$spec"
  grep -q "^export ${pname}=${pfrom}\$" "$BASE_LAUNCHER" \
    || die "base launcher shape drifted: expected 'export ${pname}=${pfrom}' in $BASE_LAUNCHER. Re-derive this patch table before running."
done
say "    patch table    4/4 bare scalars found in the base launcher"

# The chat-template line, checked HERE too so DRY_RUN=1 catches a drifted base
# before anything is downloaded.
CHAT_OLD="'actor_rollout_ref.model.custom_chat_template=\${oc.env:RELEX_QWEN_CHAT_TEMPLATE}'"
grep -qF "$CHAT_OLD" "$BASE_LAUNCHER" \
  || die "base launcher no longer carries the RELEX ChatML override line. This driver rewrites it to null so the eval prompt matches training; re-derive the template patch before running."
say "    chat template  RELEX ChatML override located, will be rewritten to null"

# GPUs. The real count, before any shim is put on PATH.
NVIDIA_SMI_REAL="$(command -v nvidia-smi 2>/dev/null || true)"
[[ -n "$NVIDIA_SMI_REAL" ]] || die "nvidia-smi is not on PATH; this driver needs GPUs"
GPU_COUNT="$("$NVIDIA_SMI_REAL" -L 2>/dev/null | wc -l | tr -d ' ')"
say "    GPUs detected  ${GPU_COUNT:-0}"
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && (( GPU_COUNT >= 2 )) \
  || die "need at least 2 GPUs (one pair); detected ${GPU_COUNT:-0}"

IFS='|' read -r -a PAIRS <<< "$PAIRS_CSV"
(( ${#PAIRS[@]} >= 1 )) || die "PAIRS_CSV=$PAIRS_CSV parsed to zero pairs"
for pair in "${PAIRS[@]}"; do
  [[ "$pair" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "PAIRS_CSV entry '$pair' is not a comma-separated GPU list"
  IFS=',' read -r -a _ids <<< "$pair"
  for gid in "${_ids[@]}"; do
    (( gid < GPU_COUNT )) || die "PAIRS_CSV references GPU $gid but only $GPU_COUNT GPUs exist"
  done
done
say "    GPU pairs      ${#PAIRS[@]} (${PAIRS[*]})"

GPU_MIN_MIB="$("$NVIDIA_SMI_REAL" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
                | tr -d ' ' | sort -n | head -1)"
if [[ "$GPU_MIN_MIB" =~ ^[0-9]+$ ]]; then
  GPU_MIN_TOTAL_GIB=$(( GPU_MIN_MIB / 1024 ))
  say "    GPU memory     ${GPU_MIN_TOTAL_GIB} GiB on the smallest GPU (vLLM budget ${ROLLOUT_GPU_MEM_UTIL} of it)"
  (( GPU_MIN_TOTAL_GIB >= MIN_GPU_GIB )) \
    || die "need more than ${MIN_GPU_GIB} GiB per GPU; the smallest here is ${GPU_MIN_TOTAL_GIB} GiB. Override MIN_GPU_GIB only if you have measured it."
  (( GPU_MIN_TOTAL_GIB >= WARN_GPU_GIB )) \
    || say "    WARN: below ${WARN_GPU_GIB} GiB/GPU the KV cache left after the weights is small, so the avg@8 benchmarks will run at low concurrency."
else
  say "    WARN: could not read GPU memory from nvidia-smi; skipping the memory gate"
fi

AWS_BIN="$(command -v aws 2>/dev/null || true)"
[[ -n "$AWS_BIN" ]] || die "'aws' is not on PATH. Install the awscli v2 self-contained zip (it bundles its own python and cannot perturb the pinned torch/vllm stack), then re-run."
say "    aws            $AWS_BIN"

TIMEOUT_BIN="$(command -v timeout 2>/dev/null || command -v gtimeout 2>/dev/null || true)"
[[ -n "$TIMEOUT_BIN" ]] || say "    WARN: no 'timeout' binary; aws downloads will run unbounded (a stalled multipart can hang for hours)"

# ===========================================================================
# 2. Benchmark table + model roster.
# ===========================================================================
BENCH_TABLE="$(awk '/^BENCHES=\(/{f=1;next} f&&/^\)/{exit} f' "$SCRIPT_DIR/ood_run_all.sh" \
  | sed -e 's/#.*$//' -e 's/^[[:space:]]*"//' -e 's/"[[:space:]]*$//' -e 's/[[:space:]]*$//' \
  | grep -v '^[[:space:]]*$')"
[[ -n "$BENCH_TABLE" ]] || die "could not parse the BENCHES table out of $SCRIPT_DIR/ood_run_all.sh"

BENCH_SPECS=()
BENCH_NAMES=()
while IFS= read -r line; do
  read -r _b _n _t _p _extra <<<"$line"
  [[ -n "$_b" && -n "$_n" && -n "$_t" && -n "$_p" && -z "${_extra:-}" ]] \
    || die "BENCHES row '$line' in ood_run_all.sh does not have exactly 4 fields (name n temp top_p)"
  [[ "$_n" =~ ^[0-9]+$ ]] || die "BENCHES row '$line' has a non-integer n"
  BENCH_SPECS+=("$_b $_n $_t $_p")
  BENCH_NAMES+=("$_b")
done <<< "$BENCH_TABLE"
(( ${#BENCH_SPECS[@]} >= 8 )) || die "parsed only ${#BENCH_SPECS[@]} benchmarks from ood_run_all.sh, expected the full suite"

# In-domain goes FIRST and is not part of that table: it is the training val set
# itself (DeepScaleR's held-out split), scored with verl's own validation
# sampling defaults (n=1, temperature 0, top_p 1.0), which is what makes it a
# cross-check of the in-training val rather than a second opinion.
ALL_SPECS=("$IN_DOMAIN_BENCH 1 0 1.0" "${BENCH_SPECS[@]}")
ALL_NAMES=("$IN_DOMAIN_BENCH" "${BENCH_NAMES[@]}")
say "    benchmarks     ${#ALL_SPECS[@]} (${ALL_NAMES[*]})"

# Roster: base, then <arm><step> for every arm x step. A tag is the arm name
# followed by the step, so the arm and step are recovered by table lookup rather
# than by string surgery.
TAGS=("base")
TAG_ARMS=("-")
TAG_STEPS=("-")
# shellcheck disable=SC2086
for a in $ARMS; do
  [[ "$a" =~ ^[A-Za-z0-9_-]+$ ]] || die "ARMS entry '$a' is not a plain name"
  # shellcheck disable=SC2086
  for s in $EVAL_STEPS; do
    [[ "$s" =~ ^[0-9]+$ ]] || die "EVAL_STEPS entry '$s' is not an integer"
    TAGS+=("$a$s"); TAG_ARMS+=("$a"); TAG_STEPS+=("$s")
  done
done
(( ${#TAGS[@]} >= 2 )) || die "ARMS x EVAL_STEPS is empty"
say "    models         ${TAGS[*]}"

# ===========================================================================
# 3. R2 preflight: credentials, listing, key layout, step existence -- for
#    EVERY arm, before a single byte is downloaded.
# ===========================================================================
say ""
say "--- preflight: R2"

[[ -f "$SECRETS_FILE" ]] || die "secrets file not found: $SECRETS_FILE (needs R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT)"
# shellcheck disable=SC1090
{ set -a; source "$SECRETS_FILE"; set +a; } || die "could not source $SECRETS_FILE"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing from $SECRETS_FILE}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing from $SECRETS_FILE}"
: "${R2_ENDPOINT:?R2_ENDPOINT missing from $SECRETS_FILE}"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"

# The same secrets file is what the ENGINE reads, and the engine's gate is
# stricter than ours. Check it here, before the download, so a stripped-wrong
# secrets file costs a second instead of a multi-GiB pull followed by N jobs
# that each die in under a second.
: "${HF_TOKEN:?HF_TOKEN missing from $SECRETS_FILE. The engine requires it, and the base model is pulled from the hub for the base column.}"
: "${WANDB_API_KEY:?WANDB_API_KEY missing from $SECRETS_FILE. The engine requires the variable to exist even when WANDB_MODE=offline.}"
[[ -z "${VAST_API_KEY:-}" ]] \
  || die "VAST_API_KEY is set (probably a verbatim copy of the laptop secrets file). The engine refuses to start with it present. Re-strip $SECRETS_FILE on the laptop."
say "    engine creds   HF_TOKEN + WANDB_API_KEY present, no VAST_API_KEY"
say "    bucket         $R2_CKPT_BUCKET (pinned; R2_BUCKET from secrets.env is deliberately ignored)"

# 256MB parts. At the 8MB default a large shard is thousands of parts and R2
# rejects that many, and a silently-missing part is the failure mode that has cost
# this project hours. Same setting on the read side keeps ranged GETs sane.
if [[ "${AWS_CONFIGURE_CHUNKS:-1}" == "1" ]]; then
  "$AWS_BIN" configure set default.s3.multipart_chunksize 256MB 2>/dev/null || true
  "$AWS_BIN" configure set default.s3.multipart_threshold 256MB 2>/dev/null || true
fi

MANIFEST_DIR="$OOD_EVAL_ROOT/manifests"
mkdir -p "$MANIFEST_DIR"
R2_PLAN="$OOD_EVAL_ROOT/r2_plan.tsv"
: > "$R2_PLAN"
PLAN_RC=0

# shellcheck disable=SC2086
for a in $ARMS; do
  prefix="$R2_ROOT/$RUN_ID/$a"
  listing="$OOD_EVAL_ROOT/r2_listing_$a.txt"
  say "    listing        s3://$R2_CKPT_BUCKET/$prefix/ ..."
  if ! "$AWS_BIN" s3api list-objects-v2 \
        --bucket "$R2_CKPT_BUCKET" --prefix "$prefix/" \
        --endpoint-url "$R2_ENDPOINT" \
        --query 'Contents[].[Key,Size]' --output text > "$listing" 2>"$OOD_EVAL_ROOT/r2_listing_$a.err"; then
    say "    listing FAILED for arm $a. aws said:"
    sed -n 1,20p "$OOD_EVAL_ROOT/r2_listing_$a.err" | tee -a "$DRIVER_LOG"
    die "cannot list s3://$R2_CKPT_BUCKET/$prefix/ (credentials, endpoint, or bucket wrong)"
  fi
  [[ -s "$listing" ]] && ! grep -qx "None" "$listing" \
    || die "s3://$R2_CKPT_BUCKET/$prefix/ listed ZERO objects. Wrong prefix, or arm '$a' has not saved a checkpoint yet."

  R2_LISTING="$listing" MANIFEST_DIR="$MANIFEST_DIR" EVAL_STEPS="$EVAL_STEPS" ARM="$a" \
  python3 - >> "$R2_PLAN" <<'PY'
"""Derive, per (arm, step), the real actor/ key prefix and what to download.

Reads the listing rather than hardcoding a layout, so both the
<exp>/<regime>/checkpoints/global_step_N/actor/ shape and the flatter
<exp>/global_step_N/actor/ shape resolve. Writes one download manifest per
(arm, step) (relative-path TAB size, optimizer shards excluded) plus a plan line
for bash. Exits non-zero if anything requested is missing or incomplete.
"""

import os
import re
import sys

listing = os.environ["R2_LISTING"]
mdir = os.environ["MANIFEST_DIR"]
steps = os.environ["EVAL_STEPS"].split()
arm = os.environ["ARM"]

pat = re.compile(r"^(?P<pre>.*?/global_step_(?P<step>\d+)/actor/)(?P<rest>.+)$")
shard_pat = re.compile(r"^model_world_size_(\d+)_rank_(\d+)\.pt$")
found = {}
for raw in open(listing, encoding="utf-8"):
    line = raw.rstrip("\n")
    if not line or line == "None":
        continue
    parts = line.split("\t")
    if len(parts) != 2:
        parts = line.rsplit(None, 1)
    if len(parts) != 2:
        continue
    key, size_s = parts[0], parts[1]
    try:
        size = int(size_s)
    except ValueError:
        continue
    m = pat.match(key)
    if not m:
        continue
    st = str(int(m.group("step")))
    found.setdefault(st, {}).setdefault(m.group("pre"), []).append((m.group("rest"), size))

print("AVAILABLE\t{arm}\t{steps}".format(arm=arm, steps=" ".join(sorted(found, key=int))))

bad = 0
for st in steps:
    st = str(int(st))
    if st not in found:
        print(f"MISSING\t{arm}\t{st}")
        bad += 1
        continue
    prefixes = sorted(found[st])
    if len(prefixes) > 1:
        print("AMBIGUOUS\t" + arm + "\t" + st + "\t" + " ".join(prefixes))
        bad += 1
        continue
    pre = prefixes[0]
    objs = found[st][pre]
    total = sum(sz for _, sz in objs)
    keep = [(r, sz) for r, sz in objs if not os.path.basename(r).startswith("optim")]
    dl = sum(sz for _, sz in keep)

    # SHARD COMPLETENESS. "at least one model_*.pt" is not enough: the trainer
    # mirrors to R2 WHILE it trains, so a step can be listed with only some of
    # its rank shards uploaded. The merger reads world_size from
    # fsdp_config.json and then opens model_world_size_<W>_rank_<i>.pt for every
    # i, so a half-uploaded step passes a naive check, costs a full download,
    # and dies in the merge. Recover W from the shard NAMES (exactly the set of
    # files the merger will open), require one W and all of 0..W-1.
    shard_ranks = {}
    for r, _sz in keep:
        m2 = shard_pat.match(r)
        if m2:
            shard_ranks.setdefault(int(m2.group(1)), set()).add(int(m2.group(2)))
    ws = 0
    shards_ok = 0
    if len(shard_ranks) == 1:
        ws = next(iter(shard_ranks))
        got = shard_ranks[ws]
        shards_ok = int(got == set(range(ws)))
        if not shards_ok:
            missing_ranks = sorted(set(range(ws)) - got)
            print("SHARDS\t{arm}\t{st}\t{ws}\t{miss}".format(
                arm=arm, st=st, ws=ws, miss=",".join(map(str, missing_ranks))))
    elif len(shard_ranks) > 1:
        print("SHARDS\t{arm}\t{st}\t0\tmixed world sizes {w}".format(arm=arm, st=st, w=sorted(shard_ranks)))

    # The merger calls AutoConfig/AutoTokenizer on <actor>/huggingface, so those
    # two files specifically are what "the huggingface dir is here" has to mean.
    rels = {r for r, _ in keep}
    has_model = int(ws > 0)
    has_hf = int("huggingface/config.json" in rels and "huggingface/tokenizer_config.json" in rels)
    has_fsdp = int("fsdp_config.json" in rels)
    with open(os.path.join(mdir, f"{arm}_step_{st}.tsv"), "w", encoding="utf-8") as fh:
        for r, sz in sorted(keep):
            fh.write(f"{r}\t{sz}\n")
    print(
        "STEP\t{arm}\t{st}\t{pre}\t{n}\t{total}\t{dl}\t{nk}\t{hm}\t{hh}\t{hf}\t{ws}\t{sh}".format(
            arm=arm, st=st, pre=pre, n=len(objs), total=total, dl=dl, nk=len(keep),
            hm=has_model, hh=has_hf, hf=has_fsdp, ws=ws, sh=shards_ok,
        )
    )
    if not (has_model and has_hf and has_fsdp and shards_ok):
        bad += 1

for st in steps:
    st = str(int(st))
    if st in found:
        pre = sorted(found[st])[0]
        rest = sorted(found[st][pre])[0][0]
        print(f"PROBEKEY\t{arm}\t{pre}{rest}")
        break

sys.exit(1 if bad else 0)
PY
  rc=$?
  (( rc != 0 )) && PLAN_RC=1
done

tee -a "$DRIVER_LOG" < "$R2_PLAN" > /dev/null
while IFS=$'\t' read -r kind a rest; do
  [[ "$kind" == "AVAILABLE" ]] && say "    steps in R2    arm $a: ${rest:-none}"
done < "$R2_PLAN"

if (( PLAN_RC != 0 )); then
  while IFS=$'\t' read -r kind a b c d; do
    case "$kind" in
      MISSING)   say "    MISSING: $a/global_step_$b is NOT in R2" ;;
      AMBIGUOUS) say "    AMBIGUOUS: $a/global_step_$b resolves to more than one actor prefix: $c" ;;
      SHARDS)    say "    STILL UPLOADING: $a/global_step_$b is world_size $c and is missing rank shard(s) $d" ;;
    esac
  done < "$R2_PLAN"
  while IFS=$'\t' read -r kind a st _pre _n _tot _dl _nk hm hh hf _ws sh; do
    [[ "$kind" == "STEP" ]] || continue
    (( hm )) || say "    INCOMPLETE: $a/global_step_$st has no model_world_size_<W>_rank_<i>.pt shard"
    (( hh )) || say "    INCOMPLETE: $a/global_step_$st is missing huggingface/config.json or huggingface/tokenizer_config.json (the merger opens both)"
    (( hf )) || say "    INCOMPLETE: $a/global_step_$st has no fsdp_config.json"
    (( sh )) || say "    INCOMPLETE: $a/global_step_$st does not have every rank shard yet (see STILL UPLOADING above)"
  done < "$R2_PLAN"
  say "    If a step is merely still uploading, wait for the trainer to finish that"
  say "    checkpoint and re-run; nothing has been downloaded."
  die "requested steps [$EVAL_STEPS] are not all present and complete in R2 for arms [$ARMS]."
fi

# Real head-object probe on a key we know exists, so a listing-only permission
# cannot pass for a working read credential.
PROBE_KEY="$(awk -F'\t' '$1=="PROBEKEY"{print $3}' "$R2_PLAN" | head -1)"
[[ -n "$PROBE_KEY" ]] || die "no probe key resolved from the listings"
if "$AWS_BIN" s3api head-object --bucket "$R2_CKPT_BUCKET" --key "$PROBE_KEY" \
     --endpoint-url "$R2_ENDPOINT" --query 'ContentLength' --output text >/dev/null 2>&1; then
  say "    head-object    OK on a known key"
else
  die "head-object failed on a key the listing returned. The credential can list but not read; fix R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY."
fi

# ===========================================================================
# 4. Disk gate, on measured sizes. Peak is ONE pulled checkpoint plus ALL
#    merges, because each pull is deleted the moment its merge succeeds.
# ===========================================================================
say ""
say "--- preflight: disk"

MAX_DL_GIB=0
N_CKPTS=0
while IFS=$'\t' read -r kind a st _pre nobj tot dl _nk _hm _hh _hf ws _sh; do
  [[ "$kind" == "STEP" ]] || continue
  g_tot=$(( (tot + 1073741823) / 1073741824 ))
  g_dl=$(( (dl + 1073741823) / 1073741824 ))
  say "    $a/global_step_$st: $nobj objects, world_size $ws (all shards present), ${g_tot} GiB listed, ${g_dl} GiB to pull (optimizer shards excluded)"
  (( g_dl > MAX_DL_GIB )) && MAX_DL_GIB=$g_dl
  N_CKPTS=$(( N_CKPTS + 1 ))
done < "$R2_PLAN"
(( N_CKPTS > 0 )) || die "no checkpoints resolved from the R2 plan"

MERGE_TOTAL_GIB=$(( MERGE_GIB * N_CKPTS ))
NEED_GIB=$(( MAX_DL_GIB + MERGE_TOTAL_GIB + BASE_CACHE_GIB + DISK_SLACK_GIB ))
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

say ""
say "    This driver excludes the optimizer shards on download and DELETES each"
say "    pulled checkpoint as soon as its merge succeeds, so the peak is one pull"
say "    plus all merges:"
say "        ${MAX_DL_GIB} GiB  largest single pull        -> $CKPT_ROOT"
say "      + ${MERGE_TOTAL_GIB} GiB  ${N_CKPTS} merged HF models, kept   -> $MERGED_ROOT"
say "      + ${BASE_CACHE_GIB} GiB  HF cache for $BASE_MODEL -> $HF_CACHE_DIR"
say "      + ${DISK_SLACK_GIB} GiB  logs, vLLM scratch, caches -> $OOD_EVAL_ROOT"
say "      = ${NEED_GIB} GiB total, charged per filesystem below"

# Charge each requirement to the filesystem that actually holds it, so an
# overridden CKPT_ROOT or HF_HOME on a small volume cannot pass a gate that only
# ever looked at OOD_EVAL_ROOT.
DISK_CLAIMS="$OOD_EVAL_ROOT/.disk_claims.tsv"
: > "$DISK_CLAIMS"
claim() {  # claim <path> <gib>
  local p="$1" g="$2" dev
  mkdir -p "$p" 2>/dev/null || true
  dev="$(df -Pk "$p" 2>/dev/null | awk 'NR==2{print $1}')"
  [[ -n "$dev" ]] || die "could not read the filesystem for $p"
  printf '%s\t%s\t%s\n' "$dev" "$g" "$p" >> "$DISK_CLAIMS"
}
claim "$CKPT_ROOT"     "$MAX_DL_GIB"
claim "$MERGED_ROOT"   "$MERGE_TOTAL_GIB"
claim "$HF_CACHE_DIR"  "$BASE_CACHE_GIB"
claim "$OOD_EVAL_ROOT" "$DISK_SLACK_GIB"

DISK_FAIL=0
while IFS=$'\t' read -r dev need where; do
  [[ -n "$dev" ]] || continue
  free_gib="$(df -Pk "$where" 2>/dev/null | awk 'NR==2{print int($4/1048576)}')"
  say "    $dev (via $where): ${free_gib:-unknown} GiB free, ${need} GiB required"
  if ! [[ "$free_gib" =~ ^[0-9]+$ ]] || (( free_gib < need )); then
    say "    INSUFFICIENT DISK on $dev"
    DISK_FAIL=1
  fi
done < <(awk -F'\t' '{n[$1]+=$2; if(!($1 in p)) p[$1]=$3}
                     END{for (d in n) printf "%s\t%s\t%s\n", d, n[d], p[d]}' "$DISK_CLAIMS")
(( DISK_FAIL == 0 )) \
  || die "not enough free disk. Free space, or point CKPT_ROOT / MERGED_ROOT / OOD_EVAL_ROOT / HF_HOME at a bigger volume."

say ""
RAM_GIB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ "$RAM_GIB" =~ ^[0-9]+$ ]]; then
  say "    host RAM       ${RAM_GIB} GiB, merge needs about ${MERGE_RAM_GIB} GiB"
  (( RAM_GIB >= MERGE_RAM_GIB )) \
    || die "host RAM ${RAM_GIB} GiB is below the ${MERGE_RAM_GIB} GiB the merge needs. Merging would OOM after the download."
else
  say "    WARN: could not read host RAM from 'free -g'; the ${MERGE_RAM_GIB} GiB merge gate is unchecked"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  say ""
  say "=== DRY_RUN=1: every preflight gate passed, stopping before the first download ==="
  exit 0
fi

# ===========================================================================
# 5. Benchmark data. ood_prep.py owns every OOD dataset definition. We only ask
#    it for the sets we do not already have, and we hand-build the in-domain dir
#    (DeepScaleR's held-out test split is not one of ood_prep.py's benchmarks,
#    it is the training val set).
# ===========================================================================
say ""
say "--- data"

[[ -f "$INDOMAIN_DATA_DIR/test.parquet" && -f "$INDOMAIN_DATA_DIR/train.parquet" ]] \
  || die "in-domain parquet missing in INDOMAIN_DATA_DIR=$INDOMAIN_DATA_DIR. Prepare it with:
       python3 research/scripts/prepare_rlvr_math.py --dataset deepscaler --local_save_dir $INDOMAIN_DATA_DIR
     using the SAME --train-cap/--val-size/--seed the training run used, or the held-out split will not be the one the model was validated on."

IN_DOMAIN_DIR="$OOD_DATA_ROOT/$IN_DOMAIN_BENCH"
mkdir -p "$IN_DOMAIN_DIR" || die "could not create $IN_DOMAIN_DIR (is OOD_DATA_ROOT=$OOD_DATA_ROOT writable?)"
# -f -n, not a bare ln: a dangling symlink left by an earlier run on a wiped
# data dir fails an -e test and then fails 'ln -s' with "File exists".
for f in test.parquet train.parquet; do
  ln -sfn "$INDOMAIN_DATA_DIR/$f" "$IN_DOMAIN_DIR/$f" \
    || die "could not link $INDOMAIN_DATA_DIR/$f into $IN_DOMAIN_DIR"
  [[ -f "$IN_DOMAIN_DIR/$f" ]] || die "$IN_DOMAIN_DIR/$f does not resolve to a real file"
done
say "    in-domain      $IN_DOMAIN_DIR (DeepScaleR held-out split, the training val set)"

MISSING_BENCHES=()
for b in "${BENCH_NAMES[@]}"; do
  [[ -f "$OOD_DATA_ROOT/$b/test.parquet" ]] || MISSING_BENCHES+=("$b")
done
if (( ${#MISSING_BENCHES[@]} > 0 )); then
  say "    building ${#MISSING_BENCHES[@]} benchmark parquets via ood_prep.py: ${MISSING_BENCHES[*]}"
  ( cd "$VERL_DIR" && OOD_ROOT="$OOD_DATA_ROOT" MATH_TRAIN="$INDOMAIN_DATA_DIR/train.parquet" \
      python3 "$SCRIPT_DIR/ood_prep.py" --only "${MISSING_BENCHES[@]}" ) 2>&1 | tee -a "$OOD_EVAL_ROOT/prep.log"
fi
STILL_MISSING=()
for b in "${BENCH_NAMES[@]}"; do
  [[ -f "$OOD_DATA_ROOT/$b/test.parquet" ]] || STILL_MISSING+=("$b")
done
(( ${#STILL_MISSING[@]} == 0 )) \
  || die "ood_prep.py did not produce: ${STILL_MISSING[*]} (see $OOD_EVAL_ROOT/prep.log)"
say "    benchmarks     all ${#BENCH_NAMES[@]} parquets present under $OOD_DATA_ROOT"

# ===========================================================================
# 6. Launcher. Two generated files, both regenerated every run:
#      a) a patched copy of the base launcher (bare exports no env var can
#         override, plus the chat-template rewrite), living next to the engine so
#         its $HERE still resolves,
#      b) a thin wrapper adding the Hydra keys ood_eval.sh cannot forward.
#    Plus the nvidia-smi -L shim: the engine hard-detects NGPUS_PER_NODE from
#    nvidia-smi -L, which ignores CUDA_VISIBLE_DEVICES, so without the shim a
#    2-GPU slice on a 4-GPU box asks Ray for 4 GPUs and dies.
# ===========================================================================
say ""
say "--- launcher"

PATCHED="$VERL_DIR/examples/grpo_trainer/ckpt_eval.gen.sh"
cp "$BASE_LAUNCHER" "$PATCHED" || die "could not copy $BASE_LAUNCHER"
for spec in "${PATCH_SPEC[@]}"; do
  IFS='|' read -r pname pfrom pto <<< "$spec"
  sed -i.bak -e "s/^export ${pname}=${pfrom}\$/export ${pname}=${pto}/" "$PATCHED" \
    || die "sed failed for ${pname}"
  rm -f "$PATCHED.bak"
  grep -q "^export ${pname}=${pto}\$" "$PATCHED" || die "${pname} patch missed (wanted ${pto})"
  say "    patched ${pname}: ${pfrom} -> ${pto}"
done

# THE PROMPT. Same rewrite the training launcher performs, for the same reason,
# gated the same way. Python, not sed: the line carries ${oc.env:...}.
BASE_LAUNCHER="$BASE_LAUNCHER" PATCHED="$PATCHED" python3 - <<'PY' || die "chat-template patch failed"
import os
import sys

base_p, patched_p = os.environ["BASE_LAUNCHER"], os.environ["PATCHED"]
OLD = "'actor_rollout_ref.model.custom_chat_template=${oc.env:RELEX_QWEN_CHAT_TEMPLATE}'"
NEW = "'actor_rollout_ref.model.custom_chat_template=null'"

src = open(base_p, encoding="utf-8").read()
if src.count(OLD) != 1:
    sys.exit(f"base launcher shape drifted: expected exactly one occurrence of {OLD}, found {src.count(OLD)}")

out = open(patched_p, encoding="utf-8").read().replace(OLD, NEW)
if OLD in out or NEW not in out or "oc.env:RELEX_QWEN_CHAT_TEMPLATE" in out:
    sys.exit("template rewrite did not take")
open(patched_p, "w", encoding="utf-8").write(out)
print("    patched custom_chat_template: RELEX ChatML -> null (the model's own template)")
PY
chmod +x "$PATCHED"

LAUNCHER="$OOD_EVAL_ROOT/ckpt_eval_launcher.gen.sh"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Generated by ckpt_eval.sh. Do not edit, it is overwritten every run.
set -uo pipefail
export COMM_EFF_ENABLED=false
export CKPT_R2_ENABLED=false
export ROLLOUT_GPU_MEM_UTIL="\${ROLLOUT_GPU_MEM_UTIL:-$ROLLOUT_GPU_MEM_UTIL}"
exec bash "$PATCHED" \\
  actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \\
  actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \\
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \\
  "\$@"
EOF
chmod +x "$LAUNCHER"
say "    launcher       $LAUNCHER (max_model_len $MAX_MODEL_LEN)"

SHIM_DIR="$OOD_EVAL_ROOT/shim"
mkdir -p "$SHIM_DIR"
cat > "$SHIM_DIR/nvidia-smi" <<EOF
#!/usr/bin/env bash
# Generated by ckpt_eval.sh. Makes a bare 'nvidia-smi -L' report only the GPUs
# in CUDA_VISIBLE_DEVICES, which the real binary ignores. Everything else is
# passed through to the real binary by absolute path, so this cannot recurse.
REAL="$NVIDIA_SMI_REAL"
if [[ "\$#" -eq 1 && "\$1" == "-L" && -n "\${CUDA_VISIBLE_DEVICES:-}" ]]; then
  n=\$(awk -F, '{print NF}' <<< "\$CUDA_VISIBLE_DEVICES")
  # NOT 'exec ... | head': in a pipeline exec only replaces the left subshell,
  # so the script would fall through and print the unfiltered list as well.
  "\$REAL" -L | head -n "\$n"
  exit 0
fi
exec "\$REAL" "\$@"
EOF
chmod +x "$SHIM_DIR/nvidia-smi"
say "    shim           $SHIM_DIR/nvidia-smi (real binary: $NVIDIA_SMI_REAL)"

# Offline by default even though WANDB_API_KEY exists (the engine demands the
# variable, not a live connection). Every score this driver reads comes from
# train.log, so the online path buys nothing and costs something: the rc=1
# atexit teardown race has silently dropped final-step values before.
export WANDB_MODE="${WANDB_MODE:-offline}"
say "    wandb          WANDB_MODE=$WANDB_MODE (scores are read from train.log, not WandB)"

export VERL_DIR OOD_EVAL_ROOT OOD_DATA_ROOT LAUNCHER SHIM_DIR

# ===========================================================================
# 7. Model preparation: pull, size-verify, merge, delete shards.
#    The result is handed back in the global PREPARED_MODEL rather than on
#    stdout: a multi-GiB pull takes long enough that its progress lines have to
#    reach the terminal live, and a command substitution would swallow them.
# ===========================================================================
PREPARED_MODEL=""

merge_is_complete() {  # true only for a merged model vLLM can actually load
  local out="$1" wbytes wgib nfiles
  [[ -f "$out/config.json" ]] || { say "    merge check: no config.json in $out"; return 1; }
  [[ -f "$out/tokenizer_config.json" ]] || { say "    merge check: no tokenizer in $out"; return 1; }
  nfiles="$(find "$out" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | wc -l | tr -d ' ')"
  (( nfiles > 0 )) || { say "    merge check: no weight files in $out"; return 1; }
  wbytes=0
  local f fsz
  while IFS= read -r f; do
    fsz="$(file_size "$f")"
    [[ "$fsz" =~ ^[0-9]+$ ]] || fsz=0
    wbytes=$(( wbytes + fsz ))
  done < <(find "$out" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null)
  wgib=$(( wbytes / 1073741824 ))
  if (( wgib < MERGE_MIN_GIB )); then
    say "    merge check: $out holds only ${wgib} GiB of weights across $nfiles files, expected at least ${MERGE_MIN_GIB}"
    return 1
  fi
  # The merged model must carry the model's OWN chat template through to vLLM.
  python3 - "$out" <<'PY' || { say "    merge check: merged tokenizer has no usable chat template"; return 1; }
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
assert tok.chat_template, "merged model ships no chat_template"
r = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
assert "<|im_start|>" not in r, "merged model renders ChatML; the eval prompt would not match training"
PY
  printf 'merged %s  %s GiB across %s files\n' "$(date -Iseconds)" "$wgib" "$nfiles" > "$out/.merge_ok"
  say "    merge check: $nfiles weight files, ${wgib} GiB, tokenizer + native chat template present"
  return 0
}

pull_and_merge() {  # pull_and_merge <arm> <step> <tag>  -> sets PREPARED_MODEL
  local arm="$1" step="$2" tag="$3"
  local out="$MERGED_ROOT/$tag"
  local ck="$CKPT_ROOT/$arm/global_step_$step/actor"
  local manifest="$MANIFEST_DIR/${arm}_step_$step.tsv"
  PREPARED_MODEL=""

  # RESUME ON SUCCESS ONLY. config.json is NOT a success marker: transformers'
  # save_pretrained writes the config BEFORE the weight shards, so a merge
  # killed part way leaves a directory holding a config and no weights. Keying
  # resume on that hands a broken model to vLLM. .merge_ok is written by
  # merge_is_complete, after the weights are on disk and checked.
  if [[ -f "$out/.merge_ok" ]]; then
    say "    merged model already present and verified for $tag: $out"
    PREPARED_MODEL="$out"; return 0
  fi
  if [[ -f "$out/config.json" ]]; then
    say "    $out has a config but no .merge_ok: a previous merge did not finish. Redoing it."
    rm -rf "$out"
  fi

  local pre dl_bytes
  pre="$(awk -F'\t' -v a="$arm" -v s="$step" '$1=="STEP" && $2==a && $3==s {print $4}' "$R2_PLAN")"
  dl_bytes="$(awk -F'\t' -v a="$arm" -v s="$step" '$1=="STEP" && $2==a && $3==s {print $7}' "$R2_PLAN")"
  [[ -n "$pre" ]] || { say "    NO PLAN for $arm step $step"; return 1; }

  local secs=$(( dl_bytes / (R2_MIN_MB_S * 1048576) + 600 ))
  (( secs < 1800 )) && secs=1800

  local attempt
  for attempt in 1 2; do
    mkdir -p "$ck"
    say "    pull  s3://$R2_CKPT_BUCKET/$pre -> $ck (attempt $attempt, timeout ${secs}s)"
    local -a cmd=("$AWS_BIN" s3 cp "s3://$R2_CKPT_BUCKET/$pre" "$ck/"
                  --recursive --exclude "optim*" --only-show-errors
                  --endpoint-url "$R2_ENDPOINT")
    if [[ -n "$TIMEOUT_BIN" ]]; then
      "$TIMEOUT_BIN" "$secs" "${cmd[@]}" >> "$OOD_EVAL_ROOT/pull.log" 2>&1
    else
      "${cmd[@]}" >> "$OOD_EVAL_ROOT/pull.log" 2>&1
    fi
    local rc=$?
    if (( rc == 124 )); then
      say "    pull TIMED OUT after ${secs}s. Sample list-parts and 'ps -o time=' together:"
      say "    both flat means a hang, both climbing means merely slow (raise R2_MIN_MB_S)."
    elif (( rc != 0 )); then
      say "    aws s3 cp exited $rc (see $OOD_EVAL_ROOT/pull.log)"
    fi

    # Size-verify every expected object. A part can go missing silently, so a
    # zero exit from aws is not evidence that the bytes are all here.
    local bad=0 rel sz got
    while IFS=$'\t' read -r rel sz; do
      [[ -n "$rel" ]] || continue
      if [[ ! -f "$ck/$rel" ]]; then
        say "    MISSING after pull: $rel"; bad=$(( bad + 1 )); continue
      fi
      got="$(file_size "$ck/$rel")"
      if [[ "$got" != "$sz" ]]; then
        say "    SIZE MISMATCH $rel: local $got, R2 $sz"; bad=$(( bad + 1 ))
      fi
    done < "$manifest"

    if (( bad == 0 )); then
      say "    pull verified: $(wc -l < "$manifest" | tr -d ' ') objects match R2 byte for byte"
      break
    fi
    say "    pull incomplete ($bad objects wrong) on attempt $attempt"
    (( attempt == 2 )) && { say "    GIVING UP on $arm step $step"; return 1; }
  done

  [[ -f "$ck/fsdp_config.json" ]] || { say "    no fsdp_config.json in $ck"; return 1; }
  say "    merge $arm/global_step_$step -> $out"
  ( cd "$VERL_DIR" && OMP_NUM_THREADS=8 python3 -m verl.model_merger merge \
      --backend fsdp --local_dir "$ck" --target_dir "$out" ) >> "$OOD_EVAL_ROOT/merge.log" 2>&1

  # The merger writes the tokenizer from <ck>/huggingface. Belt and braces: if
  # it did not, copy it before the shards go away, because vLLM loads the
  # tokenizer from MODEL_PATH and a missing one fails at boot on every bench.
  if [[ ! -f "$out/tokenizer_config.json" && -d "$ck/huggingface" ]]; then
    cp -n "$ck/huggingface/"* "$out/" 2>/dev/null || true
    say "    copied tokenizer/config files from $ck/huggingface"
  fi

  if ! merge_is_complete "$out"; then
    say "    MERGE FAILED for $arm step $step (see $OOD_EVAL_ROOT/merge.log); pulled shards kept at $ck for inspection"
    return 1
  fi

  say "    merge OK, deleting pulled shards $ck (peak disk stays at one checkpoint)"
  rm -rf "$ck"
  rmdir "$CKPT_ROOT/$arm/global_step_$step" 2>/dev/null || true
  PREPARED_MODEL="$out"
  return 0
}

# ===========================================================================
# 8. Eval. Per (model, bench), skipped when a result already exists.
# ===========================================================================

has_result() {  # has_result <tag> <bench>
  # -a because train.log carries carriage returns and progress-bar bytes that
  # can make grep decide the file is binary and skip the match.
  local f="$OOD_EVAL_ROOT/$1/$2/train.log"
  [[ -f "$f" ]] && grep -qa "acc/mean@" "$f" 2>/dev/null
}

run_tag() {  # run_tag <tag> <model_path>
  local tag="$1" model="$2" i=0
  local -a pids=()
  if [[ "$model" != "$BASE_MODEL" && ! -f "$model/.merge_ok" ]]; then
    say "    SKIP $tag: no verified merged model at '$model'"; return 1
  fi
  local spec b n t p gpus
  for spec in "${ALL_SPECS[@]}"; do
    read -r b n t p <<<"$spec"
    if has_result "$tag" "$b"; then
      say "    skip $tag/$b (result already on disk)"
      continue
    fi
    gpus="${PAIRS[$(( i % ${#PAIRS[@]} ))]}"
    say "    start $tag/$b on GPUs $gpus (n=$n temp=$t top_p=$p)"
    bash "$SCRIPT_DIR/ood_eval.sh" "$model" "$b" "$tag" "$gpus" "$n" "$t" "$p" &
    pids+=("$!")
    i=$(( i + 1 ))
    if (( ${#pids[@]} >= ${#PAIRS[@]} )); then
      wait "${pids[@]}"
      pids=()
    fi
  done
  if (( ${#pids[@]} > 0 )); then
    wait "${pids[@]}"
  fi
  local ok=0
  for spec in "${ALL_SPECS[@]}"; do
    read -r b _n _t _p <<<"$spec"
    has_result "$tag" "$b" && ok=$(( ok + 1 ))
  done
  if (( ok == ${#ALL_SPECS[@]} )); then
    say "    DONE $tag: $ok/${#ALL_SPECS[@]} benchmarks"
  else
    say "    PARTIAL $tag: $ok/${#ALL_SPECS[@]} benchmarks (re-run this script to resume)"
  fi
}

for idx in "${!TAGS[@]}"; do
  tag="${TAGS[$idx]}"
  arm="${TAG_ARMS[$idx]}"
  step="${TAG_STEPS[$idx]}"
  say ""
  say "=== model $tag  $(date -Iseconds) ==="
  if [[ "$tag" == "base" ]]; then
    model="$BASE_MODEL"
  else
    # Do not pay for a multi-GiB pull if every bench for this tag is already done.
    pending=0
    for spec in "${ALL_SPECS[@]}"; do
      read -r b _n _t _p <<<"$spec"
      has_result "$tag" "$b" || pending=$(( pending + 1 ))
    done
    if (( pending == 0 )); then
      say "    all ${#ALL_SPECS[@]} benchmarks already done for $tag, no checkpoint pull needed"
      continue
    fi
    if pull_and_merge "$arm" "$step" "$tag"; then
      model="$PREPARED_MODEL"
    else
      say "    SKIP $tag: checkpoint could not be prepared"
      continue
    fi
  fi
  run_tag "$tag" "$model"
done

# ===========================================================================
# 9. Table.
# ===========================================================================
RESULTS="$OOD_EVAL_ROOT/RESULTS_$RUN_ID.tsv"
say ""
OOD_EVAL_ROOT="$OOD_EVAL_ROOT" \
TAGS_CSV="$(IFS=,; echo "${TAGS[*]}")" \
BENCH_CSV="$(IFS=,; echo "${ALL_NAMES[*]}")" \
IN_DOMAIN_BENCH="$IN_DOMAIN_BENCH" \
RUN_ID="$RUN_ID" \
RESULTS_TSV="$RESULTS" \
python3 "$SCRIPT_DIR/tabulate_arms.py" | tee -a "$DRIVER_LOG"

say ""
say "=== capability audit finished  $(date -Iseconds) ==="
say "    table   $RESULTS"
say "    log     $DRIVER_LOG"
say "    figure  python3 $SCRIPT_DIR/plot_dense_vs_compressed.py --results $RESULTS"

# The completion marker is SUCCESS-ONLY. An unconditional touch here would tell
# a watcher the audit is done while cells are still dots, which is exactly the
# mistake the per-cell resume logic is built to avoid.
CELLS_DONE=0
CELLS_TOTAL=0
for tag in "${TAGS[@]}"; do
  for spec in "${ALL_SPECS[@]}"; do
    read -r b _n _t _p <<<"$spec"
    CELLS_TOTAL=$(( CELLS_TOTAL + 1 ))
    has_result "$tag" "$b" && CELLS_DONE=$(( CELLS_DONE + 1 ))
  done
done
rm -f "$OOD_EVAL_ROOT/CKPT_EVAL_DONE" "$OOD_EVAL_ROOT/CKPT_EVAL_PARTIAL"
if (( CELLS_DONE == CELLS_TOTAL )); then
  say "    cells   $CELLS_DONE/$CELLS_TOTAL, complete"
  touch "$OOD_EVAL_ROOT/CKPT_EVAL_DONE"
else
  say "    cells   $CELLS_DONE/$CELLS_TOTAL, INCOMPLETE. Re-run this script to resume the rest."
  touch "$OOD_EVAL_ROOT/CKPT_EVAL_PARTIAL"
fi
