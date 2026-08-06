#!/usr/bin/env bash
# run_98_wedge_probe.sh
#
# Sequential step-1 wedge probe driver. Runs ON THE BOX inside tmux, one short
# GPU probe per per-boundary compression pattern, on a 4x H200 box.
#
# WHAT IT MEASURES
#   The step-1 sampler-trainer wedge (rollout_corr/kl and friends) of
#   Qwen3-8B-Base under different per-boundary PRF exact-k compression
#   patterns. The wedge is fully determined at step 1 because the pre-update
#   policy equals the init policy, so TOTAL_TRAINING_STEPS=1 per arm
#   suffices. Probes run at 3072 response length for speed. The per-token
#   wedge mechanism is length independent and all arms are compared at equal
#   length. Known anchors at 15360 response: all-7-compressed wedge 17.23,
#   five-compressed [4,9,23,27,31] wedge 16.88.
#
# ARM TABLE
#   Boundaries sit after decoder layers [4,9,14,19,23,27,31] (36 layers over
#   8 stages). Compressed cuts run p=0.95, dense cuts 0.0.
#     ref7  all seven cuts compressed (reference)
#     s4 s9 s14 s19 s23 s27 s31  exactly one cut compressed (diagnostic)
#     alt4  cuts 4,14,23,31 compressed (alternating cross-check)
#     alt3  cuts 9,19,27 compressed (complementary cross-check)
#
# ATTRIBUTION LOGIC
#   The single-cut arms are DIAGNOSTIC probes for per-cut attribution. The
#   deployable min-3-compressed constraint applies to the final chosen
#   config, not to these probes. If per-cut contributions are additive, the
#   best k-compressed deployable config is the k cuts with the smallest
#   single-cut wedge. That candidate is then verified by one confirmation
#   arm after this sweep.
#
# MECHANICS
#   Copy this file to /workspace and run it from there. The arm launcher
#   hard-resets the /workspace/verl checkout on every invocation, so never
#   execute this driver from a path inside that checkout. Each arm invokes
#   the env-driven idempotent launcher
#   /workspace/verl/examples/grpo_trainer/run_qwen3_8b_prf_exactk_1000.sh
#   in a subshell (arm env cannot leak across arms) under a 40 minute hard
#   timeout (60 for the first arm launched, which also pays the model pull,
#   the parquet prep and the first vLLM boot). The timeout kills only the
#   engine SHELL: the engine backgrounds the trainer, so the orphan can
#   outlive its arm; the drain step (poll, then ray stop --force, then a
#   pid-exact kill of whatever still holds GPU memory) is what actually
#   enforces the cap. After each arm, whatever the exit code, the driver
#   greps the arm log binary-safe for the step-1 metrics line and appends
#   one TSV row to /workspace/98_wedge_results.tsv, then waits for every
#   GPU to drain below 5000 MiB before the next arm. rc=1 from the trainer
#   at exit is NORMAL (atexit teardown race). Arm success is judged only by
#   the presence of the step-1 rollout_corr/kl metrics line, never by the
#   exit code.
#
#   The launcher lives INSIDE the checkout it re-fetches, and on a transient
#   fetch failure it moves the checkout aside before dying. Each arm
#   therefore re-clones the branch first when the launcher file is missing,
#   so one network blip fails one arm instead of the rest of the night. A
#   restarted driver skips arms that already have an OK row in the TSV.
#
#   NOTE on aws: the launcher's aws preflight is unconditional in the
#   current script (it does not consult CKPT_R2_ENABLED), so a box without
#   aws on PATH fails every arm in seconds. This driver checks once up
#   front and refuses to start instead.
#
#   Usage, inside tmux on the box:
#     cp /workspace/verl/examples/grpo_trainer/run_98_wedge_probe.sh /workspace/
#     bash /workspace/run_98_wedge_probe.sh
set -uo pipefail

# ---------------------------- fixed paths and knobs -------------------------
WORK="/workspace"
SECRETS="/root/.config/verl-research/secrets.env"
LAUNCHER="$WORK/verl/examples/grpo_trainer/run_qwen3_8b_prf_exactk_1000.sh"
MASTER_LOG="$WORK/98_wedge_probe.log"
TSV="$WORK/98_wedge_results.tsv"
PIDFILE="$WORK/98_wedge_probe.pid"
PROBE_BRANCH="exp/97-qwen3-8b-16k-densemid-1000"
REPO_URL="https://github.com/shamanez/verl.git"
ARM_TIMEOUT_S=2400        # 40 minutes hard cap per arm
ARM_TIMEOUT_FIRST_S=3600  # arm 1 also pays model pull + parquet prep + first vLLM boot
DRAIN_LIMIT_MIB=5000      # every GPU must be below this before the next arm
DRAIN_CAP_S=360           # 6 minute polling cap before forcing ray stop
DRAIN_POLL_S=15

# name|COMM_EFF_MASK_P_BY_BOUNDARY (Hydra list literal, no spaces)
ARMS=(
  "ref7|[0.95,0.95,0.95,0.95,0.95,0.95,0.95]"
  "s4|[0.95,0.0,0.0,0.0,0.0,0.0,0.0]"
  "s9|[0.0,0.95,0.0,0.0,0.0,0.0,0.0]"
  "s14|[0.0,0.0,0.95,0.0,0.0,0.0,0.0]"
  "s19|[0.0,0.0,0.0,0.95,0.0,0.0,0.0]"
  "s23|[0.0,0.0,0.0,0.0,0.95,0.0,0.0]"
  "s27|[0.0,0.0,0.0,0.0,0.0,0.95,0.0]"
  "s31|[0.0,0.0,0.0,0.0,0.0,0.0,0.95]"
  "alt4|[0.95,0.0,0.95,0.0,0.95,0.0,0.95]"
  "alt3|[0.0,0.95,0.0,0.95,0.0,0.95,0.0]"
)

# ---------------------------- secrets FIRST ---------------------------------
# WANDB_API_KEY must be in the environment BEFORE the launcher's offline-mode
# check runs (a prior relaunch missed this and the run went wandb-offline).
if [[ ! -f "$SECRETS" ]]; then
  echo "FATAL: secrets file $SECRETS is missing." >&2
  echo "       Without it WANDB_API_KEY is unset and every arm silently goes wandb-offline." >&2
  echo "       Install the secrets file, then relaunch this driver." >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$SECRETS"
set +a
[[ -n "${WANDB_API_KEY:-}" ]] \
  || echo "WARN: WANDB_API_KEY still unset after sourcing $SECRETS, arms will log wandb-offline" >&2

# ---------------------------- pidfile ---------------------------------------
if [[ -f "$PIDFILE" ]]; then
  OLD_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "FATAL: another driver instance appears alive (pid $OLD_PID from $PIDFILE)." >&2
    echo "       If that is stale, remove $PIDFILE and relaunch." >&2
    exit 1
  fi
fi
echo "$$" > "$PIDFILE" || { echo "FATAL: cannot write $PIDFILE" >&2; exit 1; }
trap 'rm -f "$PIDFILE"' EXIT

# ---------------------------- helpers ---------------------------------------
progress() {
  local msg
  msg="[98-wedge $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"
  echo "$msg"
  echo "$msg" >> "$MASTER_LOG"
}

# Pull one "key:value" metric out of a verl console metrics line.
# The line format is "step:N - key:val - key:val ..." with no spaces in values.
metric_of() {
  local line="$1" key="$2" val
  val="$(printf '%s\n' "$line" | grep -aoE "${key}:[^ ]+" | head -1 | sed "s|^${key}:||")"
  if [[ -n "$val" ]]; then printf '%s' "$val"; else printf 'NA'; fi
}

gpu_max_used_mib() {
  local v max=0 seen=0
  while IFS= read -r v; do
    v="${v//[[:space:]]/}"
    [[ "$v" =~ ^[0-9]+$ ]] || continue
    seen=1
    (( v > max )) && max="$v"
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  # An unreadable nvidia-smi must read as NOT drained, never as drained.
  if (( seen == 0 )); then echo 999999; else echo "$max"; fi
}

append_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"
}

# True iff the TSV already carries an OK row for this arm, so a restarted
# driver resumes instead of re-burning finished arms.
arm_done() {
  local name="$1"
  [[ -f "$TSV" ]] || return 1
  awk -F'\t' -v n="$name" '$1 == n && $8 == "OK" {found=1} END {exit found ? 0 : 1}' "$TSV"
}

# The arm launcher lives INSIDE the checkout it re-fetches. When an arm's
# re-fetch fails transiently, the launcher moves the checkout aside and dies
# mid-clone; without this restore every later arm would fail in milliseconds
# on a missing file with nothing left to heal it. A restore failure fails
# only THIS arm; the next arm retries.
restore_checkout() {
  [[ -f "$LAUNCHER" ]] && return 0
  progress "launcher missing at $LAUNCHER; re-cloning $PROBE_BRANCH"
  if [[ -e "$WORK/verl" ]]; then
    mv "$WORK/verl" "$WORK/verl.broken.$(date +%s)" 2>/dev/null || return 1
  fi
  git clone --depth 1 --single-branch -b "$PROBE_BRANCH" "$REPO_URL" "$WORK/verl" >> "$MASTER_LOG" 2>&1 \
    && [[ -f "$LAUNCHER" ]]
}

run_arm() {
  local name="$1" vec="$2" arm_to="${3:-$ARM_TIMEOUT_S}" rc dur line
  local armlog="$WORK/runs/98-wedge-${name}/train.log"
  local t0=$SECONDS

  if ! restore_checkout; then
    progress "ARM $name FAIL: launcher missing and re-clone failed (network?)"
    append_row "$name" "NA" "NA" "NA" "NA" "NA" "NA" "FAIL_NOLAUNCHER"
    return 1
  fi

  # Rotate any previous arm log so a stale metrics line can never be read as
  # this invocation's result (and so nothing tee-truncates it mid-run).
  if [[ -f "$armlog" ]]; then
    mv "$armlog" "${armlog}.prev.$(date +%s)" 2>/dev/null || true
  fi

  progress "ARM $name START vector=$vec timeout=${arm_to}s log=$armlog"

  # Subshell so the arm env cannot leak into later arms or the driver.
  (
    export RUN_ID="98-wedge-${name}"
    export PROJECT_NAME="98-wedge-probe"
    export EXPERIMENT_NAME="98-wedge-${name}"
    export WANDB_RUN_GROUP="98-wedge-probe"
    export COMM_EFF_MASK_P_BY_BOUNDARY="$vec"
    export TOTAL_TRAINING_STEPS=1
    export VAL_BEFORE_TRAIN=False
    # -1 and not a huge positive number: ray_trainer fires BOTH the checkpoint
    # save and validation on is_last_step whenever the freq is positive, and
    # step 1 IS the last step here. A positive SAVE_FREQ would therefore write
    # one full 8B FSDP checkpoint (roughly 98 GB) per arm with no R2 sweeper,
    # and a positive TEST_FREQ would run a full val pass per arm. -1 disables
    # both outright and is the engine's own documented default.
    export SAVE_FREQ=-1
    export TEST_FREQ=-1
    export CKPT_R2_ENABLED=false
    export MAX_RESPONSE_LENGTH=3072
    export MAX_MODEL_LEN=4096          # MAX_PROMPT_LENGTH(1024) + MAX_RESPONSE_LENGTH(3072)
    export MIN_DISK_GIB=150
    export EXPECT_GPUS=4
    export MIN_RAM_GIB=1200
    export BRANCH="$PROBE_BRANCH"
    export LOG="$armlog"
    exec timeout -k 60 "$arm_to" bash "$LAUNCHER"
  ) >> "$MASTER_LOG" 2>&1
  rc=$?
  dur=$(( SECONDS - t0 ))

  # A timeout kills only the engine SHELL: the engine backgrounds the trainer
  # (python3 ... > "$LOG" & then wait), so on TERM the trainer survives as an
  # orphan that still holds the GPUs and still appends to the arm log. Drain
  # (which escalates to ray stop and a pid-exact kill) BEFORE reading the
  # verdict, so the row reflects the log's final state and the orphan cannot
  # bleed into the next arm.
  if (( rc == 124 || rc == 137 )); then
    progress "ARM $name TIMEOUT rc=$rc after ${dur}s; trainer may be orphaned, draining before the verdict"
    drain_gpus "$name" || true
  fi

  # rc is recorded but NEVER used to judge success: rc=1 at exit is the
  # normal atexit teardown race. The step-1 metrics line is the verdict.
  line="$(grep -a 'rollout_corr/kl:' "$armlog" 2>/dev/null | tail -1)"

  local kl k3 gn mr sc rl status
  if [[ -n "$line" ]]; then
    kl="$(metric_of "$line" 'rollout_corr/kl')"
    k3="$(metric_of "$line" 'rollout_corr/k3_kl')"
    gn="$(metric_of "$line" 'actor/grad_norm')"
    mr="$(metric_of "$line" 'comm_eff/mask_ratio')"
    sc="$(metric_of "$line" 'critic/score/mean')"
    rl="$(metric_of "$line" 'response_length/mean')"
    status="OK"
  else
    kl="NA" k3="NA" gn="NA" mr="NA" sc="NA" rl="NA" status="FAIL"
  fi
  append_row "$name" "$kl" "$k3" "$gn" "$mr" "$sc" "$rl" "$status"
  progress "ARM $name DONE rc=$rc dur=${dur}s status=$status rollout_corr/kl=$kl"
}

drain_gpus() {
  local name="$1" waited=0 max
  while :; do
    max="$(gpu_max_used_mib)"
    if (( max < DRAIN_LIMIT_MIB )); then
      progress "drain OK after ${waited}s (max GPU used ${max} MiB)"
      return 0
    fi
    (( waited >= DRAIN_CAP_S )) && break
    sleep "$DRAIN_POLL_S"
    waited=$(( waited + DRAIN_POLL_S ))
  done
  progress "drain cap ${DRAIN_CAP_S}s hit (max GPU used ${max} MiB), forcing ray stop"
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >> "$MASTER_LOG" 2>&1
  else
    progress "WARN: ray not on PATH, cannot force stop"
  fi
  sleep 60
  max="$(gpu_max_used_mib)"
  if (( max < DRAIN_LIMIT_MIB )); then
    progress "drain OK after ray stop (max GPU used ${max} MiB)"
    return 0
  fi
  # ray stop misses the orphaned main_ppo driver and any vLLM engine cores it
  # spawned. Escalate by killing the EXACT pids still holding GPU memory
  # (functional targeting from nvidia-smi, never name matching, so wandbsync
  # and sshd are untouchable here). On some container images nvidia-smi hides
  # the per-process list; then this finds nothing, so also reap the one known
  # orphan shape by its module path, a literal that appears in no other
  # tenant's command line and not in this driver's own.
  local pid pids
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -aoE '[0-9]+' | sort -u)"
  if [[ -n "$pids" ]]; then
    progress "GPU still pinned; killing GPU-holding pids: $(tr '\n' ' ' <<< "$pids")"
    while IFS= read -r pid; do
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      [[ "$pid" == "$$" ]] && continue
      kill -9 "$pid" 2>/dev/null || true
    done <<< "$pids"
  fi
  if command -v pkill >/dev/null 2>&1; then
    pkill -9 -f 'verl\.trainer\.main_ppo' 2>/dev/null || true
  fi
  sleep 30
  max="$(gpu_max_used_mib)"
  if (( max < DRAIN_LIMIT_MIB )); then
    progress "drain OK after pid-exact kill (max GPU used ${max} MiB)"
    return 0
  fi
  progress "DRAIN_STUCK after arm $name (max GPU used ${max} MiB), continuing anyway"
  append_row "$name" "NA" "NA" "NA" "NA" "NA" "NA" "DRAIN_STUCK"
  return 1
}

final_report() {
  {
    echo "==== raw TSV: $TSV ===="
    cat "$TSV"
  } | tee -a "$MASTER_LOG"
  python3 - "$TSV" <<'PY' 2>&1 | tee -a "$MASTER_LOG"
import sys

path = sys.argv[1]
with open(path) as f:
    lines = [ln.rstrip("\n") for ln in f if ln.strip()]

rows = [ln.split("\t") for ln in lines[1:]]
rows = [r for r in rows if len(r) == 8]

def fval(s):
    try:
        return float(s)
    except ValueError:
        return None

ok = {}
for r in rows:
    if r[7] == "OK" and fval(r[1]) is not None:
        ok[r[0]] = fval(r[1])  # a rerun's later row wins

print("==== arms sorted by rollout_corr/kl (OK arms only) ====")
for name, kl in sorted(ok.items(), key=lambda kv: kv[1]):
    print(f"  {name:>5}  rollout_corr/kl={kl}")
failed = sorted({r[0] for r in rows} - set(ok))
if failed:
    print(f"  (no OK row for: {', '.join(failed)})")

singles = ["s4", "s9", "s14", "s19", "s23", "s27", "s31"]
have = [s for s in singles if s in ok]
print("==== additivity raw numbers (interpretation left to the reader) ====")
for s in singles:
    print(f"  {s:>4}: {ok.get(s, 'MISSING')}")
if len(have) == len(singles):
    print(f"  sum of the 7 singles: {sum(ok[s] for s in singles)}")
else:
    print(f"  sum of the 7 singles: MISSING ({len(have)}/7 available)")
print(f"  ref7 (all 7 compressed): {ok.get('ref7', 'MISSING')}")
print(f"  alt4 (cuts 4,14,23,31): {ok.get('alt4', 'MISSING')}")
print(f"  alt3 (cuts 9,19,27): {ok.get('alt3', 'MISSING')}")
print("  note: no dense-floor arm in this sweep, raw numbers only")

if len(have) >= 3:
    best3 = sorted(have, key=lambda s: ok[s])[:3]
    idx = {s: i for i, s in enumerate(singles)}
    vec = ["0.0"] * 7
    for s in best3:
        vec[idx[s]] = "0.95"
    layers = ", ".join(s[1:] for s in sorted(best3, key=lambda s: idx[s]))
    print("==== suggested best-3-compressed set (3 lowest single-cut rollout_corr/kl) ====")
    print(f"  arms: {', '.join(best3)}  (cuts after layers {layers})")
    print(f"  confirmation vector: COMM_EFF_MASK_P_BY_BOUNDARY=[{','.join(vec)}]")
else:
    print(f"==== only {len(have)} OK single-cut arms, no best-3 suggestion ====")
PY
}

# ---------------------------- preflight -------------------------------------
mkdir -p "$WORK/runs" || { echo "FATAL: cannot create $WORK/runs" >&2; exit 1; }
touch "$MASTER_LOG" || { echo "FATAL: cannot write $MASTER_LOG" >&2; exit 1; }

progress "driver start pid=$$ arms=${#ARMS[@]} branch=$PROBE_BRANCH timeout=${ARM_TIMEOUT_S}s/arm"

restore_checkout || {
  progress "FATAL: launcher $LAUNCHER not found and cloning $PROBE_BRANCH failed"
  exit 1
}
command -v timeout >/dev/null 2>&1 || { progress "FATAL: coreutils timeout not on PATH"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { progress "FATAL: nvidia-smi not on PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { progress "FATAL: python3 not on PATH"; exit 1; }
# The launcher's aws preflight is unconditional even with CKPT_R2_ENABLED=false,
# so fail once here instead of ten times below.
command -v aws >/dev/null 2>&1 || {
  progress "FATAL: aws not on PATH and the launcher's aws gate does not consult CKPT_R2_ENABLED"
  exit 1
}

if [[ ! -f "$TSV" ]]; then
  printf 'arm\trollout_corr_kl\trollout_corr_k3_kl\tactor_grad_norm\tcomm_eff_mask_ratio\tcritic_score_mean\tresponse_length_mean\tstatus\n' > "$TSV"
fi

# ---------------------------- main loop -------------------------------------
ARMS_RUN=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r ARM_NAME ARM_VEC <<< "$arm"
  if arm_done "$ARM_NAME"; then
    progress "ARM $ARM_NAME already has an OK row in $TSV, skipping (rm the TSV row to force a rerun)"
    continue
  fi
  # The first arm this invocation actually launches also pays the model pull,
  # the parquet prep and the first vLLM boot, so it gets the longer cap.
  if (( ARMS_RUN == 0 )); then ARM_TO="$ARM_TIMEOUT_FIRST_S"; else ARM_TO="$ARM_TIMEOUT_S"; fi
  ARMS_RUN=$(( ARMS_RUN + 1 ))
  run_arm "$ARM_NAME" "$ARM_VEC" "$ARM_TO"
  drain_gpus "$ARM_NAME" || true
done

final_report
progress "sweep complete, results in $TSV"
