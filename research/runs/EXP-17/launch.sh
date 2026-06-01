#!/usr/bin/env bash
# EXP-17 — masked p=0.9 + clean_cadence=20, 2-epoch characterization.
# Runs inside the Vast.ai container (instance 38877541, 4x H200).
#
# code_change=true but target_modules=[] => NO method patch is shipped. The
# heart of this run is pure-config: every knob already exists in the canonical
# launcher (read via ${VAR:-default}). We therefore run the box's existing
# /workspace/verl tree (on branch vast-ai-workload) directly — we do NOT ship
# or apply an exp.bundle (there is no code delta to apply, and re-cloning a
# marker-only branch would needlessly replace the box's working tree). The
# exp/17-masked-clean-every20 branch exists+pushed purely as a crash-survival
# anchor; any ad-hoc on-box instrumentation is captured via commit-hotfix.sh.
set -euo pipefail

# Configure git identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

cd /workspace/verl

# PREFER THE CANONICAL LAUNCHER + ENV/HYDRA OVERRIDES (VAST_README.md §"Stability contract").
# This scenario maps to the comm-eff launcher; we override ONLY the knobs this cell varies via
# its ${VAR:-default} env vars. The launcher runs under `set -x`, so train.log records the
# fully-expanded main_ppo command — the analyst extracts that into resolved_params.txt.
#
# Token budget: max_token_len=98304 (the three *_MAX_TOKEN_LEN_PER_GPU knobs) is the EXP-16-proven
# perf setting (MFU 0.75%->13.86%, step 129s->37s, peak ~62 GB/GPU). anchor+spectral are OFF here,
# so no ~3 GB anchor clone / spectral GPU-EMA-SVD alloc; 4x H200 >=140 GB fits with >2x headroom.
# These three knobs are ADDITIONAL overrides prepended to the verbatim issue env block (which is
# preserved unchanged below). NGPUS_PER_NODE auto-detects to 4 via `nvidia-smi -L`.
PPO_MAX_TOKEN_LEN_PER_GPU=98304 \
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=98304 \
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=98304 \
PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=grpo_mask_channel_p0p9_rescale_clean_every20_2epoch \
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 \
COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=20 COMM_EFF_ANCHOR_ENABLED=false COMM_EFF_SPECTRAL_ENABLED=false \
TOTAL_EPOCHS=2 TOTAL_TRAINING_STEPS=116 TEST_FREQ=10 VAL_BEFORE_TRAIN=True USE_DYNAMIC_BSZ=True \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  > /workspace/runs/EXP-17/train.log 2>&1

echo "$(date -Iseconds) done" > /workspace/runs/EXP-17/done.flag
