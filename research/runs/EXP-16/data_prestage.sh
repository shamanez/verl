#!/usr/bin/env bash
# EXP-16 data pre-stage — runs IN PARALLEL with the env install. Pure network →
# HF cache only (NO pip, so it can't corrupt the concurrent install's
# site-packages). Downloads the Qwen training model + raw GSM8K dataset into the
# shared HF cache on the big disk. Writes /workspace/data.DONE when finished.
set +e
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_PROGRESS_BARS=1
source /root/.config/verl-research/secrets.env 2>/dev/null
LOG=/workspace/data_download.log
: > "$LOG"
exec >> "$LOG" 2>&1
echo "==== EXP-16 data prestage start $(date -u +%FT%TZ) ===="
echo "HF_HOME=$HF_HOME ; hf=$(command -v hf)"

echo "-- download model Qwen/Qwen2.5-1.5B-Instruct --"
hf download Qwen/Qwen2.5-1.5B-Instruct
echo "model_download_rc=$?"

echo "-- download dataset openai/gsm8k (main) --"
hf download openai/gsm8k --repo-type dataset
echo "dataset_download_rc=$?"

echo "-- HF cache size --"
du -sh "$HF_HOME" 2>/dev/null
echo "-- model snapshot files --"
find "$HF_HOME/hub" -name '*.safetensors' 2>/dev/null | head
echo "==== EXP-16 data prestage done $(date -u +%FT%TZ) ===="
touch /workspace/data.DONE
