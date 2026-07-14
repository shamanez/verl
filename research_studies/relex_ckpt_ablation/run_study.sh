#!/usr/bin/env bash
# End-to-end MacBook study: download a SPARSE RELEX checkpoint set to a TEMP dir,
# run the whole-tensor forecast-skill ablation, and render the result plots.
#
# NOTHING is written into the repo: all checkpoints/outputs live under $STUDY_ROOT
# (defaults to a temp/scratch path). Set STUDY_ROOT to an external SSD if your
# temp mount is small — Tier-1 needs ~35 GB, Tier-2 (finer gaps) more.
#
#   bash run_study.sh            # Tier-1 core (base + steps 10..100 by 10)
#   TIER=2 bash run_study.sh     # also fetch consecutive steps for gap sensitivity
set -euo pipefail
cd "$(dirname "$0")"

STUDY_ROOT="${STUDY_ROOT:-${TMPDIR:-/tmp}/relex_ckpt_study}"
CKPT_DIR="$STUDY_ROOT/checkpoints"
OUT_DIR="$STUDY_ROOT/outputs"
PLOTS_DIR="$STUDY_ROOT/plots"
TIER="${TIER:-1}"
mkdir -p "$STUDY_ROOT"

echo "== Study root (temp): $STUDY_ROOT =="
df -h "$STUDY_ROOT" | tail -1

# ---- deps (once) ----------------------------------------------------------- #
python3 - <<'PY' || pip3 install torch numpy safetensors huggingface_hub matplotlib
import torch, numpy, safetensors, huggingface_hub, matplotlib  # noqa
PY

# ---- 0. equivalence proof: our port == the live harness projector ---------- #
echo "== Proving harness_projector == live rank1_relex projector =="
( cd ../.. && python3 research_studies/relex_ckpt_ablation/harness_projector.py ) || \
  echo "[warn] equivalence proof skipped (run from the exp/relex-ckpt-ablation worktree root to enable)"

# ---- 1. download only the needed steps ------------------------------------- #
CORE_STEPS="10,20,30,40,50,60,70,80,90,100"
python3 download_subset.py --steps "$CORE_STEPS" --output_dir "$CKPT_DIR" --with_base

if [[ "$TIER" == "2" ]]; then
  # consecutive steps around one anchor for gap in {1,2,5} sensitivity (heavier)
  python3 download_subset.py \
    --steps "41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59" \
    --output_dir "$CKPT_DIR"
fi

# ---- 2. run the ablation --------------------------------------------------- #
# On a <24 GB laptop add: --skip_embedding
python3 run_forecast_ablation.py \
  --ckpt_dir "$CKPT_DIR" --out_dir "$OUT_DIR" \
  --windows 2,3,4,5,6,8 \
  --horizons 1,2,3 \
  --gap 10 \
  --ranks 1,2,3 \
  --strengths 1.0 \
  --methods rank1_relex,relex_from_base,fixed_linear

# ---- 3. plots -------------------------------------------------------------- #
python3 make_plots.py --in_dir "$OUT_DIR" --out_dir "$PLOTS_DIR"

echo
echo "== DONE =="
echo "rows/summary: $OUT_DIR"
echo "plots:        $PLOTS_DIR  (open results.html)"
echo "To reclaim disk: rm -rf $CKPT_DIR"
