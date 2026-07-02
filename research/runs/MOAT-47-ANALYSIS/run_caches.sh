#!/bin/bash
# EXP-47 cache builds: regime T (per-tick band-80) then regime S (per-step band-60 + paper_linear).
# Sequential (SAFE default per plan) — each ring buffer is tens of GB. Runs under nohup on box 43511290.
set -u
cd /workspace/verl/research
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python3
LOG=runs/MOAT-47-ANALYSIS/analysis.log
TRACE=/workspace/trace/EXP-57

echo "=== [analyst] STEP 2 REGIME-T (per-tick, ext-delta, band-80) START $(date -u) ===" >> "$LOG"
$PY scripts/moat_scorecard.py --trace-root "$TRACE" \
    --cadence per-tick --n-ticks 160 \
    --method hold_stale,naive_linear,damped_linear \
    --lam-grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --delta 5,10,20,25,35,40 --h 1,2,5,10,20,30,40 \
    --operating-point 20,20 --also 10,10 \
    --out runs/MOAT-47-ANALYSIS/scorecard-pertick/ --ram-gb 48 >> "$LOG" 2>&1
echo "=== [analyst] STEP 2 REGIME-T END rc=$? $(date -u) ===" >> "$LOG"

echo "=== [analyst] STEP 3 REGIME-S (per-step=global-step, +paper_linear, band-60) START $(date -u) ===" >> "$LOG"
$PY scripts/moat_scorecard.py --trace-root "$TRACE" \
    --cadence per-step --n-ticks 80 \
    --method hold_stale,naive_linear,damped_linear,paper_linear \
    --lam-grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --delta 5,10,20 --h 1,2,5,10,20,30,40 \
    --paper-anchor-frac 0.25 --paper-stride 2 \
    --operating-point 10,10 --also 5,5 \
    --out runs/MOAT-47-ANALYSIS/scorecard-perstep/ --ram-gb 48 >> "$LOG" 2>&1
echo "=== [analyst] STEP 3 REGIME-S END rc=$? $(date -u) ===" >> "$LOG"

echo "=== [analyst] BOTH REGIMES DONE $(date -u) ===" >> "$LOG"
