# EXP-38 · big-math arm — PLACEHOLDER (not yet run)

This directory is the **pre-named home** for the future **Big-Math** drift-probe data
(`gshasiri/Big-Math-RL-Verified-filtered`), the sibling of the `../gsm8k/` arm.

**Why it exists now:** so when the Big-Math run happens (in a NEW session), its data has an
unmistakable destination that can NEVER be confused with the GSM8K data.

## What to do (in a new session)

1. Run the SAME 75-step dense GRPO temporal-drift + boundary probe as GSM8K, but on Big-Math
   (data source `gshasiri/Big-Math-RL-Verified-filtered`). Reuse the `exp/38-dense-drift-probe`
   instrumentation (already merged-ready) — only the dataset changes.
2. Download ALL its artifacts into THIS directory, mirroring `../gsm8k/`:
   ```
   big-math/
     captures/rank0/manifest.jsonl + tick_<gs>_<tick>/<role>/*.pt   (.pt + nested captures/ are gitignored)
     captures/sidecar_layernorms.jsonl
     sidecar_grpo.jsonl
     train.log
     DATASET.json   (already here — dataset=big-math)
   ```
3. Analyze (dataset auto-detected from this `DATASET.json`):
   ```
   python3 research/scripts/exp38_drift_analysis.py research/runs/EXP-38/big-math
   #  -> research/reports/comm-eff-grpo/exp38-dense-drift-big-math.html
   ```
4. Joint/comparative report: once BOTH `../gsm8k/` and `big-math/` exist, produce a combined report
   with the two datasets in clearly-separated, dataset-tagged sections.

## HARD RULE (operator)

**Never mix the two datasets.** GSM8K data lives ONLY in `../gsm8k/`; Big-Math data lives ONLY here.
No merging of tensors or drift curves across datasets — ever.
