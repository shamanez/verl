# reports/dense-run-behaviour/

**Dedicated output home for the EXP-38 "dense GRPO temporal-drift / dense-run-behaviour" analysis.**
Everything the analysis produces lands here — one clean folder, nothing else mixed in.

## What goes here (all OUTPUTS)
- `exp38-dense-drift-gsm8k.html` — ARM A (GSM8K) standalone report (+ `…_findings.json`)
- `exp38-dense-drift-big-math.html` — ARM B (Big-Math) standalone report (+ `…_findings.json`)
- `exp38-dense-drift-joint.html` — the GSM8K↔Big-Math comparative report (B4)
- any standalone plot/image files and auxiliary notes/markdown from the analysis

Reports are self-contained (base64-embedded plots, no external assets).

## How to produce them
Follow the cold-start runbook in [`.claude/plans/38.md`](../../.claude/plans/38.md) →
section **"⏭️ NEXT BIG THING — ANALYSIS RUNBOOK (cold-start, self-contained)"**.
Commands write here via `--out reports/dense-run-behaviour/…`.

## Inputs (read-only; never copied here)
- Captures + sidecars: `runs/EXP-38/gsm8k/` and `runs/EXP-38/big-math/` (1071 fp32 tensors each + manifest + GRPO/layernorm sidecars + train.log)
- **Style reference** (input only, stays in its own folder): `../comm-eff-grpo/why-grpo-fails-sft-works.html` (the EXP-37 verdict report)

## Hard rule
**Never mix the two datasets' tensors/curves.** Outputs are dataset-tagged in the filename; the joint
report keeps GSM8K and Big-Math in separate, clearly-labelled panels.
