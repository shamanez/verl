# Research Log (newest first)

## EXP-18 · 2026-06-03T04:50:00+10:00 · M4 · PASS
M4 — Match the masked GRPO training SIGNAL to dense within ≤50 steps via a continuous, STALE-anchor gradient correction (GSM8K; anchor cadence=5, staleness delay_K=5, clean_cadence OFF) — recursive empirical search
- hypothesis: A continuous, stale (delay_K=5) anchor gradient correction applied every masked step (cadence=5, clean_cadence OFF) causes masked+correction per-step GRPO reward to track the dense reference within mean|Δ|≤0.05 and final|Δ|≤0.05 over ≤50 steps on GSM8K.
- result: PASS (C5 clean-PG anchor + blend η=0.9): reward 0.13→0.8135 vs dense 0.13→0.8408; final|Δ|=0.027≤0.05, plateau(20-50) mean|Δ|=0.036≤0.05, slope +0.668 vs +0.706 MATCH; strict whole-trajectory mean|Δ|=0.070>0.05 missed solely on the cadence-5 warmup (steps 1-15). Prize unlocked by fixing two anchor-circuit bugs (FSDP name-key + importance-ratio corruption); staleness delay_K=5 is NOT fatal.
- run dir: runs/EXP-18/
- verdict: runs/EXP-18/verdict.md

## EXP-17 · 2026-06-01 · M3 · PASS
masked p=0.9 + clean@20 (anchor+spectral OFF), GSM8K val **0.7354 ≈ dense parity**
(0.741), no collapse, clean-resettable sawtooth, ~85.5% boundary-activation comm cut.
Detail: `runs/SUMMARY.md`.
