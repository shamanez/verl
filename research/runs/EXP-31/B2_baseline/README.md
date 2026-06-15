# B2 — the comm-eff SOTA baseline (migrated from EXP-30, 2026-06-15)

These are the **ground-truth settings of the current SOTA** (`B2` = `delayed_ef` λ=1, β_anc=0;
val@50 ≈ 0.74–0.75 = parity with dense at ~5% gradient-comm cost). Every EXP-31 cell — and future
experiments — holds this substrate FIXED and varies ONLY how the stale anchor gradient is used.

- `resolved_params_B2.txt` — the canonical knob set (the substrate; ground truth, not prose).
- `launch_B2.sh` — the launcher that produced B2.
- `verdict.md` — the EXP-30 6-run one-knob decomposition + findings F1–F5 (the analysis that shaped the levers).
- `beat_dense/{program,feasibility}.md` — the beat-dense analysis (input to the lever design).
- `metrics/` — B2 baseline metric data (for diff_against_baseline). Canonical W&B run: `u9okvgzz`.
