# B2 — the comm-eff SOTA baseline (migrated from EXP-30, 2026-06-15)

These are the **ground-truth settings of the current SOTA** (`B2` = `delayed_ef` λ=1, β_anc=0;
val@50 ≈ 0.74–0.75 = parity with dense at ~5% gradient-comm cost). Every EXP-31 cell — and future
experiments — holds this substrate FIXED and varies ONLY how the stale anchor gradient is used.

**To REPRODUCE B2 on a box:** run `examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh`
(the self-contained, authoritative baseline launcher — pins exactly the knobs in
`resolved_params_B2.txt`). The issue-#31 anchor-usage levers are env overrides on top of it.

- `resolved_params_B2.txt` — the canonical knob set (the substrate; ground truth, not prose).
- `launch_B2.sh` — the launcher that produced B2.
- `verdict.md` — the EXP-30 6-run one-knob decomposition + findings F1–F5 (the analysis that shaped the levers).
- `metrics/` — B2 baseline metric data (for diff_against_baseline). Canonical W&B run: `u9okvgzz`.
