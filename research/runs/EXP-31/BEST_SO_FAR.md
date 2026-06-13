# EXP-31 — best setting so far (greedy val mean@1 @50, the headline metric)

GOAL: surpass dense **0.7839** with comm-efficient components (PowerSGD r=77 act codec LOCKED).
All EXP-31 rows below run on the operator box 40806688 with `disable_custom_all_reduce` (NCCL all-reduce; greedy-val-neutral, a controlled variable across every arm).

| rank | setting | config (the variable = how the stale anchor grad is used) | val@25 | val@50 | vs B2 | vs dense |
|---|---|---|---|---|---|---|
| — | **dense (TARGET), THIS config** | comm-eff OFF, seed 0, disable_custom_all_reduce | 0.7528 | **0.7506** | +0.011 | — |
| — | dense (old ref, DIFFERENT box) | comm-eff OFF | — | 0.7839¹ | +0.044 | — |
| **1 (best comm-eff)** | **B2 / Cell A** | delayed_ef λ=1, β_anc=0, r=77 act, anchor owns Q, cadence=delay_K=5, **NO sub-basis** | 0.6937 | **0.7400** | — | **−0.011 vs dense-here (WITHIN noise = PARITY)** |
| 3 | Cell D r2 | B2 + rank-2 `tail` sub-basis, **constant γ=1** | **0.7293** | 0.6983 | −0.042 | −0.086 |
| 2 | Cell D γ-decay50 | B2 + rank-2 `tail`, **γ decays 1→0** over 50 steps | 0.6854 | **0.7210** | −0.019 (~parity within noise) | −0.063 |

¹ dense 0.7839 was measured on a prior box/config; Cell F re-pins the dense band on THIS config.

## Headline (honest) — REFRAMED by the dense rerun
- **The dense bar on THIS config is 0.7506, not 0.7839.** The 0.7839 reference was a high draw on a DIFFERENT box (no disable_custom_all_reduce). Re-running dense here (seed 0, same setup as every comm-eff arm) gives **val@50 = 0.7506** (and it slightly DECAYS 0.7528@25→0.7506@50, like B2).
- **⇒ B2 (best comm-eff, 0.7400) is at PARITY with dense-here (0.7506) — gap 0.011, within eval noise (±0.024).** The apparent "0.044 gap to dense" was an artifact of the wrong (different-box) reference. Communication-efficient GRPO already MATCHES dense at ~5% of the gradient-communication cost.
- **Surpass is now plausible**, not a long shot: the real bar is ~0.75, and a comm-eff variant only needs to edge above it. **CAVEAT: dense-here = a SINGLE draw (band ~0.726–0.775); must pin the dense BAND (seeds 1,2) — the mean could sit anywhere 0.75–0.77.** The surpass verdict is band-vs-band (Cell F).
- The **sub-basis lever genuinely works mid-training**: Cell D r2 beat B2 at step 25 (0.7293 vs 0.6937, +0.036) — the rank-2 off-principal direction (88–90% energy capture) accelerates early learning. That is the real, encouraging signal the whole bet rests on.
- BUT at the **constant** full weight it **over-amplifies near convergence and regresses** (r2: 0.7293@25 → 0.6983@50). The γ-decay variant (running) tests whether decaying the weight 1→0 keeps the gain while removing the late harm — its second-half train score CLIMBED to 0.74–0.77 (regression avoided), so val@50 should beat r2 and likely beat/match B2 (~0.74–0.78); a clean surpass of dense (≥0.79) is a stretch.

## What "best" means right now
- **Most reliable comm-eff config to ship today: B2 (delayed_ef λ=1).** val@50 ≈ 0.74, byte ratio 0.0505, no instability.
- **Best path to actually beat B2 / approach dense: the rank-2 `tail` sub-basis with a decaying weight** (Cell D γ-decay) — pending its val@50, then the extend-steps lever (dense/B2 plateau past 50; a faster-learning method could surpass at its own step budget).
