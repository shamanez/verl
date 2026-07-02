# EXP-47 verdict — MOAT ANCHOR linear / damped-linear projection lane

VERDICT: PASS
date: 2026-07-03
kind: analysis (GPU-free replay over the EXP-57 fp32 trace; code_change=true, additive under research/scripts/)
box: EXTERNAL 43511290 (operator-managed, never provisioned/torn-down by this issue)

## Why PASS
Every correctness + completeness gate holds in BOTH regimes, the report renders offline,
and all six decisive questions are answered (direction-agnostic — a negative direction is
still a PASS). The science direction here is **favorable**: OOS-damped linear projection
beats both references at the operating point.

## Lead — per-scalar linearity R² (the MUST metric; Wang et al. 2026 arXiv:2601.04537)
- **Regime S (per-step, PAPER-COMPARABLE): R² median = 0.535, Pr(R²>0.7) = 0.335, n_excluded_const = 4,227,167.**
- Regime T (per-tick): R² median = 0.535, Pr(R²>0.7) = 0.337, n_excluded_const = 4,220,204.
- Placement: our GSM8K GRPO run lands **between** the Wang SFT-on-GSM8K floor (0.426 / 0.259)
  and the nearest RL analog R1-Distill-Qwen-1.5B GRPO/DeepScaleR (0.845 / 0.794) — moderate
  linearity, clearly above SFT, well below the strong-RL analog. This answers the plan's open
  question (it could have landed anywhere in [0.43, 0.85]): **~0.535, moderate.**
- **R²-vs-ratio coupling (the steering signal): Spearman ρ = −0.75 (per-step), −0.67 (per-tick)
  over 43 groups** — high-R² groups have LOWER OOS-damped ratio, i.e. they project better.
  A strong, decisive coupling that future MOAT optimization can steer by.

## Decisive answers
1. **Operating point (PRIMARY, per-step Δ=10,h=10 global steps).** GLOBAL median weight_proj_ratio:
   hold_stale = 1.000 (identity), naive_linear = **1.158 (HURTS)**, OOS-**damped_linear = 0.940**
   with λ*=0.3. Damped **beats naive AND beats hold-stale — projection HELPS.** Per-tick (Δ=20,h=20):
   naive = 1.158, damped = 0.940.
2. **Does wider Δ help? (per-tick extended sweep Δ∈{5,10,20,25,35,40}).** NO. best_delta = **5**
   (the smallest). The ratio rises monotonically with Δ (Δ5=0.871, Δ10=0.906, Δ20=0.940,
   Δ25=0.951, Δ35=0.967, Δ40=0.971 at h=20). Wider anchors monotonically HURT — decisive
   negative. Regime S agrees (best_delta=5 over {5,10,20}).
3. **h_safe** (max h with GLOBAL OOS-damped median < 1.0): **30 global steps** (per-step),
   **40 ticks** (per-tick, the whole grid). Naive by contrast is safe only to h=2 (per-step) /
   h=5 (per-tick). OOS-damped projects ~15× farther than naive.
4. **Breakers: NONE.** Every block_type / super_block / layer / special group has OOS-damped
   ratio < 1.0 at the operating point in the per-step regime (no group breaks the global
   conclusion; regimes agree).
5. **λ-selection.** OOS λ* distribution (per-step, @op, over groups): {0.1:1, 0.2:142, 0.3:437,
   0.4:46, 0.5:10, 0.6:2}. The selector picks **moderate damping (~0.2–0.4)** — it does NOT
   collapse to λ=0 (hold-stale) or λ=1 (naive). OOS λ* == in-sample oracle λ (0.3) at the
   operating point — no honesty gap there.
6. **Paper-protocol readout (Wang §6.2 Eq.4 / App. E.1).** paper_linear (wide proportional
   window t0=⌊0.25t⌋, Δ_resolved≈0.75·t, β∈[1.01, 3.66]) is WORSE than fixed-short-Δ naive at
   short h (h=1: paper 1.009 vs naive 0.888) but CROSSES OVER to beat naive by h≈10 (h=10: paper
   1.111 < naive 1.158; h=20: 1.271 < 1.404) — a long-window noise-averaged slope helps at long
   horizons. **OOS-damped beats BOTH at every h.** β climbs monotonically with h (Fig. 5-style):
   moderate β helps relative to naive, larger β still climbs past 1 under the hold-stale baseline
   (the paper's benefit is vs more-training, a different comparator — stated in the report).

## Correctness + completeness gates (all PASS)
- SELFTEST: GO on the box (18 invariants incl. damped λ1==naive / λ0==hold_stale in BOTH regimes,
  cadence-reindex, OOS leakage guard fires, damped off-path parity 0.0, per-scalar R² off-path
  parity ≤4e-16 + [0,1] bounds + constant exclusion, paper anchor-rule + cross-path parity 4e-14,
  real-trace hold_stale identity 0.0 + off-path parity 0.0 + determinism byte-identical).
- Regime T (per-tick, band-80, Δ→40): EMIT: GO. Regime S (per-step, band-60, +paper_linear): EMIT: GO.
- SCHEMA: GO for BOTH dirs on the BOX and again on the LAPTOP after rsync (portability). Rows carry
  the 29 base keys + the #47 superset (cadence/unit, anchor_mode/delta_resolved/beta, r2_median/
  r2_frac_gt_0.7/n_excluded_const, lam_star). Full taxonomy present: global/338 matrix/10 block_type/
  5 super_block/28 layer/252 layer_block/4 special (lm_head tied=true).
- paper_linear rows present for every h at the strided anchor set (frac25, per-window β aggregated).
- Soft cross-check: per-tick naive @(Δ=20,h=20) = 1.1580 ≈ #45's 1.158 (Δ ≤ 0.01) — the band-80
  rebuild did not perturb the shared cell.
- report.html renders fully offline (84 KB, 8 inline SVG, ZERO <script>/<img>/external refs); leads
  with the R² readout vs the Wang anchors + histogram + depth×block R² heatmap + R²-vs-ratio coupling,
  then both regimes side-by-side, then the paper-equivalence panel, then the verdict.
- Code is ADDITIVE under research/scripts/ only (moat_scorecard.py + new moat_report.py); no verl/
  source, no shared metric-math/taxonomy/tick_select edits.

## Artifacts
- runs/MOAT-47-ANALYSIS/scorecard-perstep/  (regime S: scorecard.jsonl 50 MB, visuals.json, meta.json)
- runs/MOAT-47-ANALYSIS/scorecard-pertick/  (regime T: scorecard.jsonl 90 MB, visuals.json, meta.json)
- runs/MOAT-47-ANALYSIS/report.html         (self-contained; opens offline)
- runs/MOAT-47-ANALYSIS/analysis.log, selftest.log
- stats caches persist ON THE BOX (scorecard-*/stats_cache.npz) as a shared asset for #48/#49/#56.

## Feeds #56
Structured, schema-verified block/layer/special tables in both cadence units + the linearity-R²
block + the paper-equivalence panel — ready for the MOAT verdict rollup.
