# EXP-31 — Stale-anchor geometry sizing (Cell ANALYSIS, laptop, non-blocking)

Read of stored EXP-30 artifacts — NOT a re-run, no GPU. There are no raw
δ / G_anchor / Q tensors stored, so a literal δ-SVD is impossible; the equivalent
is computed from the per-fire MEDIAN scalars (`stepA_fires.jsonl`) and the full
per-target distribution (`stepA_fires_targets.jsonl`, 196 targets × 8 fires).
The plan's `geometry_audit.py` is hardwired for EXP-26 raw fp32 dumps and lacks the
`--delta-svd/--q-rotation/--honest-bytes` flags, so the quantities are computed
inline per the plan's §"Verification commands" fallback. Post-warmup = fires 2–8
(fire 1 / step 3 is the cold-fallback window, no m5).

---

## 1. Rank confirmation — stable rank ≈ 2, r_sb = 2 is the right default

Pooled per-target `m7_stable_rank = ‖G‖²_F/‖G‖²₂` over all 7 post-warmup fires
(n = 1372 targets):

| stat | value |
|---|---|
| median | **1.933** |
| mean | 2.111 |
| p25 / p75 | 1.577 / 2.478 |
| p90 / p95 / p99 | 3.083 / 3.507 / 4.552 |
| min / max | 1.079 / 5.946 |

Per-fire median stable rank stays in a tight band **1.774 – 2.054** (matches the
verdict F3 "1.8–2.05"). Fraction of targets with stable rank:

- **≤ 2: 53.6 %**   ≤ 2.5: 75.4 %   ≤ 3: 88.6 %   **≤ 4: 97.7 %**   ≤ 6: 100 %

top-1 % coordinate mass median **0.597** (heavily concentrated; ambient = 1536).

**Verdict: rank ≈ 2 is confirmed.** The median target is stable-rank ~1.9 and over
half sit at ≤ 2. `r_sb = 2` captures the median direction and the bulk of the
distribution. A `r_sb = 4` sub-basis would cover 97.7 % of targets' full stable
rank (vs 53.6 % at r=2), so r=4 is the natural REVISE step if r=2 under-captures —
but it is NOT the launch default: the central tendency is rank-2, and an over-sized
sub-basis risks harvesting noise tail directions (stable rank is a *soft* upper
bound; the energy past direction 2 is small — top-1 % mass already 0.60 at rank 2).
**Launch Cell D at r_sb = 2** (plan default holds); promote to 4 only on a measured
r=2 under-capture.

---

## 2. Off-principal characterization — the act basis misses the true rank-2 direction

The F1 within-pair geometry, recovered algebraically from the stored m5 medians
(δ = G_anc_rep − G_comp_ring, so ‖G_anc_rep‖ and cos(G_anc_rep, G_comp_ring) follow
from r = ‖δ‖/‖G_comp‖ = `m5_ratio` and c = `m5_cos`):

| quantity | this run (median over fires 2–8) | F1 claim |
|---|---|---|
| m5_cos(δ, G_comp_ring) | **−0.950** (range −0.98…−0.72) | −0.92…−0.98 |
| m5_ratio ‖δ‖/‖G_comp_ring‖ | **1.053** (range 1.02…1.39) | ≈ 1.05 |
| derived ‖G_anc_rep‖/‖G_comp_ring‖ | **0.327** | ≈ 0.33 |
| derived cos(G_anc_rep, G_comp_ring) | **0.0002** | ≈ 0 |

This reproduces F1 exactly from the scalars: the true gradient proxy (G_anc_rep) is
~1/3 the magnitude of the compressed gradient and **essentially orthogonal** to it.
The codec error (δ) is ~92 % of the compressed-gradient energy and ⊥ the true
direction. So the true rank-~2 direction lives almost entirely OUTSIDE the
act-spanned G_comp subspace — exactly the off-principal energy a `tail`
(act-deflated grad) sub-basis is built to harvest.

Per-direction lag-projection energies `m4_j1..j5` (median over fires 2–8):
j1 **+0.086**, j2 **+0.200**, j3 **+0.115**, j4 **+0.295**, j5 **+0.169** — all
non-zero (the K-delayed self-correlation survives ≥5 ticks; verdict Q1), so the
stale-anchor signal the sub-basis reads is NOT decohered noise at the K=5 horizon.

**Off-principal share to harvest:** EXP-26's weight-space proxy put act-basis
update-energy capture at ~0.318 → off-principal share **0.682**. This run's
independent cross-check (G_anc_rep ≈ 0.33× G_comp, cos ≈ 0) is fully consistent:
the act basis captures ~1/3 of the update energy, leaving ~2/3 for a tail sub-basis
to recover. **Confirmed, not refined down** — the 0.682 off-principal share stands.

---

## 3. Honest-byte denominator — the standing ~4× (Cell C must beat this, not 19.8×)

Measured forward fast-path ratio `comm_eff/bytes_ratio` = **0.0505** (n=73 ticks,
0.05037–0.05053) ⇒ 19.8× nominal forward savings. **That number is retired** — it
excludes the anchor circuit. Honest amortized inter-stage cost (dense per-tick
boundary transfer = unit 1.0; H=1536, r_act=77, cadence=5, 196 matrices):

| component | per-tick cost vs dense=1.0 | arithmetic |
|---|---|---|
| forward fast path (measured) | 0.0505 | `comm_eff/bytes_ratio` |
| anchor-M DP-all-reduce (full-H, 196 mats, /cadence) | 0.2000 | 1.0 / 5 |
| Q-broadcast (2·r/H, amortized /cadence) | 0.0201 | (2·77/1536) / 5 |
| **total** | **0.2706** | sum |

**Honest amortized inter-stage savings = 1 / 0.2706 = 3.70× ≈ the standing ~4×.**
The anchor-M term (full-rank gradient, amortized only over cadence=5) dominates and
caps realized savings near ~4× regardless of the 19.8× forward codec.

Sensitivity: M=0.5× → 5.9×; M=2× (2-pass all-reduce) → 2.1×. The ~4× figure is the
honest center; **this is the Cell C denominator**, NOT 19.8×.

Cell C leverage: compressing the *correction* δ to r_δ ∈ {16,8} columns costs only
~0.004 / 0.002 per tick — a real win (→ ~13× optimistic) ONLY IF that compressed δ
also *replaces* the full-rank anchor-M traffic. If M stays full-rank, the ~4×
denominator is unchanged and r_δ buys nothing on the honest axis. Cell C's success
therefore hinges on whether the merger can transmit the correction in r_δ columns
*instead of* the full M, not in addition to it.

---

## 4. Verdict line

- **recommended `cell_D_subbasis_rank` = 2** (median stable rank 1.93; 53.6 % of
  targets ≤ 2; top-1 % mass 0.60 already at rank 2). REVISE to 4 only on a measured
  r=2 under-capture (4 covers 97.7 % of the distribution).
- **`cell_C_r_delta` starting point = 16** (then 8). But the honest ~3.7× denominator
  is only beaten if the compressed δ *replaces* the full-rank anchor-M traffic;
  otherwise Cell C is a NULL on the savings axis by construction.
- **Geometry SUPPORTS the surpass-dense bet directionally but does not guarantee it.**
  The act basis demonstrably misses a ~0.68 off-principal share of a stable-rank-~2
  true direction, and the stale-anchor signal is coherent at K=5 — so there IS a
  rank-2 tail for the Cell D additive sub-basis to harvest. The caution: B2 already
  re-injects that direction via δ and only reaches parity (0.7528 vs dense 0.7839);
  the sub-basis must inject it MORE CLEANLY than B2's drift-limited residual to clear
  the dense band, which the geometry permits but cannot prove. This analysis sets the
  rank; the headline is decided by Cell D + Cell F, not here.
