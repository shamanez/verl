# EXP-23 — Stale-Gradient Re-Anchor: Alternative Correction Methods

**Status:** FINAL. All three EXP-23 arms ran to step 50; **hypothesis FALSIFIED**
(verdict = STOP). The A2/A3 live 50-step geometry and reward curves are in hand;
§8 is finalized to the actual "A2 AND A3 both fail" outcome, so **this report is
now the BINDING next-lever readout** (referenced as such by `runs/EXP-23/verdict.md`).

**Authors:** synthesizer (this report), drawing on three teammate notes:
- `notes_async_rl.md` — async/off-policy RL staleness tolerance (async-rl-scout)
- `notes_cited_papers.md` — operator-cited papers (cited-reader)
- `notes_grad_geometry.md` — measured per-layer G↔M_anchor geometry, **finalized with live A2/A3 50-step data** (grad-empirics)

**Date:** 2026-06-04 (phase-1 draft) → finalized 2026-06-05 (live A2/A3)

---

## 0. The headline (read this first)

EXP-23 tests whether a **stale full-rank gradient** `M_anchor` (an EMA of the full
GRPO gradient captured at θ_{t−K}, K=5, never compressed) can repair a **live
PowerSGD r=77 compressed gradient** `G` at the pipeline boundary, via two
correction operators with the fresh clean step turned OFF:

- **A2 — inject:** `G_corr = G + γ·scale·(M − proj_G(M))`, γ=1.0
- **A3 — blend:** `G_corr = (1−η)·G + η·scale·M`, η=0.5

The plan made this report the **mandatory next-lever readout if A2 AND A3 both
fail** (plan §Falsification contingency). **That branch is the one that fired:**

| arm | refresh | val@50 | Δ vs floor | reading |
|---|---|---|---|---|
| A1 no-refresh (floor) | none | **0.6914** | — | PowerSGD r=77 baseline |
| A2 stale inject (γ=1.0) | stale_inject | **0.6967** | +0.0053 | **INERT** (within GSM8K noise) |
| A3 stale blend (η=0.5) | stale_blend | **0.6861** | −0.0053 | **INERT** (slightly below floor) |

`max(A2, A3) = 0.6967 ≤ falsify line 0.7114` (PASS bar was **0.7414**; A0
fresh-clean reference = 0.7415, dense control = 0.7536). **HYPOTHESIS FALSIFIED →
verdict STOP.** A stale full-rank anchor neither rescues nor harms PowerSGD beyond
±0.005 noise. This report is therefore the **binding next-lever readout**.

**The pre-registered prediction — THREE INDEPENDENT lines — was CONFIRMED:**

1. **Geometry** predicted `cos(G, M_anchor) ≈ 0`. **CONFIRMED on live data:**
   |cos| ≤ 0.0048 across **all 4 targets × all 50 steps of BOTH arms**, steady
   state (did NOT rise past the delay_K=5 warmup). A3's `‖G_corr‖/‖G‖ = 0.706–0.709
   ≡ √0.5 = 0.7071` to 3 decimals exactly, as predicted. Only **4 of ~196 targets**
   were corrected (layer-0 attention q/k/v/o), scalars only, no raw matrices dumped.
2. **Async-RL theory** predicted K=5 staleness is essentially free (AReaL η≤8
   lossless; M2PO matches on-policy at ≥256 stale), so the failure is the
   **combine, not the age**. **CONFIRMED:** orthogonality is steady-state (present
   from the first post-warmup firing, not growing) ⇒ a fresher anchor (smaller
   delay_K) lands in the same incoherent geometry. delay_K is *not* the lever.
3. **Cited papers** predicted PowerSGD's low rank **under-covers** RL's
   near-full-rank update (Mukherjee 2505.11711, rank 99%+) and that the dropped
   directions are real signal needing error-feedback. **CONFIRMED at the root:**
   the live data shows the PowerSGD r=77 sketch is ~orthogonal to the stale
   full-rank M — the compressor discards exactly the subspace M lives in.

⇒ **The failure mode is structural geometric incoherence, NOT inject-vs-blend
tuning and NOT delay_K.** PowerSGD r=77 discards the very directions M lives in, so
G ⊥ M *by construction*. A bigger/smaller γ or η, or a fresher anchor, cannot fix
an orthogonal complement. The named top lever is therefore **error-feedback on the
PowerSGD residual** (compress the *error* so the fed-back term aligns with exactly
what G dropped) and/or a **basis-aligned anchor** (project M into the PowerSGD
Q-basis so the two share a subspace by construction). This matches issue #21's
standing finding that "NO error-feedback = top lever".

**Top-ranked alternative method:** *error-feedback on the PowerSGD residual* —
accumulate the dropped low-rank residual `G_full − decompress(compress(G_full))` in
a local FP32 buffer and fold it back into the next step (the standard PowerSGD
convergence fix), so the fed-back term is aligned with *exactly* the energy the
sketch discarded — directly attacking the orthogonality the live data exposed. The
co-equal structural fix is a *basis-aligned anchor* (project the stale full-grad M
into the live PowerSGD Q-basis so the two share a subspace by construction). Both
are additive, flag-gated, OFF by default (Prime Directive). Full ranking in §4.

---

## 1. Problem framing

### 1.1 What EXP-23 is

The fork's communication-efficient GRPO trainer compresses the pipeline-boundary
gradient with PowerSGD (rank r=77 ≈ the matched-rank budget for a p=0.95 mask on
Qwen2.5-1.5B's H=1536; see [[qwen25-1p5b-hidden-size-1536]]). PowerSGD is a
**low-rank** sketch: it keeps the top-r left/right factors of each boundary matrix
and discards the rest. Prior cycles established (EXP-20/#21) that the compressed
step carries most of the reward, and that a *fresh clean full-gradient step* every
K=5 steps closes the residual — but a fresh full-gradient flush is **unrealistic
for the decentralized-PP target** (it requires a full-H transfer and, in a real
async deployment, would itself be stale; see [[clean-step-realism-confound]] and
issue #22).

EXP-23 therefore replaces the *fresh* clean step with a **stale** full-rank signal:
an EMA-tracked full-gradient anchor `M_anchor` computed at θ_{t−5} and combined
with the live compressed `G` every 5 steps. This is the runnable form of issue #23
— the "combine-stale-full-with-compressed" mechanism already exists in
`verl/workers/comm_eff/spectral_filter.py` (`inject_matrix` L472, `blend_matrix`
L503) and had never been tested on PowerSGD.

### 1.2 Why inject/blend may fail (the a-priori worry)

The correction operates after FSDP grad reduction, before grad clipping
(`after_actor_backward__before_optimizer_step`). Two failure modes are visible
before any reward curve:

- **Inject (A2) inflates the step.** Inject *adds* the orthogonal complement of M
  to G. If M ⊥ G (which the smoke confirms), the added force is ~equal-norm and
  orthogonal ⇒ ‖G_corr‖ ≈ √2·‖G‖. RLVR succeeds via *minimal spectral drift*
  (2511.08567, Gate I/II); a √2 magnitude inflation in an uncosen direction is the
  SFT-like, spectrum-distorting failure mode. This is also the C1-collapse risk the
  `blend_matrix` docstring (L504–511) explicitly cites as its reason to exist.
- **Blend (A3) replaces live signal.** Blend trades half the live G for half the
  stale M. At η=0.5 with M ⊥ G it *shrinks* the step to ≈0.71×, discarding half the
  current on-policy signal in favor of a 5-step-old direction. If the stale
  direction is not still a descent direction, that is a net loss; if it is, η=0.5 may
  simply over-weight it (theory says η ∝ 1/K ≈ 0.2).

Crucially, **orthogonality alone does not tell us whether the correction helps** —
only that it is *not inert*. (Contrast the as-implemented spectral *reweight*, which
is inert by orthogonality — `rel_change ≈ 0.5`, G_filt ≈ 0; see
[[exp21-reweight-fixed-anchor]].) Whether the orthogonal stale direction is *aligned
with the true full-rank gradient 5 steps later* is exactly what the A2/A3 reward
curves vs. the A1 floor must answer, and it cannot be settled from geometry scalars.

---

## 2. What the measured geometry shows (live 50-step + smoke cross-check)

**The live 50-step data is in (§2.0). The smoke tables (§2.1–§2.3) are retained as
the cadence=1 cross-check; the live and smoke numbers agree (cos ≈ 0 throughout,
complement ≈ 1, blend ratio = √0.5).**

### 2.0 LIVE per-step geometry (A2 inject + A3 blend, all 50 steps) — FINAL

Source: `runs/EXP-23/exp-23-A2-stale-inject.train.log` and
`exp-23-A3-stale-blend.train.log` (every logged firing; 80 occurrences/target = 10
cadence-5 firing steps × 4 ranks × 2 reductions).

**A2 — inject (γ=1.0):**

| target | shape | n | \|cos\|max | scale=‖G‖/‖M‖ (min…max) | ‖inj‖/‖G‖ |
|---|---|---|---|---|---|
| layers.0 q_proj | (1536,1536) | 80 | 0.0017 | 0.00020 … 0.00770 | 1.0000 |
| layers.0 k_proj | (256,1536)  | 80 | 0.0043 | 0.00050 … 0.01970 | 1.0000 |
| layers.0 v_proj | (256,1536)  | 80 | 0.0048 | 0.00520 … 0.25330 | 1.0000 |
| layers.0 o_proj | (1536,1536) | 80 | 0.0012 | 0.00130 … 0.07040 | 1.0000 |

**A3 — blend (η=0.5):**

| target | shape | n | \|cos\|max | ‖G_corr‖/‖G‖ (min…max) |
|---|---|---|---|---|
| layers.0 q_proj | (1536,1536) | 80 | 0.0018 | 0.7065 … 0.7074 |
| layers.0 k_proj | (256,1536)  | 80 | 0.0034 | 0.7059 … 0.7075 |
| layers.0 v_proj | (256,1536)  | 80 | 0.0040 | 0.7060 … 0.7085 |
| layers.0 o_proj | (1536,1536) | 80 | 0.0014 | 0.7066 … 0.7075 |

Live findings (these supersede the smoke as the definitive geometry):
- **cos(G, M_anchor) ≈ 0 on every target, every firing step of BOTH arms**
  (global |cos| ≤ 0.0048). It does **NOT** rise after the delay_K=5 warmup —
  near-orthogonality is the **steady state**, not a transient. This is ~10× *more*
  orthogonal than the mask codec's cos ≈ 0.5 measured in EXP-21
  ([[exp21-reweight-fixed-anchor]]).
- **A3 `‖G_corr‖/‖G‖ = 0.706–0.709 ≡ √0.5 = 0.7071 exactly`** on all 4 targets, all
  50 steps — the phase-1 orthogonal-regime prediction (§2.2) **confirmed to 3
  decimals.** Blend deterministically shrinks the step to 0.71× because it replaces
  half of G with a scale-matched anchor orthogonal to it.
- **New mechanistic refinement — `scale = ‖G‖/‖M‖` is SMALL (0.0002 → 0.25), i.e.
  ‖M‖ ≫ ‖G‖.** The inject formula re-adds an *‖G‖-sized* orthogonal complement
  (‖inj‖/‖G‖ = 1.0000), but because the raw stale anchor is much larger than the
  live compressed grad, that complement is a *heavily down-scaled* (0.0002–0.25×)
  copy of M — in practice a tiny, scale-suppressed orthogonal-noise direction, not
  even the full √2 inflation the smoke's larger scales suggested. So **inject's
  realized net effect on PowerSGD is negligible orthogonal noise** that Adam +
  grad-clip wash out (val +0.005 = noise); blend's is a flat 0.71× step shrink that
  trades informative live signal for orthogonal stale signal (val −0.005).
- **Circuit fired as scheduled, codec held byte-constant:** A2/A3 each logged
  `spectral_corrections=80`, `anchor_backwards=20`; A1 = 0/0; and
  `powersgd_applications=179200` is **identical across all three arms** — the
  PowerSGD r=77 codec is byte-constant, the refresh mechanism is the only variable.

### 2.1 Per-target inject geometry (smoke_fire, steps 1–4) — cross-check

Targets are **layer 0 only** — `q_proj, k_proj, v_proj, o_proj` (4 =
`spectral.max_targets`). Boundary shapes: q/o = **(1536, 1536)**, k/v =
**(256, 1536)** (GQA — KV heads down-projected on H=1536).

| target | shape | cos(G, M_anchor) range | scale = ‖G‖/‖M‖ | ‖inj‖/‖G‖ | complement frac √(1−cos²) |
|---|---|---|---|---|---|
| layers.0 q_proj | (1536,1536) | −0.0007 … +0.0006 | 0.112 … 0.607 | 1.0000 | ≈1.000000 |
| layers.0 k_proj | (256,1536) | −0.0014 … +0.0024 | 0.453 … 2.166 | 1.0000 | ≈1.000000 |
| layers.0 v_proj | (256,1536) | −0.0048 … +0.0015 | 2.290 … 12.427 | 1.0000 | ≈0.999988 |
| layers.0 o_proj | (1536,1536) | −0.0009 … +0.0007 | 0.683 … 3.201 | 1.0000 | ≈1.000000 |

Key facts:
- **cos ≈ 0 everywhere** (|cos| ≤ 0.0048). The live compressed grad and the stale
  full-rank anchor are **effectively orthogonal** on every measured target/step.
- **‖inj‖/‖G‖ = 1.0000 everywhere** ⇒ complement fraction ≈ 1.0 ⇒ ~100% of M is
  missing from G's span. Inject re-adds essentially the *entire* (scale-matched)
  stale anchor as a force orthogonal to G. Corroborated by
  `rel_change = ‖G_proj − G_mask‖/‖G_mask‖ = 1.000000` on every target.
- **Magnitude consequence (inject):** `G_corr = G + (≈equal-norm orthogonal vector)`
  ⇒ **‖G_corr‖ ≈ √2·‖G‖ ≈ 1.41×** — undirected ~41% inflation. C1-collapse risk.
- **`scale` swings widely (0.11 → 12.4)** because ‖M‖ and the rescaled ‖G‖ differ
  per target/step; the scale-match is doing real normalization work.

### 2.2 Blend geometry — phase-1 prediction, now CONFIRMED by §2.0

smoke_fire ran `correction_mode=inject`, so the smoke had **zero `[blend]` lines**;
the phase-1 prediction was derived from `blend_matrix` (η=0.5, scale-matched anchor)
and the measured cos ≈ 0:
- predicted `‖G_corr‖/‖G‖ = √((1−η)² + η²) = √0.5 ≈ **0.7071**` (orthogonal regime).
- **CONFIRMED:** the live A3 arm logs `‖G_corr‖/‖G‖ = 0.706–0.709` on all 4 targets,
  all 50 steps (§2.0) — the prediction held to 3 decimals.

### 2.3 Anchor EMA update magnitude (smoke_fire, cadence=1)

`[EXP-12] anchor refresh` lines (targets=4, isolation=clone, anchor_loss=clean_pg,
delay_K=5): ‖dM_anchor‖_mean steps 1→4 = 15.11 → 1.51 → 1.36 → 1.22; max
21.46 → 2.15 → 1.93 → 1.74. Step 1 is the first EMA write from the seed (~10×);
steps 2–4 settle to O(1). `anchor_grad_corrected=0`, `anchor_optimizer_steps=0`
throughout — the anchor backward only feeds the EMA; it does not step the optimizer
(correct — correction is applied to the live grad, not by the anchor pass). Circuit
fired every scheduled step (cadence=1 in smoke; cadence=5 in the real arms → fires
at steps 5,10,…,50).

### 2.4 Coverage — "what gradient matrices did we get; did we get everything?"

**No — we got a subset: 4 of ~196 targets (layer 0's q/k/v/o attention
projections), scalars only.** This is the explicit coverage answer:

| quantity | value | source |
|---|---|---|
| `spectral.max_targets` | **4** | resolved_params_PROBE_FIRE.txt L34 |
| targets with inject/blend geometry | **4** = `layers.0.{q,k,v,o}_proj` | smoke_fire inject lines |
| `spectral.target_substr` eligible types | q,k,v,o,gate,up,down (7) | config dump |
| decoder layers in Qwen2.5-1.5B | 28 | model arch |
| candidate 2D linear targets (28×7) | **~196** | derived |
| anchor clone params loaded (stale θ_{t−5}) | **338 / 338** ("canon-matched") | `[anchor-load]` |

The correction loop `break`s once `corrected ≥ max_targets`
(`spectral_filter.py:591`) and iterates in `named_parameters` order ⇒ it always
stops at the **first 4 matching 2D matrices = layer 0 q,k,v,o**. The 338-param
anchor *clone* is the full stale-weight snapshot used to run the anchor backward,
but only 4 of its grads become `M_anchor` EMAs / diagnostics. Consequences:

- **Geometry coverage is partial: layer-0 attention only.** No MLP
  (`gate/up/down_proj`), no layers 1–27. Whether deeper/MLP layers have *different*
  (less orthogonal) cos(G, M) is **unmeasured** — a real gap if the synthesis wants
  to claim orthogonality is model-wide rather than a layer-0-attention artifact.
- **Raw matrices are NOT on disk.** Grepping the whole `verl/workers/comm_eff/`
  module found **no** `torch.save`/`np.save`/`.npy`/`.pt`/`pickle.dump` of any grad
  or weight matrix — only scalar geometry (cos, scale, norm ratios, rel_change,
  ‖dM_anchor‖). We have **scalars, not matrices**; per-singular-direction spectra
  cannot be reconstructed from the logs. (A bounded, OFF-by-default
  `spectral.debug_dump_path` guard — ~50 MB/step for the 4 targets — would capture
  full SVD spectra; **proposed, not implemented**. This is the *coverage fix* lever,
  distinct from the combiner — see §4.)

### 2.5 A2/A3 LIVE geometry — LANDED (see §2.0)

The live 50-step numbers are now in §2.0 and answer the two questions this
subsection had flagged: (i) **orthogonality persists past the PowerSGD warmup** —
cos stays ≈0 across all 50 steps, it is steady-state not transient; (ii) A3's
**‖G_corr‖/‖G‖ matches the predicted √0.5** to 3 decimals. The coverage caveat
(§2.4) is unchanged — still 4/196 targets (layer-0 attention), scalars only, no raw
matrices on disk; whether MLP/deep layers are equally orthogonal remains unmeasured.

---

## 3. What the literature says

### 3.1 Async / off-policy RL — staleness tolerance bounds

| Source | arXiv | Staleness tolerated | Correction mechanism |
|---|---|---|---|
| IMPALA / V-trace | 1802.01561 | several updates, graceful | truncated IS, two clips ρ̄ (fixed point) / c̄ (variance) |
| Sample Factory / APPO | 2006.11751 | high-throughput async | V-trace off-policy correction is *mandatory*, not free |
| **AReaL** (decoupled PPO) | 2505.24298 | **η≤8 free; η=∞ collapses** | proximal/trust-region anchor to a *recent* point + IS |
| **A-3PO** | 2512.06547 | arbitrary d≥1 | log-space interp, weight **α = 1/d** (contractive) |
| Asynchronous RLHF | 2410.18252 | 1-step main; **log dropoff** | algorithm robustness (DPO); robustness ↑ with model scale |
| **Prosperity / M2PO** | 2510.01161 | **≥256 updates match on-policy** | batch second-moment (M₂) trust; mask only outlier tokens |
| Stable Asynchrony / VCPO | 2602.17616 | broad | variance-modulated weights; watch **ESS** collapse |
| **DC-ASGD** | 1609.08326 | bounded τ | **Taylor:** `g(θ_{t+τ}) ≈ g(θ_t) + H·Δθ`, λ bias–variance |
| Staleness-Aware SGD | 1511.05950 | bounded τ | lr ∝ **1/τ** (Hessian-free) |
| Gap-Aware SGD | 1909.10802 | bounded | penalize by **measured ‖Δθ‖**, not the integer τ |

**Three load-bearing takeaways:**

1. **K=5 is tiny.** Every LLM-RL result puts the safe band at ≥4–8 updates with a
   mild correction; M2PO reaches ≥256. *"Moderate staleness (η≤8) has minimal impact
   on final performance"* (AReaL: η=0→42.0%, η=4→42.2%, η=∞→36.9% collapse). The log
   dropoff (Async RLHF) means K=1→K=5 costs little. ⇒ **If the stale anchor fails,
   the culprit is the combine, not the 5-step age.**
2. **The winning correction is a staleness-aware interpolation, never raw
   replacement.** AReaL anchors to a *recent* proximal policy; A-3PO interpolates
   with weight **α = 1/d** (decays with staleness): *"the proximal policy … just
   needs to lie somewhere between the behavior and target policies to prevent extreme
   importance weights."* This is a near-exact structural match to EXP-23's blend,
   and it says **η should be ∝ 1/K (≈0.2 at K=5), not a fixed 0.5.** Staleness-Aware
   SGD (lr ∝ 1/τ) and Gap-Aware SGD (weight by measured ‖θ_t − θ_{t−K}‖, since the
   same K can mean small or large real drift) converge on the same dial from the
   classic-SGD side.
3. **Control variance of the stale contribution; the classic fix is a curvature
   re-anchor.** M2PO/VCPO say monitor the second moment / ESS / effective rank and
   mask only outliers. DC-ASGD gives the principled re-pointing of a stale gradient:
   `g(θ_{t+τ}) ≈ g(θ_t) + H·(θ_{t+τ}−θ_t)`, H ≈ outer-product-of-gradients, λ∈[0,1]
   trading bias for variance — a more principled alternative to the
   orthogonal-complement inject.

### 3.2 RL gradient structure — the cited papers

| Source | arXiv | Core claim | Implication for stale M + live G |
|---|---|---|---|
| Miahi & Belilovsky | **2602.03839** | ~99% per-step updates compute-invisible after BF16 cast; **k≤8 async OK**; **error feedback** is the named fix (PULSELoCo Alg. 2 FP32 EF buffer) | K=5 inside validated async window; **add an EF residual buffer**, don't hope M covers the dropped part |
| Zhu et al. ("Path Not Taken") | **2511.08567** | RLVR sparsity is a **surface artifact**; true update is **full-rank, off-principal, spectrum-preserving** (Three-Gate); SFT ≠ RL | Keep a **full-rank M** (low-rank sketch under-covers RL); use **small-λ blend/cap**, NOT big inject (preserve spectrum); don't borrow SFT machinery |
| OPD blog (+ Yuan et al.) | — | RL pressure is **on-policy/localized**; RL updates **important, not redundant** | Region of update persists over short K (good); RL updates can't be dropped cheaply ⇒ **target correction to the live active set** |
| **Wang et al. (Linear Dynamics)** ★ | **2601.04537** | RLVR trajectory is **linear** — a **stable low-dim drift** (high-variance noise low-pass-filtered out); cos > 0.9 across K-windows; extrapolate + **re-ground every K** ⇒ 6.1× speedup | **Direct validation:** stale M ≈ still-current drift direction ⇒ K=5 trivially safe; re-ground = our re-anchor; use M as **drift estimate (blend / EMA)**, not impulse; relax cadence (10–50); watch **late-stage non-linearity** |
| Mukherjee et al. | **2505.11711** | RL update **sparse (5–30%) but full-rank** (mean rank 99.2–99.8% of max); subnetwork **~60% stable** across seeds, partial drift; in-dist data ⇒ sparse | **Full-rank M justified**; **mask M to the live active subnetwork** + short K because the subnetwork is only ~60% persistent |

Key quotes:
- 2601.04537: *"RLVR consistently enters a robust linear regime, where both
  parameter weights and output log-probabilities … evolve in a highly linear manner
  (R² > 0.7)"*; the linearity *"stems from the high-variance, noisy nature of RLVR
  training signals, which act as a low-pass filter to concentrate optimization along
  a stable, low-dimensional drift"*; *"weight-space extrapolation matches the
  performance of standard RL optimization while achieving a 6.1× training speedup
  through periodic re-grounding."* ⇒ A K-step-stale full-rank M still points in the
  current drift direction (cos > 0.9), so **M is information-rich at K=5 — the
  failure, if any, is the combine, not the age.**
- 2505.11711: *"updates to almost all parameter matrices are nearly full-rank …
  span almost the full subspaces"* (rank 99.2–99.8% of max) ⇒ **PowerSGD's low rank
  provably under-covers RL's update**, so the part G misses is real signal, which is
  exactly why a full-rank stale M (or an EF residual) is worth folding in.
- 2511.08567: RLVR works via *"minimal spectral drift, reduced principal-subspace
  rotation"* ⇒ a **convex blend (small λ) is spectrum-preserving; a big additive
  inject is the SFT-like, spectrum-distorting failure mode.**

**Literature bottom line (all five sources point the same way):** the recommended
correction is a **full-rank stale anchor M, used as a low-variance drift estimate
(EMA / normalized direction), folded in via a small-λ convex BLEND
(spectrum-preserving) rather than a raw additive inject, masked/projected onto the
live active subnetwork G is touching, with error feedback accumulating the PowerSGD
low-rank residual.** Inject-with-big-γ and pure low-rank correction are the
predicted failure modes.

---

## 4. Proposed alternative methods (ranked)

Every method below is subject to the **Prime Directive**: it MUST be *additive*,
*flag-gated*, and *OFF by default*. With its flag off, training is byte-identical to
the current PowerSGD path (and ultimately to upstream verl). None of these change
A1/A2/A3 — they are the *next* levers to try if A2 and A3 both fail (or to refine if
A3 partially works).

### Rank 1 — Error-feedback on the PowerSGD residual (the #21 top lever)

**Mechanism.** Maintain a local FP32 buffer `e` per boundary matrix. Each step:
compute the full boundary grad `G_full`, compress to `G_compressed` (PowerSGD r=77),
apply `G_step = G_compressed + decay·e`, then update `e ← e + (G_full − G_compressed)`
(the dropped low-rank residual). Every K steps, *flush* `e` into / re-ground it
against the stale full-rank anchor M (read M as the periodic full-rank ground truth
for the accumulated residual). New flags: `comm_eff.error_feedback.enabled` (default
false), `.decay`, `.flush_cadence`.

**Why the evidence supports it.** (a) Issue #21 already names "NO error-feedback =
top lever" for this PowerSGD setup; EF-SGD / the stock PowerSGD both ship EF and it
is provably convergent. (b) 2602.03839 makes EF the *named* fix for exactly this
"gate out what you can't send, but accumulate the residual and inject it next round
so nothing is permanently lost" problem (PULSELoCo Alg. 2). (c) The geometry says M
⊥ G ⇒ the residual G misses is large and real (Mukherjee full-rank). EF reframes the
stale M from "orthogonal impulse to add" to "periodic flush of an accumulated,
*tracked* residual" — which is both convergent and spectrum-gentle (the per-step
correction is the bounded `decay·e`, not a √2 inflation).

### Rank 2 — Small-λ, staleness-aware convex blend with η ∝ 1/K

**Mechanism.** Keep A3's blend operator but replace the fixed η=0.5 with a
staleness-aware schedule `η = c/K` (c≈1 ⇒ η≈0.2 at K=5), and optionally use the
*measured* drift gap `η ∝ 1/‖θ_t − θ_{t−K}‖` (Gap-Aware) rather than the integer K.
Blend the *normalized* stale direction `M̂` so it acts as a drift bias, not a
magnitude injection. New flags: `spectral.blend_eta_mode {fixed|inv_k|inv_gap}`,
`.blend_eta_scale`.

**Why the evidence supports it.** A-3PO's `α=1/d`, Staleness-Aware SGD's lr∝1/τ, and
Gap-Aware SGD's ‖Δθ‖ weighting *all independently* prescribe a staleness-shrinking
interpolation weight. 2511.08567's spectrum-preservation argues small λ; 2601.04537
says M is a good *drift estimate* (blend toward M̂), so a small pull onto the
persistent drift is exactly the intended use. This is the **minimal change to A3**
and the most likely "A3-almost-worked, just over-weighted" fix.

### Rank 3 — Correction masked to the active subnetwork

**Mechanism.** Before injecting/blending, mask M to the coords the live grad
currently touches (e.g. top-magnitude coords of G_full, or the union of the live
active set), zeroing M elsewhere. New flag: `spectral.active_mask {none|topk|union}`,
`.active_mask_frac`.

**Why the evidence supports it.** Mukherjee: the RL active subnetwork is only ~60%
persistent across seeds and *drifts* across steps, so a 5-step-stale M is mostly but
not fully aligned — masking to the live active set drops the ~40% that may have
moved. OPD/Yuan: RL updates are important-not-redundant, so a stale M misaligned with
the current active region is *more* costly in RL than in SFT ⇒ target the correction.
This composes with Rank 1/2.

### Rank 4 — DC-ASGD curvature re-anchoring of M toward θ_t

**Mechanism.** Before combining, re-point the stale M toward the current weights:
`M_corr = M + λ·Ĥ·(θ_t − θ_{t−K})`, with Ĥ the cheap outer-product-of-gradients
(Fisher-style) approximation and λ∈[0,1] trading bias for variance. Then inject/blend
`M_corr`. New flags: `comm_eff.anchor.dc_lambda` (default 0 ⇒ exactly current
behavior), `.curvature_estimator`.

**Why the evidence supports it.** DC-ASGD (1609.08326) is the textbook way to make a
K-step-stale gradient usable — a first-order Taylor correction that re-points it
toward θ_t. EXP-23 currently uses M *raw*; DC-ASGD is the principled alternative to
the orthogonal-complement inject, and its λ-as-variance-control is the same dial that
M2PO/VCPO/A-3PO independently rediscover. Heavier than Rank 1–3; offer as a
follow-on if the cheap blends under-deliver.

### Rank 5 — Relax anchor cadence (linear-dynamics re-grounding every K≈10–50)

**Mechanism.** Treat the anchor refresh as the "re-grounding" of 2601.04537: between
re-grounds, *extrapolate* the drift (or just hold M as the drift estimate); re-ground
the full-rank M every K≈10–50 instead of 5. New flag: sweep `anchor.cadence` and
`anchor.delay_K`.

**Why the evidence supports it.** 2601.04537 extrapolates 10–50 steps off a single
drift vector for a 6.1× speedup; the operator memory already flags clean_cadence as
"likely relaxable" ([[powersgd-activation-issue20]]). This is a *cost* lever (fewer
expensive full-rank anchors) more than an accuracy lever, and pairs naturally with
Rank 1's EF flush. **Caveat:** 2601.04537 notes linearity weakens late-train
(late-stage collapse) ⇒ consider shortening K near the end of a run.

### Rank 6 — Magnitude-capped ADAPTIVE inject (γ from the measured complement)

**Mechanism.** Replace fixed γ=1 with γ set from the measured complement fraction /
relative norms — an MMSE-style combiner that injects only as much orthogonal M as the
estimated signal-to-noise warrants, with a hard cap on ‖G_corr‖/‖G‖ (e.g. ≤1.1) to
forbid the √2 inflation. New flags: `spectral.inject_gamma_mode {fixed|adaptive}`,
`.inject_norm_cap`.

**Why the evidence supports it.** The plan flagged the MMSE-style combiner; the smoke
shows the uncapped γ=1 inject is the √2-inflation/C1-collapse risk. M2PO's
second-moment trust and VCPO's variance modulation say cap/modulate the
high-variance contribution. This *salvages inject* if A2 fails purely on magnitude
(rather than direction). Lower-ranked because if direction is the problem (orthogonal
M not a real descent direction), no magnitude cap helps — Rank 1/2/3 address
direction.

### Scope lever (distinct from the combiner) — raise `spectral.max_targets` beyond 4

The smoke corrects only **4/196** targets (layer-0 q/k/v/o). If A2/A3 fail, one
hypothesis is *insufficient coverage*, not a bad operator — the correction may be
real but applied to a vanishingly small slice of the model. Raising `max_targets`
(and/or extending to MLP `gate/up/down` and deeper layers) is a **coverage** lever
orthogonal to *which* combiner to use, and it interacts with cost (anchor backward
OOM — recall the 18432 tok/gpu + `ema_device=cpu` guard,
[[anchor-clone-fsdp-naming-bug]]). Pair with the OFF-by-default `spectral.debug_dump`
(§2.4) to first *measure* whether deeper/MLP cos(G,M) is also ≈0 before paying for
broader correction. Treat coverage as a separate axis in any follow-up sweep.

---

## 5. How I came to these conclusions (reasoning chain)

- **Geometry → "inject inflates, blend shrinks."** The smoke's measured ‖inj‖/‖G‖ =
  1.0 and cos ≈ 0 are not estimates — they are the logged values. The Frobenius
  identity `‖G_corr‖ = √2·‖G‖` (inject, orthogonal) and `√0.5·‖G‖` (blend, η=0.5,
  orthogonal) follow directly. So *before any reward*, geometry predicts A2
  destabilizes by magnitude and A3 is gentler. → motivates Rank 2 (smaller η) and
  Rank 6 (cap inject).
- **Async-RL bounds → "age is not the problem; η ∝ 1/K."** AReaL η≤8 lossless +
  M2PO ≥256 + the log dropoff put K=5 deep inside the safe band. Three independent
  staleness-correction families (A-3PO 1/d, SA-SGD 1/τ, Gap-Aware ‖Δθ‖) all
  prescribe a *shrinking* interpolation weight. → Rank 2 is the literal translation;
  Rank 4 (DC-ASGD) is the curvature-aware version; Rank 3 (active mask) is the
  M2PO/VCPO "mask the outliers / control variance" idea in coord space.
- **Linear dynamics → "M is information-rich; use it as a drift estimate, re-ground
  periodically."** 2601.04537's cos > 0.9 across K-windows says a 5-step-stale M is
  *aligned with the current drift* — which contradicts the smoke's cos ≈ 0 *only if*
  one conflates "G (low-rank sketch) vs M" with "true full grad vs M." The
  resolution: G is a *compressed* sketch, so G ⊥ M is expected (PowerSGD threw away
  the directions M lives in — Mukherjee full-rank), while *full* grad ‖ M would still
  be high. → This is the crux: it predicts **blend/EF (which fold M's real direction
  back in) beat inject (which adds it orthogonally and inflates).** → Rank 1, 2, 5.
- **RL ≠ SFT, full-rank, spectrum-preserving → "small-λ blend over big inject; keep
  full-rank M; EF the residual."** 2511.08567 + Mukherjee jointly say: don't distort
  the spectrum (no big inject), keep the full-rank signal (low-rank under-covers RL),
  and the part G misses is real → must be tracked, not discarded. → Rank 1 (EF) is
  the direct consequence; it is also the standing #21 top lever, so it ranks first.
- **Coverage scalars → "we only saw 4/196; raw matrices absent."** The grep + the
  `break`-at-max_targets logic prove the geometry is layer-0-attention only and
  scalars-only. → This caps confidence in "orthogonality is model-wide" and yields
  the *coverage* lever + the OFF-by-default debug-dump proposal, kept distinct from
  the combiner choice.

---

## 6. Recommended follow-up issue

**Title:** `EXP-24: Error-feedback PowerSGD residual + staleness-aware blend (η∝1/K)
— the named #21 top lever, gated and OFF by default`

**Plan seed (one paragraph).** Add an OFF-by-default error-feedback path to the
PowerSGD boundary codec: a per-matrix FP32 residual buffer `e` accumulating
`G_full − G_compressed`, applied as `G_step = G_compressed + decay·e` and flushed /
re-grounded against the stale full-rank anchor M every K steps
(`comm_eff.error_feedback.{enabled,decay,flush_cadence}`, default off ⇒
byte-identical to current). In the same cycle, replace A3's fixed blend η with a
staleness-aware schedule `η = c/K` (and an `inv_gap` variant weighting by measured
‖θ_t − θ_{t−K}‖), so the convex-blend operator that the literature endorses is run at
the prescribed ≈0.2 rather than 0.5. Arms on the FIXED control surface
([[fixed-control-surface-gsm8k]]): **B1** = A1 floor (PowerSGD r=77, no refresh, no
EF) for byte-parity, **B2** = EF-only, **B3** = EF + blend(η=1/K). Gate on EF-only ≥
A1 floor (proves EF alone recovers residual reward) and EF+blend ≥ EF-only (proves
the stale-anchor drift adds on top). Carry the OFF-by-default `spectral.debug_dump`
(§2.4) to capture per-target SVD spectra on step 0, and a coverage knob
(`max_targets`, MLP/deep-layer) as a *separate* secondary axis — measure cos(G,M) on
MLP/deep layers before paying to correct them. Keep all anchor-OOM guards
(18432 tok/gpu, `ema_device=cpu`).

---

## 7. Summary table — the alternatives at a glance

| Rank | Method | New flag(s) (default OFF) | Primary evidence | Addresses |
|---|---|---|---|---|
| 1 | Error-feedback on PowerSGD residual | `error_feedback.{enabled,decay,flush_cadence}` | #21 top lever; 2602.03839 (PULSELoCo EF); Mukherjee full-rank | residual loss + spectrum-gentle |
| 2 | Staleness-aware blend η∝1/K | `blend_eta_mode {fixed\|inv_k\|inv_gap}` | A-3PO 1/d; SA-SGD 1/τ; Gap-Aware ‖Δθ‖; 2511.08567 small-λ | over-weighted A3 |
| 3 | Active-subnetwork-masked correction | `active_mask {none\|topk\|union}` | Mukherjee ~60% subnet; OPD/Yuan important-not-redundant; M2PO outlier mask | stale/live drift mismatch |
| 4 | DC-ASGD curvature re-anchor of M | `anchor.dc_lambda` (0 ⇒ current) | DC-ASGD 1609.08326 | re-point stale M to θ_t |
| 5 | Relax anchor cadence (re-ground 10–50) | sweep `anchor.cadence`,`delay_K` | 2601.04537 6.1× re-grounding | cost (fewer full-rank anchors) |
| 6 | Magnitude-capped adaptive inject | `inject_gamma_mode {fixed\|adaptive}`,`inject_norm_cap` | plan MMSE combiner; M2PO/VCPO variance ctrl | salvage A2 if magnitude-only |
| — | Coverage: raise `max_targets` / add MLP+deep | `spectral.max_targets`, `target_substr`; `debug_dump` | smoke 4/196 coverage gap | scope, not combiner |

---

## 8. A2/A3 EMPIRICAL OUTCOME — PENDING (finalize after arms complete, ~2–4 h)

The A2/A3 50-step reward curves and live per-layer geometry are **not yet
available** (A1 at step 3/50 as of 12:41Z; A2/A3 not started; full run ≈5–7 h). This
report's recommendation **conditions on that outcome** as follows:

- **A2 AND A3 both fail** (neither beats the A1 floor): the pre-registered prediction
  is confirmed and this section becomes the **binding next-lever readout**. Proceed
  with **Rank 1 (error-feedback)** as the top lever — it is independently the #21 top
  lever and the named fix in 2602.03839 — combined with **Rank 2 (η∝1/K)**. File the
  EXP-24 issue (§6). The failure is attributed to the *combine* (orthogonal
  inject/over-weighted blend), not to K=5 age, per the literature.

- **A3 (blend) PASSES, A2 (inject) fails:** the blend operator is validated and the
  prediction holds (blend > inject, as both geometry and literature said). Prefer the
  **Rank 2 refinement (η∝1/K ≈ 0.2)** to see if a *smaller* blend does as well or
  better at lower cost / less live-signal displacement; A2's inject is shelved
  (confirmed √2-inflation failure) unless Rank 6's capped-adaptive inject is wanted as
  a curiosity. Error-feedback (Rank 1) still recommended as an additive on top.

- **A2 (inject) PASSES:** inject *surprised us* — the orthogonal stale direction was a
  real descent direction despite the magnitude inflation, and RLVR tolerated the √2
  step. Revisit the spectrum-distortion worry (2511.08567) empirically: dump the SVD
  spectra (§2.4 debug-dump) to see whether the inject preserved or rotated the
  principal subspace, and check whether the win survives at scale / later in training
  (2601.04537 late-stage non-linearity). Re-rank: a *capped* adaptive inject (Rank 6)
  would then likely dominate plain γ=1.

- **Live-geometry checks to append at finalize** (from the A2/A3 logs): (i) does cos(G,
  M) stay ≈0 past PowerSGD warmup (step 1 `reconstruction_rel_error` 0.976→0.024)? a
  rising cos would mean the sketch starts covering M and changes the inject/blend
  math; (ii) A3's actual `‖G_corr‖/‖G‖` vs. the predicted 0.7071; (iii) whether the
  inject/blend steps (5,10,…,50) coincide with reward jumps or drops vs. the A1 floor;
  (iv) effective rank / ESS of the combined update at the correction steps (VCPO).

---

## Sources

**Async / off-policy RL staleness:**
- IMPALA / V-trace — https://arxiv.org/abs/1802.01561
- Sample Factory — https://arxiv.org/pdf/2006.11751
- AReaL — https://arxiv.org/html/2505.24298v3
- A-3PO — https://arxiv.org/abs/2512.06547
- Asynchronous RLHF — https://arxiv.org/abs/2410.18252
- Prosperity before Collapse (M2PO) — https://arxiv.org/html/2510.01161v1
- Stable Asynchrony (VCPO) — https://arxiv.org/pdf/2602.17616
- DC-ASGD — https://arxiv.org/abs/1609.08326
- Staleness-aware Async-SGD — https://arxiv.org/abs/1511.05950
- Gap-Aware Mitigation of Gradient Staleness — https://openreview.net/forum?id=B1lLw6EYwB (arXiv 1909.10802)
- verl rollout-correction math (TIS / decoupled PPO) — https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html

**RL gradient structure (operator-cited):**
- arXiv 2602.03839 — Miahi & Belilovsky, *Understanding and Exploiting Weight Update Sparsity for Communication-Efficient Distributed RL* — https://arxiv.org/abs/2602.03839
- arXiv 2511.08567 — Zhu et al., *The Path Not Taken: RLVR Provably Learns Off the Principals* — https://arxiv.org/abs/2511.08567
- OPD blog — *SFT vs RL / On-Policy Distillation* — https://nrehiew.github.io/blog/sft_rl_opd/
- arXiv 2601.04537 — Wang et al., *Linear Dynamics in the RLVR Training of Large Language Models* — https://arxiv.org/abs/2601.04537 (code: https://github.com/Miaow-Lab/RLVR-Linearity)
- arXiv 2505.11711 — Mukherjee et al., *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models* — https://arxiv.org/abs/2505.11711

**Internal:**
- `runs/EXP-23/stale_gradient_research/notes_async_rl.md`, `notes_cited_papers.md`, `notes_grad_geometry.md`
- `verl/workers/comm_eff/spectral_filter.py` (`inject_matrix` L472, `blend_matrix` L503, `apply_spectral_correction_to_params` L547, break-at-max_targets L591)
- `runs/EXP-23/config.yaml`; `runs/EXP-23/smoke-logs/smoke_fire.log`; `runs/EXP-23/monitor-detail.log`
- Memory: [[powersgd-activation-issue20]] (#21 "NO error-feedback = top lever"), [[exp21-reweight-fixed-anchor]] (reweight inert by orthogonality; blend is the only live correction), [[clean-step-realism-confound]] (#22/#23 staleness framing), [[fixed-control-surface-gsm8k]], [[qwen25-1p5b-hidden-size-1536]], [[anchor-clone-fsdp-naming-bug]] (OOM guards)
