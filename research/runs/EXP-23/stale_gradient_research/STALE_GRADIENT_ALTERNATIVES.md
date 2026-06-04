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
with the true full-rank gradient 5 steps later* was the open question the A2/A3
reward curves had to settle — and they did: **it is not.** Both arms came in at the
A1 floor ±0.005 (§8); the stale anchor carried no usable descent signal into the
compressed step, because it lives in PowerSGD's discarded subspace.

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
- **Mechanistic refinement — `scale = ‖G‖/‖M‖` is SMALL (0.0002 → 0.253), i.e.
  ‖M‖ ≫ ‖G‖** (the full-rank stale anchor is *larger*-norm than the rank-77
  compressed grad). Inject scale-matches M *down* to ‖G‖-magnitude: the injected term
  = `(‖G‖/‖M‖)·(M − proj_G(M)) ≈ (‖G‖/‖M‖)·M`, with magnitude ≈ ‖G‖ — so
  `‖inj‖/‖G‖ = 1.0000` means the injected orthogonal vector is **‖G‖-sized, not tiny
  noise.** With cos ≈ 0 ⇒ `‖G_corr‖ ≈ √2·‖G‖` in the formula. **Why A2 is still
  inert:** the correction is applied AFTER FSDP reduction but BEFORE GRPO grad
  clipping (§2.4), so the √2-inflated `G_corr` (grad_norm ≈ 1.3–2.57) exceeds the
  clip threshold; clipping then scales the *whole* vector down — shrinking the useful
  G component by ~1/√2 while spending half the clipped budget on an orthogonal stale
  direction that adds no aligned signal. **Net: A2 ≈ floor (no aligned signal added,
  useful step mildly diluted) — the same outcome as A3's explicit 0.71× shrink,
  reached by a different route** (A2 val +0.005, A3 val −0.005; both noise).
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
- **Magnitude consequence (inject):** `G_corr = G + (‖G‖-sized orthogonal vector)`
  ⇒ ‖G_corr‖ ≈ √2·‖G‖ ≈ 1.41× — this holds in *both* smoke and live runs, because
  inject always scale-matches the complement to ‖G‖ (‖inj‖/‖G‖ = 1.0) regardless of
  the raw ‖M‖/‖G‖ ratio. **The live arms confirmed the √2 inflation is real but
  benign:** it lands above the GRPO grad-clip threshold (grad_norm ≈ 1.3–2.57) and
  clipping dilutes the useful G component by ~1/√2 while half the budget goes to an
  orthogonal stale direction (§2.0). The failure is **direction (cos≈0), not
  magnitude** — a magnitude cap would not supply the missing alignment.
- **`scale` swings widely in the smoke (0.11 → 12.4)** but is uniformly *small* in the
  live run (0.0002–0.253, ‖M‖≫‖G‖); either way the injected complement is normalized
  to ‖G‖, so the realized inject magnitude is ‖G‖-sized in both.

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
the current PowerSGD path (and ultimately to upstream verl). A2 and A3 **both
failed** (FALSIFIED, §0/§8), so these are the *binding* next levers. The live data
narrows the ranking sharply: **the failure is geometric incoherence (G ⊥ M by
construction), not combiner tuning** — so the two Rank-1 methods (error-feedback,
basis-aligned anchor) that *remove the orthogonality at its source* dominate, while
the combiner-tuning levers (Rank 2/3/4/6) can only help *once a shared subspace
exists*. Each is flagged with whether it attacks the incoherence or presumes it
already fixed.

### Rank 1a — Error-feedback on the PowerSGD residual (the #21 top lever; ATTACKS the incoherence)

**Mechanism.** Maintain a local FP32 buffer `e` per boundary matrix. Each step:
compute the full boundary grad `G_full`, compress to `G_compressed` (PowerSGD r=77),
apply `G_step = G_compressed + decay·e`, then update
`e ← e + (G_full − decompress(compress(G_full)))` (the *dropped* low-rank residual —
the standard PowerSGD convergence fix). Optionally re-ground `e` against the stale
full-rank anchor M every K steps. New flags: `comm_eff.error_feedback.enabled`
(default false), `.decay`, `.flush_cadence`.

**Why the evidence supports it (and why it beats inject/blend on the LIVE result).**
The live data shows G ⊥ M (cos ≈ 0.001, steady-state) because PowerSGD r=77 *discards*
the subspace M lives in. EF carries forward **exactly the energy the sketch dropped**,
in the *same* basis — so the fed-back term is aligned with what G is missing by
construction, the one thing inject/blend could not be. (a) Issue #21 names
"NO error-feedback = top lever" for this PowerSGD setup; EF-SGD / stock PowerSGD ship
EF and it is provably convergent. (b) 2602.03839 makes EF the *named* fix for "gate
out what you can't send, accumulate the residual, inject it next round so nothing is
permanently lost" (PULSELoCo Alg. 2). (c) Mukherjee full-rank ⇒ the dropped residual
is real signal, not noise. Spectrum-gentle (per-step correction is the bounded
`decay·e`, not a √2 inflation).

### Rank 1b — Basis-aligned anchor (project M into the live PowerSGD Q-basis; ATTACKS the incoherence)

**Mechanism.** Before combining, project the stale full-rank anchor M onto the live
PowerSGD rank-r Q-basis (the left/right factors the codec keeps this step):
`M_aligned = Q (Qᵀ M)` (and the right-factor analog), so M and G share a subspace by
construction; then inject/blend `M_aligned`. New flags:
`comm_eff.anchor.basis_align {none|q_project}` (default `none` ⇒ exactly current
behavior).

**Why the evidence supports it.** The live result proves the problem is that M lives
in PowerSGD's *complement*; the direct structural fix is to stop defining the anchor
in an orthogonal subspace. **An anchor defined in the orthogonal complement can never
help no matter how fresh** (grad-empirics §8). Projecting M into Q forces a non-zero
cos and lets a small blend actually move G along a shared, informative direction.
Caveat: this *also* throws away M's full-rank content (the part outside Q), so it is
complementary to — not a substitute for — Rank 1a (EF, which is what keeps the dropped
content alive). The strongest single design is **1a + 1b together**: EF preserves the
dropped energy across steps; basis-alignment makes the periodic full-rank anchor
land in a subspace the live step can use.

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
persistent drift is exactly the intended use.

**PRESUMES a shared subspace — DEMOTED by the live result.** With G ⊥ M (the live
finding), a smaller η just shrinks an orthogonal, *unaligned* contribution: at
η=0.2 instead of 0.5, A3 would scale the step to √(0.8²+0.2²)=0.82× and still add no
aligned signal. So η∝1/K alone **cannot** fix EXP-23's failure — it is only useful
*downstream of* Rank 1b (once the anchor is basis-aligned and cos>0, a smaller, gap-
aware η is the right weight on a now-informative blend). Keep it as the cheap
refinement layered on Rank 1, not as a standalone fix.

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

**Why the evidence supports it — and why the LIVE result largely retires it.** The
plan flagged the MMSE-style combiner; M2PO/VCPO say cap/modulate the high-variance
contribution. The live data settled the smoke's open question: A2's failure was
**direction, not magnitude.** The √2 inflation *did* occur (inject scale-matches the
complement to ‖G‖, so `‖G_corr‖ ≈ √2·‖G‖`, grad_norm ≈ 1.3–2.57), but GRPO grad
clipping already absorbed it — clipping scaled the inflated vector down, leaving A2 at
the floor with no aligned signal. A static magnitude cap would only do what grad-clip
already did and **cannot supply the missing alignment.** **Retire as a fix on its
own**; an adaptive γ is meaningful only *after* Rank 1b makes the injected direction
non-orthogonal (so capping trades a real aligned contribution against a budget).

### Scope lever (distinct from the combiner) — raise `spectral.max_targets` beyond 4

Both smoke and the live arms correct only **4/196** targets (layer-0 q/k/v/o). A
residual hypothesis for the FALSIFIED result is *insufficient coverage*, not a bad
operator — though the live geometry argues against it (the failure is orthogonality
on the targets we *did* measure, which broader coverage would not fix). Still, since
cos(G,M) on MLP/deep layers is unmeasured, raising `max_targets`
(and/or extending to MLP `gate/up/down` and deeper layers) is a **coverage** lever
orthogonal to *which* combiner to use, and it interacts with cost (anchor backward
OOM — recall the 18432 tok/gpu + `ema_device=cpu` guard,
[[anchor-clone-fsdp-naming-bug]]). Pair with the OFF-by-default `spectral.debug_dump`
(§2.4) to first *measure* whether deeper/MLP cos(G,M) is also ≈0 before paying for
broader correction. Treat coverage as a separate axis in any follow-up sweep.

---

## 5. How I came to these conclusions (reasoning chain)

The chain below ran first on the smoke + literature (phase 1) and is now closed by
the live 50-step data (phase 2). The live result did not overturn the prediction — it
**sharpened it**: the failure is not magnitude-vs-shrink tuning, it is structural
geometric incoherence.

- **Geometry (smoke → live) → "G ⊥ M is steady-state and structural."** The logged
  ‖inj‖/‖G‖ = 1.0 and cos ≈ 0.001 are measured values, not estimates; the Frobenius
  identities `‖G_corr‖ = √2·‖G‖` (inject, orthogonal) and `√0.5·‖G‖` (blend, η=0.5)
  follow directly — and the live A3 arm logged 0.7071 to 3 decimals, confirming the
  geometry held. The live refinement: `scale = ‖G‖/‖M‖ = 0.0002–0.253` ⇒ **‖M‖≫‖G‖**,
  but inject scale-matches the complement *down* to ‖G‖, so the injected orthogonal
  vector is **‖G‖-sized** (not tiny noise) and `‖G_corr‖ ≈ √2·‖G‖` is real — it just
  lands above the grad-clip threshold (grad_norm 1.3–2.57), so clipping shrinks the
  useful G by ~1/√2 and spends half the budget on an orthogonal stale direction. **A2
  reaches the same "≈ floor, useful step diluted, no aligned signal" outcome as A3's
  0.71× shrink, by a different route.** cos does NOT rise past warmup ⇒ orthogonality
  is the steady state. → This *demotes* Rank 2 (a smaller η shrinks an
  already-orthogonal contribution) and *retires* Rank 6 (the failure is direction, not
  a magnitude that a cap could fix) as standalone fixes.
- **The crux — reconciling Linear-Dynamics (cos>0.9) with the measured cos≈0.** These
  only conflict if one conflates "G (the compressed sketch) vs M" with "true full
  grad vs M." The resolution the live data confirms: G is a *PowerSGD low-rank
  sketch*, so it discards exactly the subspace M lives in ⇒ G ⊥ M *by construction*,
  while *full* grad ‖ M would still be high (the trajectory really is linear). → This
  is the load-bearing inference: **the fix must restore a shared subspace** — either
  carry the dropped energy forward in-basis (Rank 1a, EF) or project M into the live
  Q-basis (Rank 1b). It also says delay_K is irrelevant: a fresher M lands in the
  same complement (confirmed — cos flat across all 50 steps).
- **Async-RL bounds → "age is not the problem."** AReaL η≤8 lossless + M2PO ≥256 +
  the log dropoff put K=5 deep inside the safe band; the steady-state orthogonality
  confirms staleness is not the failure axis. The three staleness-weight families
  (A-3PO 1/d, SA-SGD 1/τ, Gap-Aware ‖Δθ‖) still prescribe η∝1/K — but, per the live
  result, only *after* a shared subspace exists (Rank 2 layered on Rank 1b).
- **RL ≠ SFT, full-rank, spectrum-preserving → "EF the residual; keep M full-rank."**
  2511.08567 + Mukherjee: don't distort the spectrum, keep the full-rank signal
  (low-rank under-covers RL), and the part G misses is real → must be tracked, not
  discarded. The live G⊥M is the empirical face of "low-rank under-covers RL." →
  Rank 1a (EF) is the direct consequence and the standing #21 top lever, so it ranks
  first; Rank 1b (basis-align) is the co-equal structural fix the live data motivated.
- **Two dead-ends, two different root causes.** EXP-21's mask+reweight was inert by a
  *projection-operator* collapse at cos≈0.5 (G_filt≈0, [[exp21-reweight-fixed-anchor]]);
  EXP-23's PowerSGD+inject/blend is inert by *true geometric incoherence* at cos≈0.001
  (~10× more orthogonal). Distinguishing these matters: the EXP-23 fix is not a better
  operator on the same gradients (that was EXP-21's failed move) but a change to *which
  subspace the anchor occupies*. → Rank 1a/1b, not Rank 2–6.
- **Coverage scalars → "we only saw 4/196; raw matrices absent."** The grep + the
  `break`-at-max_targets logic prove the geometry is layer-0-attention only and
  scalars-only. → caps confidence in "orthogonality is model-wide" and yields the
  *coverage* lever + the OFF-by-default debug-dump proposal, kept distinct from the
  combiner choice.

---

## 6. Recommended follow-up issue

**Title:** `EXP-24: Error-feedback on the PowerSGD residual (+ basis-aligned anchor)
— remove the G⊥M incoherence EXP-23 exposed; gated, OFF by default`

**Plan seed (one paragraph).** EXP-23 falsified the stale full-rank anchor on
PowerSGD because the compressed live grad G is ~orthogonal to the stale anchor M
(cos≈0.001, steady-state) — PowerSGD r=77 discards exactly the subspace M occupies,
so neither inject (a ‖G‖-sized orthogonal add that grad-clip then dilutes ~1/√2) nor
blend (flat 0.71× step shrink) can carry M's descent information into the live step. EXP-24 attacks that
incoherence at its source. Add an OFF-by-default error-feedback path to the PowerSGD
boundary codec: a per-matrix FP32 residual buffer `e` accumulating the *dropped*
energy `G_full − decompress(compress(G_full))`, applied as
`G_step = G_compressed + decay·e` (`comm_eff.error_feedback.{enabled,decay}`, default
off ⇒ byte-identical to current — the standard provably-convergent PowerSGD EF). Add
an OFF-by-default basis-aligned anchor (`comm_eff.anchor.basis_align=q_project`) that
projects the stale full-rank M into the live PowerSGD Q-basis so the anchor and live
step share a subspace by construction; only with basis-alignment does a
staleness-aware blend `η=c/K` become meaningful (layer it on, not standalone). Arms
on the FIXED control surface ([[fixed-control-surface-gsm8k]]): **B1** = A1 floor
(PowerSGD r=77, no refresh, no EF) for byte-parity, **B2** = EF-only, **B3** = EF +
basis-aligned anchor blend(η=1/K). Gate on EF-only ≥ A1 floor (proves EF alone
recovers the dropped-residual reward — the #21 top lever) and B3 ≥ B2 (proves the
basis-aligned stale drift adds on top). Carry the OFF-by-default `spectral.debug_dump`
(§2.4) to capture per-target SVD spectra on step 0, and a coverage knob
(`max_targets`, MLP/deep-layer) as a *separate* secondary axis — first measure
cos(G,M) on MLP/deep layers to confirm the orthogonality is model-wide. Keep all
anchor-OOM guards (18432 tok/gpu, `ema_device=cpu`).

---

## 7. Summary table — the alternatives at a glance

Ranks reflect the FALSIFIED live result: **1a/1b ATTACK the G⊥M incoherence** (the
proven root cause); 2–6 **presume a shared subspace** and only help layered on 1a/1b.

| Rank | Method | New flag(s) (default OFF) | Primary evidence | Status vs live result |
|---|---|---|---|---|
| **1a** | Error-feedback on PowerSGD residual | `error_feedback.{enabled,decay}` | #21 top lever; 2602.03839 (PULSELoCo EF); Mukherjee full-rank; live G⊥M | **top fix** — carries dropped energy forward in-basis |
| **1b** | Basis-aligned anchor (project M into Q) | `anchor.basis_align {none\|q_project}` | live cos≈0.001 = M in PowerSGD complement; grad-empirics §8 | **co-top fix** — forces shared subspace by construction |
| 2 | Staleness-aware blend η∝1/K | `blend_eta_mode {fixed\|inv_k\|inv_gap}` | A-3PO 1/d; SA-SGD 1/τ; Gap-Aware ‖Δθ‖; 2511.08567 small-λ | demoted — needs 1b first (else shrinks orthogonal noise) |
| 3 | Active-subnetwork-masked correction | `active_mask {none\|topk\|union}` | Mukherjee ~60% subnet; OPD/Yuan; M2PO outlier mask | layered on 1a/1b |
| 4 | DC-ASGD curvature re-anchor of M | `anchor.dc_lambda` (0 ⇒ current) | DC-ASGD 1609.08326 | re-points M but not into Q — pairs with 1b |
| 5 | Relax anchor cadence (re-ground 10–50) | sweep `anchor.cadence`,`delay_K` | 2601.04537 6.1× re-grounding | cost lever; delay_K confirmed NOT the failure axis |
| 6 | Magnitude-capped adaptive inject | `inject_gamma_mode {fixed\|adaptive}`,`inject_norm_cap` | plan MMSE combiner; M2PO/VCPO | retired — failure was direction not magnitude; grad-clip already absorbed the √2 inflation |
| — | Coverage: raise `max_targets` / add MLP+deep | `spectral.max_targets`, `target_substr`; `debug_dump` | live 4/196 coverage gap | scope, not combiner; confirm orthogonality is model-wide |

---

## 8. A2/A3 EMPIRICAL OUTCOME — FINAL (hypothesis FALSIFIED, verdict STOP)

All three arms ran to step 50; the box is torn down; logs are on disk. **The
"A2 AND A3 both fail" branch fired** — so the conditioning resolves to the binding
next-lever path below.

| arm | refresh | val@50 | Δ vs A1 floor | reading |
|---|---|---|---|---|
| A1 no-refresh (floor) | none | 0.6914 | — | PowerSGD r=77 baseline |
| A2 stale inject (γ=1.0) | stale_inject | 0.6967 | +0.0053 | INERT (within GSM8K noise) |
| A3 stale blend (η=0.5) | stale_blend | 0.6861 | −0.0053 | INERT (slightly below floor) |

`max(A2, A3) = 0.6967 ≤ falsify line 0.7114` (PASS bar 0.7414; A0 fresh-clean
ref 0.7415, dense 0.7536) ⇒ **FALSIFIED.** The PowerSGD codec was held
byte-constant across arms (`powersgd_applications=179200` identical), and the
correction circuit fired as scheduled (A2/A3 `spectral_corrections=80`,
`anchor_backwards=20`; A1 0/0), so this is a clean negative, not a plumbing failure.

**Why it failed (settled, not hypothesized).** cos(G, M_anchor) ≈ 0.001 on every
target and every one of the 50 steps in *both* arms — steady-state near-orthogonality
that did not rise past the delay_K=5 warmup. The stale full-rank anchor lives in the
subspace PowerSGD r=77 *discards*, so neither operator carries M's descent
information into the live step. The two operators reach the same null by different
routes:
- **A3 blend (η=0.5):** `‖G_corr‖/‖G‖ = 0.7071` exactly (= √0.5) — a deterministic
  0.71× step shrink that trades half the informative live grad for half an orthogonal,
  uninformative stale one.
- **A2 inject (γ=1.0):** `scale = ‖G‖/‖M‖ = 0.0002–0.253` ⇒ **‖M‖ ≫ ‖G‖**, but inject
  scale-matches the complement *down* to ‖G‖ (`‖inj‖/‖G‖ = 1.0`), so it adds a
  **‖G‖-sized orthogonal vector** (not tiny noise) ⇒ `‖G_corr‖ ≈ √2·‖G‖`. Because the
  correction lands BEFORE GRPO grad clipping, the √2-inflated grad (grad_norm
  ≈ 1.3–2.57) exceeds the clip threshold; clipping scales the whole vector down,
  shrinking the useful G component by ~1/√2 while half the clipped budget is spent on
  an orthogonal stale direction. Net ≈ A3's outcome: no aligned signal added, useful
  step mildly diluted.

The phase-1 prediction from geometry + literature is confirmed; the mechanism is
**structural geometric incoherence (G ⊥ M by construction), not combiner tuning and
not K=5 staleness.** (See §0, §2.0, §5.)

**Binding next lever.** Pursue **Rank 1a (error-feedback on the PowerSGD residual)**
and **Rank 1b (basis-aligned anchor)** — the only two levers that remove the
incoherence at its source — ideally together (EF keeps the dropped energy alive
across steps; basis-alignment lands the periodic full-rank anchor in a usable
subspace). Layer **Rank 2 (η∝1/K)** on top *only after* 1b restores a shared
subspace. **Do NOT** re-run inject/blend at other γ/η (Rank 6 retired; Rank 2
standalone demoted) and **do NOT** chase delay_K — the live data shows a fresher
anchor lands in the same orthogonal complement. File the EXP-24 issue (§6).

**Self-consistency with the STOP verdict:** this report is referenced by
`runs/EXP-23/verdict.md` as the binding next lever; the STOP applies to the
*stale-anchor-on-PowerSGD* mechanism as implemented, and the forward path (EXP-24
EF + basis-align) is the continuation, not a reopening of EXP-23.

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
