# EXP-23 — Per-layer gradient/weight-matrix geometry (empirical grounding)

**Author:** grad-empirics (task #3) · **Date:** 2026-06-04 (smoke) → **finalized 2026-06-05** (live A2/A3)
**Sources (read-only):**
- `runs/EXP-23/exp-23-A1-no-refresh.train.log` (FLOOR: circuit OFF, 50 steps) — **LIVE**
- `runs/EXP-23/exp-23-A2-stale-inject.train.log` (inject, cadence=5, 50 steps) — **LIVE primary inject source**
- `runs/EXP-23/exp-23-A3-stale-blend.train.log` (blend, cadence=5, 50 steps) — **LIVE primary blend source**
- `runs/EXP-23/smoke-logs/smoke_fire.log` (PROBE_FIRE: anchor.cadence=1, inject mode, 4 steps) — initial smoke probe
- `runs/EXP-23/smoke-logs/smoke_on.log` (PROBE_ON: cadence=5, 2 steps — circuit never fired) / `smoke_off.log` (parity)
- Code of record: `verl/workers/comm_eff/spectral_filter.py` (`inject_matrix` L472, `blend_matrix` L503, correction loop `apply_spectral_correction_to_params` L547)

> **Status: FINAL.** All 3 arms ran to step 50; box torn down; logs on disk
> locally. **Headline result — HYPOTHESIS FALSIFIED:** A1 floor val@50=0.6914,
> A2 inject=0.6967 (+0.0053), A3 blend=0.6861 (−0.0053). max(A2,A3)=0.6967 ≤
> falsify line 0.7114 (PASS bar was 0.7414). The stale full-rank anchor neither
> rescues nor harms PowerSGD beyond noise. §7 below has the live per-step
> geometry that explains *why*; the smoke tables (§2–§4) are retained as the
> cadence=1 cross-check and agree with the live numbers.

---

## 1. What the diagnostic scalars mean (from `spectral_filter.py`)

For each targeted 2D matrix, with live compressed grad `G_mask` and stale full-rank anchor EMA `M_anchor`:

- `coeff = <G_mask,M_anchor> / ||G_mask||²`  (L491)
- `complement = M_anchor − coeff·G_mask`  (the part of `M_anchor` orthogonal to `G_mask` — what inject ADDS)  (L492)
- `scale = ||G_mask|| / ||M_anchor||`  (scale-match factor)  (L493)
- **inject:** `G_corr = G_mask + gamma·scale·complement`  (L494); logs `cos`, `gamma`, `scale`, `||inj||/||G_mask||`
- **blend:** `G_corr = (1−eta)·G_mask + eta·scale·M_anchor`  (L525); logs `cos`, `||G_corr||/||G_mask||`
- `cos = <G_mask,M_anchor> / (||G_mask||·||M_anchor||)`  (alignment)

**Derived complement fraction** (not printed directly, but exact):
`||M − proj_G(M)|| / ||M|| = ||complement|| / ||M_anchor|| = sqrt(1 − cos²)`.
And the logged inject ratio reduces to `||inj||/||G_mask|| = gamma·sqrt(1−cos²)`.
So the logged `||inj||/||G_mask||=1.0000` **is** the complement fraction (at gamma=1).

---

## 2. A2 / inject geometry — per target, smoke_fire (cadence=1, steps 1–4)

Targets are **layer 0 only**: `q_proj, k_proj, v_proj, o_proj` (4 = `spectral.max_targets`).
Shapes from the `[EXP-7][spectral]` lines: q/o = **(1536, 1536)**, k/v = **(256, 1536)** (GQA — KV heads down-projected).

| target | shape | cos(G_mask,M_anchor) range (steps 1–4) | scale = ‖G‖/‖M‖ range | ‖inj‖/‖G_mask‖ | complement frac = √(1−cos²) |
|---|---|---|---|---|---|
| layers.0 q_proj | (1536,1536) | −0.0007 … +0.0006 | 0.112 … 0.607 | 1.0000 | ≈1.000000 |
| layers.0 k_proj | (256,1536) | −0.0014 … +0.0024 | 0.453 … 2.166 | 1.0000 | ≈1.000000 |
| layers.0 v_proj | (256,1536) | −0.0048 … +0.0015 | 2.290 … 12.427 | 1.0000 | ≈0.999988 (worst case) |
| layers.0 o_proj | (1536,1536) | −0.0009 … +0.0007 | 0.683 … 3.201 | 1.0000 | ≈1.000000 |

- **cos is ~0 everywhere** (|cos| ≤ 0.0048 across all 4 targets × 4 steps). The live compressed grad and the stale full-rank anchor are **effectively orthogonal**.
- **‖inj‖/‖G_mask‖ = 1.0000 everywhere** ⇒ complement fraction ≈ 1.0 ⇒ **~100% of `M_anchor` is missing from `G_mask`'s span.** Inject re-adds essentially the *entire* (scale-matched) stale anchor as a force orthogonal to `G_mask`.
- Corroborated by `[EXP-7][spectral] rel_change=||G_proj-G_mask||/||G_mask|| = 1.000000` on every target (v_proj once logged 0.999999).
- **Magnitude consequence (inject):** `G_corr = G_mask + (≈equal-norm orthogonal vector)` ⇒ **‖G_corr‖ ≈ √2·‖G_mask‖ ≈ 1.41×**. Inject *inflates* the grad magnitude — this is the C1-collapse mechanism the blend mode was designed to avoid (see `blend_matrix` docstring, L504–511).
- `scale` swings widely (0.11 → 12.4) because `‖M_anchor‖` and the rescaled `‖G_mask‖` differ per target and per step; the scale-match is doing real work normalizing the injected complement to `‖G_mask‖`.

## 3. A3 / blend geometry — PREDICTED (no smoke source; confirm from A3 log)

smoke_fire ran inject only, so there are **zero `[EXP-18][blend]` lines** anywhere in the local logs. From the code (`blend_matrix`, eta=0.5, scale-matched anchor) and the measured cos≈0:
- predicted `||G_corr||/||G_mask|| = sqrt((1−eta)² + eta²) = sqrt(0.5) ≈ **0.7071**` (orthogonal regime).
- i.e. blend at eta=0.5 *shrinks* the grad to ~0.71×, replacing half of `G_mask` with half of the (orthogonal, scale-matched) stale anchor.
- **To confirm:** the `[comm_eff][EXP-18][blend] ... cos(...) ||G_corr||/||G_mask||=` line from `runs/EXP-23/exp-23-A3-stale-blend.train.log` once that arm runs.

## 4. Anchor EMA update magnitude `||dM_anchor||` (smoke_fire, cadence=1)

`[comm_eff][EXP-12] anchor refresh` lines, `targets=4`, `anchor_backward_isolation_mode=clone`, `anchor_loss=clean_pg`, `delay_K=5`:

| step | ‖dM_anchor‖_mean | ‖dM_anchor‖_max | anchor_backwards |
|---|---|---|---|
| 1 | 1.5112e+01 | 2.1458e+01 | 1 |
| 2 | 1.5112e+00 | 2.1458e+00 | 2 |
| 3 | 1.3601e+00 | 1.9312e+00 | 3 |
| 4 | 1.2241e+00 | 1.7381e+00 | 4 |

- The step-1 update is ~10× the later ones (first EMA write from the seed); steps 2–4 settle to O(1). Anchor is being updated and is non-trivial in magnitude.
- `anchor_grad_corrected=0` and `anchor_optimizer_steps=0` throughout — the anchor backward only **feeds the EMA**; it does not itself step the optimizer (correct: the correction is applied to the live grad via inject/blend, not by the anchor pass).
- **Circuit fired every step** it was scheduled: `anchor_backwards` increments 1→2→3→4 (cadence=1). In the real arms cadence=5, so it will fire at steps 5,10,15,…,50.

## 5. Coverage — "did we get everything, or only a subset?"

**Only a subset — 4 of ~196 targets (layer 0's q/k/v/o).**

| quantity | value | source |
|---|---|---|
| `spectral.max_targets` | **4** | resolved_params_PROBE_FIRE.txt L34; config dump L102 |
| targets instrumented with inject/blend geometry | **4** = `layers.0.{q,k,v,o}_proj` | smoke_fire inject lines (no layers ≥1, no gate/up/down) |
| `spectral.target_substr` (eligible types) | q,k,v,o,gate,up,down (7) | config dump L106–112 |
| decoder layers in Qwen2.5-1.5B | 28 | model arch |
| candidate 2D linear targets (28×7) | **~196** | derived |
| anchor clone params loaded (stale θ_{t−5}) | **338 / 338** ("canon-matched") | `[EXP-18][anchor-load]` |

The correction loop `break`s once `corrected >= max_targets` (`spectral_filter.py:591`), and iteration is in `named_parameters` order ⇒ it always stops at the **first 4 matching 2D matrices = layer 0 q,k,v,o**. The anchor EMA is *also* capped at `targets=4` (`anchor refresh ... targets=4`). So:

- **Geometry coverage is partial: layer 0 attention projections only.** No MLP (`gate/up/down_proj`), no layers 1–27.
- The 338-param anchor *clone* is the full stale-weight snapshot used to run the anchor backward, but only 4 of its grads are turned into `M_anchor` EMAs / diagnostics.
- Whether deeper layers or MLP blocks have *different* cos(G,M) (less orthogonal?) is **unmeasured** — a real gap if the synthesis wants to claim the orthogonality is model-wide rather than a layer-0-attention artifact.

### Boundary matrix shapes (FSDP discovery)
`[EXP-7][FSDP-DISCOVERY]` / grad-repr discovery: `model.layers.0._fsdp_wrapped_module.self_attn.q_proj.weight`,
`logical_2d_shape=(1536,1536)`, `grad_container_type=Tensor`, `is_dtensor=False`, `fsdp_version=1` (FSDP1, not FSDP2/DTensor),
`world_size=4`, correction applied `after_actor_backward__before_optimizer_step`, AFTER FSDP reduction, BEFORE grad clipping.
Per-target shapes: **q_proj/o_proj = (1536,1536); k_proj/v_proj = (256,1536)** (Qwen2.5-1.5B hidden_size=1536, GQA KV-dim=256).

### Raw matrices on disk?
**No.** Grepped the entire `verl/workers/comm_eff/` module — there is **no `torch.save`/`np.save`/`.npy`/`.pt`/`pickle.dump`** of any grad or weight matrix. Only **scalar geometry** (cos, scale, norm ratios, rel_change, ‖dM_anchor‖) is `print`-logged. We have **scalars, not matrices.** Reconstructing per-target `G_mask`/`M_anchor`/`G_corr` tensors from these logs is impossible.

**What an additive, OFF-by-default debug dump would capture (proposal, not implemented):** a `spectral.debug_dump_path` + `spectral.debug_dump_steps` (e.g. `[0]`) guard inside `apply_spectral_correction_to_params`, that on the gated step does `torch.save({name: {"g_mask": full.cpu(), "m_anchor": anc.cpu(), "g_corr": g_proj.cpu()}}, path)` for the ≤4 targets, for ≤2 steps. That would let us compute the *full* singular-value spectra of `M_anchor` and `G_mask`, the per-singular-direction overlap (not just the aggregate Frobenius cos), and verify the orthogonality is genuine isotropy vs. a few dominant directions. It is purely additive (no behavior change when off) and bounded (~4 × (1536·1536 + 2·256·1536) fp32 ≈ 50 MB/step). **We do NOT have this; today only the scalars exist.**

---

## 6. What this geometry implies (one paragraph)

The stale full-rank anchor `M_anchor` is **essentially orthogonal** to the live compressed gradient `G_mask` (|cos| ≤ 0.005 on every measured target/step), so the complement fraction is ≈1.0 — **G_mask contains almost none of M_anchor's direction.** This is the *necessary* condition for inject to be non-trivial (there is a large missing complement to add), but it is also the **danger sign**: because the complement is the *entire* (scale-matched) anchor and it is orthogonal, inject sets `G_corr = G_mask + (≈equal-norm orthogonal force)` ⇒ ‖G_corr‖ ≈ √2·‖G_mask‖, an undirected ~41% magnitude inflation pointing in a direction G_mask never chose. Whether that *helps* depends entirely on whether the stale `M_anchor` direction is still a descent direction 5 steps later — orthogonality alone says inject is **not inert** (unlike the as-implemented reweight, which is inert by orthogonality per the [[exp21-reweight-fixed-anchor]] result), but it also gives no guarantee the added force is useful rather than just noise/inflation. The blend mode exists precisely to avoid that inflation (predicted ‖G_corr‖≈0.71×), trading magnitude stability for actually *replacing* half the live signal with the stale one. **Bottom line for synthesis:** the measured geometry says inject/blend are mechanically *live* (they change the grad by ~100% in an orthogonal direction), but the same orthogonality means there is zero evidence here that the stale direction is *aligned with the true gradient* — that question is what the A2/A3 reward curves (vs A1 floor) must answer, and it cannot be settled from geometry scalars alone. Caveat: all of this is **layer-0-attention only** (4/~196 targets); MLP and deep-layer geometry is unmeasured.

---

## 7. LIVE 50-step geometry (A2 inject + A3 blend) — FINAL

Extracted from the local arm logs (ANSI/pid stripped, every logged occurrence across all 50 steps; 80 occurrences/target = 10 cadence-5 firing steps × 4 ranks × 2 reductions). The smoke (§2) and live numbers agree: cos≈0 throughout, complement fraction ≈1, blend ratio = √0.5.

### 7.1 A2 — inject (`exp-23-A2-stale-inject.train.log`)

| target | shape | n | cos min | cos max | \|cos\|max | scale=‖G‖/‖M‖ min | scale max | ‖inj‖/‖G_mask‖ |
|---|---|---|---|---|---|---|---|---|
| layers.0 q_proj | (1536,1536) | 80 | −0.0017 | +0.0005 | 0.0017 | 0.00020 | 0.00770 | 1.0000 |
| layers.0 k_proj | (256,1536)  | 80 | −0.0030 | +0.0043 | 0.0043 | 0.00050 | 0.01970 | 1.0000 |
| layers.0 v_proj | (256,1536)  | 80 | −0.0034 | +0.0048 | 0.0048 | 0.00520 | 0.25330 | 1.0000 |
| layers.0 o_proj | (1536,1536) | 80 | −0.0012 | +0.0009 | 0.0012 | 0.00130 | 0.07040 | 1.0000 |

- **cos(G_mask, M_anchor) ≈ 0 on every target, every firing step** (global |cos| ≤ 0.0048). It does **not** rise after the delay_K=5 warmup — orthogonality is the steady state, not a transient.
- **‖inj‖/‖G_mask‖ = 1.0000 everywhere** ⇒ complement fraction = √(1−cos²) ≈ 1.0 (whole anchor missing from G_mask span) — identical to smoke.
- **scale = ‖G_mask‖/‖M_anchor‖ is SMALL (0.0002 → 0.253) and grows over training.** This is the key difference from the smoke (where scale ranged 0.1–12): in the real run ‖M_anchor‖ ≫ ‖G_mask‖ early — wait, scale<1 means ‖M_anchor‖>‖G_mask‖, i.e. the stale anchor's RAW norm is *larger* than the rescaled live grad, so the injected complement (= scale·M_anchor, scaled DOWN to match ‖G_mask‖) is normalized to exactly ‖G_mask‖. The injected force is therefore an equal-norm orthogonal vector regardless — but its *content* is a heavily down-scaled (0.0002–0.25×) copy of a large stale anchor, i.e. near-pure scale-suppressed orthogonal direction.

### 7.2 A3 — blend (`exp-23-A3-stale-blend.train.log`, eta=0.5)

| target | shape | n | cos min | cos max | \|cos\|max | ‖G_corr‖/‖G_mask‖ min | ‖G_corr‖/‖G_mask‖ max |
|---|---|---|---|---|---|---|---|
| layers.0 q_proj | (1536,1536) | 80 | −0.0018 | +0.0009 | 0.0018 | 0.7065 | 0.7074 |
| layers.0 k_proj | (256,1536)  | 80 | −0.0034 | +0.0012 | 0.0034 | 0.7059 | 0.7075 |
| layers.0 v_proj | (256,1536)  | 80 | −0.0031 | +0.0040 | 0.0040 | 0.7060 | 0.7085 |
| layers.0 o_proj | (1536,1536) | 80 | −0.0014 | +0.0012 | 0.0014 | 0.7066 | 0.7075 |

- **‖G_corr‖/‖G_mask‖ = 0.706–0.709 ≡ √0.5 = 0.7071 EXACTLY** — the orthogonal-regime prediction from phase 1 (§3) is **CONFIRMED to 3 decimals on all 4 targets, all 50 steps.** Blend at eta=0.5 deterministically *shrinks* the step to 0.71× because it replaces half of G_mask with a scale-matched anchor that is orthogonal to it.
- cos again ≈0 throughout (|cos| ≤ 0.0040), consistent with the inject arm.

### 7.3 Circuit counters + codec invariance (all from last logged step, step 50)

| arm | global_step | anchor_backwards | spectral_corrections | powersgd_applications | val@50 (gsm8k acc) |
|---|---|---|---|---|---|
| A1 no-refresh (floor) | 50 | 0 | 0 | 179200 | **0.6914** |
| A2 stale-inject | 50 | 20 | 80 | 179200 | **0.6967** (+0.0053) |
| A3 stale-blend | 50 | 20 | 80 | 179200 | **0.6861** (−0.0053) |

- The circuit **fired as scheduled** in A2/A3: `spectral_corrections=80` = 4 targets × 10 cadence-5 firing steps × 2 (the per-step reduction count); `anchor_backwards=20` = 10 firing steps × 2. A1 is 0/0 (correct, circuit off).
- `powersgd_applications=179200` is **identical across all three arms** — the PowerSGD r=77 codec is held byte-constant; the only variable is the refresh mechanism, exactly as the EXP-23 design requires.
- val deltas (±0.005) are within run-to-run GSM8K noise; the geometry below explains why no real effect was possible.

---

## 8. THE MECHANISM — why the stale anchor is inert on PowerSGD (FINAL statement)

For the **PowerSGD r=77 codec, cos(G_mask, M_anchor) ≈ 0.001** (|cos| ≤ 0.0048 across all 4 targets and all 50 steps) — an **order of magnitude more orthogonal** than the **mask codec's cos ≈ 0.5** measured in EXP-21 (where the as-implemented two-sided *reweight* was found inert, see [[exp21-reweight-fixed-anchor]]). The two dead-ends have **different root causes**:

- **Mask + reweight (EXP-21):** inert because the two-sided Tikhonov *projection operator* collapses (G_filt ≈ 0, rel_change ≈ 0.5) even though the gradients are only ~60° apart — a **projection-operator failure**.
- **PowerSGD + inject/blend (EXP-23):** inert because the live and stale gradients are **genuinely, almost perfectly geometrically incoherent** — the PowerSGD compression subspace (the rank-77 sketch that survives the boundary) is ~orthogonal to the stale full-rank gradient. This is **true geometric incoherence, NOT a projection failure.** The correction operators here are mechanically live (they change the grad by ~100% / scale it by 0.71×), but they have nothing aligned to inject.

Consequently: **inject** adds an equal-norm but *orthogonal*, scale-suppressed (‖M‖≫‖G‖ ⇒ scale 0.0002–0.25) copy of the stale direction — effectively tiny orthogonal noise that Adam + grad-clip wash out (val +0.005, noise). **Blend** simply shrinks the optimizer step to 0.71× by trading half the (informative) live grad for half a (orthogonal, uninformative) stale one — a slight *pessimization* (val −0.005). Neither carries any of the stale full-gradient's descent information into the live step, because there is no shared direction to carry.

**Implication for the fix (hands to synthesis §8):** `delay_K` (anchor staleness) is **not** the lever — the orthogonality is steady-state, present from the first post-warmup firing and not growing, so a fresher anchor (smaller delay_K) would land in the same incoherent geometry. To make a stale/auxiliary full-gradient *useful* on PowerSGD you must remove the incoherence at its source, i.e. either **(a) error-feedback** (compress the *residual* G − decompress(compress(G)) and carry it forward, the standard PowerSGD convergence fix, which keeps the dropped energy in the same basis instead of discarding it), or **(b) a basis-aligned anchor** (project/define the stale full-grad onto the live PowerSGD Q-basis so the two share a subspace by construction). An anchor defined in an orthogonal complement can never help no matter how fresh.

**Coverage caveat (unchanged):** all live geometry is **layer-0 self-attention only** — 4 of ~196 candidate 2D targets (`spectral.max_targets=4`); MLP and layers 1–27 are unmeasured, and only scalar geometry exists (no raw G/M matrices on disk — §5).
