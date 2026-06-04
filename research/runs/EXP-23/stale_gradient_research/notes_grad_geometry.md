# EXP-23 — Per-layer gradient/weight-matrix geometry (empirical grounding)

**Author:** grad-empirics (task #3) · **Date:** 2026-06-04
**Sources (read-only):**
- `runs/EXP-23/smoke-logs/smoke_fire.log` (PROBE_FIRE: anchor.cadence=1, inject mode, 4 steps) — **primary inject geometry source**
- `runs/EXP-23/smoke-logs/smoke_on.log` (PROBE_ON: anchor.cadence=5, 2 steps — circuit never fired, see below)
- `runs/EXP-23/smoke-logs/smoke_off.log` (PROBE_OFF: PowerSGD r=77, circuit OFF — no diagnostics, as expected)
- `runs/EXP-23/smoke-logs/resolved_params_PROBE_FIRE.txt`
- Code of record: `verl/workers/comm_eff/spectral_filter.py` (`inject_matrix` L472, `blend_matrix` L503, correction loop `apply_spectral_correction_to_params` L547)

> **Status of LIVE 50-step arm data (A2 inject / A3 blend):** PENDING. As of the
> last monitor poll (`monitor-detail.log`, POLL 2 @ 12:37Z) **A1 is running**
> (anchor OFF — the floor, emits NO inject/blend lines) and **A2/A3 have not
> started** (`A2_log: NOT_CREATED_YET`, `A3_log: NOT_CREATED_YET`). The numbers
> below are from the **smoke probe**, which exercises the identical inject
> circuit on the identical model/codec — they are representative of what A2 will
> log, but they are not the 50-step arm numbers. **A3 blend has NO smoke source**
> (smoke_fire used `correction_mode=inject`); the blend prediction below is
> derived from the code + measured cos, to be confirmed when A3 logs land.

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
