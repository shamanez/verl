# EXP-21 — Original spectral correction (`reweight`) on the FIXED anchor

> **Confirmatory negative-control / de-confounding run.** Closes the one open loop
> left by EXP-18/M4: *"spectral `reweight` was only ever tested on the BUGGY anchor —
> never re-run on the fixed true-gradient anchor."* This run re-runs it on the fixed
> anchor. Operator-requested ("run a quick test with the previous spectral + anchor
> method"), with the explicit constraint: **use the correct FSDP anchor.**

## Verdict: **reweight is orthogonality-INERT — fixing the anchor changed nothing.**

The original two-sided Tikhonov spectral projection (`correction_mode=reweight`),
run on the **fixed** true-gradient anchor (`anchor_pg_loss` + `_canon`), produces
**flat reward at the masked floor** — statistically identical to running it on the
old broken anchor. This **isolates the dead end to the projection operator itself**,
not the anchor, and retroactively de-confounds the EXP-16/17 `spectral_baseline` result.

## What was tested
Byte-identical to the proven EXP-18 C5 **blend** command EXCEPT the single mode flip
`spectral.correction_mode=blend → reweight` (and `blend_eta` dropped, ignored in reweight).
- Fixed anchor VERIFIED in the running code before launch (SHA-256 local==remote + greps):
  `def anchor_pg_loss`=1, `def _canon`(anchor.py)=1, `def _canon`(spectral_filter.py)=1,
  `_build_anchor_pg_loss`(transformer_impl.py)=2. Payload = local `vast-ai-workload` @ `88bfeba2c`.
- Knobs: `mask.p=0.9 rescale=true`, `anchor.enabled=true cadence=5 delay_K=5`,
  `spectral.correction_mode=reweight alpha=0.5 tau=0.01 beta_anc=0.0 svd_mode=full basis_cache=cache`,
  `clean_cadence=0`, `total_training_steps=50`, val OFF.

## Result (reward = `critic/score/mean`; WandB run `shamanework-pl/comm_eff_curve_match_m4/j8rnzan2`)

| curve | anchor | correction | n | last | mean | mean\|Δ vs dense\| | final\|Δ\| | verdict |
|---|---|---|---|---|---|---|---|---|
| dense_ref | — | none | 50 | 0.841 | 0.731 | 0.000 | 0.000 | bar |
| blend C5 | **fixed** | blend η0.9 | 50 | 0.813 | 0.661 | **0.070** | 0.027 | ✅ recovers dense |
| **reweight (this run)** | **fixed** | reweight | 43 | 0.118 | **0.137** | **0.577** | 0.714 | ✗ inert (floor) |
| spectral_baseline (cached) | buggy | reweight | 50 | 0.145 | 0.135 | 0.596 | 0.696 | ✗ inert (floor) |

```
step |  dense | blend  | reweight-FIXED | reweight-BUGGY
   5 |  0.396 |  0.187 |     0.121      |    0.153
  10 |  0.750 |  0.583 |     0.136      |    0.152
  20 |  0.778 |  0.735 |     0.149      |    0.137
  40 |  0.799 |  0.752 |     0.138      |    0.135
  43 |  0.832 |  0.782 |     0.118      |    0.138
```

**reweight-FIXED ≈ reweight-BUGGY** (mean 0.137 vs 0.135). Fixing the anchor made no
difference to reweight. On the *same* fixed anchor, **blend climbs to 0.81, reweight
stays at 0.14.**

## The mechanism — `rel_change` with a WARM true-gradient anchor
The spectral diagnostic `rel_change = ‖G_proj − G_mask‖/‖G_mask‖` held at **0.48–0.50**
throughout — including **after** the anchor warmed (it fired 17 backward refreshes,
cadence=5, on 196 target matrices). With `α=0.5`, `G_proj = 0.5·G_mask + 0.5·G_filt`,
so `rel_change ≈ 0.5` ⟺ `G_filt ≈ 0`: the projection of `G_mask` onto `M_anchor`'s SVD
basis is empty. Since `cos(G_mask, g_true) ≈ 0.08` (near-orthogonal), `G_mask` lies
almost entirely outside the true gradient's principal subspace, so projecting onto it
recovers ~nothing — **regardless of how correct or fresh `M_anchor` is.** The projection
operator cannot *synthesize* the missing orthogonal direction; it can only *reweight*
components `G_mask` already has. Blend works precisely because it **replaces/adds** the
true gradient rather than projecting onto it.

## Why this matters
- **De-confounds** EXP-16/17: `spectral_baseline` failed by orthogonality, NOT because
  the anchor was broken. Both anchors → same floor.
- **Isolates** the M4 win: blend-vs-reweight on an identical correct anchor is a clean,
  single-variable contrast (`project` vs `replace`). mean|Δ| 0.070 (blend) vs 0.577 (reweight).
- Confirms the orthogonality theory holds on the corrected circuit — `reweight`/`inject`
  remain documented dead-ends; `blend` is the live mechanism.

## Run lifecycle / infra note
- Instance 39246336 (4×H200, $15.53/hr). Reached **step 43/50** then the Stop hook
  auto-tore-down on `teardown_reason: no-heartbeat-30min`. **Box destruction
  independently VERIFIED via `vastai show instances` → 0 instances. No money leak.**
- The premature stop is a **monitoring-infra gap**, not an experiment failure: the
  training-log-monitor SSH-polls the box but does **not** refresh the heartbeat file the
  Stop hook's 30-min staleness check reads, so the hook judged the run stale and killed it.
  The 7 missing steps (44–50) do **not** change the conclusion — 43 steps of dead-flat
  reward (σ≈0.012) + `rel_change` locked at 0.5 + the blend comparator at 0.81 are
  conclusive. (Future re-runs: have the watcher touch the heartbeat, or raise the
  staleness threshold, to let a clean run finish 50.)

## Provenance
- WandB: `shamanework-pl/comm_eff_curve_match_m4`, run `j8rnzan2` (43 steps).
- Local curve: `runs/EXP-21/metrics/curvematch_reweight_fixedanchor_c5_d5.jsonl`.
- Code: `vast-ai-workload` @ `88bfeba2c` (merged #11; carries the fixed anchor).
- Box train log unrecoverable (teardown preceded rsync) — WandB is the source of truth.
