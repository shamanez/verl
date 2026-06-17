# EXP-34 — signed_ema(α=0.5) β_anc sweep {0.25, 0.50, 0.75} — RESULTS (INTERIM)

**Status:** INTERIM as of 2026-06-17 ~12:31Z — cells 1 & 2 complete, **cell 3 (β=0.75) still running**.
Box: operator-provided team-account Vast instance **41292294** (4×H200). `vast_account=team`.
W&B project: `verl_compression_research_beta_sweep_signed_ema`.

## Method validity (confirmed at runtime)
- Merger firing: `[comm_eff][merger] correction_mode=signed_ema alpha=0.5 corrected=196` on all 196 matrices; `spectral_corrections` +392/step (196×2 mini-batches).
- Resolved config carries `correction_mode=signed_ema`, `signed_ema_alpha=0.5`, per-cell `beta_anc`, `val_before_train=False`, project name — verified (last-wins over the B2 wrapper's delayed_ef/β=0 exports).
- signed_ema forward path intact (`spectral_filter.py:393 signed_ema_matrix`); commit `421567ec6` changed the dataclass *default*, not the implementation. Identical signed_ema math to EXP-32.
- Box-compat break-glass `DISABLE_CUSTOM_ALL_REDUCE=true` (NCCL all-reduce) — required; attempt 1 crashed at vLLM `custom_all_reduce.cuh:455` KV-cache init. Greedy-val-neutral.

## Results (greedy val-core/openai/gsm8k/acc/mean@1)

| cell | β_anc | val@25 | **val@50** (headline) | val@55 (end-of-train, unplanned) |
|---|---|---|---|---|
| signed_ema_b0p25 | 0.25 | 0.7271 | **0.7612** | 0.7384 |
| signed_ema_b0p50 | 0.50 | 0.7430 | **0.7635** | (pending) |
| signed_ema_b0p75 | 0.75 | (pending) | (pending) | (pending) |

**Reference points** (not re-run): EXP-32 signed_ema β=0 val@50 = **0.7271**; B2 delayed_ef SOTA = **0.7528**; dense band ≈ 0.75–0.78 (apples-to-apples draw 0.7839).

## Headline (vs the +0.024 surpass bar over the β=0 reference 0.7271 ⇒ bar = 0.7511)
- Cell 1 (β=0.25): val@50 0.7612 = **+0.0341** over ref — clears bar; also > B2.
- Cell 2 (β=0.50): val@50 0.7635 = **+0.0364** over ref — clears bar; also > B2.
- Both nominally surpass. **Two-cell consistent positive signal** that β_anc > 0 lifts signed_ema — contrary to EXP-33's *flat* β curve on the delayed_ef merger.

## Caveat (important — not yet a PASS verdict)
- **Eval-noise–blurred.** Cell 1's val@50 (0.7612) vs its own val@55 (0.7384) differ by 0.0228 ≈ the ±0.024/draw eval noise, from the same weights 5 steps apart. So each val@50 is a single noisy draw; the "+0.034" may be a high draw.
- The averaged late-draw level for β-averaged signed_ema is ≈ 0.75 (≈ B2 / dense band), i.e. a likely real **lift over the β=0 reference (0.7271)** but a **tie with B2/dense** at the exact magnitude.
- A **replicate** (re-run a winning β at a fresh seed/draw) is warranted before promoting — likely a REVISE the analyst/operator green-lights.
- Final PASS/STOP is the analyst's call after cell 3 + the plan predicate (`best_cell_val@50 − 0.7271 > 0.024`, weighing noise).

## Correctness
No NaN/Inf, no length-ignition (response_length stayed ~150–220, well below cap), no collapse on cells 1–2. `bytes_ratio = 0.0504` (PowerSGD r=77).

## val@55 handling (operator directive 2026-06-17)
No end-of-training val@55 going forward (`TOTAL_TRAINING_STEPS=50`, not 55 — see FIXED_CONTROL_SURFACE / `no-end-of-training-val55` memory). Cells 1 & 2 already ran total=55 so their val@55 (0.7384, and cell-2's pending) exist but are **informational only — ignored for the headline** (val@50). Cell 3 (in-flight on total=55) is honored by tearing the box down the instant its **val@50** is captured (skips steps 51–55 + val@55).

## Teardown
Box `41292294` is torn down with the **team key** (vast_account=team) the instant **cell 3's val@50 is captured + rsynced** (NOT waiting for step 55 / done.flag) — minimizes box cost AND skips val@55. NO keep-warm. Analyst runs offline from the laptop after teardown. (Backstop: teardown Stop hook on verdict / >30-min stale heartbeat / budget, team-key auth via `vast_account=team`.)
