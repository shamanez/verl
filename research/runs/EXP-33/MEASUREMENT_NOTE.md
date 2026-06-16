# EXP-33 — val_before_train measurement optimization (operator-directed 2026-06-16)

## What changed
Only the **control cell C0 (`b0p00`)** runs `trainer.val_before_train=True`.
Cells **C1–C4 (`b0p25`, `b0p50`, `b0p75`, `b1p00`) run `val_before_train=False`** and
reuse C0's step-0 validation value.

**C0 val@0 (the shared reference for all cells): `val-core/openai/gsm8k/acc/mean@1 = 0.0819`.**

## Why this is exact, not an approximation
`val_before_train` validates at global_step 0, **before any gradient is applied**. At that
point the model is the untrained base `Qwen2.5-1.5B-Instruct` — **identical for every cell**.
`beta_anc` only affects the gradient *merger* applied at the optimizer step (step ≥1), so it
has **zero** effect on val@0. Greedy (deterministic) validation ⇒ val@0 is **byte-identical**
across all 5 cells. Re-running it per cell is pure wasted compute (one full greedy generation
pass over the GSM8K test set, ~1319 problems, per cell).

## Why this does NOT break the controlled-variable / off-axis-parity contract
`val_before_train` is a **measurement-cadence knob** in the same category as `test_freq` and
`total_training_steps` — both of which the plan (§Success criteria) explicitly lists as
**allowed deltas**. It runs no backward pass and touches no weights, so the **training
trajectory is unchanged** and `val@25`/`val@50` (the actual β→accuracy curve) are unaffected.
The training / communication / generation axes remain byte-identical to B2.

## Instruction for the analyst
Treat `trainer.val_before_train` as an **allowed measurement delta** (alongside
`experiment_name`, `project_name`, step count, `test_freq`). Do **NOT** classify C1–C4 as
off-axis-parity failures on this basis — **score them normally**. Use **C0's val@0 = 0.0819**
as the step-0 reference for every cell when tabulating the curve.

## Clean mechanism for FUTURE sweeps (no wrapper edit needed)
This run applied the skip via a box-side edit of the b2_sota wrapper (the in-flight driver
couldn't be changed). For future config-only sweeps, the driver should instead pass it as a
**Hydra passthrough** for non-control cells — last-wins over the wrapper's hard
`export VAL_BEFORE_TRAIN=True`, exactly like the `beta_anc` passthrough:

```bash
# control cell: full val (gets val@0 once)
... bash <b2_sota_wrapper> actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.00
# every other cell: skip the redundant val@0
... bash <b2_sota_wrapper> actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.25 trainer.val_before_train=false
```
