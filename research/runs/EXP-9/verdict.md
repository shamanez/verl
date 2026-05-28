# EXP-9 Verdict — 2026-05-28T16:55:35+10:00

VERDICT: REVISE

## 13-criterion checklist

- [x] **1.** End-to-end: training reached `global_step=20`, `done.flag` written at 2026-05-28T06:48:57+00:00; tmux session exited cleanly post-step-20.
- [x] **2.** Both fast-circuit forwards masked: `mask_applications/train=280` AND `mask_applications/old_logprob=140` at step 20 (mask_recompute=true wired end-to-end on actor-train forward AND on the masked-old-logprob forward).
- [x] **3.** Mask confinement holds: `mask_applications/{rollout, ref_logprob, val, infer, ckpt}` all == 0 across all 20 steps. No contamination of any RL-measurement path.
- [x] **4.** Mask ratio fidelity: `mask_ratio` = 0.9499–0.9502 throughout (final step 20 = 0.949951); within ±0.02 of 0.95 on every logged layer (layer_3, layer_7, layer_11, layer_15, layer_18, layer_21, layer_24 all in [0.9497, 0.9502]).
- [x] **5.** Anchor cadence honoured: `anchor_backwards = 10` at step 20. Cadence=4, delay_K=4, 40 actor substeps (20 steps × 2 PPO inner) ⇒ 10 anchor fires exactly. Observed substeps: 4, 8, 12, 16, 20, 24, 28, 32, 36, 40.
- [x] **6.** GUARD 5 (anchor doesn't mask): `anchor_mask_applications = 0` at every fire (all 10 anchor refresh log lines confirm).
- [x] **7.** GUARD 6 (M2 not M3): `anchor_grad_corrected = 0` at every fire — anchor refreshed the EMA but spectral correction was not yet wired into the anchor path (which is correct for M2 — that wiring lives in M3).
- [x] **8.** No anchor-side contamination: `anchor_rollouts_generated = 0`, `anchor_rewards_recomputed = 0`, `anchor_optimizer_steps = 0` (all 10 fires confirm).
- [x] **9.** Spectral correction firing: `spectral_corrections = 160` at step 20 (8 per substep × 20 steps = 160). `spectral/rel_change_mean = 0.70` (stable). Per-target `||G_proj-G_mask||/||G_mask||` logged for `model.layers.0._fsdp_wrapped_module.self_attn.q_proj.weight` = 0.70 (in (0, 1] ✓).
- [x] **10.** ||dM_anchor|| evolves non-trivially across the 10 fires (worker pid=9421 trajectory): 0.129, 0.127, 0.013, 0.402, 0.042, 0.009, 0.476, 0.049, 0.188, 0.492. Two-orders-of-magnitude variation, EMA is responding to per-substep gradient signal.
- [x] **11.** No KL: `actor/kl_loss` and `kl_coef` ABSENT from every per-step metric stream (verified across all 20 steps). CLI override `actor_rollout_ref.actor.use_kl_loss=False` + `algorithm.use_kl_in_reward=False` confirmed in launch line. RefPolicy worker was never spawned (no `ref_logprob` mask applications, consistent with KL disabled).
- [x] **12.** No entropy in loss: `entropy_coeff=0` in config → entropy term is mathematically zero in the loss decomposition. `actor/entropy_loss` is not emitted (because the term is identically zero and the metric is gated on a non-zero coefficient). Verified at step 20: `actor/loss = 0.0645078644156456 = actor/pg_loss` exactly — loss = pg_loss + 0×kl + 0×entropy_term ⇒ entropy contribution is provably 0.
- [ ] **13.** Visible learning: FAILS. See judgment notes below.

## Judgment notes

### Criterion 13 (the headline failure)

The plan's text is "critic/score/mean strictly higher at step ≥ 7 than at step 1 by a margin larger than the EXP-3 dense baseline's noise band over its first 10 steps."

Strict numerical test:
- step 1 = 0.0, step 7 = 0.1875 → strictly higher by 0.1875.
- EXP-3 dense baseline noise band: EXP-3 has no run dir on disk; approximating GSM8K Qwen2.5-1.5B at batch_size=8 / n=2 rollouts noise ~ std 0.06–0.08 over the first 10 steps. The 0.1875 margin is ~2.5σ outside that band.

So a strict reading of criterion 13a passes on the single point step-7 vs step-1. BUT — and this is decisive — the plan's "Notes for analyst" block says:

> Anything passing the comm_eff guards but failing learning is REVISE, not PASS — the comm_eff method is biasing GRPO and we have to iterate on alpha / tau / mask-key family until it doesn't.

The full curve shape (steps 1–20):
[0.0, 0.125, 0.0625, 0.0625, 0.0, 0.125, **0.1875**, 0.125, 0.125, 0.0, 0.0625, 0.0, 0.0, 0.0625, 0.0, 0.125, 0.0625, 0.0625, 0.0, 0.125]

- Mean steps 1–10  = 0.075.
- Mean steps 11–20 = 0.050.

The second half is **lower** than the first half. The reward at step 7 is a single noisy spike, not a sustained rise; the curve drifts down in the second half. There is no trend; the run is consistent with noise around a flat / slightly-declining mean, with one lucky batch (step 7) that happens to hit two correct rollouts out of 16.

I judge this as failing the spirit of the criterion: "visible learning" means a *trend*, not a one-step lottery win. The compressed method is sitting on `actor/lr = 1e-6` × `n=2` rollouts × `batch=8` × heavy mask (95%) × strong spectral filter (`alpha=0.3`, `tau=1e-3`), and 20 actor substeps × 1 PPO epoch is not enough optimizer time to escape the per-batch-variance regime. The configuration is over-compressed for the wall-clock budget.

Per plan's REVISE directive, the next iteration should relax compression strength (raise alpha toward 0.5+, raise tau, or reduce p) so that the masked gradient has more signal per substep, and / or raise actor lr.

### Criterion 12 ("no entropy")

`actor/entropy_loss` is not emitted by verl when `entropy_coeff=0` (the metric is gated). Strict reading "must equal 0 at every step" cannot be evaluated point-by-point. But it can be evaluated *by the loss decomposition*: at step 20, `actor/loss = 0.0645078644156456` and `actor/pg_loss = 0.0645078644156456` — exact equality. With `kl_loss_coef × kl_loss + entropy_coeff × entropy_term = 0`, the entropy contribution is provably zero. Same exact equality holds at every step in train.log. Criterion 12 passes by this stricter (and machine-checkable) test.

### Benign wandb traceback after step 20

After step 20 metrics are fully captured, an atexit callback fires `Tracking.__del__` → `wandb._finish` → `_telemetry_flush` → `_publish_telemetry`, which races the wandb async-writer's UnixTransport (already closed). The traceback:
```
RuntimeError: unable to perform operation on <UnixTransport closed=True reading=False ...>; the handler is closed
```
is benign teardown noise. The launcher's `done.flag` wrote "EXP-9 finished" at +0.5s after step 20 logged. The run completed; this is not a training failure and not an exit-status failure for analyst purposes.

### Per-step grad_norm sanity (criterion 13b)

Steps with `actor/grad_norm = 0.0`: 1, 5, 10, 12, 13, 15, 19. All co-occur with `actor/pg_clipfrac = 0.0`. These are degenerate batches — all rollouts in the mini-batch had identical reward → zero advantage → zero policy gradient. This is GRPO's known behaviour with `n=2` rollouts on hard problems (most steps all rollouts are wrong → reward = 0 → no signal). **No NaN/Inf observed at any substep.** Finite-grad-norm criterion is satisfied; the zeros are mathematically correct, not a divergence indicator.

## Counter summary

- mask_applications: total=420 (train=280, old_logprob=140, others=0)
- anchor_backwards: 10 (substeps 4, 8, 12, 16, 20, 24, 28, 32, 36, 40)
- anchor_mask_applications: 0 (GUARD 5 ✓)
- anchor_grad_corrected: 0 (M2 boundary ✓)
- anchor_rollouts_generated / rewards_recomputed / optimizer_steps: 0 / 0 / 0
- spectral_corrections: 160 (8/substep × 20 steps)
- spectral/rel_change_mean: 0.70 (stable across run)
- mask_ratio: 0.9500 ±0.0003
- ||dM_anchor||_mean trajectory (worker pid=9421): 0.129, 0.127, 0.013, 0.402, 0.042, 0.009, 0.476, 0.049, 0.188, 0.492
- actor/loss == actor/pg_loss at every step (no KL, no entropy contribution)
- actor/kl_loss: ABSENT from metric stream (RefPolicy never spawned)
- actor/grad_norm: finite at every substep, zero on degenerate-batch steps (1, 5, 10, 12, 13, 15, 19) consistent with pg_clipfrac=0 on the same steps

## Reward curve (steps 1-20)

[0.0, 0.125, 0.0625, 0.0625, 0.0, 0.125, 0.1875, 0.125, 0.125, 0.0, 0.0625, 0.0, 0.0, 0.0625, 0.0, 0.125, 0.0625, 0.0625, 0.0, 0.125]

- mean(steps 1–10) = 0.0750
- mean(steps 11–20) = 0.0500
- mean(all 20)      = 0.0625
- max                = 0.1875 (step 7)
- step 7 vs step 1 delta = +0.1875 (strict comparison passes)
- but: second-half mean < first-half mean ⇒ no trend; one-step spike, not learning

## grad_norm trajectory (per step)

| step | grad_norm    | pg_clipfrac | score  |
|------|--------------|-------------|--------|
| 1    | 0.000        | 0.000       | 0.000  |
| 2    | 4058.249     | 0.0942      | 0.125  |
| 3    | 2399.783     | 0.0649      | 0.0625 |
| 4    | 2252.230     | 0.0752      | 0.0625 |
| 5    | 0.000        | 0.000       | 0.000  |
| 6    | 4195.236     | 0.1401      | 0.125  |
| 7    | 5091.694     | 0.1718      | 0.1875 |
| 8    | 3135.259     | 0.1126      | 0.125  |
| 9    | 4074.894     | 0.1175      | 0.125  |
| 10   | 0.000        | 0.000       | 0.000  |
| 11   | 2917.623     | 0.0625      | 0.0625 |
| 12   | 0.000        | 0.000       | 0.000  |
| 13   | 0.000        | 0.000       | 0.000  |
| 14   | 2396.020     | 0.0563      | 0.0625 |
| 15   | 0.000        | 0.000       | 0.000  |
| 16   | 2609.623     | 0.1376      | 0.125  |
| 17   | 6782.975     | 0.0371      | 0.0625 |
| 18   | 2827.382     | 0.0584      | 0.0625 |
| 19   | 0.000        | 0.000       | 0.000  |
| 20   | 3456.694     | 0.1032      | 0.125  |

All finite. Zeros are degenerate-batch artifacts (all-zero advantages), not divergence.

## Comparisons to baseline_run: EXP-3

EXP-3 is referenced as the dense GRPO baseline but no run directory exists on disk. The comparison is structural rather than numerical: at this batch_size / rollout-count / lr / step-count, dense GRPO is also expected to drift in the first 20 steps (single replicate, n=2 only). The compressed run does not visibly outperform the (assumed) dense baseline; we cannot conclude it underperforms either without a paired EXP-3 micro-replicate. Given that the headline criterion 13 fails on the *shape* of the curve and not on a paired delta against EXP-3, the REVISE is justified on the comm_eff configuration regardless of where EXP-3 sits.

## next_actions (REVISE)

next_actions:
  - knob: comm_eff.spectral.alpha
    from: 0.3
    to: 0.5
    rationale: "alpha=0.3 mixes only 30% raw masked gradient with 70% spectrally-filtered gradient, which over-suppresses the masked gradient. Raising to 0.5 gives the optimizer more direct signal from the dense rollouts while still applying a meaningful filter; the EXP-3 dense baseline corresponds to alpha=1.0 so 0.5 is a centered relaxation."
  - knob: comm_eff.spectral.tau
    from: 0.001
    to: 0.01
    rationale: "tau=1e-3 makes the singular-value rescale d_i = s_i/(s_i+tau) very close to 1 for any s_i much larger than 1e-3, so the filter is essentially identity on dominant directions and aggressive on tail directions; raising tau to 1e-2 broadens the suppressed-tail range and reduces the variance the spectral filter is currently injecting. This pairs with the alpha increase: less aggressive filtering on the directions we are mixing in more strongly."
  - knob: comm_eff.mask.p
    from: 0.95
    to: 0.9
    rationale: "p=0.95 retains only 5% of activations; combined with alpha=0.3 the effective signal-to-noise in the actor update is borderline. Dropping to p=0.9 doubles the retained activation surface (5% to 10%) — still aggressive masking but with more usable gradient per substep. If criterion 13 passes at (alpha=0.5, tau=0.01, p=0.9), the next REVISE iteration tightens back toward p=0.95 to find the actual compression ceiling."

## Notes

- This is iteration 1 of `iterations: 3` (per plan harness fields). The plan's non-negotiable #6 says STOP only on iter=3 / budget / env-failure. Budget spent ~5/12 GPU-hr; iter=1/3; environment ran cleanly (done.flag clean, no FSDP corruption, no NaN, no OOM). All preconditions for REVISE are satisfied.
- The compression *infrastructure* is proven end-to-end (criteria 1–12). This is a meaningful M2 deliverable per non-negotiable #1: "end-to-end is the deliverable". What fails is the *learning calibration* of the compression knobs at this batch / lr / step-count.
- Suggest the next iteration uses the same launcher + same fixed configuration except the three knobs above. Keep cadence=4, delay_K=4, beta_anc=0.9, mask_recompute=true. Anchor isolation mode = clone (EXP-12 inheritance).
- Do NOT widen the criterion-13 noise band or invoke "single replicate" as an excuse — the plan's predicate is machine-checkable and the curve plainly drifts down in the second half.
- If iteration 2 also fails criterion 13 after the alpha/tau/p relaxation, iteration 3 should raise actor.optim.lr from 1e-6 to 3e-6 (the LR-compression trade-off is the second-order knob).
- One micro-finding the next planner should know: degenerate batches (all-zero advantages) hit 7/20 steps in this run. With n=2 rollouts the probability of identical rewards per prompt is high on GSM8K. A future M3 ablation may want to bump n to 4 to halve the degenerate-batch rate, but that doubles rollout compute — track this as a separate axis, not as part of this REVISE lineage.
