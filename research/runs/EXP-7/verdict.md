VERDICT: REVISE

# Verdict EXP-7 — 2026-05-28T04:55:00+10:00

## Result
VERDICT: REVISE

## Success criteria
- [ ] (1) Unit test `tests/workers/comm_eff/test_spectral_filter.py` (alpha=1 no-op, alpha=0 Tikhonov, shape preservation, determinism) — NOT EVALUABLE on Vast (observed: no unit-test output in either train log; this box is a codex-verify/CI artifact, not a Vast-cell artifact, and no test result reached this run dir).
- [x] (2) Smoke reaches `global_step=2`; `actor/grad_norm` finite every substep; no NaN/Inf (observed: both cells hit `training/global_step:2`, "Training Progress: 100%|2/2"; `actor/grad_norm:0.0` at step1 and step2 — finite; no NaN/Inf in any loss/grad_norm/reward/log_prob field; the only Tracebacks are post-run wandb-socket + DataLoader-worker teardown noise, after the step-2 summary). NOTE: grad_norm is finite but identically 0.0 — see criteria 4/5.
- [ ] (3) Logs the gradient representation (Tensor/DTensor/FlatParameter/local shard), its shape vs logical 2D matrix, and correction point relative to FSDP reduction + clipping, for ≥1 target matrix (observed: ABSENT — exhaustive grep of both logs for `p.grad`/`DTensor`/`FlatParameter`/`placement`/`device_mesh`/`reduction`/`clip`/`G_proj`/`G_mask`/`rel_change` returns nothing beyond the config dump. The headline FSDP discovery output was never emitted). THIS IS THE LOAD-BEARING DELIVERABLE PER `## Notes for analyst`.
- [ ] (4) Per-target `||G_proj - G_mask|| / ||G_mask||` in `(0, 1]` for `alpha=0.3` (observed: ratio not just <=0 but UNDEFINED — `actor/comm_eff/spectral_corrections:0.0` at step1 AND step2; the filter never fired on a non-trivial gradient. No rel_change line logged. Ratio==0 / silent no-op → REVISE per plan's own analyst note, target: in (0,1]).
- [ ] (5) ≥1 actor parameter changes step0→step2 (observed: NOT DEMONSTRATED — gradient was identically 0.0 every substep, so AdamW applied (lr=1e-6 × ~0 grad); no param-delta line logged and a zero gradient cannot be claimed to move a param meaningfully. Target: ≥1 confirmed delta).
- [x] (6) `enabled=false` regression is a true no-op matching dense/EXP-5 (observed: disabled cell trajectory `pg_loss:0.0 / grad_norm:0.0 / spectral_corrections:0.0 / rewards:0.0` is identical in structure to the spectral_on cell; `spectral_corrections:0.0` in both → enabled=true added no spurious behavior on a zero gradient. Confirmed equal in this degenerate regime; a non-degenerate equality could not be tested because no cell produced a non-zero gradient).

## Metrics summary
- spectral_on actor/grad_norm: 0.0 (step1), 0.0 (step2) — finite (target: finite, non-trivially demonstrate correction)
- spectral_on actor/comm_eff/spectral_corrections: 0.0 (step1), 0.0 (step2) — target >0
- spectral_on critic/rewards/mean: 0.0 (step1), 0.0 (step2) — root cause of zero gradient
- spectral_on critic/advantages/mean: 0.0 (step1), 0.0 (step2) — zero reward variance ⇒ zero GRPO advantage
- spectral_on actor/pg_loss: 0.0 / actor/loss: 0.0 (both steps)
- disabled actor/grad_norm: 0.0 (step2); disabled critic/rewards/mean: 0.0 (step2); disabled spectral_corrections: 0.0
- FSDP grad-representation discovery lines: 0 found in either log
- per-target rel_change `||G_proj-G_mask||/||G_mask||`: not logged (filter never fired)
- budget: lifetime_spent_usd 3.1442 / monthly_cap 1500 — NOT exhausted; this is the 1st analyst pass (0 prior REVISE cycles; iterations cap 3)

## Comparisons to baseline_run: EXP-3
`diff_against_baseline.py runs/EXP-7 --baseline EXP-3` could not run: `baseline not found: runs/EXP-3` (no EXP-3 run dir on disk; EXP-3/EXP-5 are referenced by id only, per the plan's Background pointers). The intended dense/EXP-5 regression comparison therefore reduces to the within-run `enabled=false` cell, which matches the spectral_on cell in this zero-gradient regime (criterion 6). A meaningful dense-vs-spectral grad-norm comparison was impossible because no cell produced a non-zero gradient.

## Diagnosis (why everything is 0.0)
Benign, not a divergence: the 2-step GSM8K smoke on the base Qwen2.5-1.5B-Instruct model, with only batch=8 / rollout_n=2 and 256-token responses, produced zero reward variance (`critic/rewards/{mean,max,min}:0.0`). Zero reward variance ⇒ zero GRPO advantage ⇒ zero pg_loss ⇒ zero actor gradient. The spectral filter, wired in, saw an all-zero gradient and (correctly) performed no correction, so `spectral_corrections` stayed 0 and the `||G_proj-G_mask||/||G_mask||` ratio is undefined (0/0). The filter was therefore never exercised on a real matrix, so the headline FSDP gradient-representation discovery (criterion 3), the active-and-bounded ratio (criterion 4), and the param-delta (criterion 5) cannot be claimed. The smoke proves the dense path is not broken (criteria 2, 6) but does not deliver the experiment's actual deliverable.

This is a REVISE, not a STOP: hypothesis is NOT falsified (no NaN/Inf, no shape change, no FSDP corruption — grad_norm finite throughout), budget is intact, and this is the first analyst pass. The fix is concrete: make the filter fire on a genuinely non-zero gradient and emit the instrument-first discovery log.

## next_actions (REVISE only)
- knob: reward_signal
  from: "2-step GSM8K smoke on base instruct model (zero reward variance ⇒ zero gradient)"
  to: "use a reward-bearing prompt subset (or raise rollout_n / batch so ≥1 rollout group has non-uniform reward) so a non-zero GRPO advantage and a non-zero actor gradient reach the spectral filter; assert critic/rewards std > 0 before claiming the correction fired. If a reward-bearing batch is too costly for a 2-step smoke, inject a synthetic non-zero gradient on ≥1 target matrix for the discovery proof."
  rationale: "The filter is a no-op on an all-zero gradient by construction; criteria 3/4/5 are unmeasurable until a real non-zero gradient matrix reaches the correction point."
- knob: instrumentation
  from: "no FSDP grad-representation discovery output emitted (criterion 3 absent from logs)"
  to: "log, unconditionally for ≥1 target 2D matrix on the first substep, type(p.grad), p.grad.shape vs logical 2D shape, DTensor placements/device_mesh (or FlatParameter slice / local shard), FSDP version, and whether correction runs before/after gradient reduction and before/after clipping — gated on neither reward nor grad magnitude so it fires even on the zero-gradient smoke"
  rationale: "The headline deliverable per `## Notes for analyst` is the gradient-representation finding; it must be emitted from the wiring point regardless of gradient magnitude, otherwise a clean smoke that proves only finiteness scores REVISE."
- knob: assertions
  from: "spectral_corrections counter silently stays 0; ratio undefined"
  to: "emit per-target ||G_proj - G_mask|| / ||G_mask|| and assert spectral_corrections > 0 on ≥1 substep when a non-zero gradient is present; record the ratio even if it lands slightly >1 (per pinned codex CONCERN#2, treat ratio>1 as a NOTE not a fail; the failure mode is ratio==0)"
  rationale: "Makes criterion 4 directly machine-checkable on the next run and respects the pinned amplification caveat on the (0,1] upper bound."

## Notes
- The orchestrator-pinned codex CONCERN#2 (the (0,1] upper bound is not provably <=1 because the anchor projection can spectrally amplify) was NOT triggered here — no ratio was logged at all. Carry the pin forward: on the rerun, a logged ratio slightly >1 is a note, not an auto-fail; ratio==0 remains the real failure mode.
- The orchestrator's earlier note (PROGRESS line 58) recorded the intended discovery point as "AFTER reduction / BEFORE clipping (DTensor full_tensor unshard)". That intent is plausible but UNVERIFIED in the logs — no DTensor/full_tensor/reduction line was actually emitted. The rerun must produce that evidence, not assert it.
- analyze.py emitted a stub `verdict=PASS` because it found no `metrics/*.jsonl`; that default was IGNORED. The real evidence is the inline `step:N - ...` lines in train_spectral_on.log / train_disabled.log. Do not trust the analyze.py stub for this run shape.
- Both runs completed cleanly (done.flag, done_spectral_on.flag, done_disabled.flag; instance torn down). No divergence, no OOM, no FSDP shard/reduce error. The Tracebacks in both logs are post-step-2 wandb/DataLoader teardown noise, not training failures.
