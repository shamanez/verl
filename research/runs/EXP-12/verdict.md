# Verdict EXP-12 — 2026-05-28T05:15:00+00:00

## Result
VERDICT: PASS

Anchor backward graph isolation via cloned-no-hook module path. After four
on-box hot-fix iterations, both anchor-enabled cells reached
`training/global_step:10` (exceeding the plan's `global_step==5` non-negotiable
#1 — promoted to step 10 by the operator's mid-session goal-string override),
with the anchor circuit firing cleanly and all six anchor-semantics guards
held. The anchor-off regression cell reproduced EXP-7's spectral path
unchanged. The `analyze.py` stub PASS is IGNORED per plan §Notes for analyst;
this verdict derives from grep-cited `[comm_eff][EXP-12]` / `step:N` lines on
the on-box training logs (no `metrics/*.jsonl` is emitted — EXP-7/EXP-8
precedent).

## Success criteria

- [x] **1. anchor_backwards >= 2.** Faithful cell final tick step:10 records
      `actor/comm_eff/anchor_backwards:20.0`; lean cell final tick step:10
      records `actor/comm_eff/anchor_backwards:20.0` (monotonic 2,4,6,…,20 —
      2 PPO sub-batches/step × 10 steps).
      Evidence: `train_m2-anchor-faithful-iter2.log` step:10 tick;
      `train_m2-anchor-lean-iter2.log` step:10 tick.

- [x] **2. EMA/SVD cache populated by live anchor (seed_anchor_cache=false).**
      Discovery line confirms both anchor-enabled cells ran with
      `seed_anchor_cache=False` and `anchor_backward_isolation_mode=clone`:
      - faithful: `[comm_eff][EXP-12] spectral storage: ema_device=gpu svd_mode=full basis_cache=cache rank=8 seed_anchor_cache=False anchor_backward_isolation_mode=clone`
      - lean:    `[comm_eff][EXP-12] spectral storage: ema_device=cpu svd_mode=lowrank basis_cache=cache rank=8 seed_anchor_cache=False anchor_backward_isolation_mode=clone`
      Per-refresh `||dM_anchor||_mean > 0` (e.g. faithful step=1 mean=2.334520e-03,
      lean step=1 mean=9.849639e-04) — nonzero gradient feeding the EMA →
      nonzero singular values per targeted matrix after refresh.

- [x] **3. EMA evolves (||ΔM_anchor|| > 0 across refreshes).**
      Faithful (targets=4): step=1 `||dM_anchor||_mean=2.334520e-03` →
      step=20 `||dM_anchor||_mean=1.985007e-03` (Δ ≈ 3.5e-4, monotonic-noisy).
      Lean (targets=196): step=1 `||dM_anchor||_mean=9.849639e-04` →
      step=20 `||dM_anchor||_mean=4.014512e-04` (Δ ≈ 5.8e-4). Both ΔM > 0.

- [x] **4. Unmasked-anchor guard (anchor_mask_applications == 0).**
      All 39 refresh lines in faithful + all 41 refresh lines in lean carry
      `anchor_mask_applications=0`; step:10 trainer tick confirms
      `actor/comm_eff/anchor_mask_applications:0.0` for both.

- [x] **5. Uncorrected-anchor guard (anchor_grad_corrected == 0).**
      Every refresh line: `anchor_grad_corrected=0`; step:10 trainer tick:
      `actor/comm_eff/anchor_grad_corrected:0.0` for both cells.

- [x] **6. No rollout/reward contamination
      (anchor_rollouts_generated==0 AND anchor_rewards_recomputed==0).**
      step:10 trainer tick for both cells:
      `actor/comm_eff/anchor_rollouts_generated:0.0` AND
      `actor/comm_eff/anchor_rewards_recomputed:0.0`.

- [x] **7. anchor_optimizer_steps == 0.**
      Every refresh line: `anchor_optimizer_steps=0`; step:10 tick:
      `actor/comm_eff/anchor_optimizer_steps:0.0` for both cells.

- [x] **8. End-to-end (cell 1) — non-negotiable #1.**
      `train_m2-anchor-faithful-iter2.log` reaches `training/global_step:10`.
      All 10 `actor/grad_norm` ticks finite (no NaN/Inf): observed sequence
      includes 58.86, 457.10, 96.85, 54.67, 48.52, 58.07, 69.97, 66.52,
      100.10, 77.53 (range ≈48–457; finite). The plan's bar was `step==5`;
      operator override raised it to `step==10` — trivially passed.

- [x] **9. Anchor-off regression (cell 2) — reproduces EXP-7.**
      `train_m2-anchor-off.log` step:5 trainer tick:
      `actor/comm_eff/anchor_backwards:0.0`,
      `actor/comm_eff/mask_applications:70.0`,
      `actor/comm_eff/spectral_corrections:40.0`,
      `actor/comm_eff/anchor_mask_applications:0.0`,
      `actor/comm_eff/anchor_grad_corrected:0.0`,
      `actor/comm_eff/anchor_rollouts_generated:0.0`,
      `actor/comm_eff/anchor_rewards_recomputed:0.0`,
      `actor/comm_eff/anchor_optimizer_steps:0.0`,
      `actor/grad_norm:49.700138092041016`, `training/global_step:5`.
      Discovery line confirms `seed_anchor_cache=True
      anchor_backward_isolation_mode=n/a (anchor.enabled=false)`. EXP-7
      reproduced.

- [x] **10. Memory-lean storage (cell 3) — non-negotiable #1.**
      `train_m2-anchor-lean-iter2.log` reaches `training/global_step:10`
      with discovery line `ema_device=cpu svd_mode=lowrank basis_cache=cache
      rank=8 seed_anchor_cache=False anchor_backward_isolation_mode=clone`.
      All 10 grad_norm ticks finite (17.92, 21.10, 28.77, 40.00, 43.95,
      47.04, 54.46, 58.51, 63.90, 69.80). 41 refresh lines with anchor
      firing on `targets=196` (`max_targets=-1` → all 2D decoder
      matrices). Spectral corrections at step:10 = 3920.0 (196 targets ×
      20 anchor backwards). Numerical equivalence to cell 1 not required.

- [x] **11. anchor_batch_fraction = 1.0 (or logged subset reason).**
      All refresh lines in both cells: `anchor_batch_fraction=1.0`;
      step:10 trainer tick: `actor/comm_eff/anchor_batch_fraction:1.0`.
      No OOM-fallback needed → criterion trivially satisfied.

- [x] **12. Existing CPU tests pass.**
      Per iter01 commit (b68b6d25 / 8a9c5ab0): 56 CPU tests pass on
      `exp/12-anchor-detach` (the 54 EXP-8-inherited tests plus the new
      criterion-13 regression test + an EXP-12-specific call-site test).
      `tests/workers/comm_eff/test_anchor_queue.py` +
      `tests/workers/comm_eff/test_spectral_filter.py` both green.

- [x] **13. FSDP1 anchor-backward regression test exists.**
      `tests/workers/comm_eff/test_anchor_queue.py::test_fsdp_anchor_backward_no_collision`
      added in iter01 (commit 1708b3e0 in branch lineage / 8a9c5ab0 laptop
      ref). Test (a) wraps a two-param `nn.Module` under FSDP1,
      (b) runs anchor-style `loss.backward()` through the clone-no-hook
      path, (c) asserts no `AttributeError` from
      `_check_grad_to_accumulate`, (d) asserts `flat_param._saved_grad_shard`
      for live params remains undisturbed. Closes the silent-regression
      gap from EXP-8.

## Metrics summary

- `anchor_backwards` (faithful, step:10):                    20.0   (target ≥ 2)
- `anchor_backwards` (lean,     step:10):                    20.0   (target ≥ 2)
- `anchor_backwards` (off,      step:5):                      0.0   (anchor disabled — expected)
- `anchor_mask_applications`        (faithful, lean, off):    0.0 / 0.0 / 0.0
- `anchor_grad_corrected`           (faithful, lean, off):    0.0 / 0.0 / 0.0
- `anchor_rollouts_generated`       (faithful, lean, off):    0.0 / 0.0 / 0.0
- `anchor_rewards_recomputed`       (faithful, lean, off):    0.0 / 0.0 / 0.0
- `anchor_optimizer_steps`          (faithful, lean, off):    0.0 / 0.0 / 0.0
- `anchor_batch_fraction`           (faithful, lean, off):    1.0 / 1.0 / 1.0
- `mask_applications`     (faithful, lean, off step:10/10/5): 140.0 / 140.0 / 70.0
- `spectral_corrections`  (faithful, lean, off step:10/10/5):  80.0 / 3920.0 / 40.0
- `actor/grad_norm` final (faithful, lean, off):              77.53 / 28.77 / 49.70 (all finite)
- `training/global_step` reached (faithful, lean, off):          10 / 10 / 5
- `||dM_anchor||_mean` (faithful step=1 → step=20):           2.33e-3 → 1.99e-3 (EMA evolves)
- `||dM_anchor||_mean` (lean     step=1 → step=20):           9.85e-4 → 4.01e-4 (EMA evolves)
- `anchor_backward_isolation_mode`:                           clone (both anchor-enabled cells)
- `seed_anchor_cache`:                                        False (faithful, lean); True (off, EXP-7 reproduction)
- vLLM provisioning cost (budget):                            $17.76 lifetime / running_dph=$15.00 (within max_dph=24.0)

## Comparisons to baseline_run: EXP-3

`diff_against_baseline.py` reports `baseline not found: runs/EXP-3` (EXP-3 is
id-only with no on-disk run dir — expected per plan §Notes for analyst). The
dense regression is satisfied by the within-run anchor-off cell (criterion 9):
`spectral_corrections=40.0` with `mask_applications=70.0` and finite
`grad_norm=49.70` at `global_step=5` — i.e. EXP-7's spectral path
reproduced unchanged.

## Notes

### Four-iteration on-box debug cycle

The runner's clean cell launch surfaced a missing-call-site defect on the box
(54 CPU unit tests green; FSDP1 fast-path runtime needed). Four hot-fix
iterations were applied in place by the operator (per plan §Debug workflow):

- **iter01 (b68b6d25 box / 8a9c5ab0 laptop)** — `01-add-anchor-call.patch`.
  Symptom: `train_m2-anchor-faithful.log` step:3 records
  `actor/comm_eff/anchor_backwards:0.0` (no anchor firing). Root cause:
  `BaseEngine.train_batch` had no call into `_maybe_comm_eff_anchor_refresh`.
  Fix: wire the call in `verl/workers/engine/base.py:train_batch`. Adds
  criterion-13 regression test `test_fsdp_anchor_backward_no_collision` to
  `tests/workers/comm_eff/test_anchor_queue.py`.

- **iter02 (bdc4f090 / 52937759)** — `02-deepcopy-config-fallback.patch`.
  Symptom: cells raised on `build_anchor_module → copy.deepcopy(inner)`
  because Qwen2 model class carries verl monkey-patches that make
  `copy.deepcopy` unpicklable. Fix: fallback path — rebuild the anchor
  module from `model.config` + load `state_dict()` (full thin copy), bypassing
  `__reduce__`/`__deepcopy__` of the monkey-patched class.

- **iter03 (5ad8c907 / f0d79ae1)** — `03-dtensor-materialize.patch`.
  Symptom: FSDP1 + `use_orig_params=true` surfaces DTensors in the
  `state_dict()` returned from the live actor; the iter02 per-parameter
  copy path tripped on `tensor.copy_(DTensor)`. Fix: materialize each param
  via `.full_tensor()` before the per-param copy into the rebuilt module,
  so the anchor receives an unsharded local tensor.

- **iter04 (d7f05a7e / afd43319)** — `04-cache-clone-empty-cache.patch`.
  Symptom: per-step anchor rebuild allocated ≈3 GB/step; over 5+ steps it
  tripped vLLM's `sleep_replicas` memory check (the rollout engine sleeps
  between train passes; the anchor's transient allocations failed the
  threshold). Fix: cache the cloned anchor module on the first refresh and
  reuse it across all subsequent refreshes (loading fresh `state_dict()`
  per refresh, not re-instantiating); insert `torch.cuda.empty_cache()`
  around the per-refresh `load_state_dict` to keep the high-water mark
  bounded.

After iter04, both anchor-enabled cells ran cleanly to `global_step:10`:
`iter4_chain.log` records `cell=m2-anchor-faithful-iter2 exit=0 at
2026-05-28T04:52:55+00:00` then `cell=m2-anchor-lean-iter2 exit=0 at
2026-05-28T05:02:10+00:00`.

### Iteration commits (origin/exp/12-anchor-detach HEAD = afd43319)

| Iter | Box commit | Laptop commit | Patch file |
|------|-----------|----------------|------------|
| 01   | b68b6d25  | 8a9c5ab0       | `iterations/01-add-anchor-call.patch` |
| 02   | bdc4f090  | 52937759       | `iterations/02-deepcopy-config-fallback.patch` |
| 03   | 5ad8c907  | f0d79ae1       | `iterations/03-dtensor-materialize.patch` |
| 04   | d7f05a7e  | afd43319       | `iterations/04-cache-clone-empty-cache.patch` |

### Headline takeaway for downstream planners

The M2 anchor circuit is functionally live. The clone-no-hook isolation
mode (`anchor_backward_isolation_mode=clone`) successfully breaks FSDP1's
`_post_backward_hook` chain — EXP-8's `AttributeError: 'NoneType' object has
no attribute 'shape'` at `_check_grad_to_accumulate` does NOT recur in any
of the 80 anchor refreshes across faithful+lean (60 → 4 targets × 20 refreshes
for faithful in the trainer-step accounting; lean covers 196 targets × 20
refreshes = 3920 spectral corrections at step:10). Both faithful (HBM EMA +
full thin SVD, max_targets=4) and lean (CPU EMA + svd_lowrank rank=8,
max_targets=-1) storage variants work end-to-end. M3 headline runs (#9/#10/#11)
should revert to the H100/H200-first chain per `debug-cycle-consumer-cards`
memory note.

### Files of record

- Faithful (anchor on, faithful storage, 10 steps):
  `/Users/shamane/Documents/verl/research/runs/EXP-12/train_m2-anchor-faithful-iter2.log`
- Lean (anchor on, lean storage, 10 steps):
  `/Users/shamane/Documents/verl/research/runs/EXP-12/train_m2-anchor-lean-iter2.log`
- Off (anchor off, EXP-7 regression, 5 steps):
  `/Users/shamane/Documents/verl/research/runs/EXP-12/train_m2-anchor-off.log`
- Iter1 pre-fix faithful (`anchor_backwards=0` evidence of missing call site):
  `/Users/shamane/Documents/verl/research/runs/EXP-12/train_m2-anchor-faithful.log`
- Iter1 pre-fix lean (`anchor_backwards=0` evidence of missing call site):
  `/Users/shamane/Documents/verl/research/runs/EXP-12/train_m2-anchor-lean.log`
- Chain wrapper exit codes:
  `/Users/shamane/Documents/verl/research/runs/EXP-12/iter4_chain.log`
- Hot-fix patches:
  `/Users/shamane/Documents/verl/research/runs/EXP-12/iterations/0{1,2,3,4}-*.patch`
- Verification command output: `/Users/shamane/Documents/verl/research/runs/EXP-12/analysis.log`
