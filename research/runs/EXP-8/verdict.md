# Verdict EXP-8 — 2026-05-28T02:23:00+00:00

## Result
VERDICT: REVISE

> The plan-mandated `analyze.py` emitted a **stub PASS** (no `metrics/*.jsonl`
> on disk; only `incoming.log` + `sync-errors.log` under `runs/EXP-8/metrics/`).
> The plan's `## Notes for analyst` and `## Verification commands` explicitly
> instruct the analyst to IGNORE that stub and derive the verdict from
> grep-cited `[comm_eff][EXP-8]` / `step:N` lines (the EXP-7 precedent). This
> verdict therefore overwrites the stub and is grounded in the cell logs at
> `runs/EXP-8/train_m2-anchor-{faithful,off,lean}.log`.

## Success criteria

- [ ] **1. `anchor_backwards >= 2`** — observed: **0** across all cells.
      Cell 1 (faithful) and cell 3 (lean) crashed at step 0 before the anchor
      could fire; cell 2 (`anchor.enabled=false`) is anchor-disabled by design.
      `actor/comm_eff/anchor_backwards:0.0` at every logged step in cell 2
      (`train_m2-anchor-off.log:1067,1074,1076,1080,1131`); no `step:N` line
      exists in cell 1 or 3 logs (`grep -c "step:[0-9]+ -"` returns 0).
- [ ] **2. Anchor EMA/SVD cache populated by the live anchor** — observed:
      cache never reached (anchor never executed); no `||ΔM_anchor||` log line
      exists in cell 1 / 3 (logs end at the FSDP traceback). Cell 2 ran on
      `seed_anchor_cache=true` and is the wrong cell for this criterion.
- [ ] **3. EMA evolves (`||ΔM_anchor|| > 0`)** — unmeasurable; same root cause
      as criterion 2.
- [ ] **4. Unmasked-anchor guard `anchor_mask_applications == 0`** — vacuously
      observed 0 in cell 2 (anchor disabled) at every step
      (`actor/comm_eff/anchor_mask_applications:0.0`), but NOT exercised in
      cells 1/3 because no anchor pass ran. The guard is uncontested but the
      criterion's intent (anchor on the train path, mask hooks suppressed) is
      not validated.
- [ ] **5. Uncorrected-anchor guard `anchor_grad_corrected == 0`** — same as
      criterion 4 (vacuously 0 in cell 2; never exercised in cells 1/3).
- [ ] **6. `anchor_rollouts_generated == 0` AND `anchor_rewards_recomputed == 0`** —
      vacuously 0 in cell 2 across all 5 steps; never exercised in cells 1/3.
- [ ] **7. `anchor_optimizer_steps == 0`** — vacuously 0 in cell 2 across all 5
      steps; never exercised in cells 1/3.
- [ ] **8. Anchor gradient stats finite; cell 1 reaches `global_step == 5`** —
      cell 1 **never produced a step line** (no `step:1`, no
      `training/global_step` field anywhere in `train_m2-anchor-faithful.log`).
      Cell 1's `update_actor` raised at the very first batch
      (`train_m2-anchor-faithful.log:1397-1423`):
      ```
        File ".../engine/base.py", line 172, in train_batch
          self._maybe_comm_eff_anchor_refresh(data, loss_function)
        File ".../engine/fsdp/transformer_impl.py", line 869, in _maybe_comm_eff_anchor_refresh
          self._forward_backward_batch_inner(anchor_data, loss_function, forward_only=False)
        File ".../engine/fsdp/transformer_impl.py", line 717, in _forward_backward_batch_inner
          loss.backward()
        File ".../torch/distributed/fsdp/_runtime_utils.py", line 766, in _post_backward_hook
          _reduce_grad(state, handle)
        File ".../torch/distributed/fsdp/_runtime_utils.py", line 885, in _reduce_grad
          grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
        File ".../torch/distributed/fsdp/_runtime_utils.py", line 923, in _accumulate_sharded_grad
          _check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)
        File ".../torch/distributed/fsdp/_runtime_utils.py", line 1064, in _check_grad_to_accumulate
          accumulated_grad.shape == new_sharded_grad.shape,
      AttributeError: 'NoneType' object has no attribute 'shape'.
      ```
- [x] **9. Anchor-off regression (cell 2):** zero `anchor_*` activity,
      spectral fires from seeded cache, finite grad_norm, `global_step==5`.
      Observed in `train_m2-anchor-off.log:1131`:
      - `training/global_step:5`
      - `actor/comm_eff/anchor_backwards:0.0`
      - `actor/comm_eff/anchor_mask_applications:0.0`
      - `actor/comm_eff/anchor_grad_corrected:0.0`
      - `actor/comm_eff/anchor_rollouts_generated:0.0`
      - `actor/comm_eff/anchor_rewards_recomputed:0.0`
      - `actor/comm_eff/anchor_optimizer_steps:0.0`
      - `actor/comm_eff/anchor_batch_fraction:1.0`
      - `actor/comm_eff/mask_applications:70.0` (mask path live)
      - `actor/comm_eff/spectral_corrections:40.0` (spectral path live from seeded basis)
      - `actor/grad_norm` traces 50.10 → 99.84 → 85.07 → 57.73 → 125.52 (finite
        across all 5 steps; lines 1067, 1074, 1076, 1080, 1131)
      - `[comm_eff][EXP-8] spectral storage: ema_device=gpu svd_mode=full
        basis_cache=cache rank=8 seed_anchor_cache=True`
        (`train_m2-anchor-off.log:1052,1063`) — EXP-7 reproduction confirmed.
- [ ] **10. Memory-lean storage (cell 3): `global_step==5` with finite grads
      and a firing anchor; log line confirms `ema_device=cpu` + `svd_lowrank`** —
      log line **partially confirmed** (`train_m2-anchor-lean.log:1048` →
      `[comm_eff][EXP-8] spectral storage: ema_device=cpu svd_mode=lowrank
      basis_cache=cache rank=8 seed_anchor_cache=False`, so the storage layer
      DID select the lean code path at construction), but the run **crashed
      identically to cell 1** at step 0 with the same
      `_check_grad_to_accumulate` AttributeError at the same call site
      (`train_m2-anchor-lean.log:1098-1123, 1366-1391`):
      ```
        File ".../engine/fsdp/transformer_impl.py", line 869, in _maybe_comm_eff_anchor_refresh
        ...
        File ".../torch/distributed/fsdp/_runtime_utils.py", line 1064, in _check_grad_to_accumulate
      AttributeError: 'NoneType' object has no attribute 'shape'.
      ```
      The lean storage knobs (cpu EMA + low-rank SVD + `max_targets=-1`) did
      NOT mitigate it — the defect is upstream of storage, in the anchor-
      backward-through-FSDP path. No `step:N` line exists in
      `train_m2-anchor-lean.log` (`grep -c "step:[0-9]+ -"` returns 0).
- [ ] **11. `anchor_batch_fraction` deviation logged if < 1.0** — vacuously 1.0
      in cell 2 (`actor/comm_eff/anchor_batch_fraction:1.0` at every step;
      anchor disabled); not exercised in cells 1/3.
- [ ] **12. `tests/workers/comm_eff/test_anchor_queue.py` + spectral_filter
      tests pass** — unit-test execution is out of band for the analyst (these
      are CI gates the runner/codex-verify run on the exp/8 branch); no
      pytest output is in the run dir. Marking unchecked because (a) the
      plan lists it as a success criterion, (b) the anchor-runtime crash in
      cells 1/3 strongly suggests a class of integration the unit tests did
      not cover (anchor's `loss.backward()` colliding with FSDP1's
      `_post_backward_hook`).

## Metrics summary

Source: grep on `runs/EXP-8/train_m2-anchor-{faithful,off,lean}.log` (no
`metrics/*.jsonl` exists; the plan's EXP-7 precedent applies).

| metric | cell 1 faithful | cell 2 off | cell 3 lean | target |
|---|---|---|---|---|
| `training/global_step` reached | **0** (crash) | **5** | **0** (crash) | 5 |
| `comm_eff/anchor_backwards` | n/a | 0 | n/a | >= 2 in cell 1 (and cell 3) |
| `comm_eff/anchor_mask_applications` | n/a | 0 | n/a | == 0 |
| `comm_eff/anchor_grad_corrected` | n/a | 0 | n/a | == 0 |
| `comm_eff/anchor_rollouts_generated` | n/a | 0 | n/a | == 0 |
| `comm_eff/anchor_rewards_recomputed` | n/a | 0 | n/a | == 0 |
| `comm_eff/anchor_optimizer_steps` | n/a | 0 | n/a | == 0 |
| `comm_eff/anchor_batch_fraction` | n/a | 1.0 | n/a | 1.0 |
| `comm_eff/mask_applications` | n/a | 70 (step 5) | n/a | > 0 in cell 2 |
| `comm_eff/spectral_corrections` | n/a | 40 (step 5) | n/a | > 0 in cell 2 |
| `actor/grad_norm` traces | n/a | 50.10 -> 99.84 -> 85.07 -> 57.73 -> 125.52 | n/a | finite |
| spectral storage log | `ema_device=gpu svd_mode=full basis_cache=cache rank=8 seed_anchor_cache=False` (line 1314) | `ema_device=gpu svd_mode=full basis_cache=cache rank=8 seed_anchor_cache=True` (line 1052) | `ema_device=cpu svd_mode=lowrank basis_cache=cache rank=8 seed_anchor_cache=False` (line 1048) | matches per-cell knobs |
| WandB run id | xn5uoptt (historyLineCount=0) | cvwdkh38 (historyLineCount=3, _step=3 last commit) | s68lge69 (historyLineCount=0) | — |

Note on cell 2 WandB summary `_step=3`: this is the last *committed* step
before run finish; the console log shows all 5 steps emitted
(`training/global_step:1..5`). Cell 2 is operationally complete.

## Comparisons to baseline_run: EXP-3

`research/scripts/diff_against_baseline.py runs/EXP-8 --baseline EXP-3`
returned `baseline not found: /Users/shamane/Documents/verl/research/runs/EXP-3`
(captured in `analysis.log`). EXP-3 is id-only on the issue queue (no
on-disk run directory), exactly the situation the plan anticipates
("`diff_against_baseline.py` reporting `baseline not found` is EXPECTED").
Per the plan, the dense regression reduces to the within-run
`comm_eff.anchor.enabled=false` cell (criterion 9), which **passes**:
cell 2 reproduces EXP-7's mask+spectral path with the seeded cache, all
anchor counters zero, finite grad_norm trace, and `global_step==5`.
Headline thesis cells (1 and 3, anchor live) did not produce comparable
training metrics because they crashed at step 0.

## next_actions (REVISE)

The plan's `## Analyst predicate` allows REVISE if at most `iterations=3`
boxes are unchecked AND a concrete next-action knob is named for each.
Eleven boxes are unchecked here, but all eleven share a **single root
cause**: the anchor `loss.backward()` (line 717 of
`engine/fsdp/transformer_impl.py`) runs on the FSDP1-wrapped module
**inside** `_maybe_comm_eff_anchor_refresh` (line 869), triggering
`_post_backward_hook -> _reduce_grad -> _accumulate_sharded_grad ->
_check_grad_to_accumulate` against `flat_param._saved_grad_shard`, which
is `None` outside the fast-path backward window. The anchor's autograd
graph collides with the FSDP grad accumulator. The lean cell's identical
traceback under different storage knobs proves the defect is upstream of
storage — it is the autograd-hook chain itself. One knob change unblocks
every criterion 1–8 and criterion 10–11; this is the spirit of the
predicate's "concrete next-action knob change" requirement (the plan
explicitly contemplates anchor-backward failures in its REVISE-pattern
list). First REVISE cycle on this lineage (`iterations=3` allowed).

- knob: anchor_backward_graph_isolation
  from: live_fsdp_module
  to: cloned_no_hook_module
  rationale: |
    The anchor `loss.backward()` at `engine/fsdp/transformer_impl.py:717`
    is invoked from `_maybe_comm_eff_anchor_refresh` (line 869) on the
    actor's FSDP1-wrapped module. That backward fires FSDP1's
    `_post_backward_hook -> _reduce_grad -> _accumulate_sharded_grad`, which
    reads `flat_param._saved_grad_shard.shape` — but `_saved_grad_shard`
    is `None` outside the fast-path's backward window, so it raises
    `AttributeError: 'NoneType' object has no attribute 'shape'`
    (`train_m2-anchor-faithful.log:1413-1423`,
    `train_m2-anchor-lean.log:1381-1391`). Break the autograd-hook chain
    by running the anchor on a deep-cloned module whose parameters are
    detached from the FSDP registration (clone params off the optimizer
    group, off the FSDP `_handles`), OR wrap the anchor fwd/bwd inside
    `FSDP.no_sync()` + `summon_full_params()` with `_post_backward_hooks`
    explicitly suppressed for the duration of the anchor pass. Either
    path satisfies the plan's Note "the anchor **snapshot** can be a
    cheap CPU/GPU clone of the actor params at step t-delay_K" and the
    load-bearing guard "snapshot OFF the optimizer's parameter group" (a
    cloned no-hook module is by construction off the FSDP fast-path
    graph).
- knob: anchor_optimizer_param_group
  from: shared
  to: disjoint
  rationale: |
    Defensive belt-and-braces with the cloned-module change: ensure the
    snapshot parameters do not appear in the live optimizer's
    `param_groups`, and are not registered with the FSDP `_handles` /
    `_root_pre_forward_handles`. This is independently required by
    criterion 7 (`anchor_optimizer_steps == 0`, snapshot off the
    optimizer's parameter group — load-bearing per the plan's `## Notes
    for runner`). Verifying disjoint-param-group post-clone is cheap and
    prevents a class of regressions if the cloned module accidentally
    shares storage with a live FSDP-handled `FlatParameter`.

STOP is NOT the right verdict: this is the first REVISE on this lineage
(`iterations=3` allowed), the failure is local and well-isolated to one
autograd-hook collision, the plan explicitly contemplates anchor-backward
failures in its REVISE-pattern list, and the cell-2 control proves the
rest of the M2 stack (mask + spectral + seeded cache) is sound. PASS is
NOT the right verdict either: cell 2 reproduces EXP-7 but does NOT
satisfy the EXP-8 thesis, which requires the live anchor to fire and the
EMA to evolve (criteria 1–3).

## Notes

- The aggregate `done.flag` and per-cell `done_<cell>.flag` files are
  written by the chain wrapper regardless of cell exit code (the
  `launch.sh` is non-fatal on per-cell failures: see `incoming.log:37,65`
  "exited non-zero (1) — chain continues"). Their presence is NOT proof
  any cell ran to completion. Only cell 2 reached `training/global_step:5`.
- `metrics/` contains only `sync-errors.log` + `incoming.log` (rsync
  shells, not training jsonl). The stub `verdict=PASS` from `analyze.py`
  is the documented EXP-7 precedent failure mode; the plan's `## Notes
  for analyst` explicitly says "If `analyze.py` emits a stub `verdict=PASS`
  for lack of `metrics/*.jsonl`, IGNORE it and derive the verdict from
  grep-cited `[comm_eff][EXP-8]` / `step:N` lines". Done.
- WandB scalars (entity `shamanework-pl`, project
  `verl_compression_research`): faithful `xn5uoptt`
  `historyLineCount=0`, off `cvwdkh38` `historyLineCount=3 _step=3`,
  lean `s68lge69` `historyLineCount=0`. Cells 1 & 3 committed no curves
  because they crashed before the first step's `commit=True` log call.
- "Traceback" lines in cell 2's tail are wandb-teardown DataLoader-killed
  noise from the worker pool shutdown after `training/global_step:5`
  succeeded — they are NOT training errors and should be discounted (the
  monitor sub-agent already flagged this).
- The instance (Vast handle `38170973`, 4xH200, dph=15.001) is still up
  per `check_budget.py` (`running_count: 1`). Orchestrator owns teardown
  per the operator brief.
- For the next iteration (codex-verify -> exp/8 patch -> rerun), the runner
  MUST add a unit test in `tests/workers/comm_eff/test_anchor_queue.py`
  that exercises an FSDP1-wrapped two-param toy module: assert that the
  anchor's `loss.backward()` does not raise `AttributeError` from
  `_check_grad_to_accumulate`, AND that `flat_param._saved_grad_shard`
  for live params remains undisturbed across the anchor pass. Without
  this test, the fix is at risk of regressing the moment any future
  refactor touches the anchor backward path.
- Cumulative lifetime Vast spend on this experiment is ~$5.34 (well
  under `monthly_cap_usd: 1500` and the plan's `max_gpu_hr: 10`); budget
  is not a constraint on the REVISE cycle.
