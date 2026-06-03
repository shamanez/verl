# EXP-20 PowerSGD — Distributed & Collective Correctness Review (sync_basis)

**Reviewer:** mathematical-checker (Task #3; ADVERSARIAL drill-down on claim 9 of `math_validity_review.md` — claims independently re-derived from code, not taken on the runner's word)
**Date:** 2026-06-04
**Reviewed commit:** `f748dbc1c63ef9824a3115b091ed025fe210cf9b` (`origin/exp/20-powersgd-activation`)
**Scope:** the `sync_basis=true` consensus path — DP process-group selection, collective safety / deadlock, cross-rank determinism, cross-rank clean-step/cadence agreement, the FSDP-shards-params-not-activations premise, and the `orth(allreduce(V)) = consensus top-r of pooled gram` identity.

## Primary artifacts (re-read independently)
- `verl/workers/comm_eff/powersgd_activation.py:458-519` — `maybe_update_basis` (the consensus all-reduce).
- `verl/workers/comm_eff/powersgd_activation.py:541-563, 587-639` — `_dp_group`/`set_dp_group`, `verify_basis_agreement_across_ranks`.
- `verl/workers/engine/fsdp/transformer_impl.py:599-601` — `get_data_parallel_group()` (the group binding source of truth).
- `verl/workers/engine/fsdp/transformer_impl.py:205-223` — `_init_device_mesh` (`ulysses_device_mesh is None` unless SP>1).
- `verl/workers/engine_workers.py:760-764` — `powersgd.set_dp_group(engine.get_data_parallel_group())`.
- `verl/trainer/ppo/ray_trainer.py:1274,1331` — `batch.meta_info["comm_eff_global_step"] = self.global_steps`.

---

## D1 — DP process group is correct (runner's "WORLD under SP=1" claim, independently verified): **VALID**

I did NOT trust the runner; I read `get_data_parallel_group` myself (`transformer_impl.py:599-601`):
```python
def get_data_parallel_group(self):
    if self.ulysses_device_mesh is not None:
        return self.ulysses_device_mesh.get_group(mesh_dim="dp")
    else:
        return torch.distributed.group.WORLD
```
- `ulysses_device_mesh` is set **only** when `ulysses_sequence_parallel_size > 1` (`_init_device_mesh:216-219`); otherwise it stays `None`. The EXP-20 launcher runs SP=1, and `_comm_eff_register_powersgd_hooks` (`transformer_impl.py:773-781`) *raises* if SP>1. So `ulysses_device_mesh is None` ⇒ `get_data_parallel_group()` returns `group.WORLD`. The runner's claim holds, and it is enforced by an assert, not just configured. ✓
- `get_data_parallel_size() = world_size // sp_size = 4 // 1 = 4` (`line 596-597`). DP group = all 4 ranks.
- The launcher's TP=2 is a *rollout-only* vLLM mesh; it is NOT in the FSDP training process group (the FSDP `device_mesh` is built from `world_size`/`fsdp_size`, `create_device_mesh`, with no TP dim). So pooling over `group.WORLD` pools over exactly the 4 DP ranks whose data shards we want to consensus over — not over any TP replica. ✓
- `engine_workers.py:760-764` binds it explicitly: `powersgd.set_dp_group(engine.get_data_parallel_group())`. This is the **same** group object the loss-norm all_reduce uses in `_forward_backward_batch_inner` (`get_data_parallel_group()`), so the basis-consensus reduction and the loss normalization reduce over the identical rank set — consistent. The `set_dp_group` is a pure setter (no collective); `_dp_group()` returns the bound group or `None`→WORLD fallback. Forward-safe: a future SP>1/TP/PP config would inject a narrower DP subgroup and the all-reduce would reduce over the DP subgroup only. ✓

`test_set_dp_group_none_is_world` confirms the default/override behavior in-process.

## D2 — FSDP shards parameters, not activations ⇒ per-rank M_i is the full local activation: **VALID (premise holds)**

This premise is what makes the per-rank grams `C_i = M_iᵀ M_i` genuinely *local* (and hence makes sync necessary).
- FSDP (`FULL_SHARD` / HSDP) shards **parameters**; each FSDP unit all-gathers its params for the forward, computes, and reshards. It does NOT shard or all-gather the **activation** tensor — the hidden state flowing between blocks is each rank's own complete micro-batch activation. Activation resharding across ranks is precisely what Ulysses sequence-parallelism does, and SP=1 here (D1), so no SP slicing of the token axis occurs.
- The dispatch scatters a **different data shard** to each DP rank (different prompts/completions), so each rank's boundary activation `M_i` of shape `(N_i, H)` is the full local activation for a *distinct* data slice. Hence `C_i = M_iᵀ M_i` are distinct per-rank local grams, and a per-rank `orth(V_i)` would land on different subspaces ⇒ the bases would DIVERGE across ranks after the first update. This is the exact failure mode `sync_basis=true` repairs, and it confirms the runner's framing is correct (not a strawman). ✓
- The boundary-set placement is identical across ranks because `decoder_boundary_indices(L, pp_size)` is a pure function of the (same) model's layer count and `pp_size` — every rank wraps the same `Qwen2DecoderLayer` instances and registers the same boundary indices. ✓

## D3 — `orth(allreduce(V)) = consensus top-r of the POOLED gram` (the core math, adversarially re-derived): **VALID**

The consensus identity has a precondition the implementation must satisfy: **every rank's `V_i` must be built from the SAME `Q_t`.** I checked both the base case and the inductive step:

- **Base case (t=0):** all ranks bootstrap `Q_0 = init_basis(seed_L)` — drawn on **CPU in fp32** with a `torch.Generator` seeded by `(base_seed·1_000_003 + layer·7919)&0x7FFFFFFF`, then `orth`. Pure function of (seed, layer), device/RNG-state independent ⇒ bit-identical `Q_0` on every rank. ✓ (`test_determinism_multi_rank`)
- **Inductive step:** assume all ranks share `Q_t`. The forward hook reads this shared `Q_t` (the basis only mutates in `maybe_update_basis`, post-backward), so each rank's sketch is `V_i = M_iᵀ(M_i Q_t) = C_i Q_t` with the SAME `Q_t`. Then (`maybe_update_basis:507-512`):
  ```python
  Vsum = V.to(torch.float32)
  torch.distributed.all_reduce(Vsum, op=SUM, group=group)   # Vsum = Σ_i V_i
  q_new = orthonormalize(Vsum.to(self.qr_dtype), eps=...)
  ```
  `Σ_i V_i = Σ_i C_i Q_t = (Σ_i C_i) Q_t = (M_globᵀ M_glob) Q_t`, where `M_glob` is the row-stack of all ranks' activations and `M_globᵀ M_glob = Σ_i C_i` is the **pooled** (global-batch) activation gram. So `Vsum = C_global Q_t` — exactly one block-power-iteration step on the *pooled* SPD gram. `orth(Vsum)` therefore drives `Q` toward the top-`r` right-singular subspace of the globally-pooled activations (the consensus top-`r` basis, Eckart-Young-optimal for the pooled matrix). And because every rank holds the identical `Vsum` after all-reduce and runs the same deterministic `orth`, every rank produces the bit-identical `Q_{t+1}` — re-establishing the inductive hypothesis. ✓
- **SUM not MEAN is correct:** `orth` is scale-invariant, so `orth(Σ V_i) = orth((1/W)Σ V_i)`; summing the raw `V_i` gives the pooled *direction* with no need for per-rank count re-weighting. Dividing by `world_size` would be harmless but unnecessary; the code's choice is correct. (A subtle point the runner gets right: you must pool the **raw V**, NOT average per-rank `orth(V_i)` frames — averaging orthonormal bases is not a subspace operation and would be wrong. The code reduces V before orth. ✓)
- **Wrong alternatives the code avoids:** (a) all-reduce of `M` is impossible (different `N_i` per rank) and unneeded; (b) averaging per-rank `Q_i` is meaningless; (c) no-sync diverges. The code does the one correct thing. ✓

## D4 — Cross-rank determinism of `Q_{t+1}` (runner reports q_cross_rank_max_rel_dev=0.0): **VALID, with a precise determinism caveat**

- After `all_reduce`, every rank holds the **same** `Vsum` tensor by the semantics of all-reduce (all ranks receive the identical reduced result). `Vsum.to(self.qr_dtype)` (fp32→fp32 by default) is a deterministic cast. `orthonormalize` is a pure function: `torch.linalg.qr` plus the **sign canonicalization** (`q *= sign(diag(R))`, `powersgd_activation.py:113-116`) that removes QR's column-sign ambiguity, plus the deterministic degenerate-column repair (seeded by matrix shape). So given identical `Vsum`, every rank computes the bit-identical `Q_{t+1}`. ✓
- **Determinism caveat (states the actual guarantee precisely):** the *cross-rank* agreement within a single run holds by all-reduce semantics regardless of NCCL's internal reduction order — every rank gets the same bytes out of the collective, so they orth the same input. (NCCL SUM is not guaranteed bit-reproducible *across different runs/topologies*, but that does not matter here: we need all 4 ranks of THIS run to agree, which all-reduce guarantees by construction.) The sign canonicalization is what makes `orth` itself rank-agnostic (without it, QR sign flips could differ). So `q_cross_rank_max_rel_dev=0.0` (or ≤1e-6) is the expected, mathematically-justified result. ✓
- **Verification is enforced, not assumed:** `verify_basis_agreement_across_ranks` (`powersgd_activation.py:587-639`) all-gathers a per-boundary fp64 checksum (`Q ⊙ index-ramp` summed, sensitive to any sign/permutation/value change) over the FIXED boundary set and **RAISES** if `max_rel_dev > atol=1e-6`. Called once after the first update (`engine_workers.py:952-957`) on a symmetric gate (all ranks reach it on the same cadence step). So a broken consensus fails the probe loudly rather than silently training 4 divergent codebooks. ✓ (`test_basis_checksums_deterministic_and_sensitive`, `test_verify_agreement_single_rank_short_circuits`.)

## D5 — Collective safety / no deadlock: **VALID**

The deadlock hazard with collectives is asymmetric participation. The code is symmetric on every axis:
- **Fixed boundary iteration.** `maybe_update_basis` iterates `self._boundary_for_update()` = `sorted(self.boundary_indices)` (`powersgd_activation.py:522-532`), NOT the rank-local `self._sketch.keys()` (which could differ/be missing per rank). `boundary_indices` is identical on every rank (pure function of model + pp_size, D2). So all ranks issue the all_reduce for the SAME boundary set in the SAME order. ✓ (`test_boundary_for_update_is_fixed_sorted` — even with a sketch missing a boundary, the update set stays `[0,1,2]`.)
- **Zero-fill for missing boundaries.** If a rank lacks a local sketch for some boundary, under `do_sync` it contributes a correctly-shaped **zero V** (`line 498-503`) so it still participates in that boundary's all_reduce. The SUM is unaffected (adding 0). ✓
- **No asymmetric early-return.** Under `do_sync` the code does NOT early-return on an empty local sketch (`line 478-479` guard is `not do_sync and not self._sketch`). So a rank with no local sketches still walks the full collective sequence. ✓
- **Update decision is cross-rank identical.** `do_sync` depends on `self.sync_basis` (same config on all ranks) and `torch.distributed.is_initialized()` (true on all ranks). The cadence/clean-step gates (`gs<=0`, `gs%cadence`, `is_clean_step`) are keyed on `global_step`, which is the trainer's `self.global_steps` broadcast identically via `batch.meta_info["comm_eff_global_step"]` (D6). So every rank takes the same branch — either all all-reduce or all skip. No rank can issue a collective the others don't. ✓
- **Lockstep call site.** `maybe_update_basis` and `verify_basis_agreement_across_ranks` are called from the `finally:` of `train_mini_batch` (`engine_workers.py:941-957`), which every DP rank runs together under the Ray dispatch. So the symmetric collective set is sufficient — no rank is off doing something else. ✓
- **CI/single-process safety.** `do_sync = sync_basis and torch.distributed.is_initialized()` ⇒ False without dist init, so the single-process path issues NO collectives and behaves like the local update (`test_sync_basis_single_process_equivalent_to_local`). ✓

A rank-relative iteration over `self._sketch` would mismatch the collective and hang (all GPUs pinned-idle — the stall signature the plan's rescue trigger watches for). The code explicitly avoids it.

## D6 — Clean-step / cadence decided identically across ranks: **VALID**

- `comm_eff_global_step` is set on the batch meta_info by the trainer: `batch.meta_info["comm_eff_global_step"] = self.global_steps` (`ray_trainer.py:1274` and `:1331`). The same batch (with the same meta_info) is dispatched to every DP worker via Ray, so every rank's `update_actor`/`compute_log_prob` reads the identical `global_step` (`_comm_eff_thread_global_step`, `engine_workers.py:804-839`). ✓
- `is_clean_step(global_step)` (`state.py:431-453`) and the cadence modulo in `maybe_update_basis` are pure functions of that identical `global_step` + identical config (`clean_cadence`, `update_cadence`). So all ranks agree on (a) whether this step is clean (⇒ no hooks, no sketch, no update, on every rank) and (b) whether this is a basis-update cadence step. This is what guarantees the lockstep collective participation in D5. ✓

---

## Bottom line (Task #3)

**The distributed / sync_basis consensus path is correct, and the runner's claims hold under independent (adversarial) re-derivation.** The DP all-reduce reduces over `group.WORLD`, which I confirmed *from `get_data_parallel_group` itself* equals the data-parallel group when SP=1 (and SP=1 is enforced by an assert, not merely configured); it is the same group used for loss normalization, and the launcher's TP=2 is a separate rollout-only mesh outside the FSDP PG. FSDP shards parameters, not activations (SP=1, no Ulysses slicing), so each rank's `M_i` is the full local activation for a distinct data shard — making the per-rank grams genuinely local and sync genuinely necessary. The consensus math is exact: with the seed-identical bootstrap (base case) and the bit-identical post-sync basis (inductive step) guaranteeing every rank's `V_i = C_i Q_t` uses the *same* `Q_t`, `orth(all_reduce_SUM(V_i)) = orth((Σ_i C_i) Q_t) = orth(C_global Q_t)` is one block-power-iteration step on the pooled global-batch activation gram, driving `Q` to the consensus top-`r` subspace and yielding a bit-identical `Q_{t+1}` on every rank (summing raw V, not averaging orth'd frames, is the correct and the only correct pooling). Cross-rank determinism is guaranteed within the run by all-reduce semantics + the deterministic sign-canonicalized `orth`, and it is *enforced* (not assumed) by `verify_basis_agreement_across_ranks`, which raises on any >1e-6 divergence. Collective safety holds on every axis: the all-reduce iterates the FIXED sorted boundary set with zero-fill for missing boundaries, no asymmetric early-return under sync, an identical cross-rank update/clean decision keyed on the trainer-broadcast `global_step`, and a lockstep `finally:`-block call site — so no rank can issue a collective the others don't, and the single-process path issues none at all.

**No defects.** The expected on-box result is `powersgd_q_cross_rank_max_rel_dev ≈ 0.0` (the runner's report), and the code fails loudly if it is not. The only contingencies (all satisfied and asserted): SP=1 (asserted at hook registration), distributed initialized (the multi-GPU run), and the trainer stamping `comm_eff_global_step` identically (it does, from a single counter).
