# EXP-20 PowerSGD — Distributed / Collective Correctness Review (sync_basis)

**Owner:** distributed-correctness (Task #3). READ-ONLY (no verl/ edits). Verified INDEPENDENTLY from code — the runner's commit-message claims were NOT taken on trust.
**Reviewed commit:** `f748dbc1c63ef9824a3115b091ed025fe210cf9b` (`origin/exp/20-powersgd-activation`, "[EXP-20] PowerSGD shared-codebook consensus: sync_basis=true + collective-safety", committed **2026-06-04 04:01**).
**Scope (the 4 points the team-lead asked me to verify, each answered CONFIRM/CHALLENGE):**
1. all-reduce(SUM the RAW per-rank V) then orth ⇒ pooled-gram top-r subspace; the code reduces V, NOT per-rank orth'd Q.
2. DP-group binding via `set_dp_group(get_data_parallel_group())` correct at SP=1 (==WORLD) AND forward-safe for SP>1/TP/PP (reduces over the DP subgroup only).
3. deadlock-safety: identical collective sequence over the FIXED sorted `boundary_indices` with zero-fill, in lockstep from `train_mini_batch`'s `finally`.
4. `verify_basis_agreement_across_ranks` (fp64 checksum, RAISE on >1e-6) is a sufficient proof of invariant #4.

> **Note on this file:** the mathematical-checker briefly drilled into this same area and wrote an earlier draft here; they have stood down and Task #3 is mine. I re-derived all four points independently from the code; my conclusions and theirs agree on the static math (CONFIRM on all four). I have folded their independent derivation in verbatim as **Appendix A** (attributed) so the second pass is preserved. **What I add beyond their draft are two HIGH-severity findings (§HIGH-1, §HIGH-2) their draft did not surface, because it took the runner's `q_cross_rank_max_rel_dev=0.0` at face value and assumed the sync=true path was the one that ran.**

---

## BOTTOM LINE

The `sync_basis=true` consensus path is **mathematically and collectively correct as written** — CONFIRM on all four assigned points. **But two HIGH findings qualify the runner's claims and gate trusting the sweep:**

- **HIGH-1 — the consensus path + the verifier have NEVER been exercised by any multi-process artifact (run OR test).** The only on-disk probe (`ce_powersgd_probe_2s_gsm8k.log`, mtime **02:36**) ran against the *earlier* commit `def451e5e` (**02:12**) with **`sync_basis=false`**; the consensus commit landed **04:01**, ~85 min later. The probe log prints `sync_basis=False` on all 4 workers and has **zero** "cross-rank Q agreement"/`q_cross_rank_max_rel_dev` lines. And the unit suite (`tests/workers/comm_eff/test_powersgd_activation.py`, 21 tests) is **entirely single-process** — no `init_process_group`/`spawn`/`mp`, no real `all_reduce`/`all_gather` with world>1; the sync tests run at world≤1 where the verifier short-circuits to `0.0`. So the reported `0.0` is an *expectation about code that has not run with >1 process* — neither a run nor a test substantiates it. My CONFIRMs on points 1/3 are therefore **static-analysis** confidence, not on-box-observed.

- **HIGH-2 — the verifier is mis-gated: it asserts cross-rank Q identity with NO `sync_basis` guard.** Under `sync_basis=false` (a *supported* diagnostic mode — config docstring, constructor default, AND the plan's own step-2 config text), each DP rank legitimately holds a *different* local `orth(V_i)`. The verifier is called on every `did_update` (gated only on `did_update` at `engine_workers.py:952`, not on `sync_basis`), so on the first non-clean update it all-gathers divergent checksums, computes `max_rel_dev ≫ 1e-6`, and **RAISES `RuntimeError("basis Q DIVERGED…")`**, re-raised at the call site ⇒ **running this commit with `sync_basis=false` on ≥2 GPUs hard-crashes.** One-line fix: gate verify on `self.sync_basis`.

For the **intended launch** (`sync_basis=true`, the launcher/config default) the path is correct and will not hit the HIGH-2 crash — but it ships code no multi-process artifact has touched, so the plan's own hard-invariant #4 is currently **ungated by evidence**. Strong recommendation before the 50-step sweep: (a) re-run the ≤2-step probe with `sync_basis=true` on ≥2 GPUs and confirm the agreement line prints with metric ≤1e-6; (b) apply the one-line verifier `sync_basis` guard.

---

## Topology established (premise for all four points)

- `ulysses_sequence_parallel_size` defaults to **1** (`config/engine.py:256`); launcher never overrides (probe cmdline confirms no SP arg) ⇒ **SP=1**, and `_comm_eff_register_powersgd_hooks` *raises* if SP>1 (`transformer_impl.py:776-781`) — enforced, not just configured.
- SP=1 ⇒ `_init_device_mesh` leaves `ulysses_device_mesh = None` (`transformer_impl.py:212`).
- ⇒ `get_data_parallel_group()` returns **`torch.distributed.group.WORLD`** (`transformer_impl.py:599-603`); `get_data_parallel_size() = world//1 = world` (`:596-597`).
- FSDP mesh = `create_device_mesh(world, fsdp_size)`; launcher sets no `fsdp_size` ⇒ single `["fsdp"]` dim over all ranks (`utils.py:51-53`). **FSDP shards parameters, not activations; no DP-replica dim, no TP, no PP in the training mesh.** Each rank gets a distinct data micro-batch shard (`prepare_micro_batches(..., dp_group=WORLD)`, `transformer_impl.py:858`).
- Probe ran `n_gpus_per_node=4 nnodes=1` ⇒ world=4. Rollout TP=2 is a separate vLLM `rollout_device_mesh` (`engine_workers.py:590-604`), disjoint from the actor PG; never reaches `maybe_update_basis`.

**⇒ WORLD == DP == FSDP-shard group == all ranks** — confirmed in code AND in the probe's actual launch args.

---

## Point 1 — reduce RAW V then orth ⇒ pooled-gram top-r; NOT averaging orth'd Q — **CONFIRM**

- Per-rank sketch is the **raw** `(MᵀM)Q`, never orthonormalized before the reduce: `Y = M@q_act` (`powersgd_activation.py:341`); `contrib = M32.t() @ Y` = `Mᵀ(MQ) = (MᵀM)Q` (`:376`); accumulated raw across micro-batches (`:382`). There is **no per-rank QR/orth** on `V`.
- Reduce sums raw, orth **once**: `Vsum = V` (`:508`); `all_reduce(Vsum, ReduceOp.SUM, group)` (`:511-513`) ⇒ `Σ_ranks Σ_mb (MᵢᵀMᵢ)Q = (M_globᵀM_glob)Q`; `q_new = orthonormalize(Vsum)` (`:514`). This is exactly one block-power-iteration step on the pooled gram → the global top-r right-singular subspace.
- **SUM not MEAN is correct:** `orthonormalize` is scale-invariant, so `orth(Σ V_i) = orth((1/W)Σ V_i)`; the `/world` in the task framing is purely cosmetic. The code's SUM is correct.
- **The "average orthonormal frames" trap is avoided** — and this is the crux of point 1: averaging per-rank `orth(V_i)` is *not* a subspace operation (the mean of orthonormal matrices is gauge-dependent and is neither orthonormal nor the pooled top-r). The code reduces the **pre-orth** sketch and orthonormalizes the pooled result. This is the correct — and only correct — pooling.
- **Bit-identical across ranks:** identical `Q_t` (base case = seed-deterministic CPU-fp32 bootstrap, `:134-154`; inductive step = the prior consensus) + identical `Vsum` (all-reduce output) + deterministic sign-canonicalized `orth` (`:113-118`) ⇒ identical `Q_{t+1}`.

**What would refute me:** `contrib` being an `orth(...)`, or the reduce operating on post-orth `q_new`, or `AVG` plus a non-scale-invariant re-scale. None hold.

## Point 2 — DP-group binding correct at SP=1 and forward-safe for SP>1/TP/PP — **CONFIRM**

- Engine binds the DP group explicitly: `engine_workers.py:760-764` → `powersgd.set_dp_group(engine.get_data_parallel_group())` (pure setter, `powersgd_activation.py:558-564`). Both collectives use `group=self._dp_group()`.
- At SP=1 the bound group IS WORLD == DP, so the reduction pools over exactly the data-shard ranks — and it is the *same* group used for loss normalization (`transformer_impl.py:852-853`), so basis consensus and loss-norm reduce over an identical rank set (consistent).
- **Forward-safe:** SP>1 ⇒ `get_data_parallel_group()` returns `ulysses_device_mesh.get_group("dp")` — the DP *subgroup* only, NOT world (`transformer_impl.py:600-601`). The codec does not hard-code WORLD; it consumes whatever DP group the engine hands it. So a future SP>1 (or any config that introduces a non-DP dim into the training mesh) would correctly reduce the sketch over the DP subgroup, which is the right pooling set (you consensus over data-shard replicas, not over SP-sliced partitions of the same sample).

**CAVEAT C2 (LOW):** the bind is wrapped in a broad `except Exception` that, on failure, logs a warning and leaves `_dp_process_group=None` → `_dp_group()` falls back to WORLD (`engine_workers.py:784-785` + `powersgd_activation.py:215,556`). Harmless at SP=1 (fallback==correct group); under a future SP>1/TP/PP mesh a swallowed bind error would silently reduce over the WRONG (too-large) group — a correctness bug, not a hang. Harden (fail-loud, or assert `dp_ws==world` only when SP==1) before any SP>1 use.

**What would refute me:** a launcher setting SP>1, or an `fsdp_size` producing a `["ddp","fsdp"]` mesh whose DP semantics ≠ WORLD. Neither present (checked launcher + probe args).

## Point 3 — deadlock-safety: identical collective sequence, fixed sorted set, zero-fill, lockstep — **CONFIRM (static); UNVERIFIED with >1 process (HIGH-1)**

Lockstep chain traced end-to-end:
1. `update_actor` dispatched to all actor ranks in lockstep via `make_nd_compute_dataproto_dispatch_fn(mesh_name="actor")` (`engine_workers.py:866`).
2. **`global_step` identical on every rank:** driver stamps `batch.meta_info["comm_eff_global_step"]=self.global_steps` (`ray_trainer.py:1274,1331`); `meta_info` broadcasts identically (private key chosen to avoid colliding with the per-sample `global_steps` column, `engine_workers.py:816-821`); read via `tu.get` (`:830`).
3. **Gate is a pure function of that global_step:** `is_clean_step` (`state.py:446-452`) and the cadence modulo in `maybe_update_basis` (`powersgd_activation.py:465-469`) ⇒ identical update-or-skip on every rank; no rank all-reduces while another skips.
4. **All-reduce iterates a FIXED, identically-ordered set with zero-fill:** `_boundary_for_update()` = `sorted(self.boundary_indices)` (`:529-530`); `boundary_indices = decoder_boundary_indices(len(layers), pp_size)` (`:661`) is a pure function of the replicated model + identical `pp_size`. The key guard (`:474-507`): under `do_sync` the code does **not** early-return on an empty local sketch (`:480-481` early-returns only when `not do_sync`); a missing boundary contributes a shaped **zero** V (`:499-504`). This is the correct fix for the "rank-local dict keys ⇒ asymmetric collective ⇒ hang" hazard the docstring names (`:449-458`).
5. **Exactly two collectives in the codec** (grepped the file): `all_reduce` (`:511-513`) and the verifier's `all_gather` (`:623-625`); both iterate the fixed set. No collective inside the forward hook or diagnostics.
6. **Lockstep call site:** both are invoked from the `finally:` of `update_actor`→`train_mini_batch` (`engine_workers.py:926-972`), which every DP rank runs together under the Ray dispatch.
7. **CI/single-process safe:** `do_sync = sync_basis and is_initialized()` ⇒ False without dist init (no collectives), matching `test_sync_basis_single_process_equivalent_to_local`.

No path found where ranks diverge into a hang for the `sync_basis=true` path. **HIGH-1 qualifier:** this is static confidence — the all-reduce has not run with >1 process (the probe used `sync_basis=false` → no-collective local path; the unit tests are single-process). A `sync_basis=true` ≥2-GPU re-probe is required to *observe* lockstep + no-hang.

## Point 4 — `verify_basis_agreement_across_ranks` sufficient proof of invariant #4 — **CONFIRM for sync_basis=true; CHALLENGE the gating + the evidentiary status**

**Mechanism is sound (CONFIRM for sync=true):**
- fp64 checksum `(Q ⊙ ramp).sum()` with a distinct per-element weight ramp (`:566-586`) is sensitive to any value/sign/permutation change; fp64 avoids the checksum itself masking ~1e-6 deviations.
- The all-gather is symmetric (fixed boundary set, zero-fill, same-length/order vector) ⇒ cannot itself deadlock (`:616-625`).
- Assert-and-`raise` on `max_rel_dev > 1e-6` is the right severity (silent divergent codebooks would corrupt the experiment) (`:626-640`); single-rank/no-dist short-circuit correct (`:610,613-615`); wired to metric `comm_eff/powersgd_q_cross_rank_max_rel_dev` (`state.py:626-628`).
- **Sufficiency:** for the boundaries it iterates at the first update, a ≤1e-6 result is strong evidence of identical consensus Q (checksum moves under any per-rank divergence). Scope notes (not defects): verifies *once* at the first update (fine for a 50-step gate); fp64-linear-functional collision is theoretically possible but negligible. As a hard-invariant-#4 gate **when sync_basis=true**, it is correct and sufficient.

**CHALLENGE — gating defect (HIGH-2):** the verifier has guards for `is_initialized`, `world<=1`, empty `idxs`, but **no `sync_basis` guard** (confirmed reading `:609-641`; `sync_basis` appears only in the error string). The call site gates only on `did_update` (`engine_workers.py:952`). Under `sync_basis=false` — a supported mode (config docstring `:332-333` "diagnostic only … keeps each rank's basis local"; constructor default `False` at `powersgd_activation.py:188`; plan step-2 config text literally `sync_basis=false` at `plans/20.md:82`) — the first non-clean update produces 4 *different* local `orth(V_i)`, the all-gather sees divergent checksums, and it RAISES → re-raised at `:969-972` → the run crashes. Asserting cross-rank identity is valid only when `sync_basis=true`; under `false`, per-rank divergence is expected and correct. **Fix:** `if did_update and powersgd.sync_basis and not _checked:` (or guard the verifier body). One line.

**CHALLENGE — evidentiary status (HIGH-1):** the verifier's correctness is moot until it runs with >1 process. No on-disk run (probe = `sync_basis=false`) and no unit test (all single-process; the sync/verify tests run at world≤1 where it short-circuits to `0.0`) has exercised it with a real multi-rank all-gather. The reported `q_cross_rank_max_rel_dev=0.0` is unsubstantiated by any artifact.

---

## What the on-disk probe DID validate (local codec health, outside the consensus path)

The 2-step, 4-GPU probe (`sync_basis=false`, commit `def451e5e`) ran clean and shows (metrics inline in `train.log`/probe log):
- `powersgd_q_cond ≈ 1.0000003` every boundary, both steps (orthonormal basis healthy; no collapse).
- `powersgd_basis_updates: 1 → 2` (cadence=1, clean_cadence=0: every step updates) ⇒ `maybe_update_basis` ran and returned `did_update=True`.
- `logical_pp_bytes_powersgd_y_only: 102.0` (budget-match target); `powersgd_applications: 3584 → 7168`; no NaN/Inf, no OOM, all 4 GPUs live.
- `reconstruction_rel_error` step1 ≈ 0.97 (cold random-bootstrap basis), step2 dropping (layer_3 → 0.025) as the *local* (unsynced) block-power-iteration warms.

This confirms the local codec runs; it says nothing about the consensus path (HIGH-1). Note: with `sync_basis=false`, the basis updated locally on 4 ranks with different shards and the verifier did NOT raise — which is only consistent with the verifier not yet existing at the commit the probe ran (`def451e5e`, pre-consensus), corroborating the 02:36-vs-04:01 timeline; on the *current* commit `f748dbc1`, the same `sync_basis=false` config would crash (HIGH-2).

---

## Risk register (by severity)

| Sev | Finding | Where | Impact | Action |
|---|---|---|---|---|
| **HIGH** | Consensus path + verifier **never run with >1 process** (no run; unit tests single-process); reported `q_cross_rank_max_rel_dev=0.0` unsubstantiated | probe mtime 02:36 vs commit 04:01; `sync_basis=False` + 0 agreement lines; `test_powersgd_activation.py` has no `init_process_group`/`spawn`/world>1 | collective-safety + consensus correctness are static-only; lockstep/no-hang/`0.0` unobserved | **Re-run the ≤2-step probe with `sync_basis=true` on ≥2 GPUs BEFORE the sweep**; confirm the `[comm_eff][EXP-20] cross-rank Q agreement` line prints and metric ≤1e-6 (the plan's hard-invariant #4). |
| **HIGH** | Verifier mis-gated: asserts cross-rank Q identity with **no `sync_basis` guard** | `powersgd_activation.py:609-641` (no guard) + `engine_workers.py:952` (gated on `did_update` only) | commit `f748dbc1` with `sync_basis=false` (supported mode; plan step-2 text) **crashes** on first non-clean update | one-line gate verify on `self.sync_basis`; until then don't run any `sync_basis=false` arm multi-GPU on this commit |
| LOW | C2: DP-group bind in broad `except` → silent WORLD fallback | `engine_workers.py:784-785` | none at SP=1; wrong group under future SP>1/TP/PP | fail-loud / assert `dp_ws==world` only when SP==1 before any SP>1 use |
| LOW | C1: plan step-2 + constructor default `sync_basis=false`; launcher/config default `true` | `plans/20.md:82`, `powersgd_activation.py:188` vs `config/comm_eff.py:355`, `launcher.sh:280` | none for intended launch (`true` correct; config overrides constructor); interacts with HIGH-2 | analyst: record resolved `sync_basis` from `resolved_params.txt`; reconcile plan text |
| INFO | Verifier checks once + fp64-checksum (not full-Q) | `engine_workers.py:952`, `powersgd_activation.py:566-641` | sufficient for the gate; theoretical collision / post-step-1 drift unverified | none required |

---

## FINAL POSITION

CONFIRM on all four assigned points for the **intended `sync_basis=true` launch**: (1) the code reduces the RAW per-rank `V` and orthonormalizes the pooled `(M_globᵀM_glob)Q` (it does **not** average orth'd frames — the meaningless operation is correctly avoided); (2) the DP-group binding is correct at SP=1 (==WORLD) and forward-safe for SP>1/TP/PP (DP subgroup only); (3) the collective sequence is deadlock-safe — identical, fixed sorted boundary set with zero-fill, decided in lockstep from `update_actor`'s `finally` on a `global_step`-driven gate; (4) the verifier mechanism is a sound and sufficient proof of invariant #4 *when sync is on*.

I CHALLENGE the runner's claims on two HIGH points the prior draft missed: the consensus path is **untested by any multi-process artifact** (the only probe predates it and ran sync OFF; the unit suite is single-process), so the reported `0.0` cross-rank agreement is an expectation, not evidence; and the verifier is **mis-gated** and will crash the supported `sync_basis=false` mode multi-GPU. Both are fixable (a `sync_basis=true` re-probe + a one-line guard) and neither impugns the consensus math — but the science is **not yet gated by a run that exercised the path being shipped.**

---

## Appendix A — independent second pass (mathematical-checker, preserved verbatim)

*Preserved for the record; this is the earlier draft by the mathematical-checker (who has since stood down on Task #3). Its D1–D6 reach the same VALID/CONFIRM conclusions on the static math as my Points 1–4 above. It does NOT include §HIGH-1/§HIGH-2 (it assumed the runner's `0.0` and the sync=true path as run). All six unit tests it cites were independently confirmed to exist in `tests/workers/comm_eff/test_powersgd_activation.py` — but note they are single-process (see §HIGH-1).*

### D1 — DP process group is correct (runner's "WORLD under SP=1" claim): **VALID**
Read `get_data_parallel_group` (`transformer_impl.py:599-601`): `ulysses_device_mesh` is set only when SP>1 (`_init_device_mesh:216-219`), else `None` ⇒ returns `group.WORLD`. SP=1 is enforced (the powersgd hook registration raises if SP>1). `get_data_parallel_size()=4//1=4`. The launcher's TP=2 is a rollout-only vLLM mesh, not in the FSDP PG. `engine_workers.py:760-764` binds the same group used for loss-norm; `set_dp_group` is a pure setter; `_dp_group()` returns the bound group or None→WORLD; forward-safe for a future SP>1/TP/PP (narrower DP subgroup). `test_set_dp_group_none_is_world` covers the default/override.

### D2 — FSDP shards parameters, not activations ⇒ per-rank M_i is the full local activation: **VALID**
FSDP (FULL_SHARD/HSDP) shards parameters (all-gather for forward, reshard), NOT the activation tensor; activation resharding is Ulysses SP, and SP=1. The dispatch scatters a different data shard per DP rank, so each `M_i` is the full local activation for a distinct slice ⇒ `C_i = M_iᵀM_i` are distinct local grams and a per-rank `orth(V_i)` would diverge — the exact failure `sync_basis=true` repairs. Boundary placement identical via `decoder_boundary_indices(L, pp_size)`.

### D3 — `orth(allreduce(V)) = consensus top-r of the POOLED gram`: **VALID**
Precondition: every rank's `V_i` built from the SAME `Q_t`. Base case: `Q_0 = init_basis(seed_L)` on CPU/fp32, pure function of (seed, layer) ⇒ bit-identical (`test_determinism_multi_rank`). Inductive step: forward hook reads shared `Q_t` (basis mutates only in `maybe_update_basis`, post-backward), so `V_i = C_i Q_t`; `all_reduce(SUM)` ⇒ `Σ_i C_i Q_t = (M_globᵀM_glob)Q_t = C_global Q_t`; `orth(Vsum)` is one block-power-iteration on the pooled SPD gram → top-r right-singular subspace; identical `Vsum` + deterministic `orth` ⇒ bit-identical `Q_{t+1}`. SUM not MEAN correct (orth scale-invariant). Pool RAW V, not average orth'd frames.

### D4 — Cross-rank determinism of `Q_{t+1}`: **VALID, with a determinism caveat**
After `all_reduce`, every rank holds the same `Vsum`; `orthonormalize` is pure (`qr` + sign canonicalization `q*=sign(diag(R))` + deterministic degenerate-column repair) ⇒ bit-identical `Q_{t+1}`. Caveat: cross-rank agreement within a run holds by all-reduce semantics regardless of NCCL reduction order (NCCL SUM not bit-reproducible across runs/topologies, irrelevant here). Enforced by `verify_basis_agreement_across_ranks` (fp64 checksum, RAISE on >1e-6), called once after the first update (`engine_workers.py:952-957`) on a symmetric gate (`test_basis_checksums_deterministic_and_sensitive`, `test_verify_agreement_single_rank_short_circuits`).

### D5 — Collective safety / no deadlock: **VALID**
Symmetric on every axis: fixed `sorted(boundary_indices)` iteration (not rank-local `_sketch.keys()`, `test_boundary_for_update_is_fixed_sorted`); zero-fill for missing boundaries under `do_sync` (`:498-503`); no asymmetric early-return under sync (`:478-479` guard is `not do_sync and not _sketch`); cross-rank-identical update decision keyed on broadcast `global_step`; lockstep `finally:` call site; single-process issues no collectives (`test_sync_basis_single_process_equivalent_to_local`).

### D6 — Clean-step / cadence decided identically across ranks: **VALID**
`comm_eff_global_step` set by the trainer (`ray_trainer.py:1274,1331`), same batch meta dispatched to every worker ⇒ identical `global_step` (`_comm_eff_thread_global_step`). `is_clean_step` + cadence modulo are pure functions of it + identical config ⇒ all ranks agree on clean-vs-compressed and update-cadence, guaranteeing the lockstep participation in D5.

### Bottom line (mathematical-checker)
Distributed/sync_basis consensus path is correct under independent re-derivation; DP all-reduce over `group.WORLD` == DP group at SP=1 (enforced), same group as loss-norm, TP=2 separate rollout mesh; FSDP shards params not activations so each `M_i` is full local; consensus is exact (`orth(C_global Q_t)`, one power-iteration, bit-identical `Q_{t+1}`); determinism enforced by the fp64-checksum verifier; collective safety holds on every axis. "No defects" *on the static math* (the HIGH findings above qualify what was actually run/tested).
