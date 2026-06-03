# EXP-20 PowerSGD — Distributed / Collective Correctness Review (distributed-correctness reviewer)

> **Filename note:** the team-lead assigned me `review_distributed.md`, but that file already exists and was authored by the **mathematical-checker** (their adversarial drill-down on claim 9). I did NOT create it and it is a substantive independent review, so I did not overwrite it. This is my (distributed-correctness) review under a distinct name. Where we overlap I state CONFIRM/CHALLENGE explicitly.

**Reviewer lens:** distributed & collective correctness of the PowerSGD `sync_basis` path — process group, deadlock-safety, consensus math, on-box verifier, FSDP-vs-activation sharding. Verified INDEPENDENTLY from code; the runner's commit-message claims were NOT taken on trust.
**Mode:** READ-ONLY (no verl/ edits).
**Branch / commit reviewed:** `origin/exp/20-powersgd-activation` @ `f748dbc1c63ef9824a3115b091ed025fe210cf9b` ("[EXP-20] PowerSGD shared-codebook consensus: sync_basis=true + collective-safety", committed **2026-06-04 04:01**).

---

## BOTTOM LINE (read this first)

The `sync_basis=true` consensus **code path is mathematically and collectively correct as written** — all five assigned checks are VALID on the static code, and I CONFIRM the mathematical-checker's claim 9. **But two distributed-lens findings materially qualify the runner's claims and must be surfaced:**

- **HIGH — the consensus path + the `verify_basis_agreement_across_ranks` verifier have NEVER been exercised by any run on disk.** The only probe (`ce_powersgd_probe_2s_gsm8k.log`, mtime **02:36**) ran against the *earlier* commit `def451e5e` (**02:12**) with **`sync_basis=false`**; the consensus commit `f748dbc1` landed **04:01**, ~85 min later. The probe log prints `sync_basis=False` on all 4 workers and contains **zero** "cross-rank Q agreement" / `q_cross_rank_max_rel_dev` lines. So the reported `q_cross_rank_max_rel_dev=0.0` is **not substantiated by any artifact in `runs/EXP-20/`** — it is an expectation about code that has not run multi-GPU. The collective-safety and consensus correctness I VALID below are *static-analysis* confidence, not on-box-observed.

- **HIGH (latent crash) — the verifier is mis-gated: it asserts cross-rank `Q` identity unconditionally, with NO `sync_basis` guard.** Under `sync_basis=false` (a *supported* diagnostic mode per the config docstring, and exactly what the plan's step-2 config text and the constructor default specify), each DP rank legitimately holds a *different* local `orth(V_i)`. The verifier (called on every `did_update`, gated only on `did_update` at `engine_workers.py:952`, not on `sync_basis`) would all-gather divergent checksums, compute `max_rel_dev ≫ 1e-6`, and **RAISE `RuntimeError("basis Q DIVERGED…")`**, which the call site re-raises. ⇒ **running this commit with `sync_basis=false` on ≥2 GPUs hard-crashes on the first non-clean basis update.** The fix is a one-line gate (verify only when `self.sync_basis`).

Neither finding contradicts the *math* of the consensus path; both are about **what was actually validated** and a **gating bug that breaks the off-mode**. For the experiment as the runner intends to launch it (`sync_basis=true`, the launcher/config default), the consensus path is correct and will not hit the verifier crash — but it is launching code that no on-disk run has touched, so the pre-run probe MUST be re-run with `sync_basis=true` before trusting the 50-step sweep.

---

## Topology established (premise for checks 1–5)

Actor *training* mesh at the sanctioned config:
- `ulysses_sequence_parallel_size` defaults to **1** (`config/engine.py:256`); launcher never overrides ⇒ **SP=1**. The probe command line confirms no SP arg. `_comm_eff_register_powersgd_hooks` even *raises* if SP>1 (`transformer_impl.py:776-781`), so SP=1 is enforced, not just configured.
- SP=1 ⇒ `_init_device_mesh` leaves `ulysses_device_mesh = None` (`transformer_impl.py:212`, the `if SP>1` branch at :216 is skipped).
- ⇒ `get_data_parallel_group()` returns **`torch.distributed.group.WORLD`** (`transformer_impl.py:599-603`); `get_data_parallel_size() = world//1 = world` (`:596-597`).
- FSDP mesh = `create_device_mesh(world, fsdp_size)`; launcher sets no `fsdp_size` ⇒ single `["fsdp"]` dim over all ranks (`utils.py:51-53`). **FSDP shards parameters, not activations; no DP-replica dim, no TP, no PP in the training mesh.** Each rank still gets a distinct data micro-batch shard (`prepare_micro_batches(..., dp_group=WORLD)`, `transformer_impl.py:858`).
- Probe ran `trainer.n_gpus_per_node=4 trainer.nnodes=1` ⇒ world=4. Rollout TP=2 is a separate vLLM `rollout_device_mesh` (`engine_workers.py:590-604`), disjoint from the actor PG; never reaches `maybe_update_basis`.

**⇒ WORLD == DP == FSDP-shard group == all ranks.** This premise (which checks 1/3/5 lean on) is confirmed both in code and in the probe's actual launch args.

---

## Check 1 — Process group: all-reduce reduces over the DP group — **VALID**

- Engine binds the DP group explicitly: `engine_workers.py:760-764` → `powersgd.set_dp_group(engine.get_data_parallel_group())` (pure setter, `powersgd_activation.py:558-564`). The `all_reduce` (`:511-513`) and `all_gather` (`:623-625`) both use `group=self._dp_group()`.
- At SP=1 the bound group IS WORLD == DP, so the reduction pools over exactly the data-shard ranks. ✔
- **Future-proofing correct:** SP>1 ⇒ `get_data_parallel_group()` returns `ulysses_device_mesh.get_group("dp")` — the DP *subgroup* only, not world (`transformer_impl.py:600-601`). The codec does not hard-code WORLD; it consumes the engine's DP group. ✔
- This is the *same* group the loss-norm all-reduce uses (`transformer_impl.py:852-853`), so basis consensus and loss normalization reduce over an identical rank set — consistent.

**CAVEAT (C2, LOW):** the bind is wrapped in a broad `except Exception` that, on failure, logs a warning and leaves `_dp_process_group=None` → `_dp_group()` falls back to WORLD (`engine_workers.py:784-785` + `powersgd_activation.py:215,556`). At SP=1 the fallback equals the correct group, so harmless today; under a future SP>1/TP/PP training mesh a swallowed bind error would silently reduce over the WRONG (too-large) group — a correctness bug, not a hang. Harden (fail-loud / assert `dp_ws==world` only when SP==1) before any SP>1 use.

**What would refute me:** a launcher setting SP>1 or an `fsdp_size` producing a `["ddp","fsdp"]` mesh whose DP semantics ≠ WORLD. Neither present (checked launcher + probe args). **Agrees with math-checker D1.**

---

## Check 2 — Collective-safety / deadlock — **VALID (static); UNVERIFIED on box (see HIGH-1)**

Full lockstep chain traced:
1. `update_actor` dispatched to all actor ranks in lockstep via `make_nd_compute_dataproto_dispatch_fn(mesh_name="actor")` (`engine_workers.py:866`). ✔
2. **`global_step` identical on every rank:** the single driver stamps `batch.meta_info["comm_eff_global_step"]=self.global_steps` (`ray_trainer.py:1274,1331`); `meta_info` is broadcast identically (a private key chosen to avoid colliding with the per-sample `global_steps` column, `engine_workers.py:816-821`); read back via `tu.get` (`:830`). ✔
3. **Gate is a pure function of that global_step:** `is_clean_step` (`state.py:446-452`) and the cadence modulo in `maybe_update_basis` (`powersgd_activation.py:465-469`) take only `global_step` + identical config ⇒ identical update-or-skip on every rank. No rank can all-reduce while another skips. ✔
4. **All-reduce iterates a FIXED, identically-ordered set with zero-fill:** `_boundary_for_update()` = `sorted(self.boundary_indices)` (`:529-530`), and `boundary_indices = decoder_boundary_indices(len(layers), pp_size)` (`:661`) is a pure function of the replicated model + identical `pp_size`. The key deadlock guard (`:474-507`): under `do_sync` the code does **not** early-return on an empty local sketch (`:480-481` only early-returns when `not do_sync`); a missing boundary contributes a shaped **zero** V (`:499-504`). This is the correct fix for the "rank-local dict keys ⇒ asymmetric collective ⇒ hang" hazard the docstring names (`:449-458`). ✔
5. **Exactly two collectives in the whole codec** (grepped the file): `all_reduce` (`:511-513`) and the verifier's `all_gather` (`:623-625`); both iterate the fixed set. No collective inside the forward hook or diagnostics (q_cond/reconstruction are purely local). No second divergence surface. ✔
6. **Verifier reached in lockstep:** gated on `did_update` (identical across ranks under `do_sync`) + a once-flag that flips on all ranks together (`engine_workers.py:952`). ✔

No path found where ranks diverge into a hang **for the `sync_basis=true` path**. The two classic break modes (rank-relative dict iteration; rank-local "skip if empty") are both explicitly avoided.

**HIGH-1 qualifier:** this is *static* confidence. The consensus all-reduce has not run on disk (the probe used `sync_basis=false`, which takes the no-collective local path). A re-run probe with `sync_basis=true` is required to *observe* lockstep + no-hang. **Agrees with math-checker D5/D6 on the static logic; I add that it is unobserved.**

---

## Check 3 — Consensus correctness: `orth(SUM(Vᵢ))` is one power-iteration on the pooled gram — **VALID**

- Per-rank sketch is the **raw** `(MᵀM)Q`, never orthonormalized: `Y = M@q_act` (`:341`); `contrib = M32.t() @ Y` = `Mᵀ(MQ) = (MᵀM)Q` (`:376`); accumulated raw (`:382`). No per-rank QR. ✔
- Reduction sums raw, orth once: `Vsum = V` (`:508`); `all_reduce(Vsum, SUM, group)` (`:511-513`) ⇒ `Σ_ranks Σ_mb (MᵢᵀMᵢ)Q = (M_globᵀM_glob)Q`; `q_new = orthonormalize(Vsum)` (`:514`). One block-power-iteration step on the pooled gram → global top-r subspace. ✔
- **`/world` correctly absent (cosmetic):** code uses `ReduceOp.SUM`; `orthonormalize` is scale-invariant (`orth(αX)=orth(X)`, α>0), so `orth(SUM)=orth(SUM/world)`. The task's `/world` is exactly this cosmetic factor. ✔
- **Bit-identical across ranks:** identical `Q_t` (base case = seed-deterministic CPU-fp32 bootstrap, `:134-154`; inductive step = prior consensus) + identical `Vsum` (all-reduce output) + deterministic sign-canonicalized `orth` (`:113-118`) ⇒ identical `Q_{t+1}`. ✔
- **The "average Q's" trap is avoided:** averaging per-rank `orth(Vᵢ)` is not a subspace op (mean of orthonormal frames is gauge-dependent and not the pooled top-r). The code reduces the *pre-orth* sketch. This is the correct — and only correct — pooling. ✔

**What would refute me:** `contrib` being `orth(...)`, or reducing post-orth `q_new`, or `AVG` + non-scale-invariant re-scale. None hold. **CONFIRM math-checker D3 / claim 9.**

---

## Check 4 — On-box verifier `verify_basis_agreement_across_ranks` — **VALID for `sync_basis=true`; MIS-GATED for `sync_basis=false` (HIGH-2)**

**Correctness of the mechanism (VALID):**
- fp64 checksum `(Q ⊙ ramp).sum()` with a distinct per-element weight ramp (`:566-586`) is sensitive to any value/sign/permutation change; fp64 avoids the checksum masking small deviations. ✔
- The all-gather is symmetric (fixed boundary set, zero-fill, same-length/order vector) ⇒ cannot itself deadlock (`:616-625`). ✔
- Assert-and-`raise` on `max_rel_dev > 1e-6` is the right severity (silent divergent codebooks would corrupt the experiment) (`:626-640`). ✔
- Single-rank/no-dist short-circuits (`:610` → None, `:613-615` → 0.0) are correct. ✔
- Wired to metric `comm_eff/powersgd_q_cross_rank_max_rel_dev` (`state.py:626-628`). ✔

**Sufficiency for invariant #4 (VALID, with two scope notes):** for the boundaries it iterates at the first update, a ≤1e-6 result is strong evidence of identical consensus `Q`. Scope limits (not defects): (a) verifies *once* at the first update (`engine_workers.py:952`), not every step — fine for a 50-step gate; (b) fp64-linear-functional collision is theoretically possible but negligible (distinct weights + fp64; the expected divergence mode shifts many elements). Net: correct + sufficient as a hard-invariant-#4 gate **when `sync_basis=true`**.

**HIGH-2 — MIS-GATED for `sync_basis=false`.** The verifier has guards for `is_initialized`, `world<=1`, empty `idxs` — but **no `sync_basis` guard** (confirmed by reading `:609-641`; `sync_basis` appears only in the error message, not in any branch). The call site (`:952`) gates only on `did_update`. So with `sync_basis=false` on ≥2 GPUs, the first non-clean update produces 4 *different* local `orth(V_i)` (different data shards), the all-gather sees divergent checksums, `max_rel_dev ≫ 1e-6`, and it **RAISES** → re-raised at `:969-972` → **the run crashes.** But `sync_basis=false` is a *supported* mode — the config docstring calls it "diagnostic only … keeps each rank's basis local" (`config/comm_eff.py:332-333`), the constructor defaults it `False` (`powersgd_activation.py:188`), and the plan's step-2 config text literally says `powersgd.sync_basis=false` (`plans/20.md:82`). Asserting cross-rank identity is only valid when `sync_basis=true`; under `false`, per-rank divergence is *expected and correct*. **Fix:** gate the verify call (or the verifier body) on `self.sync_basis` — e.g. `if did_update and powersgd.sync_basis and not _checked:`. One line. Until fixed, the `sync_basis=false` diagnostic arm is uncrunnable multi-GPU.

**What would refute HIGH-2:** a `sync_basis` short-circuit inside the verifier or at the call site. I grepped both; there is none. **This is a finding the math-checker's review (which assumed the `sync_basis=true` path) did not surface — it is orthogonal to, not a contradiction of, their D4.**

---

## Check 5 — FSDP shards params, not activations ⇒ hook's `M` is the full LOCAL activation — **VALID**

- FSDP (FULL_SHARD) shards the flat *parameter*, all-gathers it JIT for each module forward, reshards after — the module computes a full unsharded output activation. The forward hook fires on the block output (`powersgd_activation.py:306-307`, `:665`), `M = h.reshape(-1, H)` (`:333-334`) — the complete `(N,H)` hidden state for *this rank's* token batch, not a param shard and not token-sliced (SP=1; SP>1 is hard-refused, `transformer_impl.py:776-781`). ✔
- ⇒ each rank's `M_i` is the full local activation over a distinct data shard, so `(M_iᵀM_i)Q` is the genuinely-local second-moment-times-Q the consensus all-reduce pools (Check 3). The rmpad guard (`transformer_impl.py:811-817`) ensures no PAD rows enter the sketch. ✔

**What would refute me:** FSDP sharding activations (it does not) or the hook firing on a sharded intermediate (it fires on the block output). Neither holds. **CONFIRM math-checker D2.**

---

## Adversarial cross-check of math-checker claim 9 (consensus basis) — **CONFIRM**

On independent code reading I confirm claim 9 from the distributed angle: the reduced object is the **raw** `(MᵀM)Q` not per-rank `orth(Vᵢ)` (Check 3); the group is the DP group = WORLD at SP=1 (Check 1); the result is bit-identical across ranks via deterministic sign-canonicalized `orth` on an identical all-reduce output (Check 3); the collective sequence is symmetric (Check 2). No daylight between claim 9 and the code *as written*.

**The one thing I add that strengthens-by-qualifying the math-checker's bottom line:** their review states "the expected on-box result is `q_cross_rank_max_rel_dev ≈ 0.0` (the runner's report)." I checked the on-disk artifacts and **found no run that produced that number** — the consensus path postdates the only probe, which ran `sync_basis=false`. So claim 9 is correct *in theory and in the committed code*, but it is **untested on hardware**; the "0.0" is an expectation, not an observation. I do not CHALLENGE the math; I CHALLENGE the implied evidentiary status of the `0.0`.

**What would refute my confirmation of the math:** SP>1/TP/PP in the training mesh (would make WORLD≠DP), per-rank pre-orth of the sketch, or `sync_basis=false` at run time (then per-rank `orth(Vᵢ)` diverges and claim 9's "shared Q" is false). The committed code + launcher default make none of these true *for the intended `sync_basis=true` launch*; the on-disk probe, however, used `sync_basis=false`.

---

## What the on-disk probe DID validate (for completeness, outside the consensus path)

The 2-step, 4-GPU probe (`sync_basis=false`, commit `def451e5e`) DID run clean and shows, per `metrics`-in-`train.log`:
- `powersgd_q_cond ≈ 1.0000003` every boundary, both steps (orthonormal basis healthy; no collapse). ✔
- `powersgd_basis_updates: 1 → 2` (cadence=1, clean_cadence=0: every step updates). ✔
- `logical_pp_bytes_powersgd_y_only: 102.0` (budget-match target). ✔
- `powersgd_applications: 3584 → 7168`, no NaN/Inf, no OOM, all 4 GPUs live. ✔
- `reconstruction_rel_error` step1 ≈ 0.97 (near the 1.0 discard threshold at the random-bootstrap basis), step2 dropping (layer_3 → 0.025) as the local block-power-iteration warms — note this is the *unsynced* basis; the synced basis would differ. (This is a codec-health observation for the analyst, not in my distributed lens; flag that the high step-1 reconstruction error is the cold bootstrap, expected.)

This confirms the *local* codec runs, but says nothing about the consensus path (HIGH-1).

---

## Risk register (by severity)

| Sev | Finding | Where | Impact | Action |
|---|---|---|---|---|
| **HIGH** | Consensus path + verifier **never run on disk**; reported `q_cross_rank_max_rel_dev=0.0` unsubstantiated | probe log mtime 02:36 vs commit `f748dbc1` 04:01; `sync_basis=False` + 0 agreement lines in probe log | The collective-safety/consensus correctness is static-only; lockstep + no-hang + `0.0` agreement are unobserved | **Re-run the ≤2-step probe with `sync_basis=true` on ≥2 GPUs BEFORE the 50-step sweep.** Confirm the `[comm_eff][EXP-20] cross-rank Q agreement: max_rel_dev=…` line prints and the metric is ≤1e-6. This is the plan's own hard-invariant #4 gate — currently ungated by evidence. |
| **HIGH** | Verifier mis-gated: asserts cross-rank `Q` identity with **no `sync_basis` guard** | `powersgd_activation.py:609-641` (no guard) + call site `engine_workers.py:952` (gated on `did_update` only) | Running commit `f748dbc1` with `sync_basis=false` (a supported mode; plan step-2 text + constructor default) **crashes** on the first non-clean update via `RuntimeError("basis Q DIVERGED…")` | One-line fix: gate verify on `self.sync_basis` (identity is an invariant only then). Until then, do not run any `sync_basis=false` arm multi-GPU on this commit. |
| LOW | C2: DP-group bind in broad `except` → silent WORLD fallback | `engine_workers.py:784-785` | None at SP=1 (WORLD==DP); wrong group under future SP>1/TP/PP | Fail-loud / assert `dp_ws==world` only when SP==1, before any SP>1 use. |
| LOW | C1: plan step-2 + constructor default `sync_basis=false`; launcher/config default `true` | `plans/20.md:82`, `powersgd_activation.py:188` vs `config/comm_eff.py:355`, `launcher.sh:280` | None for the intended launch (`true` is correct + config overrides constructor); confusing + interacts with HIGH-2 | Analyst: record resolved `sync_basis` from `resolved_params.txt`; reconcile the plan text. |
| INFO | Verifier checks once + fp64-checksum (not full-Q) | `engine_workers.py:952`, `powersgd_activation.py:566-641` | Sufficient for the gate; theoretical collision / post-step-1 drift unverified | None required. |

---

## Final position

For the **intended launch (`sync_basis=true`)** the distributed/collective `sync_basis` path is **correct as written**: the all-reduce reduces over the DP group (= WORLD at SP=1, narrowing correctly under future SP>1), the collective sequence is symmetric and deadlock-safe, the consensus is the mathematically-correct raw-sketch pool → `orth(C_glob·Q)` yielding a bit-identical `Q` on every rank, FSDP-shards-params (not activations) holds, and the verifier mechanism is sound and sufficient. I CONFIRM the mathematical-checker's claim 9.

**However, two HIGH-severity distributed-lens caveats stand:** (1) **none of this consensus code has executed on hardware** — the only probe predates the consensus commit and ran with sync OFF, so the reported `0.0` cross-rank agreement is an expectation, not evidence; a `sync_basis=true` probe re-run is required to satisfy the plan's own hard-invariant #4. (2) The verifier is **mis-gated** and will crash the supported `sync_basis=false` mode on multi-GPU because it asserts an identity that only holds under `sync_basis=true`. Both are fixable (a re-probe and a one-line guard) and neither impugns the consensus math, but the science is **not yet gated by a run that exercised the path being shipped.**
