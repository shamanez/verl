# EXP-20 PowerSGD Activation-Compression — Mathematical Validity Review

**Reviewer:** mathematical-checker (READ-ONLY math review)
**Date:** 2026-06-04
**Reviewed commit:** `f748dbc1c63ef9824a3115b091ed025fe210cf9b` (`origin/exp/20-powersgd-activation`)
**Ground truth:** GitHub issue #20 (Parts I–XI, INF-1…INF-20) + `research/.claude/plans/20.md`
**Scope:** theoretical / mathematical fidelity of the implementation to the issue's equations and invariants. NOT a runtime/perf review.

## Files reviewed (all at the committed hash above)
- `verl/workers/comm_eff/powersgd_activation.py` — projector, block power iteration, orth, seed, sketch, basis update, sync/consensus.
- `verl/workers/comm_eff/state.py` — clean-step predicate, codec resolution, build(), counters/metrics.
- `verl/workers/engine/fsdp/transformer_impl.py` — boundary hook, `M = output[0]`, frozen-Q gating, register/unregister, no_grad ctx.
- `verl/workers/engine_workers.py` — frozen-Q gating across paired forwards, clean-step wiring, `maybe_update_basis` call site, DP-group binding.
- `verl/workers/config/comm_eff.py` — `powersgd.*` knobs + defaults + `compression_type` enum.
- `verl/workers/comm_eff/activation_mask.py` — `decoder_boundary_indices`, `find_decoder_layers` (shared boundary selection).
- `tests/workers/comm_eff/test_powersgd_activation.py` — the CPU-runnable encodings of the hard invariants.
- `research/runs/EXP-20/ce_powersgd_probe_2s_gsm8k.log` — the on-box 2-step probe (corroborating numerics).

**Note on `sync_basis` (per the task brief):** the committed branch `f748dbc` **already has** the operator's `sync_basis=true` work landed. The config default `CommEffPowerSGDConfig.sync_basis = True` (`config/comm_eff.py:355`), the consensus all-reduce, the collective-safety guard, and the cross-rank checksum verifier are all present. The probe log in `runs/EXP-20/` predates this and shows an explicit CLI `powersgd.sync_basis=false` override — it is NOT the committed default. Claim 9 is reviewed against the committed code as well as conceptually.

---

## Per-claim verdict table

| # | Claim | Verdict | Reasoning (issue ref → code) |
|---|---|---|---|
| 1 | Projector + autograd, no STE | **VALID** | See below |
| 2 | Block power iteration ⇒ reconstruction-error min | **VALID** | See below |
| 3 | Deterministic zero-comm seed bootstrap | **VALID** | See below |
| 4 | Frozen-Q across paired GRPO forwards | **VALID** | See below |
| 5 | r=H lossless | **VALID** | See below |
| 6 | fp32 QR / dtype discipline | **VALID** | See below |
| 7 | Byte-budget matching (r=102 ≡ p=0.95) | **VALID-WITH-CAVEAT** | See below |
| 8 | Clean-cadence dense debiaser | **VALID** | See below |
| 9 | Cross-DP consensus basis (sync_basis) | **VALID** | See below |
| 10 | Discrepancies / unstated assumptions | **see list** | See below |

---

### Claim 1 — Projector + autograd (Part III.7, INF-9): **VALID**

Forward (`powersgd_activation.py:339-341`):
```python
q_act = q_fp32.to(dtype=M.dtype)
Y = M @ q_act          # (N, r)
M_hat = Y @ q_act.t()  # (N, H) — in-graph through M
```
- **Q detached:** `Q` is a plain tensor held in `self._basis` (`_ensure_basis`, line 258-277). It is created by `init_basis`→`orthonormalize`, both of which call `.detach()` / build fresh tensors; it is never an `nn.Parameter`, never `requires_grad_(True)`. So `Q` carries no grad and no graph. ✓ (INF-9 "Q a fixed/detached basis")
- **M in-graph:** `M = h.reshape(-1, hidden_size)` (line 333) is a view of the live block output `h`; the matmuls `M @ q_act` and `Y @ q_act.t()` are ordinary autograd ops, so `M_hat` stays in the graph through `M`. ✓
- **No STE / no custom Function:** there is no `torch.autograd.Function`, no `detach()` on `M`, no `+ (x - x.detach())` trick anywhere in the forward. The backward is therefore the genuine autograd of two matmuls with a constant `Q`:
  `dL/dM = (dL/dM_hat) @ (Q Qᵀ)ᵀ = (dL/dM_hat) Q Qᵀ`, and since `P=QQᵀ` is symmetric this is the self-adjoint projector applied to the upstream gradient. ✓ (Part III.7 eq, INF-9)
- **Idempotency `QQᵀ·QQᵀ=QQᵀ` when `QᵀQ=I`:** holds whenever `Q` has orthonormal columns. `init_basis` returns `orth(randn)` (QR ⇒ `QᵀQ=I`), and every update sets `Q=orth(V)` (QR again). So the projector identity `P²=P` is maintained by construction. ✓ (INF-2)
- **Direct test corroboration:** `test_autograd_no_ste` asserts `‖M.grad − (g@Q)@Qᵀ‖/‖·‖ < 1e-5`; `test_r_equals_H_lossless` asserts `‖M−M_hat‖/‖M‖ < 1e-4`. Both encode exactly this claim.

Off-graph diagnostics (q_cond, reconstruction, sketch) are inside a `with torch.no_grad():` block (line 343), so they cannot perturb the graph. ✓

### Claim 2 — Block power iteration ⇒ reconstruction-error minimization: **VALID**

Sketch accumulation (`powersgd_activation.py:368-383`, inside `no_grad`):
```python
contrib = M32.t() @ Y.detach().to(torch.float32)   # (H, r) = Mᵀ(MQ)
... self._sketch[layer_idx] = contrib / cur.add_(contrib)
```
Update (`maybe_update_basis`, line 507-514): `Vsum = V; [all_reduce]; q_new = orth(Vsum)`.

- **One step of subspace power iteration on SPD `C=MᵀM`:** `V = Mᵀ(MQ) = (MᵀM)Q = C Q`, then `Q ← orth(V) = orth(C Q)`. This is exactly the issue's `Q_{t+1}=orth(C Q_t)` (Part III.5 / IV-row5 / INF-5). I verified numerically that `V_exact == C@Q` to fp tolerance and that the hook's bf16-rounded `Y` path differs from the fp32-exact `(MᵀM)Q` by only ~0.17% in `‖V‖` and ~4e-4 in the resulting subspace projector — negligible and absorbed by `orth` (scale/rounding-invariant) and the cadence loop. ✓
- **Eckart–Young optimal basis:** `C=MᵀM` is SPD; its top-`r` eigenvectors = the top-`r` right singular vectors `V_r` of `M`; the projector `Q Qᵀ` with `Q=V_r` is the Frobenius-and-spectral optimal rank-`r` row-space projector (`M̂ = U_r S_r V_rᵀ`), minimizing `‖M − MQQᵀ‖_F` (INF-3, INF-4). Iterating `orth(CQ)` drives `Q` toward `V_r` linearly at rate `(σ_{r+1}/σ_r)^{2t}` (INF-5), conditional on a spectral gap. The code implements the iteration faithfully; the gap is an *empirical* precondition (INF-20), not something the code can guarantee — see Caveat in claim 7/10. ✓ (the iteration is correct; optimality is conditional on the gap, exactly as the theory states)
- **Warm-start correctness:** `Q` persists in `self._basis` across steps and is NOT cleared by `unregister()` (docstring line 217, `unregister` line 676-680 only removes hooks). `warm_start=False` re-bootstraps from the per-layer seed each update (line 481-493, diagnostic). Matches Part I.5 / IV-row8. ✓
- **Sketch folds ONLY the gradient-bearing train forward, not old-logprob recompute:** `_should_accumulate_sketch` (line 396-416) requires `grad_enabled` (captured at hook entry, line 317, *before* the no_grad diagnostics block) AND `path_tag == TRAIN_TAG`. The old-logprob recompute runs the whole forward under `torch.no_grad()` (`transformer_impl.py:864` `ctx = torch.no_grad() if forward_only else nullcontext()`; the recompute is `forward_only=True`), so `grad_enabled=False` there ⇒ no sketch. ✓ (Part V.3, "accumulate only on compressed train forwards"). `test_old_logprob_recompute_projects_but_no_sketch` encodes this.
- **Grad-checkpoint dedupe:** `_sketched_this_gen[layer_idx] == self._fwd_generation` gate (line 416). `set_context` bumps `_fwd_generation` once per micro-batch (line 297); under grad-checkpointing the boundary forward is recomputed in backward with the SAME context/generation, so a layer folds into `V` at most once per generation. ✓ (INF-12 / Part XI.3 grad-ckpt row). `test_grad_ckpt_recompute_not_double_counted` encodes this.
- **Cross-microbatch accumulation = global-step covariance:** `cur.add_(contrib)` sums over micro-batches ⇒ `V = (Σ_mb C_mb) Q = C_step Q` (INF-12). ✓

The committed `_sketch_count` is tracked but **not used to divide `V`** before `orth` — correct, because `orth` is scale-invariant (the count would only matter for a mean, which the basis direction does not need). The docstring on `maybe_update_basis` explicitly notes this. ✓ (no bug)

### Claim 3 — Deterministic zero-comm seed bootstrap (INF-13, Part III.4): **VALID**

`powersgd_layer_seed` (line 83-90): `(base_seed*1_000_003 + layer_idx*7919) & 0x7FFFFFFF`, with module constants `_PRF_MIX_BASE=1_000_003`, `_PRF_MIX_LAYER=7919`, `_MASK31=0x7FFFFFFF` (line 78-80). Exact match to the issue formula. ✓
`init_basis` (line 132-152): `torch.Generator(device="cpu").manual_seed(seed)`, `torch.randn(H, r, ..., dtype=torch.float32)`, then `orthonormalize`. Drawing on **CPU in fp32** with a seeded generator ⇒ bit-identical bytes on every rank/device irrespective of accelerator RNG state. ✓ (INF-13 device-independence)
The int31 mask guarantees a legal `manual_seed` on every backend (docstring line 87-88). `test_seed_formula_inf13` and `test_determinism_multi_rank` encode the formula + cross-"rank" identity.

### Claim 4 — Frozen-Q across paired GRPO forwards (Part V.3, INF-17): **VALID**

- **Both forwards see `Q_t`:** the basis is only ever mutated in `maybe_update_basis` (line 421-519). The forward hook *reads* `self._basis` and never writes it (it only writes the off-graph `self._sketch`). So within a global step, every forward — the old-logprob recompute and the actor-train forward — reads the same `Q_t`. ✓
- **Q advances only AFTER the gradient-bearing work:** `maybe_update_basis` is called from `engine_workers.py:941-943`, inside the `finally:` of `self.actor.train_mini_batch(data=data)` (line 926-943) — i.e. after ALL PPO mini-batch forwards/backwards of `update_actor` have run. The old-logprob recompute happens earlier (in `compute_log_prob`, a separate call). So `Q_t → Q_{t+1}` happens strictly after both paired forwards. ✓ (Part V.3 ordering)
- **Recompute does not update Q:** the recompute path is `forward_only`/`no_grad` and folds nothing into `V` (claim 2), and `maybe_update_basis` is not called from `compute_log_prob`. ✓
- **ρ≈1 at step 0:** because both forwards apply the *identical* operator `Q_t Q_tᵀ`, `logπ_old` and `logπ_new` differ only by weight change; with no weight change ρ=exp(0)=1 (INF-17). The gating (`_comm_eff_powersgd_active`, `transformer_impl.py:733-765`) projects the recompute iff `compress_recompute=true` AND `path_tag==old_logprob` AND `forward_only=True`; the train forward iff `path_tag==train` AND not `forward_only`. Both use the same frozen `Q_t`. ✓ The plan's frozen-Q hard invariant ("ρ≈1 at step 0") is the on-box check; `test_q_frozen_within_step_advances_only_on_update` encodes the single-process version (basis unchanged across forwards, advances only on `maybe_update_basis`).

### Claim 5 — r=H lossless (INF-18): **VALID**

`init_basis` clamps `r = min(int(rank), int(hidden_size))` (line 148) and `_effective_rank` does the same (line 252-255). At `r=H`, `orthonormalize` runs reduced QR on a square `(H,H)` full-rank random matrix ⇒ a **square orthogonal** `Q` with `QQᵀ = QᵀQ = I_H`, so `M_hat = M Q Qᵀ = M` exactly. I verified numerically: `‖QQᵀ − I_H‖_max ≈ 3.6e-7` and `reconstruction_rel_error ≈ 4.7e-7` at H=128. ✓ (INF-18). `test_r_equals_H_lossless` asserts `< 1e-4` at H=128.

### Claim 6 — fp32 QR / dtype (INF-14): **VALID**

- **QR in fp32:** `orthonormalize` does `work = mat.detach().to(torch.float32)` then `torch.linalg.qr(work, mode="reduced")` (line 102-110). `init_basis` draws fp32. The basis is *stored* fp32 (`_ensure_basis` keeps `torch.float32`, line 271). ✓ (INF-14 "QR in fp32")
- **Projection in activation dtype:** the forward casts `q_act = q_fp32.to(dtype=M.dtype)` and does `Y=M@q_act`, `M_hat=Y@q_act.t()` in the activation dtype (line 338-341). ✓ (INF-14 "store/QR fp32, project in activation dtype")
- **`qr_dtype` knob:** default `"fp32"`; the sketch and orth use `self.qr_dtype` (line 202-203, 513). `"bf16"` is allowed but documented as diagnostic-only (config docstring + class docstring). The committed default is fp32. ✓
- **q_cond ≈ 1 + finiteness guard:** the diagnostic computes `svdvals(Q)` and `cond = smax/smin if smin > eps else inf` (line 347-352), wrapped in try/except that returns `inf` on failure — a non-finite cond is surfaced (the basis-collapse falsifier). On the box, `q_cond ≈ 1.0000002` at every boundary (probe log), confirming orthonormality. ✓
- **Degenerate-column repair keeps full rank:** `orthonormalize` detects `|diag(R)| <= eps` columns and re-seeds them from a deterministic random complement's QR (line 117-128), and a non-finite sketch is `nan_to_num`'d before QR (line 106-107). So a rank-deficient/NaN sketch never propagates a NaN/rank-deficient basis. ✓ `test_orth_rank_deficient_repaired` + `test_orth_nonfinite_sketch_does_not_propagate_nan` encode this.
- **Sign canonicalization:** `q = q * sign(diag(R))` (line 113-116) makes QR unique (bit-identical across ranks) — supports claims 3 and 9. ✓

### Claim 7 — Byte-budget matching: **VALID-WITH-CAVEAT**

- **Arithmetic:** PowerSGD payload = `n·r` per token-layer; PRF keeps `(1−p)·H = 0.05·2048 = 102.4` coords/token. With `r=102`: `102 vs 102.4` ⇒ 0.39% apart, well within the 1% tolerance. Note the **floor convention** (`r=102` = `floor(0.05·2048)`) is what the plan/issue Part VI use; `round` would give 102 here too (0.05·2048=102.4→102). ✓
- **Logging:** `last_y_coords_per_token = q_act.shape[1]` (= effective rank, line 366) is surfaced as `comm_eff/logical_pp_bytes_powersgd_y_only` (`state.py:617`); the PRF side logs `comm_eff/logical_pp_bytes_prf = (1−p)·H` (`state.py:540`). Probe log shows `logical_pp_bytes_powersgd_y_only = 102.0`. So the analyst can assert equality. ✓

**CAVEAT (honesty of the budget — the "is Q free?" question):** the `n·r`-only accounting is honest **only in the `n→∞`, no-Q-transmission limit** (INF-16: amortized `n·r + H·r/cadence`). Two things make the headline `n·r` an *optimistic* budget in the actually-committed config:
  1. **`sync_basis=true` transmits `V` (shape `H×r`) every non-clean update.** This is an `H·r = 2048·102 ≈ 209k`-value all-reduce per boundary per step (7 boundaries). The issue's own INF-16/VII.1 says when `sync_basis=true` you must **count `H·r` basis-sync values** and provides the metric `comm_eff/logical_pp_bytes_powersgd_with_basis_sync = n·r + H·r/update_cadence`. **That metric is NOT emitted by `powersgd_metrics()`** (only the `_y_only` form is). At `update_cadence=1` the per-step amortized cost is `n·r + H·r`, and for the GSM8K token counts in the probe (`n ≈ total_nnz` ~ tens of thousands per boundary) `n·r` dominates, but `H·r` is not zero. The *seed bootstrap* is genuinely free (deterministic, zero-comm, INF-13), but the *refresh* under sync is not.
  2. The matched comparison to PRF is still fair for the **forward/activation** wire (`Y` vs masked-`h`), which is the dominant per-step traffic; the `V` all-reduce is a basis-maintenance cost the PRF mask does not incur. The analyst should either (a) also report `logical_pp_bytes_powersgd_with_basis_sync`, or (b) explicitly state the comparison is forward-payload-matched and the `H·r` consensus traffic is an extra PowerSGD-only cost. This does not invalidate the matched-budget *forward* claim, but the budget is not unconditionally `n·r` once `sync_basis=true`.

Severity: medium (a reporting/honesty gap, not a numerical bug). The plan's success box only asserts `_y_only == prf`, so the literal box passes; the caveat is that `_y_only` understates total PowerSGD traffic under the committed `sync_basis=true`.

### Claim 8 — Clean-cadence dense debiaser: **VALID**

- **Predicate:** `is_clean_step` (`state.py:431-453`): `cadence>0 and gs>0 and (gs % cadence)==0`. Exact match. The `gs<=0` guard correctly excludes the pre-train/val boundary. ✓
- **Both forwards dense on a clean step:**
  - Actor-train forward: `update_actor` sets `comm_eff_state.mask_active = not clean_step` (`engine_workers.py:917`). On a clean step `mask_active=False` ⇒ `_comm_eff_powersgd_active` returns False (`transformer_impl.py:756-757`) ⇒ no hook registered ⇒ dense. ✓
  - Old-logprob recompute: the `compress_recompute` stamp is gated on `not clean_step` (`engine_workers.py:700`), so `mask_active` is NOT set on the recompute on a clean step ⇒ dense. ✓
  - This is the "`mask_active = not clean_step` gates BOTH codecs" property the claim asks for. ✓ (Part V.4)
- **Basis update skipped (Q held):** `maybe_update_basis(is_clean_step=clean_step)` early-returns `False` and only clears stray sketch when `is_clean_step` (`powersgd_activation.py:458-463`). So `Q` is held across a clean step. ✓ (Part V.4 "no V accumulation; no Q update")
- **`clean_steps += 1`** once per trainer step on the train stamp (`engine_workers.py:920-925`), mirroring PRF. ✓
- Because no hooks fire on a clean step, no `V` accumulates anyway (belt-and-braces with the `is_clean_step` early return). ✓

### Claim 9 — Cross-DP consensus basis (sync_basis): **VALID**

The committed code does the mathematically correct thing.
- **All-reduce the RAW sketch, then orth (`powersgd_activation.py:507-514`):**
  ```python
  Vsum = V.to(torch.float32)
  if do_sync:
      torch.distributed.all_reduce(Vsum, op=SUM, group=group)
  q_new = orthonormalize(Vsum.to(self.qr_dtype), eps=...)
  ```
  With each rank's `V_i = Σ_mb (M_{i,mb}ᵀ M_{i,mb}) Q = C_i Q` (same frozen `Q` on all ranks at step start), `Σ_i V_i = (Σ_i C_i) Q = (M_globᵀ M_glob) Q` — one block-power-iteration step on the **pooled** activation gram. Every rank then orthonormalizes the *identical* `V_global` ⇒ bit-identical consensus `Q` differing only per boundary. ✓ (matches the operator's clarification + commit message #1)
- **SUM not mean is correct:** `orth` is scale-invariant, so summing the raw per-rank `V`s gives the pooled *direction* without any per-rank count re-weighting (docstring line 445-447). ✓
- **Contrast with the WRONG alternatives:** averaging per-rank `orth(V_i)`'s would average *orthonormal frames* (not a subspace operation — meaningless); all-reducing `M` directly is impossible (different `n` per rank) and unnecessary. The code reduces `V` (fixed `H×r` shape on every rank), which is the right object. ✓
- **`sync_basis=false` is the documented failure mode:** with different per-rank shards and no sync, `orth(V_i)` diverges across ranks after the first update (commit message; config docstring line 332-335). The default is correctly committed `True`. ✓
- **On-box verification exists:** `verify_basis_agreement_across_ranks` (line 587-639) all-gathers an fp64 per-boundary checksum over the FIXED boundary set and RAISES if max cross-rank rel-dev > 1e-6 — so a broken consensus fails loudly rather than silently training divergent codebooks. Called once after the first update (`engine_workers.py:952-957`). The checksum (`Q ⊙ index-ramp` summed in fp64, line 565-584) is sensitive to sign/permutation/value changes. `test_basis_checksums_deterministic_and_sensitive` encodes sensitivity. ✓

This is a genuinely correct, faithful realization of the issue's "sum V across the fast subgroup before QR" (Part III.8 / IV-row9 / INF-12) lifted from the fast-PP subgroup to the DP group.

### Claim 10 — Discrepancies / unstated assumptions

Ranked by severity:

**MEDIUM — `logical_pp_bytes_powersgd_with_basis_sync` not emitted (claim 7 caveat).** The issue specifies this metric (VII.2) and `account_basis_sync_bytes: true` as a config knob (VII.1). Neither the `account_basis_sync_bytes` knob nor the `with_basis_sync` metric is present in the committed code. Under the committed `sync_basis=true`, the headline `_y_only` budget omits the per-step `H·r` consensus all-reduce. Recommend the analyst report it as forward-payload-matched and note the extra consensus traffic, or add the metric. Not a numerical-correctness bug.

**LOW — process-group choice for the consensus all-reduce.** `set_dp_group(engine.get_data_parallel_group())` is bound in `engine_workers.py:760-764`. For the EXP-20 actor (FSDP, Ulysses SP=1, no TP/PP in the *training* mesh) `get_data_parallel_group()` returns the WORLD group and `world_size==dp_size==4`, so pooling over the DP group == pooling over the world, which is exactly the set of shards we want to consensus over. This is **correct for the sanctioned config**. The `_dp_group()` default-`None`→world fallback (line 542-554) is also correct here. The code is forward-safe (a future SP>1/TP/PP config would inject a narrower group and the all-reduce would correctly reduce over the DP subgroup only). The launcher's TP=2 is rollout-only (separate vLLM mesh), not in the FSDP training PG (commit message #2). VALID; flagged only because correctness is contingent on SP=1 (which the launcher enforces and `_comm_eff_register_powersgd_hooks` asserts, `transformer_impl.py:773-781`).

**LOW — deadlock / collective-safety.** With `sync_basis=true` every rank must issue the identical collective sequence. The code iterates the **FIXED `sorted(self.boundary_indices)`** (`_boundary_for_update`, line 522-532) — derived from `decoder_boundary_indices(L, pp_size)` on the same model, hence identical on every rank — and contributes a correctly-shaped **zero `V`** for any boundary missing locally (line 498-503), and does NOT early-return on an empty local sketch when `do_sync` (line 478-479). The verifier all-gathers a fixed-length, fixed-order vector (line 614-623). All ranks reach `maybe_update_basis`/`verify_*` on the same cadence step in lockstep (called from the `finally:` of `train_mini_batch`, which all ranks run together). So the collective set is symmetric and cannot deadlock. The plan's stall-signature risk (`single-GPU fallback`, `summon_full_params`) is mitigated. VALID. (The single-process CI path: `do_sync = sync_basis and torch.distributed.is_initialized()` ⇒ False without dist init, so CI behaves like the local path — `test_sync_basis_single_process_equivalent_to_local`.)

**LOW — the "lockstep under identical data ordering" framing.** The class/commit text is careful and CORRECT: it explicitly says per-rank `V` diverges under DP because each rank gets a different shard, which is *why* sync is required. There is no false "identical data ordering" claim in the committed PowerSGD code (that pitfall is named only to be rejected). So the task's flagged concern does not actually manifest as a bug here. VALID.

**INFORMATIONAL — sketch uses bf16-rounded `Y`.** The hook reuses the activation-dtype `Y` (bf16) cast back to fp32 for the sketch (`contrib = M32.t() @ Y.detach().to(fp32)`, line 374-375), rather than recomputing `M32 @ Q` in fp32. I measured the resulting `V` error at ~0.17% and the subspace-projector error at ~4e-4 — negligible, and `orth` + cadence absorb it. This is a deliberate compute-saving choice (reuse the already-computed `Y`), consistent with INF-14's "projection tolerates low precision; only orth needs fp32." Not a defect; noted for completeness.

**INFORMATIONAL — boundary indices differ from the docstring example.** The docstrings use `L=16, pp_size=8 → [1,3,5,7,9,11,13]`. Qwen2.5-1.5B-Instruct has **L=28** layers, so `decoder_boundary_indices(28, 8)` → `[3,7,11,15,18,21,24]` (7 boundaries), which is exactly what the probe log shows (`layer_3…layer_24`). The byte-budget claim is per-token-*layer* and is unaffected by the count; the matched comparison uses the same boundary set for both arms (PowerSGD reuses the mask's `decoder_boundary_indices`/`find_decoder_layers`, `powersgd_activation.py:60-63`, 662). Correct.

**EMPIRICAL FLAG (not an implementation defect, but decisive for the experiment) — weak spectral precondition (INF-20).** The on-box 2-step probe shows `powersgd_reconstruction_rel_error ≈ 0.72–0.97` per boundary (e.g. step 1 mean 0.967, step 2 mean 0.716). This is **bounded < 1.0** (the codec keeps more than it discards, so the plan's `< 1.0` health gate passes) but it is **high** — it says the Qwen2.5-1.5B GSM8K boundary activations are NOT strongly low-rank at `r=102` (the INF-20 spectral gap is weak/absent at this rank). The math is implemented correctly; whether the *method* works is the empirical question the 50-step sweep answers. Per the plan's analyst predicate, a reconstruction error creeping toward 1.0 with finite `q_cond` argues for REVISE toward a larger rank (r=205) rather than a code fix. Flagging so the analyst reads reconstruction error as the INF-20 precondition test, not a bug.

---

## Bottom line

**The implementation at `f748dbc` is a faithful, mathematically-correct realization of issue #20's theory.** All ten claims hold: the projector and its self-adjoint no-STE backward (1), the block-power-iteration basis update on `C=MᵀM` and its train-forward-only / grad-ckpt-deduped sketch (2), the deterministic zero-comm seed bootstrap (3), frozen-Q across the paired GRPO forwards with the post-backward update (4), r=H losslessness (5), fp32-QR/activation-dtype discipline with finiteness/degeneracy guards (6), the byte-budget arithmetic and logging (7), the clean-cadence dense debiaser gating both forwards and holding Q (8), and — notably — the **cross-DP consensus basis is correct**: it all-reduces the raw sketch `V` then orthonormalizes, yielding the pooled-gram top-`r` subspace and a bit-identical per-rank `Q`, with a deadlock-safe symmetric collective and an on-box checksum verifier (9). The committed branch already carries `sync_basis=true` + the collective-safety + cross-rank verification work.

**No INVALID claims.** One **VALID-WITH-CAVEAT** (claim 7): the headline `n·r` budget is honest for the forward/activation wire but, under the committed `sync_basis=true`, omits the per-step `H·r` basis-consensus all-reduce — the issue-specified `logical_pp_bytes_powersgd_with_basis_sync` metric and `account_basis_sync_bytes` knob are not implemented, so the analyst should report the comparison as forward-payload-matched and note the extra PowerSGD-only consensus traffic. The remaining flags are LOW/INFORMATIONAL (process-group choice contingent on SP=1, which is asserted; bf16-rounded sketch, negligible) plus one **empirical** flag that is NOT a code defect: the probe's high-but-bounded reconstruction error (~0.72–0.97) signals a weak INF-20 spectral precondition at r=102, which is exactly the thing the experiment is designed to measure and would drive a REVISE-to-larger-rank, not a code fix.
