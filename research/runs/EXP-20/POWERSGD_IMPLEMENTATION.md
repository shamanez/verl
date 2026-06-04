# PowerSGD-style Pipeline-Boundary Activation Compression in verl (GRPO)

How the EXP-20 / M6 codec is implemented in this fork, end to end. Reviewed/verified against commit `f748dbc1` (`exp/20-powersgd-activation`).

---

## 0. The one-sentence idea

At each pipeline-stage boundary, replace the block's output activation `M` (shape `(N tokens, H)`) with its projection onto a **shared, frozen, low-rank orthonormal basis** `Q` (shape `(H, r)`):

```
M̂ = (M @ Q) @ Qᵀ
```

Only the `r`-dim coordinates `Y = M @ Q` ("`n·r` payload") need cross the boundary; `Q` is a communication-free shared codebook (bootstrapped from a seed, refreshed by power iteration). Training is **vanilla GRPO** on Qwen2.5-1.5B-Instruct + GSM8K. With the codec off, the path is byte-identical to dense GRPO.

> In this experiment there is **no real pipeline parallelism** — the boundary compression is *simulated in place* via forward hooks on selected decoder blocks, while the model runs under FSDP data-parallel across 4 GPUs.

---

## 1. Components (where the code lives)

| File | Role |
|---|---|
| `verl/workers/comm_eff/powersgd_activation.py` | **The codec.** `PowerSGDActivationCompressor`: the projector, the block-power-iteration basis update, fp32 orthonormalization, the deterministic per-layer seed, the cross-DP consensus, and the health diagnostics. |
| `verl/workers/config/comm_eff.py` | **Config.** `compression_type` enum `{dense, prf_mask, powersgd}` + the `powersgd.*` knob block (dataclass + YAML schema). |
| `verl/workers/comm_eff/state.py` | **State machine.** `CommEffState`: `resolve_compression_type`, `build()` (constructs **exactly one** codec), the clean-step predicate, path tags, counters/metrics. |
| `verl/workers/engine/fsdp/transformer_impl.py` | **FSDP boundary-hook lifecycle.** Registers/unregisters the projection hooks on the boundary blocks, sets the per-forward context, gates *which* forwards are compressed (`_comm_eff_powersgd_active`). |
| `verl/workers/engine_workers.py` | **GRPO driver glue.** Frozen-Q gating across the paired forwards (old-logprob recompute + actor-train), the clean-step wiring, and the post-backward basis-update call site. |

The codec reuses the PRF mask's boundary-selection helpers (`find_decoder_layers`, `decoder_boundary_indices`) so it compresses **exactly the same boundary blocks** as the mask — required for the matched comparison.

---

## 2. The projector and its gradient (no straight-through)

Forward (in the boundary block's `register_forward_hook`, `powersgd_activation.py:_make_hook`):

```python
q_act = q_fp32.to(dtype=M.dtype)   # Q stored fp32, cast to activation dtype (bf16) for the matmul
Y     = M @ q_act                  # (N, r)  — the projected coordinates actually "sent"
M_hat = Y @ q_act.t()              # (N, H)  — reconstruction; stays in-graph through M
```

* **`Q` is detached** (a plain buffer in `self._basis`, never an `nn.Parameter`, never `requires_grad`). **`M` stays in-graph.** So PyTorch autograd of these two matmuls *is exactly* the self-adjoint projector — **no custom `autograd.Function`, no straight-through estimator**:

  ```
  dL/dM = (dL/dM̂) · Q Qᵀ
  ```
* Because `Q` is orthonormal (`QᵀQ = I`), `P = QQᵀ` is an idempotent projector (`P² = P`), so the gradient is genuinely "the upstream gradient, projected onto the kept subspace."
* The **byte budget** is `r` coordinates per token-layer (`Y` is `(N, r)`), logged as `comm_eff/logical_pp_bytes_powersgd_y_only`.

---

## 3. How `Q` is initialized — the zero-communication bootstrap

Per boundary layer `L` (`powersgd_layer_seed` + `init_basis`):

```
seed_L = (base_seed · 1_000_003 + layer_idx · 7919) & 0x7FFFFFFF
Q_L    = orthonormalize( randn(H, r) )      # drawn fp32 on CPU with a seeded Generator
```

Drawing on **CPU in fp32** with a seeded `torch.Generator` makes `Q_L` **bit-identical on every rank/device** with zero communication — both ends of a boundary independently reproduce the same starting codebook. The basis is bootstrapped lazily on the first hook fire (once `H` is known) and **persists across steps** (warm start) — it is *not* cleared by `unregister()`.

`orthonormalize()` runs `torch.linalg.qr` in **fp32** (bf16-QR loses orthogonality), canonicalizes column signs (so QR is unique/reproducible), and repairs any rank-deficient/non-finite column from a deterministic random complement (so a degenerate sketch can never propagate a NaN basis).

---

## 4. How `Q` gets UPDATED — block power iteration (the heart)

`Q` is **never touched by gradient descent.** It is refreshed by one step of *block power iteration* on the activation Gram matrix, in two phases:

**(a) Sketch accumulation — off-graph, during the forward** (`_make_hook`, inside `torch.no_grad()`):

```python
V += Mᵀ (M Q) = (MᵀM) Q          # (H, r), accumulated in fp32
```

* Folded **only on the gradient-bearing actor-train forward** (`path_tag == train` and grad enabled), never on the old-logprob recompute (that runs under `no_grad`).
* **Deduped against gradient-checkpoint recompute**: a per-micro-batch "forward generation" counter (`_fwd_generation`, bumped once per micro-batch in `set_context`) ensures each layer folds into `V` at most once, even though grad-checkpointing re-fires the hook in the backward pass.
* Summing across micro-batches makes `V = (Σ_mb MᵀM) Q = C_step Q` (the step's pooled Gram applied to `Q`).

**(b) Orthonormalize — once per step, AFTER the backward** (`maybe_update_basis`, called from `update_actor`'s `finally:`):

```python
Q ← orthonormalize(V)            # fp32 QR ; then clear V
```

`V = (MᵀM) Q` then `Q ← orth(V)` is exactly `Q_{t+1} = orth(C Q_t)` — block power iteration on the SPD matrix `C = MᵀM`. Iterating drives `Q` toward the **top-`r` right-singular subspace** of the activations, i.e. the Eckart–Young-optimal rank-`r` reconstruction basis (minimizes `‖M − MQQᵀ‖_F`), at rate `(σ_{r+1}/σ_r)^{2t}`. *(Live evidence: aggregate reconstruction error fell 0.97 → ~0.17 over the first few warm-up steps.)*

---

## 5. Cross-DP consensus — one shared codebook (`sync_basis=true`)

Under data parallelism each of the 4 ranks sees a **different data shard**, so each rank's local sketch `V_i = C_i Q` differs. To keep `Q` a *single* codebook (differing only per boundary, never per rank), `maybe_update_basis` **all-reduces the raw sketch over the DP group before orth**:

```python
all_reduce(V, op=SUM, group=dp_group)        # Σ_i V_i = (Σ_i MᵢᵀMᵢ) Q = (M_globᵀ M_glob) Q
Q ← orthonormalize(V)                          # identical input on every rank ⇒ bit-identical Q
```

* Reduce the **raw `V`** (not per-rank `orth(V_i)` — averaging orthonormal frames is meaningless). `SUM` (not mean) is fine because `orth` is scale-invariant.
* The DP group is bound via `set_dp_group(get_data_parallel_group())` (== `WORLD` here, since SP=1 and TP is rollout-only).
* **Deadlock-safe:** every rank iterates the *fixed* `sorted(boundary_indices)` and zero-fills any locally-missing boundary, so all ranks issue the identical collective sequence in lockstep.
* **Verified on-box:** `verify_basis_agreement_across_ranks` all-gathers an fp64 per-boundary `Q` checksum and raises on `> 1e-6` divergence → measured `q_cross_rank_max_rel_dev = 0.0` on all 4 ranks (live).

> Cost note: this consensus all-reduce is an `H·r` DP-axis traffic per non-clean step. The headline `logical_pp_bytes_powersgd_y_only = r` counts only the forward payload; the basis-sync traffic is a separate (uncounted) cost — see the analyst footnote.

---

## 6. GRPO integration — frozen-Q across the paired forwards

Vanilla GRPO computes the importance ratio `ρ = exp(logπ_new − logπ_old)`, where `logπ_old` comes from an **old-policy recompute** (`compute_log_prob`) and `logπ_new` from the **actor-train forward** (`update_actor`). The single GRPO-specific rule:

* **`Q` is FROZEN for the entire global step.** Both the old-logprob recompute (when `compress_recompute=true`) and the actor-train forward read the *same* `Q_t`. The hook only ever *reads* `self._basis`; it never mutates it.
* **`Q` advances to `Q_{t+1}` only AFTER the gradient-bearing work** — `maybe_update_basis` runs in `update_actor`'s `finally:`, after all PPO micro-batch forwards/backwards.
* ⇒ at step 0 (no weight change) the two forwards apply the *identical* operator `Q_t Q_tᵀ`, so **`ρ ≈ 1`** — no spurious ratio from a drifting basis (a changing `Q` would otherwise make `ρ ≠ 1` and corrupt the GRPO objective).

Everything else is unchanged: advantages come from the **uncompressed vLLM rollouts**; the loss is plain GRPO (no-KL, no-entropy); compression only enters the *policy gradient* through `M̂`. With the codec disabled the path is byte-identical to dense GRPO.

---

## 7. The clean-cadence debiaser (every k steps, dense)

```
is_clean_step = clean_cadence > 0 AND global_step % clean_cadence == 0
```

On a clean step (`clean_cadence=5` ⇒ steps 5, 10, …, 50):
* `update_actor` sets `mask_active = not clean_step = False` ⇒ **no compression hooks register** ⇒ the actor-train forward is byte-identical dense.
* `compute_log_prob` suppresses its `compress_recompute` stamp ⇒ the old-logprob recompute is **also dense**.
* `maybe_update_basis(is_clean_step=True)` **skips** — `Q` is held, no `V` accumulated.
* ⇒ AdamW refreshes its moments on the **true dense gradient** every k steps, debiasing the drift from the low-rank projection. *(Signature: grad_norm drops to ~0.4 on clean steps, e.g. 6.9 → 0.37.)*

---

## 8. FSDP integration (how the hook fires correctly)

* **Boundary selection:** `find_decoder_layers` locates the decoder `ModuleList`; `decoder_boundary_indices(L, pp_size)` picks the boundaries — for Qwen2.5-1.5B (`L=28`, `pp_size=8`) that's 7 boundaries: layers `[3,7,11,15,18,21,24]`.
* **Post-hook on the FSDP unit:** `register_forward_hook` fires *after* each FSDP-wrapped block completes, so it receives the fully-materialized `output[0]` activation, downstream of all flat-param all-gather / reshard / `summon_full_params`. The hook reads no parameters or grads, so `use_orig_params` / `no_sync` / flat-param semantics are immaterial (unlike the anchor/spectral circuits).
* **FSDP shards parameters, not activations** (and SP=1, enforced at registration), so the `M` the hook sees is the full *local* activation for that rank's shard — exactly what the per-rank sketch needs.
* **Gradient checkpointing** (`use_reentrant=False`): the boundary forward is recomputed in the backward pass. Because `Q` is frozen until the post-backward update, the recomputed `M̂` is bit-identical, and the `_fwd_generation` dedup prevents the recompute from double-folding `V`.
* **Lifecycle** (`transformer_impl.py`): `forward_backward_batch` registers the hooks on entry to a compressed forward and removes them in `finally:`; `_comm_eff_powersgd_active` gates registration to the eligible path (train, + old-logprob when `compress_recompute`), gated by the shared `mask_active` flag (so a clean step / disabled run registers nothing and is byte-identical to dense).

---

## 9. Numerics

* **fp32 QR, activation-dtype projection** (INF-14): orthonormalize/store `Q` in fp32; project (`M@Q`, `Y@Qᵀ`) in the activation dtype (bf16). bf16-QR would lose orthogonality (`QᵀQ` drifts from `I`).
* **`q_cond ≈ 1`** confirms orthonormality; a non-finite `q_cond` is the basis-collapse falsifier. (Caveat: `q_cond` is measured on the orthonormal QR *output*, so it's ~1 by construction — it detects *collapse*, not a *poorly-fit* basis; `reconstruction_rel_error` is the real basis-health metric.)
* **Activation-scale shrink is benign:** `M̂ = MQQᵀ` is a projection, so `‖M̂‖ ≤ ‖M‖`. The block output feeds the next block's input **RMSNorm**, which renormalizes each token to unit-RMS·γ and thus **absorbs the shrink** — so PowerSGD needs *no* rescale knob (unlike the PRF mask, which zeros random dims and biased the post-norm distribution).

---

## 10. Config knobs (`actor_rollout_ref.actor.comm_eff.*`)

```
compression_type: powersgd          # {dense, prf_mask, powersgd}; selects exactly one codec
clean_cadence:    5                 # every-k-steps dense refresh (shared with the mask path)
powersgd:
  rank:               102           # r — kept-coordinates/token (the byte budget)
  update_cadence:     1             # run orth(V) every (non-clean) step
  warm_start:         true          # carry Q across steps (true power-iteration warm start)
  compress_recompute: true          # compress the old-logprob recompute too (frozen Q ⇒ ρ≈1)
  sync_basis:         true          # all-reduce V over DP ⇒ single shared consensus Q  (default True)
  qr_dtype:           fp32          # orth/QR precision (fp32 required; bf16 is diagnostic-only)
  reortho_eps:        1e-6          # degenerate-column threshold for the orth repair
  seed:               0             # base_seed for the per-layer bootstrap
  pp_size:            8             # boundary count via decoder_boundary_indices
```

---

## 11. Diagnostics / metrics (per step)

| Metric | Meaning |
|---|---|
| `comm_eff/powersgd_q_cond` | max/min singular value of `Q`; ≈1 = orthonormal, non-finite = collapse |
| `comm_eff/powersgd_reconstruction_rel_error` (+ per-layer) | `‖M − M̂‖/‖M‖`; the real basis-fit / spectral-gap health signal |
| `comm_eff/powersgd_q_cross_rank_max_rel_dev` | cross-DP `Q` agreement (≈0 ⇒ one shared codebook) |
| `comm_eff/logical_pp_bytes_powersgd_y_only` | `r` — the forward payload budget (vs the PRF mask's `(1−p)·H`) |
| `comm_eff/powersgd_basis_updates` / `clean_steps` | count of `orth(V)` updates / dense clean steps |

---

## 12. Lifecycle summary (one global step)

```
compute_log_prob (old-logprob):  stamp Q_t → [if compress_recompute] compressed forward (no_grad ⇒ no sketch)
update_actor (train):            stamp Q_t → compressed forward+backward (folds V += (MᵀM)Q off-graph)
                                            ↓ FSDP grad reduce
                                            ↓ optimizer.step()  (clean step ⇒ dense, no V, no update)
   finally:                      maybe_update_basis():  all-reduce V over DP → Q ← orth(V) → Q_{t+1}
                                 verify_basis_agreement_across_ranks()  (first update)
```

So within a step both GRPO forwards see `Q_t` (⇒ `ρ≈1`); the basis advances to `Q_{t+1}` only after the gradient is applied; and every k-th step is a dense debiasing refresh.
