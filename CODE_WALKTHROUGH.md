# Code Walkthrough — Communication-Efficient GRPO on verl

> **Scope.** This file documents the **implemented** state of the two-circuit
> compression method described in [`major-goal/implementation-logic.md`](major-goal/implementation-logic.md)
> as of the merge of `exp/12-anchor-detach` into `vast-ai-workload`. It is the
> canonical map for new contributors: what lives where, how the pieces connect,
> and — equally important — what the paper's logic calls for that we have
> **not yet implemented**.
>
> If you have not read it, start with [`major-goal/implementation-logic.md`](major-goal/implementation-logic.md)
> for the mathematics and paper logic. This file is the engineering view.

---

## 1. The two-circuit method in one paragraph

GRPO's actor-update path normally runs a single dense forward/backward over the
rollout-expanded batch, then `optimizer.step()`. Our compression method
**splits** that update into two coupled circuits running on the **same process,
same rollout batch, same optimizer**:

1. **Fast (masked) circuit** — every step does an in-graph PRF activation mask
   at pipeline-boundary decoder blocks (the `h_tilde = h * mask` of Algorithm A),
   producing a noisy `G_mask`.
2. **Anchor (unmasked) circuit** — every `cadence` steps, an *unmasked*
   GRPO-actor-loss forward/backward runs from a `delay_K`-stale weight snapshot
   on a **no-hook clone** of the module, producing a clean `G_anchor`.

A **spectral filter** maintains a running EMA `M_anchor` of the anchor
gradients, SVDs it (faithfully or low-rank), and uses the SVD basis to denoise
`G_mask` via the paper's two-sided Tikhonov projection before AdamW sees it.

Disabled is a strict no-op (`comm_eff.enabled=false`); dense GRPO runs
byte-identical to upstream verl.

---

## 2. File map (what to read, in order)

| Path | Role |
|---|---|
| **Config dataclasses** | |
| `verl/workers/config/comm_eff.py` | `CommEffConfig` + `Mask`/`Anchor`/`Spectral` sub-configs, all defaults DISABLED, validated in `__post_init__` |
| `verl/trainer/config/actor/dp_actor.yaml` (and `_megatron.yaml`) | YAML schema bound to the dataclasses above |
| `verl/trainer/config/ppo_trainer.yaml` | Top-level trainer config (unchanged from upstream apart from inheriting actor schema) |
| **Per-worker state** | |
| `verl/workers/comm_eff/state.py` | `CommEffState` + `maybe_build_comm_eff_state` factory + `PATH_TAGS` + `comm_eff_metrics`; the single object that owns counters, masker, spectral filter, anchor staleness queue |
| **First circuit — masking** | |
| `verl/workers/comm_eff/activation_mask.py` | `ActivationMasker`, PRF (splitmix64) `prf_mask`, `decoder_boundary_indices`, `find_decoder_layers`; train-only forward hooks |
| **Second circuit — anchor** | |
| `verl/workers/comm_eff/anchor.py` | `AnchorStalenessQueue`, `snapshot_named_params`, `extract_target_grads`, `feed_anchor_grads_into_ema`, `anchor_should_fire`, `build_anchor_module`, `assert_anchor_module_isolated`; the CPU-testable FSDP-agnostic pieces |
| **Third circuit — spectral correction** | |
| `verl/workers/comm_eff/spectral_filter.py` | `SpectralFilter`, `compute_basis` (full / lowrank SVD), `tikhonov_weights`, `two_sided_projection`, `spectral_correct`, `apply_spectral_correction_to_params`; the paper formula, FSDP-decoupled and CPU-unit-testable |
| **Engine wiring** | |
| `verl/workers/engine/base.py` | `BaseEngine.train_batch` calls `_maybe_comm_eff_anchor_refresh` (TOP) → `forward_backward_batch` → `_maybe_comm_eff_grad_correction` → `optimizer_step`; base no-op stubs |
| `verl/workers/engine/fsdp/transformer_impl.py` | `FSDPEngine._maybe_comm_eff_anchor_refresh` (clone-no-hook live anchor) and `FSDPEngine._maybe_comm_eff_grad_correction` (`summon_full_params` → per-target full-tensor extraction → spectral correct → write back); the **only** backend currently overriding these |
| **Worker integration** | |
| `verl/workers/engine_workers.py` | `update_actor` attaches state, stamps `mask_active=True` + `path_tag="train"` around `train_mini_batch`, surfaces counters into output metrics; the other entrypoints (`compute_log_prob`, `compute_ref_log_prob`, `infer_batch`, validation, checkpoint save/load) stamp a different `path_tag` so the mask hook's assertion guards confinement |
| `verl/workers/utils/losses.py` | Unmasked GRPO-actor-loss entry point reused by the anchor pass (same `ppo_loss` as the fast path) |
| **Trainer flow** | |
| `verl/trainer/ppo/ray_trainer.py` | `_update_actor` feeds the rollout-expanded GRPO batch into `update_actor`; comment block at line ~1308 documents the anchor data flow (no separate batch, no new rollouts) |
| `verl/trainer/ppo/core_algos.py` | GRPO loss assembly the anchor reuses |
| **Tests** | |
| `tests/workers/comm_eff/test_activation_mask.py` | PRF determinism, mask ratio, boundary indices, no-rescale form, train-only confinement |
| `tests/workers/comm_eff/test_spectral_filter.py` | `alpha=1` no-op, `alpha=0` pure projection, shape preservation, determinism, lowrank monotonicity, CPU EMA path, basis-cache path |
| `tests/workers/comm_eff/test_anchor_queue.py` | Staleness ring eviction, warmup fallback, `extract_target_grads` skip rules, `test_fsdp_anchor_backward_no_collision` (criterion 13 regression) |

56 CPU tests pass on `vast-ai-workload` after the EXP-12 merge.

---

## 3. End-to-end data flow (one trainer step at `comm_eff.enabled=true`)

```
RayPPOTrainer.fit() — per trainer step
└─ rollout via vLLM ─────────────────────────────────────────────── path_tag="rollout"
└─ compute_log_prob (old policy log-prob) ────────────────────────── path_tag="old_logprob"
└─ compute_ref_log_prob (reference policy) ───────────────────────── path_tag="ref_logprob"
└─ reward / advantage assembly (core_algos)
└─ _update_actor(batch)
   └─ update_actor (engine_workers.py)
      ├─ state.mask_active = True, state.set_path_tag("train")
      └─ actor.train_mini_batch(data)
         └─ engine.train_batch (engine/base.py)
            ├─ optimizer_zero_grad
            ├─ _maybe_comm_eff_anchor_refresh(data, loss_fn)  ──┐  [ANCHOR CIRCUIT]
            │     └─ FSDPEngine override (fsdp/transformer_impl.py)
            │        ├─ anchor_step += 1
            │        ├─ summon_full_params(self.module, with_grads=True)
            │        ├─ snapshot_named_params → queue.push(step)
            │        ├─ if anchor_should_fire(step, cadence):
            │        │    ├─ stale = queue.get_stale(step, delay_K)
            │        │    ├─ state.mask_active = False (GUARD 5)
            │        │    ├─ build_anchor_module(inner) — cached clone, no FSDP hooks
            │        │    ├─ load K-stale snapshot into clone (DTensor-safe)
            │        │    ├─ self.module ← clone (swap)
            │        │    ├─ _forward_backward_batch_inner(data, loss_fn) on clone
            │        │    ├─ extract_target_grads(clone.named_parameters)  — RAW (GUARD 6)
            │        │    ├─ feed_anchor_grads_into_ema(grads, spectral) — M_anchor update
            │        │    ├─ self.module ← live (restore)
            │        │    ├─ zero clone.grads, torch.cuda.empty_cache()  — vLLM hygiene
            │        │    └─ anchor_backwards += 1
            ├─ forward_backward_batch(data, loss_fn, forward_only=False)  ─┐  [FAST/MASKED CIRCUIT]
            │     └─ ActivationMasker forward hooks fire at boundary layers
            │        when state.mask_active && state.path_tag == "train"
            │        (PRF-keyed masks, no 1/(1-p) rescale)
            ├─ _maybe_comm_eff_grad_correction()                           ─┐  [SPECTRAL CORRECTION]
            │     └─ FSDPEngine override
            │        ├─ summon_full_params(self.module)  — FSDP grads reduced
            │        ├─ for each target 2D matrix p.grad:
            │        │    ├─ compute_basis(M_anchor, svd_mode, rank)  — cached per refresh
            │        │    ├─ G_filt = U diag(d) (U^T G_mask V) diag(d) V^T
            │        │    ├─ G_proj = alpha · G_mask + (1-alpha) · G_filt
            │        │    └─ p.grad ← G_proj (in place)
            │        └─ spectral_corrections += len(targets)
            ├─ optimizer_step (grad clip + AdamW)
            └─ return outputs (metrics annotated)
      ├─ state.mask_active = False, state.set_path_tag(None)
      └─ surface comm_eff/* counters into output.meta_info["metrics"]
└─ checkpoint-engine weight sync (no mask, no anchor — path_tag stays None / "ckpt")
```

The **invariant ordering** matches `implementation-logic.md`:
*masked fwd/bwd → DP reduction (FSDP all-reduce) → spectral correction → AdamW*.
The anchor block runs **before** the masked fwd/bwd so its `G_anchor` is fed
into the EMA *before* any correction touches the masked grads (the GUARD-6
read-raw contract).

---

## 4. Per-component walkthrough

### 4.1 Config (`verl/workers/config/comm_eff.py`)

* `CommEffConfig.enabled = False` is the master switch. **No** sub-config is
  honored when this is false — `maybe_build_comm_eff_state` short-circuits to
  `None`, no RNG drawn, no buffer allocated, no hook registered.
* `Mask.{p, seed, pp_size}`: `p` is the **masked fraction** (probability of
  zero); the measured `comm_eff/mask_ratio` tracks `p ± 0.02`. `pp_size` is a
  *logical* knob (not a real PP split) used to derive boundary block indices.
* `Anchor.{cadence, delay_K}`: paper defaults `cadence=20, delay_K=20`; smoke
  defaults `cadence=1, delay_K=1` so an anchor provably fires in a short test.
  The EXP-4 scaffolding had unused fields (`every_n_steps`, `ema_decay`);
  EXP-8 amended the schema and dropped them.
* `Spectral.{alpha, tau, beta_anc, seed_anchor_cache, target_substr,
  max_targets, rank, ema_device, svd_mode, basis_cache}`: the paper knobs +
  storage layer. `alpha=1` is the no-op blend; `alpha=0` is pure projection.
  `seed_anchor_cache=true` populates `M_anchor` deterministically so the
  spectral path can be tested without the live anchor (EXP-7); `false` defers
  population to the live anchor (EXP-8/12).
* `__post_init__` validates bounds: `p ∈ [0,1]`, `rank ≥ 1`, `alpha ∈ [0,1]`,
  `tau > 0`, `beta_anc ∈ [0,1]`, `cadence ≥ 1`, `delay_K ≥ 0`, plus enum
  membership for `ema_device ∈ {gpu, cpu}`, `svd_mode ∈ {full, lowrank}`,
  `basis_cache ∈ {cache, recompute}`. **Validation only** — no allocation.

### 4.2 State (`verl/workers/comm_eff/state.py`)

* `maybe_build_comm_eff_state(config)` is **the single gate**. Returns
  `None` when disabled — caller stores `self._comm_eff_state = None`, every
  hook guards on `state is None or not state.enabled`.
* `CommEffState.build(module)` is **idempotent**, called from `update_actor`
  on the first train_batch. Lazily constructs the `ActivationMasker` (only if
  `mask.enabled` and `p > 0`) and the `SpectralFilter` (only if
  `spectral.enabled`). The anchor queue is constructed lazily inside the
  engine's `_maybe_comm_eff_anchor_refresh` on first fire.
* **Path-tag confinement (EXP-6).** `PATH_TAGS = (train, rollout, old_logprob,
  ref_logprob, val, infer, ckpt)`. Every entrypoint stamps `state.path_tag`
  before its forward. The mask hook asserts `path_tag == "train"` before
  firing — a leak onto any other path raises rather than silently corrupting
  RL measurement.
* **Counters** (numeric — `reduce_metrics` does `np.mean` on every value, a
  string crashes it): `mask_applications`, `anchor_backwards`,
  `spectral_corrections`, plus the anchor-semantics falsifiers
  `anchor_mask_applications`, `anchor_grad_corrected`,
  `anchor_rollouts_generated`, `anchor_rewards_recomputed`,
  `anchor_optimizer_steps`, `anchor_batch_fraction`.
* **Per-path mask counters** `mask_applications_by_path` are surfaced as
  `comm_eff/mask_applications/<tag>` — the analyst asserts the only nonzero
  key is `.../train`.

### 4.3 Activation masking (`verl/workers/comm_eff/activation_mask.py`)

* **PRF.** `prf_mask(shape, key, p, device, dtype)` mixes a splitmix64 hash of
  the key tuple `(boundary id, global optimizer step, optimizer substep /
  microbatch identity, sequence-shard identity, hidden size, base run seed)`
  into a per-element Bernoulli draw. The key is value-agnostic: two different
  hidden tensors with the same key/shape get the same mask. Reproducible
  across ranks and re-runs.
* **Form.** Returns `mask ∈ {0, 1}` (no `1/(1-p)` rescale — the paper writes
  the direct product; rescaling at `p=0.95` destabilises bf16). The masker
  applies it in-graph as `h_tilde = h * mask`.
* **Boundary selection.** `decoder_boundary_indices(L, pp_size)` partitions
  `L` blocks into `pp_size` contiguous shards, returns the *last* index of
  every shard *except the final shard*. `L=16, pp_size=8` → `[1,3,5,7,9,11,13]`.
  `L` and hidden_size come from `model.config`, never hardcoded.
* **Hook lifecycle.** Hooks are registered on entry to the train forward and
  removed on exit (`mask_active=False` gates the work even if a hook somehow
  survives). The hook asserts `state.path_tag == "train"` before firing —
  contamination is a loud failure.
* **Top-k is forbidden.** Only random PRF masking per the paper.

### 4.4 Spectral filter (`verl/workers/comm_eff/spectral_filter.py`)

Pure FSDP-agnostic logic on logical 2D matrices. `correct_matrix(G_mask, name)`
asserts its input is 2D so a bad unshard upstream fails here instead of
silently mangling a gradient. The math is:

```
M_anchor = beta_anc · M_anchor + (1 - beta_anc) · G_anchor    # EMA (update_anchor)
M_anchor = U S V^T                                            # compute_basis
d_i      = s_i / (s_i + tau)                                  # tikhonov_weights
X        = U^T G_mask V
G_filt   = U diag(d) X diag(d) V^T                            # two_sided_projection
G_proj   = alpha · G_mask + (1 - alpha) · G_filt              # spectral_correct
```

Storage layer (EXP-8):
* `ema_device="gpu"` keeps `M_anchor` in HBM; `="cpu"` offloads to pinned CPU
  and pulls it to GPU only inside the refresh/correct window. `M_anchor` is
  touched only on refresh, so cpu offload costs one H2D/D2H per refresh, not
  per mini-batch.
* `svd_mode="full"` is `torch.linalg.svd(full_matrices=False)`; `"lowrank"` is
  `torch.svd_lowrank(M_anchor, q=rank, niter=4)`, with an automatic fallback
  to the full SVD when `q ≥ k = min(m,n)` (the randomized algorithm adds noise
  at full rank).
* `basis_cache="cache"` computes `U/S/V` once per refresh and caches them on
  GPU for reuse across the fast PPO mini-batches; `"recompute"` recomputes per
  `correct_matrix` call (pre-EXP-8 behavior).

`apply_spectral_correction_to_params(state, target_names, grads_iter)` is the
hot-loop helper the engine calls.

### 4.5 Anchor circuit (`verl/workers/comm_eff/anchor.py` + engine override)

CPU-testable pieces (in `anchor.py`):

* **`anchor_should_fire(step, cadence, enabled)`** — pure predicate
  `enabled && (step % cadence) == 0`. Step is 1-based (engine advances before
  the check).
* **`AnchorStalenessQueue(delay_K)`** — bounded ring of `delay_K + 1`
  snapshots keyed by step. `push(step, snapshot)`; `get_stale(step, delay_K)`
  returns the `step - delay_K` snapshot or falls back to the oldest available
  during warmup.
* **`snapshot_named_params(named_params, *, target_substrs, device, detach)`** —
  detached clones of each named parameter, optionally filtered by substring,
  optionally moved off-GPU. Explicitly DECOUPLED from optimizer state.
* **`extract_target_grads(named_params, target_substrs, max_targets, full_grad_of)`** —
  iterates parameters by name, applies the same target-selection rule the
  spectral hook uses, returns `{name: full_2d_grad}` with NO correction
  applied (GUARD 6 raw read).
* **`feed_anchor_grads_into_ema(grads, spectral, state)`** — wires each raw
  grad into `SpectralFilter.update_anchor(name, G_anchor)` and reports
  `||ΔM_anchor||` per target so the engine can log EMA evolution (criterion 3).
* **`build_anchor_module(inner_module)`** — the **clone-no-hook** factory.
  Try `copy.deepcopy(inner_module)`; on `TypeError: cannot pickle 'module'
  object` (verl/HF monkey-patches install function attributes referencing
  Python modules, which are unpicklable), fall back to a **config-rebuild**:
  `type(inner)(inner.config)` then a manual per-parameter copy with
  `.full_tensor()` / `.to_local()` materialization for DTensor sources
  (FSDP1+`use_orig_params` surfaces DTensors in `state_dict()` even inside
  `summon_full_params`). Returns a plain `nn.Module` with **no FSDP `_handles`,
  no `_post_backward_hooks`, no `FlatParameter`** — the autograd-hook chain
  collision the live FSDP module suffers (EXP-8) is broken by construction.
* **`assert_anchor_module_isolated(clone, *, optimizer, fsdp_module)`** —
  runtime guard: the clone's params share **no `id()`** with the live
  optimizer's `param_groups` or the live FSDP module's `_handles`. Cheap; runs
  every refresh; protects criterion 7 (`anchor_optimizer_steps == 0`) and
  criterion 13 against future drift.

Engine-side fwd/bwd (in `verl/workers/engine/fsdp/transformer_impl.py`,
`FSDPEngine._maybe_comm_eff_anchor_refresh`):

1. Read `state`, `anchor_cfg`, `spectral` — early-return if any is None or
   disabled.
2. `state.anchor_step += 1`; lazily build the staleness queue on first call.
3. Inside `FSDP.summon_full_params(self.module, with_grads=True, writeback=True)`,
   snapshot the live module's full params into the queue.
4. `if anchor_should_fire(step, cadence)`:
   * Fetch `stale = queue.get_stale(step, delay_K)`.
   * Save `mask_active`, set `mask_active=False`, clear `path_tag` (GUARD 5).
   * Build / fetch the cached clone via `build_anchor_module(inner)` (cached
     on `self._anchor_module_cache` — built once, reused).
   * `assert_anchor_module_isolated(clone, optimizer, inner)`.
   * Move clone to the live device + dtype.
   * Manually copy `stale` weights into the clone's params (DTensor-safe).
   * Swap `self.module = clone` for the duration of the inner fwd/bwd.
   * `self._forward_backward_batch_inner(data, loss_fn, forward_only=False)` —
     the **same `ppo_loss` as the fast path**, on the clone, unmasked.
   * `extract_target_grads(clone.named_parameters(), …)` — raw 2D grads.
   * In `finally`: restore `self.module` to the live FSDP wrapper, zero the
     clone's grads, restore prior `mask_active`/`path_tag`, **assert**
     `anchor_mask_applications == 0` and `anchor_optimizer_steps ==
     opt_steps_before` (GUARDS 5 + 7).
   * `feed_anchor_grads_into_ema(grads, spectral, state)` → EMA update +
     `||ΔM_anchor||` per target.
   * `state.anchor_backwards += 1`, `state.anchor_batch_fraction = 1.0`.
   * Emit `[comm_eff][EXP-12] anchor refresh step=N fired backward
     (cadence=… delay_K=…) targets=… ||dM_anchor||_mean=… anchor_backwards=…
     anchor_backward_isolation_mode=clone` to stdout (NUMERIC values only in
     `metrics` — strings stay on stdout per the EXP-7 lesson).
   * `torch.cuda.empty_cache()` so vLLM's `sleep_replicas` memory check (which
     reads device-global free memory) sees the anchor's transient allocations
     released before the next rollout cycle.

### 4.6 Spectral correction hook (`FSDPEngine._maybe_comm_eff_grad_correction`)

* Runs AFTER `forward_backward_batch` (so grads are FSDP-reduced) and BEFORE
  `optimizer_step` (so grad clipping sees the corrected grads).
* Inside `summon_full_params(self.module, with_grads=True)`:
  * Iterate `self.module.named_parameters()`.
  * Skip params not matching `target_substr` or not 2D.
  * Cap by `max_targets`.
  * Materialize `p.grad` to a full logical 2D `Tensor` (the EXP-7 discovery:
    FSDP1 + `use_orig_params=True` → `p.grad` is already a full Tensor;
    DTensor under FSDP2 needs `.full_tensor()`).
  * `G_proj = spectral.correct_matrix(G_mask, name)`.
  * Write `G_proj` back into `p.grad` (in place).
* Logs the FSDP-grad-repr discovery once on first correction (string-valued,
  stdout only — `[comm_eff][EXP-7][FSDP-DISCOVERY]`).

### 4.7 Worker integration (`verl/workers/engine_workers.py`)

* `_maybe_comm_eff_state()` is the lazy attach (single allocation per actor
  worker).
* `update_actor` (line ~712) wraps `actor.train_mini_batch(data)` in a
  `try/finally` that stamps `mask_active=True` + `path_tag="train"` on entry
  and clears them on exit. Counters are surfaced into `output.meta_info["metrics"]`
  — the disabled path emits explicit zeros so the analyst can grep by name.
* `compute_log_prob`, `compute_ref_log_prob`, `infer_batch`, validation,
  `load_checkpoint`, `save_checkpoint` all use the `_comm_eff_path(tag)`
  context manager to stamp a non-`train` tag so a mask leak onto those paths
  raises in the hook.

### 4.8 Launcher contract (`examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`)

Unchanged from upstream's baseline shape; comm_eff is opt-in via overrides:

```
actor_rollout_ref.actor.comm_eff.enabled=true
actor_rollout_ref.actor.comm_eff.mask.enabled=true   # mask.p=0.95 etc.
actor_rollout_ref.actor.comm_eff.spectral.enabled=true
actor_rollout_ref.actor.comm_eff.anchor.enabled=true # cadence, delay_K
```

Without these overrides the launcher trains dense GRPO exactly as upstream.
The baseline acceptance test (`comm_eff.enabled=false` ⇒ byte-for-byte parity
with dense) is part of EXP-4/5's success criteria.

---

## 5. What is NOT yet implemented (compared to `implementation-logic.md`)

This is the honest gap list. Each item is either deferred (planned in a later
issue family) or out of scope (explicitly excluded by the implementation-logic
spec).

### 5.1 Deferred — on the M3 roadmap

| Capability | Where it would live | Status |
|---|---|---|
| **DP compression** (PowerSGD-64 + Streaming DiLoCo) | `verl/workers/comm_eff/dp_compress.py` (skeleton path) | Not started. Issue family 7 in `implementation-logic.md`. Explicitly OUT of scope until M2 actor-mask/anchor correctness passes. |
| **Paper-cadence K=20 long run** | New issue (#11 already planned) | #11 `M3 100-step M95+AP vs dense` exists at `status:planned`. EXP-12 was a 10-step smoke; paper cadence requires K=20 ≥ 20 steps. |
| **100-step compressed comparison vs EXP-3 dense baseline** | New issue (#9 or #11) | #9 `M2 full M95+AP two-step smoke` and #11 `M3 100-step M95+AP vs dense` are planned but not approved. |
| **Per-mini-batch anchor gradients** (the "heavier" path) | Refactor of `_maybe_comm_eff_anchor_refresh` to fire per PPO mini-batch | Not started. `implementation-logic.md` §"Anchor design note" calls this out as a later/heavier extension that multiplies anchor compute by `train_batch_size / ppo_mini_batch_size · ppo_epochs`. The current implementation is the "same-loop periodic refresh" first variant. |
| **Megatron engine integration** | `verl/workers/engine/megatron/transformer_impl.py` to override `_maybe_comm_eff_anchor_refresh` + `_maybe_comm_eff_grad_correction` | Not started. The base no-op stubs in `engine/base.py` mean Megatron + comm_eff currently runs as if comm_eff is disabled. |
| **Automodel engine integration** | `verl/workers/engine/automodel/transformer_impl.py` | Not started (same reason). |
| **DP-compression metrics surface** | `state.py` would add `comm_eff/dp_*` counters | Deferred until DP work begins. |
| **`comm_eff/anchor_batch_fraction < 1.0` OOM-fallback path** | `_maybe_comm_eff_anchor_refresh` would split anchor pass into microbatches | Plumbed (counter exists, `1.0` recorded by default) but the microbatch-accumulation path itself is not coded. Plan §"Notes for runner" §OOM fallback. |

### 5.2 Out of scope — explicitly excluded by the paper / implementation-logic

| Capability | Why excluded |
|---|---|
| **Top-k activation masking** | Paper PDF §method: random PRF masking is required; top-k introduces structured bias the spectral filter cannot remove. |
| **Forward `1/(1-p)` rescale** | Paper writes the direct product. Rescaling at `p=0.95` destabilises bf16 (simulator finding). |
| **Random projection / codebook / signed-EMA / signSGD / rank-1 projection / quantization** | `implementation-logic.md` §"Ignore for the first GRPO issue set". |
| **Separate anchor GPU / Ray rank / resource pool** | `implementation-logic.md` §"Anchor design note" mandates same-process / same-worker anchor emulation; verl already colocates actor/ref/rollout/checkpoint under Ray/FSDP. |
| **SmolLM3 / Llama-3.2-1B port of the paper SFT suite** | Paper uses these in main text. `implementation-logic.md` §"Paper Experiment Facts" says do not migrate; the GRPO target is Qwen2.5-1.5B + GSM8K. |
| **Compressed rollout / log-prob / ref / validation / checkpoint paths** | Plan §"GRPO Integration Boundaries" requires masking to be actor-train-only for the first compression tests. Confinement is enforced via the `path_tag` contamination guard. |
| **Forking GRPO into a separate algorithm** | Plan §"GRPO Integration Boundaries": the compressed path must wrap or augment actor training while preserving the same rollout/log-prob/reward/advantage/actor-update/optimizer/weight-sync sequence as normal GRPO. |

### 5.3 Known caveats (working but with caveats logged on disk)

* **Memory cost of the cached clone.** EXP-12 iter04 caches one Qwen2-sized
  clone (~3 GB params for 1.5B in bf16). For a 7B model this is ~14 GB; for a
  paper-scale `delay_K=20` queue it's `(delay_K + 1) × ~3 GB ≈ 63 GB` of
  snapshots — fine on H100/H200 (80–140 GB HBM) but tight on 24 GB consumer
  cards. The lean mode (`ema_device=cpu, svd_mode=lowrank, basis_cache=cache`)
  partially mitigates by CPU-offloading the EMA, but the staleness queue
  itself is not yet CPU-offloaded.
* **FSDP1 mandate.** All anchor + spectral hooks assume FSDP1 +
  `actor_rollout_ref.actor.fsdp_config.use_orig_params=true`. FSDP2
  (`fully_shard`) is not exercised; the `summon_full_params` codepath would
  need an FSDP2-equivalent (`fully_shard.unshard` / DTensor `.full_tensor()`
  per param).
* **`anchor.cadence=1` is smoke-only.** Paper cadence is 20. At `cadence=1` we
  fire the anchor every step (every PPO sub-batch through the trainer loop),
  which is ~10× the paper's anchor compute fraction.
* **Single FSDP1 toy-module regression test.** Criterion 13's
  `test_fsdp_anchor_backward_no_collision` covers the EXP-8 `_saved_grad_shard`
  failure mode but does not exercise the full Qwen2-scale config-rebuild
  fallback path. That is currently only covered by the EXP-12 on-box smoke.

---

## 6. Where to look when something breaks

| Symptom | Likely culprit | Where |
|---|---|---|
| `AttributeError: 'NoneType' object has no attribute 'shape'` on `flat_param._saved_grad_shard` | Anchor backward ran on a hook-registered FSDP module | `verl/workers/engine/fsdp/transformer_impl.py` — clone path not entered; check `_anchor_module_cache` + `assert_anchor_module_isolated` |
| `TypeError: cannot pickle 'module' object` in `copy.deepcopy` | HF/verl monkey-patch on the model class | `verl/workers/comm_eff/anchor.py::build_anchor_module` — the config-rebuild fallback should catch this |
| `aten.copy_.default got mixed torch.Tensor and DTensor` in clone's state_dict load | FSDP1+use_orig_params surfaces DTensor entries | `anchor.py::build_anchor_module` — the per-param `.full_tensor()` materialization path |
| `Memory usage increased after sleeping` from vLLM `sleep_replicas` | Anchor clone allocation not released | `_maybe_comm_eff_anchor_refresh` finally block — verify `torch.cuda.empty_cache()` ran |
| Mask hook fires on a non-train path | Path-tag stamp missing on that entrypoint | `engine_workers.py::_comm_eff_path` usage; check the hook's assert in `activation_mask.py` |
| `np.mean` crash on metric reduction at end of step | A string ended up in `meta_info["metrics"]` | All comm_eff metric values must be numeric. String discovery fields (`ema_device`, `svd_mode`, `anchor_backward_isolation_mode`) go to stdout only — see `state.py::spectral_metrics` docstring |
| Anchor counter stays at 0 with `enabled=true` | Call site to `_maybe_comm_eff_anchor_refresh` missing | `verl/workers/engine/base.py::train_batch` line ~166 — the EXP-12 iter01 fix |

---

## 7. Verification commands

```bash
# CPU unit tests (~30s on a laptop, 56 tests)
cd /Users/shamane/Documents/verl
pytest tests/workers/comm_eff/ tests/workers/config/ -v

# Two-step dense parity smoke (comm_eff.enabled=false; should match dense)
# Run via the Vast launcher; not local.

# Full M95+AP smoke shape (the EXP-12 10-step run)
# See `research/runs/EXP-12/launch_iter2.sh` for the exact command,
# inherited from `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
# with comm_eff.* overrides.

# Greppable proof the anchor fired in a smoke log:
grep -E 'anchor refresh step=' <log> | head
grep -oE 'actor/comm_eff/anchor_backwards:[0-9.]+' <log>
grep -oE 'actor/comm_eff/spectral_corrections:[0-9.]+' <log>
grep -oE 'training/global_step:[0-9]+' <log> | sort -u
```

---

## 8. Provenance — the EXP-12 four-iteration debug cycle

`exp/12-anchor-detach` carries six commits beyond `vast-ai-workload` at merge.
The first two are the planned scaffold; the last four are the on-box hot-fixes
applied during the EXP-12 smoke. They are useful as case studies of the FSDP1
× HF-monkey-patch × verl-vLLM-sleep interaction surface:

| Commit | What it fixed |
|---|---|
| `1708b3e0` | EXP-12 scaffold: clone-no-hook `build_anchor_module` + `assert_anchor_module_isolated` + criterion-13 regression test |
| `1de1d2c4` | YAML schema alignment (`ema_device` / `svd_mode` / `basis_cache` / `cadence` / `delay_K`) so OmegaConf merge accepts the overrides |
| `8a9c5ab0` (iter01) | `BaseEngine.train_batch` calls `_maybe_comm_eff_anchor_refresh` at the TOP — the call site was missing entirely; the EXP-12 cell 1 reached step 5 cleanly with `anchor_backwards=0` because the function was never invoked |
| `52937759` (iter02) | `build_anchor_module` deepcopy → config-rebuild fallback when `copy.deepcopy` hits `cannot pickle 'module' object` (verl monkey-patch on `_flash_attention_forward` installs unpicklable function attributes) |
| `f0d79ae1` (iter03) | DTensor materialization via `.full_tensor()` / `.to_local()` before the per-param `copy_` (FSDP1+`use_orig_params` surfaces DTensor entries in `state_dict()` even inside `summon_full_params`) |
| `afd43319` (iter04) | Cache the anchor clone on `self._anchor_module_cache` + `torch.cuda.empty_cache()` after refresh — was failing vLLM v1 `sleep_replicas` assertion `freed_bytes >= 0` because per-step clone allocation grew device memory |

Each iteration's diff is preserved at `research/runs/EXP-12/iterations/{01..04}.patch`.

After iter04 both anchor-enabled cells reached `training/global_step:10` with
the anchor firing every step (`anchor_backwards=20`, 2/step from the 2 PPO
sub-batches × 10 trainer steps), `||dM_anchor||_mean` evolving non-trivially,
all six anchor-semantics guards held, finite `actor/grad_norm` throughout.
EXP-12 verdict file: `research/runs/EXP-12/verdict.md` (PASS).

---

## 9. Read this before starting M3

* The orchestrator playbook (`research/.claude/playbooks/orchestrator.md`) is
  the autonomous loop; the entry pattern is `/goal Read .../orchestrator.md
  and execute it for issue #<N>`.
* The locked Vast.ai template is `verl-research-vllm020` hash
  `6485b9625ddd6d25a5f2f09b9f7fde17` — do not name a template hash in any
  launch.
* The fork's branch policy is in [`CLAUDE.md`](CLAUDE.md): `main` is read-only
  (tracks upstream), `vast-ai-workload` is the primary working branch, PRs
  from `exp/<N>-<slug>` branches target `vast-ai-workload`.

— end of walkthrough —
