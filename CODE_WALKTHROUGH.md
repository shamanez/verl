# Code Walkthrough — Communication-Efficient GRPO on verl

This fork is **not** vanilla verl. It bolts a **communication-efficient
training method** onto verl's existing GRPO recipe (Qwen2.5-1.5B-Instruct on
GSM8K). When the method is disabled (`comm_eff.enabled=false`), training is a
byte-for-byte no-op against upstream verl; everything below only activates
when it's on.

This file is the engineering map: what lives where, how a step flows, and
what's deliberately not built yet. Agents work from the code + the issue
queue; the project's north-star is
[`research/.claude/GOAL.md`](research/.claude/GOAL.md).

---

## 1. The method in one paragraph

GRPO's actor update normally runs one dense forward/backward over the
rollout-expanded batch, then `optimizer.step()`. The method splits that update
into two coupled circuits on the **same process, same batch, same optimizer**:

1. **Fast (compressed) circuit** — every step applies the configured **codec** at the
   pipeline-boundary decoder blocks (the one variable axis). Two codecs exist: the
   **PRF mask** (`prf_mask`) masks per-(token, dim) `h_tilde = h * mask`, keyed on each
   token's stable `(sample_id, position_id)` so it is packing-invariant across the
   old-logprob and train forwards (`mask.rescale` inverted-dropout `1/(1-p)` is **ON**
   to unbias `E[h̃]=h`; without it grad_norm explodes ~2700 vs ~0.4 dense); **PowerSGD**
   (`powersgd`, the current default) projects each boundary activation onto a shared
   low-rank basis, `h_hat = (h Q) Qᵀ`, sending only `Y = h Q`. Either produces the noisy
   gradient the rest of the step corrects.
2. **Anchor (unmasked) circuit** — every `cadence` steps, an *unmasked*
   GRPO-actor-loss forward/backward runs from a `delay_K`-stale weight
   snapshot on a **no-hook clone** of the module, producing a clean
   `G_anchor`.

A **spectral filter** keeps a running EMA `M_anchor` of the anchor gradients,
SVDs it, and uses that basis to denoise `G_mask` via a two-sided Tikhonov
projection before AdamW sees it:

```
M_anchor = β·M_anchor + (1-β)·G_anchor      # EMA
M_anchor = U S Vᵀ                           # SVD
d_i      = s_i / (s_i + τ)                  # Tikhonov weights
G_filt   = U diag(d) (Uᵀ G_mask V) diag(d) Vᵀ
G_proj   = α·G_mask + (1-α)·G_filt          # blend  (α=1 ⇒ no-op)
```

Ordering invariant: **masked fwd/bwd → FSDP all-reduce → spectral correction
→ AdamW**. The anchor block runs *before* the masked fwd/bwd so its raw
gradient feeds the EMA before any correction touches the masked grads.

**Status.** The **codec is the one variable axis** (`dense | prf_mask | powersgd`; see
`research/runs/FIXED_CONTROL_SURFACE.md`). The chosen codec is **PowerSGD-style activation
compression**: a shared low-rank orthonormal basis `Q` projects each boundary activation,
`M_hat = (M Q) Qᵀ`, so only `Y = M Q` (rank-`r` coords/token) crosses the boundary; `Q` is
updated by block power iteration on the activation Gram matrix (`Q ← orth(V_global)`,
DP-synced), frozen within a step.

The **current frontier — make this realistic via the anchor circuit (issue #25, prerequisite
for #24).** The anchor computes a clean full gradient from **stale (delayed) weights** every
few steps and folds it into the fast compressed gradient, replacing the impractical periodic
dense step. Two defects in the existing anchor path must be fixed first: (a) `M_anchor` is
EMA-updated for only a 4-of-~196-matrix slice (`extract_target_grads` breaks at `max_targets`,
`anchor.py:341-342`), not the full-network gradient; (b) the anchor backward runs on a plain
per-rank clone with **no DP all-reduce** of `G_anchor`, so `M_anchor` is a per-rank local-shard
gradient, not the global one. Then `Q` moves to the anchor and a sign-based merger
(`α·G + (1−α)·|G|·sign(M_anchor)`) combines the two circuits. The existing `inject`/`blend`/
`reweight` spectral combiners are inert here — the stale anchor gradient is ~orthogonal to the
compressed gradient — so the merger is new work.

Settled-base config: `mask.p=0.9`, `mask.rescale=true`, `mask_recompute=true`,
`clean_cadence` set (e.g. 20), `anchor.enabled=false`, `spectral.enabled=false`; no
KL, no entropy; `fsdp_config.use_orig_params=true`. Authoritative defaults:
`verl/workers/config/comm_eff.py` + the launcher
`examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`.

---

## 2. Where it lives

| Path | Role |
|---|---|
| `verl/workers/config/comm_eff.py` | `CommEffConfig` + `Mask`/`Anchor`/`Spectral` sub-configs; all defaults DISABLED; bounds validated in `__post_init__` (no allocation) |
| `verl/workers/comm_eff/state.py` | `CommEffState` + `maybe_build_comm_eff_state` factory + path-tag set + numeric counters; the single object owning masker, spectral filter, anchor queue |
| `verl/workers/comm_eff/activation_mask.py` | `ActivationMasker`, counter-based splitmix64 `prf_token_mask` (per-(token, dim), keyed on stable `(sample_id, position_id)`), `decoder_boundary_indices` selection; train-only forward hooks |
| `verl/workers/comm_eff/powersgd_activation.py` | `PowerSGDActivationCompressor`: per-boundary low-rank basis `Q` (deterministic seed, fp32 QR), `Y=MQ`/`M_hat=YQᵀ` projection hooks, block-power-iteration `Q←orth(V)` update with cross-DP sketch all-reduce (`sync_basis`) + a cross-rank agreement guard; the `powersgd` codec |
| `verl/workers/comm_eff/anchor.py` | staleness queue, snapshot/extract/feed helpers, `anchor_should_fire`, `build_anchor_module` (clone-no-hook), `assert_anchor_module_isolated` — the FSDP-agnostic, CPU-testable pieces |
| `verl/workers/comm_eff/spectral_filter.py` | `SpectralFilter`: EMA, full/lowrank SVD, Tikhonov, two-sided projection, α-blend; pure 2D-matrix logic, CPU-unit-testable |
| `verl/workers/engine/base.py` | `train_batch`: anchor refresh → fwd/bwd → grad correction → optimizer step; base no-op stubs |
| `verl/workers/engine/fsdp/transformer_impl.py` | the **only** backend overriding the two comm-eff hooks (clone-no-hook anchor refresh; `summon_full_params` → per-target full-tensor spectral correction → write-back) |
| `verl/workers/engine_workers.py` | `update_actor` stamps `mask_active=True` + `path_tag="train"` and a stable per-row `comm_eff_sample_id` on the batch (also in `compute_log_prob`) so the mask keys on each token's `(sample_id, position_id)`; the other entrypoints stamp a non-train tag so the mask hook's guard confines masking to actor-train |
| `tests/workers/comm_eff/` | CPU unit tests: PRF determinism / mask ratio / train-only confinement; spectral α-blend / projection / determinism; anchor staleness / isolation regression |

---

## 3. One trainer step (`comm_eff.enabled=true`)

```
RayPPOTrainer.fit() — per step
├─ rollout (vLLM) ............................... path_tag="rollout"
├─ compute_log_prob (old policy) ............... path_tag="old_logprob"   [masked too iff mask_recompute]
├─ compute_ref_log_prob ........................ path_tag="ref_logprob"
├─ reward / advantage assembly
└─ update_actor                                  state.mask_active=True, path_tag="train"
   └─ engine.train_batch
      ├─ optimizer_zero_grad
      ├─ [ANCHOR]  _maybe_comm_eff_anchor_refresh(data, loss_fn)
      │     summon_full_params → snapshot into staleness queue
      │     if anchor_should_fire(step, cadence):
      │       load K-stale snapshot into a cached clone-no-hook module,
      │       run unmasked fwd/bwd on the clone (mask_active=False),
      │       extract RAW target grads → feed EMA (M_anchor), restore live module,
      │       empty_cache() for vLLM sleep hygiene
      ├─ [FAST]    forward_backward_batch         ActivationMasker hooks fire at
      │                                           boundary layers iff path_tag=="train";
      │                                           per micro-batch, prepare_model_inputs
      │                                           sets the token-aligned (sample_id,
      │                                           position_id) PRF context
      ├─ [SPECTRAL] _maybe_comm_eff_grad_correction
      │     summon_full_params (grads FSDP-reduced) → per 2D target:
      │       G_proj = α·G_mask + (1-α)·G_filt → write back into p.grad
      └─ optimizer_step (grad clip + AdamW)
└─ checkpoint-engine weight sync ............... no mask, no anchor
```

---

## 4. The guards that keep it honest

The method must not contaminate the rest of GRPO. These are asserted at
runtime (a violation raises, it does not silently corrupt a measurement):

- **Path-tag confinement** — the mask hook fires only when `path_tag=="train"`.
  A leak onto rollout/logprob/ref/val/infer/ckpt raises.
- **Anchor isolation** — the clone shares no parameter `id()` with the live
  optimizer or FSDP module; `anchor_optimizer_steps`, `anchor_rollouts_generated`,
  `anchor_rewards_recomputed`, `anchor_mask_applications` must stay 0.
- **Raw-read contract** — the anchor harvests raw grads (no correction applied)
  before the masked grads are corrected.
- **Numeric-only metrics** — every `comm_eff/*` counter is numeric; string
  discovery fields go to stdout only (a string in the metric dict crashes the
  `np.mean` reduction).

---

## 5. Not yet built (gap list)

The open frontier is the realistic **anchor circuit** (issue **#25**, prerequisite for
**#24**): the anchor `M_anchor` fixes (§1 — global DP all-reduce + full target coverage),
moving the projection basis `Q` to the anchor, and the new sign-based gradient merger
(`α·G + (1−α)·|G|·sign(M_anchor)`) plus error-feedback on the PowerSGD residual. The
existing `inject`/`blend`/`reweight` spectral combiners are inert (§1), so the merger is
new work.

Deferred (later milestones):
- **DP-axis gradient compression** (Streaming-DiLoCo / cross-replica) — distinct from the
  PP-boundary activation compression here; out of scope for now.
- **Per-mini-batch anchor gradients** (the heavier variant) — current code is the
  same-loop periodic refresh.
- **Megatron / Automodel engine integration** — only the FSDP backend overrides the
  comm-eff hooks; other backends run as if disabled.
- **OOM microbatch-split for the anchor pass** — counter plumbed, path not coded.

Out of scope (excluded by the method spec):
- Top-k masking (random PRF only); separate anchor GPU/rank; non-Qwen2.5-1.5B ports;
  masking any path other than actor-train; forking GRPO into a separate algorithm.

Known caveats:
- **SP=1 / rmpad only for masking** — the per-element mask aligns its
  `(sample_id, position_id)` key to the rmpad token axis; Ulysses
  `ulysses_sequence_parallel_size>1` and the non-rmpad (padded) path raise
  `NotImplementedError` (the comm-eff launcher runs SP=1 + rmpad).
- **FSDP1 mandate** — anchor + spectral hooks assume FSDP1 +
  `use_orig_params=true`; FSDP2 (`fully_shard`) is not exercised.
- **Anchor clone memory** — one cached ~3 GB clone for 1.5B in bf16; a deep
  `delay_K` queue multiplies snapshot cost. Fine on H100/H200, tight elsewhere.

---

## 6. When something breaks

| Symptom | Likely culprit / where |
|---|---|
| `NoneType … _saved_grad_shard` on anchor backward | anchor ran on the hook-registered FSDP module — clone path not entered (`transformer_impl.py`; check `assert_anchor_module_isolated`) |
| `TypeError: cannot pickle 'module'` in deepcopy | HF/verl monkey-patch on the model class — the config-rebuild fallback in `anchor.py::build_anchor_module` should catch it |
| mixed `Tensor`/`DTensor` in clone state-load | FSDP1+use_orig_params surfaces DTensors — the per-param `.full_tensor()` path in `build_anchor_module` |
| vLLM `sleep_replicas` memory assertion | anchor clone not released — verify `torch.cuda.empty_cache()` in the refresh `finally` |
| mask hook fires off-train | a path-tag stamp is missing on that entrypoint (`engine_workers.py`) |
| `comm_eff mask token-axis mismatch` / `comm_eff_sample_id missing` | the stable per-row id wasn't stamped before micro-batching, or SP>1/non-rmpad packing — see `engine_workers._comm_eff_stamp_sample_ids` and the SP=1 guard in `transformer_impl.py` |
| `np.mean` crash on metric reduction | a string leaked into `meta_info["metrics"]` — keep comm_eff values numeric |
| anchor counter stays 0 with `enabled=true` | `_maybe_comm_eff_anchor_refresh` not called — `engine/base.py::train_batch` |

---

## 7. Verify

```bash
cd /Users/shamane/Documents/verl
pytest tests/workers/comm_eff/ tests/workers/config/ -v      # CPU unit tests

# In a training log, prove the circuits fired:
grep -E 'anchor refresh step=' <log> | head
grep -oE 'actor/comm_eff/(anchor_backwards|spectral_corrections):[0-9.]+' <log>

# Recover the EXACT settings a run used (ground truth, not prose):
python research/scripts/capture_resolved_config.py runs/<ID>   # -> resolved_params.txt
```

Branch policy and the locked Vast.ai template are in [`CLAUDE.md`](CLAUDE.md).
