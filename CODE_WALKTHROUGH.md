# Code Walkthrough — Communication-Efficient GRPO on verl

This fork is **not** vanilla verl. It bolts a **communication-efficient
training method** onto verl's existing GRPO recipe (Qwen2.5-1.5B-Instruct on
GSM8K). When the method is disabled (`comm_eff.enabled=false`), training is a
byte-for-byte no-op against upstream verl; everything below only activates
when it's on.

This file is the engineering map: what lives where, how a step flows, and
what's deliberately not built yet. The mathematics live in the human-only
`major-goal/` reference — agents work from the code + the issue queue, not
from that directory.

---

## 1. The method in one paragraph

GRPO's actor update normally runs one dense forward/backward over the
rollout-expanded batch, then `optimizer.step()`. The method splits that update
into two coupled circuits on the **same process, same batch, same optimizer**:

1. **Fast (masked) circuit** — every step applies an in-graph PRF activation
   mask at pipeline-boundary decoder blocks (`h_tilde = h * mask`, no
   `1/(1-p)` rescale), producing a noisy gradient `G_mask`.
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

Reference config (the comm-eff baseline): `mask.p=0.9`, `mask_recompute=true`,
`anchor.cadence=5`, `anchor.delay_K=5`, `spectral.alpha=0.5`, `tau=0.01`,
`beta_anc=0.9`; no KL, no entropy; `fsdp_config.use_orig_params=true`. The
authoritative defaults are in `verl/workers/config/comm_eff.py` and the
launcher `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`.

---

## 2. Where it lives

| Path | Role |
|---|---|
| `verl/workers/config/comm_eff.py` | `CommEffConfig` + `Mask`/`Anchor`/`Spectral` sub-configs; all defaults DISABLED; bounds validated in `__post_init__` (no allocation) |
| `verl/workers/comm_eff/state.py` | `CommEffState` + `maybe_build_comm_eff_state` factory + path-tag set + numeric counters; the single object owning masker, spectral filter, anchor queue |
| `verl/workers/comm_eff/activation_mask.py` | `ActivationMasker`, splitmix64 `prf_mask`, decoder-boundary index selection; train-only forward hooks |
| `verl/workers/comm_eff/anchor.py` | staleness queue, snapshot/extract/feed helpers, `anchor_should_fire`, `build_anchor_module` (clone-no-hook), `assert_anchor_module_isolated` — the FSDP-agnostic, CPU-testable pieces |
| `verl/workers/comm_eff/spectral_filter.py` | `SpectralFilter`: EMA, full/lowrank SVD, Tikhonov, two-sided projection, α-blend; pure 2D-matrix logic, CPU-unit-testable |
| `verl/workers/engine/base.py` | `train_batch`: anchor refresh → fwd/bwd → grad correction → optimizer step; base no-op stubs |
| `verl/workers/engine/fsdp/transformer_impl.py` | the **only** backend overriding the two comm-eff hooks (clone-no-hook anchor refresh; `summon_full_params` → per-target full-tensor spectral correction → write-back) |
| `verl/workers/engine_workers.py` | `update_actor` stamps `mask_active=True` + `path_tag="train"`; the other entrypoints stamp a non-train tag so the mask hook's guard confines masking to actor-train |
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
      │                                           boundary layers iff path_tag=="train"
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

## 5. Not yet implemented (gap list)

Deferred (later milestones):
- **DP gradient compression** (PowerSGD + Streaming-DiLoCo) — out of scope until
  the actor mask/anchor/spectral path is correct.
- **Paper-cadence long runs** and the **100-step compressed-vs-dense comparison**.
- **Per-mini-batch anchor gradients** (the heavier variant) — current code is the
  same-loop periodic refresh.
- **Megatron / Automodel engine integration** — only the FSDP backend overrides
  the comm-eff hooks; other backends run as if disabled.
- **OOM microbatch-split for the anchor pass** — counter plumbed, path not coded.

Out of scope (excluded by the method spec):
- Top-k masking (random PRF only); forward `1/(1-p)` rescale; separate anchor
  GPU/rank; non-Qwen2.5-1.5B ports; masking any path other than actor-train;
  forking GRPO into a separate algorithm.

Known caveats:
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
