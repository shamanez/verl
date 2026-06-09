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
   pipeline-boundary decoder blocks. Two codecs exist: the **PRF mask** (`prf_mask`,
   reference-only) masks per-(token, dim) `h_tilde = h * mask`, keyed on each token's
   stable `(sample_id, position_id)` so it is packing-invariant across the old-logprob
   and train forwards (`mask.rescale` inverted-dropout `1/(1-p)` is **ON** to unbias
   `E[h̃]=h`; without it grad_norm explodes ~2700 vs ~0.4 dense); **PowerSGD**
   (`powersgd`, the **locked base codec**) projects each boundary activation onto a
   shared low-rank basis, `h_hat = (h Q) Qᵀ`, sending only `Y = h Q`. Either produces
   the noisy gradient the rest of the step corrects.
2. **Anchor (unmasked) circuit — MANDATORY** — every `cadence` ticks, an *unmasked*
   GRPO-actor-loss forward/backward runs from a `delay_K`-stale weight snapshot on a
   **no-hook clone** of the module. Its **backward** yields a clean, DP-reduced,
   full-coverage `G_anchor` (→ the EMA `M_anchor`); its **forward** harvests boundary
   activations to recompute the PowerSGD basis `Q ← orth(V)` and broadcast it. The
   anchor is **the only thing that updates `Q`** (`anchor.owns_q`): the fast circuit's
   `maybe_update_basis` is gated off (fail-closed) and it only ever *reads* `Q`.

A **merger** keeps a running EMA `M_anchor` of the (full-coverage, DP-reduced) anchor
gradients and folds it into the fast compressed gradient `G_comp` per-coordinate,
before AdamW sees it:

```
M_anchor = β·M_anchor + (1-β)·G_anchor                # EMA (β = beta_anc)
G_corr   = α·G_comp + (1-α)·|G_comp|·sign(M_anchor)   # signed_ema merger
```

`α=1` ⇒ `G_comp` unchanged (no merge); `α=0` ⇒ pure sign-replacement. A cold-`M`
guard returns `G_comp` unchanged for any matrix whose `M` is unwarmed, so an unwarmed
sign never zeros a gradient. The dead SVD/Tikhonov/two-sided-projection ("reweight")
path was **removed** (EXP-25); `inject`/`blend` remain as alternate combiners.
Ordering invariant: **anchor refresh → compressed fwd/bwd → FSDP all-reduce → merger →
AdamW**. The anchor runs *before* the fast fwd/bwd so its raw gradient feeds the EMA
before any correction touches the fast grads.

**Status.** The base is the **anchor circuit on a PowerSGD codec** (issue #25 / EXP-25 —
result + why in `research/runs/SUMMARY.md`). The codec is **PowerSGD-style activation
compression**: a shared low-rank orthonormal basis `Q` projects each boundary activation,
`M_hat = (M Q) Qᵀ`, so only `Y = M Q` (rank-`r`=77 coords/token) crosses the boundary,
frozen within a step. **`Q` is owned + updated by the anchor** (`Q ← orth(V)` on the
anchor's stale-weight forward activations, broadcast DP-wide each refresh); the fast
circuit's basis-update is gated off (fail-closed) — it only *reads* `Q`. The anchor also
maintains the full-coverage (196 matrices), DP-reduced, `delay_K`-stale gradient EMA
`M_anchor`, and the **signed_ema merger** folds it into the fast gradient (the §1 math).
This **replaces** the impractical periodic dense `clean_cadence` step. R1 (full-coverage
DP-reduced `M`) + R2 (anchor-owns-`Q`) are the **proven substrate**; the **merger primitive**
(R3) is the open research axis — `signed_ema` is falsified, error-feedback (#24) is next.

Base config (named, not enumerated — to avoid drift): anchor on + owns `Q` + PowerSGD
codec + the `signed_ema` merger, no clean step, no KL/entropy, `use_orig_params=true`. The
exact values are the launcher `${VAR:-default}`
(`examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`) and the locked
sheet `research/runs/FIXED_CONTROL_SURFACE.md`; the ground truth of any run is its
`resolved_params.txt`. The Hydra dataclass defaults (all-OFF, for byte-identity) live in
`verl/workers/config/comm_eff.py`.

---

## 2. Where it lives

| Path | Role |
|---|---|
| `verl/workers/config/comm_eff.py` | `CommEffConfig` + `Mask`/`Anchor`/`Spectral` sub-configs; all defaults DISABLED; bounds validated in `__post_init__` (no allocation) |
| `verl/workers/comm_eff/state.py` | `CommEffState` + `maybe_build_comm_eff_state` factory + path-tag set + numeric counters; the single object owning masker, spectral filter, anchor queue |
| `verl/workers/comm_eff/activation_mask.py` | `ActivationMasker`, counter-based splitmix64 `prf_token_mask` (per-(token, dim), keyed on stable `(sample_id, position_id)`), `decoder_boundary_indices` selection; train-only forward hooks |
| `verl/workers/comm_eff/powersgd_activation.py` | `PowerSGDActivationCompressor`: per-boundary low-rank basis `Q` (deterministic seed, fp32 QR), `Y=MQ`/`M_hat=YQᵀ` projection hooks, `Q←orth(V)` block power iteration with cross-DP sketch all-reduce (`sync_basis`) + a cross-rank agreement guard; the `powersgd` codec. In `anchor_owns_q` mode the fast `maybe_update_basis` is **fail-closed** (raises if entered) — the anchor drives `Q` |
| `verl/workers/comm_eff/anchor.py` | staleness queue, snapshot/extract/feed helpers (full-coverage target set + DP all-reduce of `G_anchor`), `anchor_should_fire`, `build_anchor_module` (clone-no-hook), `assert_anchor_module_isolated` — the FSDP-agnostic, CPU-testable pieces |
| `verl/workers/comm_eff/spectral_filter.py` | `SpectralFilter`: the anchor-gradient EMA `M_anchor` + the **`signed_ema` merger** `α·G+(1−α)·\|G\|·sign(M)` (with the cold-`M` guard) and the `inject`/`blend` combiners; pure 2D-matrix logic, CPU-unit-testable (the dead SVD/Tikhonov/reweight path was removed, EXP-25) |
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
      ├─ [ANCHOR]  _maybe_comm_eff_anchor_refresh(data, loss_fn)      # MANDATORY
      │     summon_full_params → snapshot into staleness queue
      │     if anchor_should_fire(step, cadence):
      │       load delay_K-stale snapshot into a cached clone-no-hook module,
      │       run unmasked fwd/bwd on the clone (mask_active=False):
      │         backward → DP all-reduce(mean) full-coverage G_anchor → EMA M_anchor
      │         forward  → harvest boundary activations → Q ← orth(V)
      │       broadcast Q (+ M) DP-wide (receipt-verified), restore live module,
      │       empty_cache() for vLLM sleep hygiene
      ├─ [FAST]    forward_backward_batch         PowerSGD projection hooks fire at
      │                                           boundary layers iff path_tag=="train"
      │                                           (Y=hQ, h_hat=YQᵀ); Q is READ-ONLY here
      │                                           (anchor owns it). (prf_mask codec, if
      │                                           selected, masks per-(sample_id,
      │                                           position_id) instead.)
      ├─ [MERGER]  _maybe_comm_eff_grad_correction
      │     summon_full_params (grads FSDP-reduced) → per 2D target (all 196):
      │       G_corr = α·G_comp + (1-α)·|G_comp|·sign(M_anchor) → write back p.grad
      │       (cold-M guard: unwarmed M ⇒ G_comp unchanged)
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
- **`Q` ownership (R2)** — in `anchor_owns_q` mode the fast `maybe_update_basis` is
  **fail-closed** (raises if ever entered), so the fast circuit can never write `Q`;
  `Q` (and `M`) broadcasts are receipt-verified and `verify_basis_agreement_across_ranks`
  raises on divergence.
- **Anchor `M` correctness (R1)** — `M_anchor` is the *global* gradient: DP all-reduced
  (mean) before the EMA, and covers the full set of merger-corrected matrices
  (set-equality with the merger's selector, not a `max_targets` slice).
- **Raw-read contract** — the anchor harvests raw grads (no correction applied)
  before the fast grads are corrected.
- **Numeric-only metrics** — every `comm_eff/*` counter is numeric; string
  discovery fields go to stdout only (a string in the metric dict crashes the
  `np.mean` reduction).

---

## 5. Not yet built (gap list)

The anchor circuit (R1 full-coverage DP-reduced `M` + R2 anchor-owns-`Q` + R3 the
`signed_ema` merger) is **built and is the base** (issue #25). The open frontier is a
**better merger primitive**: `signed_ema` (sign-replacement) is falsified, so the next
work is **error-feedback on the PowerSGD residual** (issue **#24**, was gated on #25) —
fold the dropped `(I−QQᵀ)` energy back in instead of overriding the gradient direction.
The `inject`/`blend` combiners remain wired but are inert/net-harmful here (§1). The
result + why are in `research/runs/SUMMARY.md`.

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
