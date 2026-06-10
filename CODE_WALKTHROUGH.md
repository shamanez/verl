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
before AdamW sees it. This is how the fast circuit consumes the anchor's **full
gradients** — *which way* is method-dependent (`spectral.correction_mode`):

```
M_anchor = β·M_anchor + (1-β)·G_anchor                # EMA (β = beta_anc)

# signed_ema (FALSIFIED, EXP-25/26 — sign-agreement is a structural coin-flip):
G_corr   = α·G_comp + (1-α)·|G_comp|·sign(M_anchor)

# ef_powersgd (EXP-26 Step B — direction-preserving, NO sign term):
e_t      = clip( ef_decay·e_{t-1} + (M_anchor − P_{G_comp}(M_anchor)),  ef_clip·‖G_comp‖ )
G_corr   = G_comp + e_t                               # keeps G_comp's direction/sign
```

`ef_powersgd` re-injects the anchor-EMA's **off-subspace residual** (the energy the
rank-`r` projection dropped), detached, shape-reset on mismatch, norm-capped relative to
`‖G_comp‖`; at the limiting setting `ef_decay=0, ef_clip=0` it is **bit-identical to
plain PowerSGD** (`G_corr == G_comp`, CPU-test-enforced, as is the no-sign-term property).
For `signed_ema`: `α=1` ⇒ no merge; `α=0` ⇒ pure sign-replacement; a cold-`M` guard
returns `G_comp` unchanged for any unwarmed matrix. The dead SVD/Tikhonov/reweight path
was **removed** (EXP-25); `inject`/`blend` remain as alternate combiners; spectral OFF ⇒
`G_comp` reaches AdamW untouched (plain codec).
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
(R3) is the open research axis — `signed_ema` is falsified (EXP-25/26), **`ef_powersgd`
error-feedback is the live successor** (EXP-26: best realistic comm-eff result, parity not
yet reached — verdict + numbers in `research/runs/EXP-26/verdict.md`, next iteration #27).
The anchor's `Q` content is selectable (`powersgd.q_basis` families, EXP-26 Step C) but
**`act` is the only production basis** — update-energy content anti-converts because
rollouts run uncompressed (Step C falsified; non-act families are retained as switchable
diagnostics only).

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
| `verl/workers/comm_eff/powersgd_activation.py` | `PowerSGDActivationCompressor`: per-boundary low-rank basis `Q` (deterministic seed, fp32 QR), `Y=MQ`/`M_hat=YQᵀ` projection hooks, `Q←orth(V)` block power iteration with cross-DP sketch all-reduce (`sync_basis`) + a cross-rank agreement guard; the `powersgd` codec. In `anchor_owns_q` mode the fast `maybe_update_basis` is **fail-closed** (raises if entered) — the anchor drives `Q`. EXP-26 additions: **`q_basis` family sketch constructions** (`act` production default / `grad` / `adv` / `tail` / `hybrid` / `ticket` — `_compute_family_V`, `_build_family_Q`, `build_and_dump_family_sketches` for the passive screen; `anchor_update_basis` selects what feeds `V` in the live arm), boundary-grad harvest hooks (`G_b`, anchor-pass-only, removed in `finally`), `set_advantage_weight` plumbing, and **Step-E comm byte counters** (`reset_tick_comm_counters`, `add_amortized_q_broadcast_bytes` → `comm/bytes_*` metrics) |
| `verl/workers/comm_eff/anchor.py` | staleness queue, snapshot/extract/feed helpers (full-coverage target set + DP all-reduce of `G_anchor`), `anchor_should_fire`, `build_anchor_module` (clone-no-hook), `assert_anchor_module_isolated` — the FSDP-agnostic, CPU-testable pieces |
| `verl/workers/comm_eff/spectral_filter.py` | `SpectralFilter`: the anchor-gradient EMA `M_anchor` + the merger modes — **`ef_powersgd`** (`ef_powersgd_matrix`: direction-preserving off-subspace residual error-feedback, detached/clipped/shape-reset, NO sign term; the live method), `signed_ema` (falsified; cold-`M` guard) and the `inject`/`blend` combiners; `apply_spectral_correction_to_params` dispatches per `correction_mode` and writes back `p.grad` pre-Adam; pure 2D-matrix logic, CPU-unit-testable (the dead SVD/Tikhonov/reweight path was removed, EXP-25) |
| `verl/workers/comm_eff/capture.py` | **EXP-26 diagnostic layer (default OFF)**: `CaptureWriter` / `maybe_build_capture_writer` — fp32 tensor dumps keyed `(global_step, optimizer_tick, target)` with tick budget (`min_tick`/`max_ticks`), stratified-target subset, `rank0_only` disk guard; lazily imported only when `capture.enabled=true`, every dump site no-ops otherwise |
| `verl/workers/engine/base.py` | `train_batch`: anchor refresh → fwd/bwd → grad correction → optimizer step; base no-op stubs |
| `verl/workers/engine/fsdp/transformer_impl.py` | the **only** backend overriding the comm-eff hooks (clone-no-hook anchor refresh; `summon_full_params` → per-target full-tensor merger correction → write-back). EXP-26 additions, **all capture-gated**: the `delay_K=0` fresh-anchor MEASUREMENT probe (optionally `ppo_clip` loss) and the `G_dense` parallel-backward probe — both on isolated clones, dump-only, with `PROBE_LEAKS_INTO_OPTIMIZER` asserts; `_comm_eff_build_adv_weight` (rmpad-aligned advantage plumbing for the adv family); `_maybe_capture_g_comp_no_merger` |
| `verl/workers/engine_workers.py` | `update_actor` stamps `mask_active=True` + `path_tag="train"` and a stable per-row `comm_eff_sample_id` on the batch (also in `compute_log_prob`) so the mask keys on each token's `(sample_id, position_id)`; the other entrypoints stamp a non-train tag so the mask hook's guard confines masking to actor-train |
| `tests/workers/comm_eff/` | CPU unit tests (149): PRF determinism / mask ratio / train-only confinement; spectral α-blend / projection / determinism; anchor staleness / isolation regression; **`test_ef_powersgd_exp26.py`** (limiting-case bit-identity, no-`torch.sign` source-inspection gate, sign-preservation, clip/decay/shape-reset) + **`test_q_family_screen_exp26.py`** (family constructions, act-only inertness, dedupe, DP-consensus) |
| `research/scripts/geometry_audit.py`, `research/scripts/stepC_screen_audit.py` | laptop-side fp32-capture audits (cosines, update-capture/off-principal, sign-agreement, family ranking) — consume `runs/<ID>/captures/`, never run on the box |

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
      │                    (q_basis selects WHAT feeds V: act = activation 2nd moment
      │                     [production]; grad/adv/tail/hybrid/ticket = diagnostic
      │                     families; q_basis_passive=[…] builds + dumps candidate
      │                     bases WITHOUT touching the live Q — the C1 screen)
      │       broadcast Q (+ M) DP-wide (receipt-verified), restore live module,
      │       empty_cache() for vLLM sleep hygiene
      │     [capture.enabled ONLY] delay_K=0 fresh-anchor probe (clean_pg|ppo_clip
      │       loss), G_dense parallel-backward probe — isolated clones, dump-only,
      │       PROBE_LEAKS_INTO_OPTIMIZER-asserted; fp32 dumps via CaptureWriter
      ├─ [FAST]    forward_backward_batch         PowerSGD projection hooks fire at
      │                                           boundary layers iff path_tag=="train"
      │                                           (Y=hQ, h_hat=YQᵀ); Q is READ-ONLY here
      │                                           (anchor owns it); per-tick comm byte
      │                                           counters accumulate (N·r sent vs N·H
      │                                           dense-equivalent → comm/bytes_*).
      │                                           (prf_mask codec, if selected, masks
      │                                           per-(sample_id, position_id) instead.)
      ├─ [MERGER]  _maybe_comm_eff_grad_correction         (spectral.enabled only;
      │     summon_full_params (grads FSDP-reduced) →       OFF ⇒ G_comp untouched)
      │     per 2D target (all 196), dispatch correction_mode:
      │       ef_powersgd: G_corr = G_comp + clip(decayed off-subspace resid of M)
      │       signed_ema:  G_corr = α·G_comp + (1-α)·|G_comp|·sign(M)  (cold-M guard)
      │       inject/blend: alternate combiners
      │     → write back p.grad
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
- **Probes never feed the optimizer (EXP-26)** — the `G_dense` and `delay_K=0`
  fresh-anchor probes run on isolated clones with hook-strip asserts and
  `PROBE_LEAKS_INTO_OPTIMIZER` checks; they exist ONLY under `capture.enabled`.
- **EF residual properties (EXP-26)** — `e_t` is detached, shape-reset on target
  mismatch, norm-capped at `ef_clip·‖G_comp‖`; `ef_decay=0 ∧ ef_clip=0` ⇒
  bit-identical to plain PowerSGD, and the no-sign-term property is enforced by a
  source-inspection CPU test (`test_ef_powersgd_never_calls_sign`).
- **Diagnostics are OFF by default** — `capture.enabled=false`,
  `q_basis="act"`, `q_basis_passive=[]`, `ef_decay=ef_clip=0.0`: a run that sets
  none of the EXP-26 knobs is byte-identical to the EXP-25 substrate, and
  `comm_eff.enabled=false` remains byte-identical to upstream (the only new code
  on the dense path is `None`-checks).
- **Numeric-only metrics** — every `comm_eff/*` counter is numeric; string
  discovery fields go to stdout only (a string in the metric dict crashes the
  `np.mean` reduction).

---

## 5. Not yet built (gap list)

The anchor circuit (R1 full-coverage DP-reduced `M` + R2 anchor-owns-`Q`) is **built and
is the base** (issue #25). The merger axis (R3) has moved: `signed_ema` is **falsified**
(EXP-25, mechanism confirmed by the EXP-26 geometry audit — sign-agreement is a structural
coin-flip), and its successor **`ef_powersgd` error-feedback is built and measured**
(EXP-26: best realistic comm-eff result; direction-preserving; parity missed by ~2 pts;
a stochastic length-explosion ignition at `ef_clip=1.0` is the open risk). The open
frontier is the **damped EF iteration** (issue **#27**: `ef_clip 0.5`, `ef_decay 0.5`,
100 steps). The `q_basis` Q-content axis was explored and **falsified** (EXP-26 Step C —
update-energy bases anti-convert; `act` stays the only production basis; the family
machinery is retained as switchable diagnostics). The `inject`/`blend` combiners remain
wired but inert/net-harmful here (§1). Results + why live in `research/runs/SUMMARY.md`
and `research/runs/EXP-26/verdict.md`.

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
grep -oE 'actor/comm_eff/(anchor_backwards|spectral_corrections|anchor_q_updates|powersgd_basis_updates):[0-9.]+' <log>

# EXP-26 additions — prove the merger mode + measure comm volume:
grep -oE 'correction_mode=ef_powersgd ef_decay=[0-9.]+ ef_clip=[0-9.]+' <log> | head -1
grep -oE 'actor/comm/bytes_(compressed|dense_equiv|ratio):[0-9.e+-]+' <log> | tail -3
# (capture/probe/family knobs all default OFF — a clean run shows family_screen_builds:0)

# Recover the EXACT settings a run used (ground truth, not prose):
python research/scripts/capture_resolved_config.py runs/<ID>   # -> resolved_params.txt
```

Branch policy and the locked Vast.ai template are in [`CLAUDE.md`](CLAUDE.md).
