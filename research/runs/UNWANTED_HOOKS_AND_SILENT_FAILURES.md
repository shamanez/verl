# Unwanted Hooks And Silent Failures

This note records the comm-eff surfaces that are risky enough to keep testing
for, but should be removed or de-risked when the implementation is simplified.
Do not remove them blindly: several exist only because the current integration
needs global hooks, FSDP workarounds, cross-rank collectives, or runtime guards.

Scope reminder: the communication-efficient method is about the actor training
path's inter-stage activation/gradient traffic. Rollout generation can stay
ordinary vLLM/non-pipeline verl. The risky surfaces below are therefore mainly
actor-train, old-logprob recompute when intentionally compressed, anchor, and
FSDP grad-correction paths.

## High-Risk Surfaces

| File | Surface | Why it is dangerous | Why remove or de-risk it |
|---|---|---|---|
| `verl/workers/engine/fsdp/transformer_impl.py` | Anchor pass temporarily swaps `self.module` to a clone before `_forward_backward_batch_inner`. | A leaked swap would route the next train, eval, or checkpoint forward through the wrong module. Exception handling restores it today, but this is still a fragile global mutation. | Replace the swap with an explicit clone-forward path so the live engine module is never reassigned. |
| `verl/workers/comm_eff/anchor.py`, `verl/workers/engine/fsdp/transformer_impl.py` | Anchor clone must avoid live FSDP post-backward hooks. | Backward on live FSDP params can trigger `_post_backward_hook` outside the normal fast-path window and read missing `_saved_grad_shard`. That can crash or corrupt gradient state. | Keep anchor backward on an isolated no-hook module. Eventually make this a dedicated runner instead of defensive clone/sentinel cleanup. |
| `verl/workers/comm_eff/activation_mask.py`, `verl/workers/engine/fsdp/transformer_impl.py`, `verl/workers/engine_workers.py` | Activation masking uses forward hooks gated by `mask_active` and `path_tag`. | A missed unregister, wrong path tag, or stale `mask_active` value can mask rollout, ref-logprob, validation, checkpoint, or anchor paths. That would corrupt measurements without changing obvious control flow. `old_logprob` is the exception: it may be intentionally masked only when `mask_recompute=true`, and must stay clean otherwise. | Prefer an explicit activation-compression wrapper or a stricter context manager that owns registration, context, and teardown in one place. Keep per-path counters, including `old_logprob`, until hooks are gone. |
| `verl/workers/comm_eff/powersgd_activation.py`, `verl/workers/engine/fsdp/transformer_impl.py`, `verl/workers/engine_workers.py` | PowerSGD uses the same forward-hook lifecycle as masking. | The hook projects activations in-graph. It may intentionally fire on `old_logprob` only when `compress_recompute=true`; if it fires on any other non-train path, or skips the paired recompute when configured, the PPO denominator/numerator comparison or checkpoint/serving path can be wrong without a direct exception. | Reduce global hook dependence and make the compressed forward path explicit. Keep path eligibility tests until then. |
| `verl/workers/engine_workers.py` | `mask_active` is shared by both mask and PowerSGD codecs. | The name is historical and easy to misuse. A future change could assume it only controls masking and accidentally enable or disable PowerSGD on the wrong path. | Split codec-specific active flags or replace the boolean with a typed compression mode/context. |
| `verl/workers/engine/fsdp/transformer_impl.py` | Spectral correction relies on FSDP1 `summon_full_params(with_grads=True)` and `use_orig_params=true`. | Without original 2D params, FSDP1 exposes flat 1D params; target matching can skip everything and produce zero corrections. The current guard raises before this, and that guard is load-bearing. | Keep a loud guard for `use_orig_params`. Longer term, remove the FSDP1 flat-param path from comm-eff or implement a first-class original-param accessor. |
| `verl/workers/comm_eff/spectral_filter.py`, `verl/workers/comm_eff/state.py` | String discovery logs must stay out of metrics. | The trainer reduces every metric value with numeric reducers. A string field in metrics can crash late in the step, after the correction already fired. | Keep discovery in stdout/logger only, or create a typed non-reduced metadata channel. |
| `verl/workers/comm_eff/spectral_filter.py` | Seeded anchors must use a stable hash, not Python `hash()`. | Python hash salting differs per process. If anchors diverge per rank, gradient correction can become rank-local and corrupt the distributed update. | Keep stable hash code. Remove seeded-anchor fallback once the live anchor path fully owns all anchor data. |
| `verl/workers/comm_eff/anchor.py`, `verl/workers/comm_eff/spectral_filter.py` | FSDP wrap-infix canonicalization for parameter names. | Anchor clone names and live FSDP names can differ by `._fsdp_wrapped_module.`. Without canonicalization, the anchor EMA can be written under one key and read under another, silently making correction a no-op. | Replace string-key matching with a stable parameter identity map if possible. Until then, keep canonicalization tests. |
| `verl/workers/comm_eff/powersgd_activation.py` | Basis update all-reduce must iterate a fixed boundary set. | If ranks all-reduce different layer keys or a different order, distributed training can hang with all GPUs idle. | Keep fixed sorted boundary iteration. Eventually move basis sync into a collective helper with explicit shape/order validation. |
| `verl/workers/engine/fsdp/transformer_impl.py` | rmpad and SP=1 guards for mask and PowerSGD. | The token axis is assumed to be rmpad/no-padding. Padded paths or sequence parallel slicing can fold PAD tokens or rank-local token fragments into masks/sketches. | Keep loud `NotImplementedError` guards until SP>1 and padded paths have explicit token-axis mapping. |
| `verl/workers/rollout/base.py`, `verl/workers/engine_workers.py`, `verl/trainer/ppo/ray_trainer.py` | Rollout, ref-logprob, validation, infer, and checkpoint paths rely on actor/rollout separation plus path tags to stay compression-free. | These are measurement or serving paths. Any accidental compression changes rewards, logprobs, or synced weights and can make experiment results invalid. Validation mostly runs through rollout generation, but any actor logprob recompute must still use the intended `old_logprob` policy. | Keep path-tag confinement tests. Remove global hook state so these paths cannot be contaminated by construction. |
| `verl/workers/comm_eff/anchor.py`, `verl/workers/engine/fsdp/transformer_impl.py` | Anchor target extraction is capped by `spectral.max_targets` unless explicitly uncapped. | The current default cap is a smoke-test convenience. For EXP-25, `M_anchor` must cover the same full correction set the merger touches: all 2D q/k/v/o/gate/up/down matrices across all decoder layers, not boundary layers and not the first four matches. A missing target makes `sign(M)` missing or zero and can turn the correction into a silent partial no-op. | Derive anchor extraction and merger correction targets from the same substring+2D selector, set `max_targets=-1` for full runs, and assert set equality plus the expected target count on the real model. |
| `verl/workers/engine/fsdp/transformer_impl.py`, `verl/workers/comm_eff/anchor.py` | Anchor gradients are read from a per-rank clone unless explicitly DP-reduced. | A per-rank `G_anchor` makes `M_anchor` rank-local. That can be cross-rank divergent, or become identically wrong if reduced with the wrong scale. Cross-rank identity alone is not enough: a SUM-vs-mean bug is also identical but off by `dp_size`. | Before feeding EMA, mean-reduce every target over the actor DP group in a fixed sorted target order, contributing shaped zeros for missing targets. Add a real multi-GPU scale check against the same clean fast-path gradient. |
| `verl/workers/engine/fsdp/transformer_impl.py`, `verl/workers/comm_eff/anchor.py` | Anchor clone loading depends on canonical key matching. | If `loaded < total`, the cached clone keeps random-init parameters for missed keys. The anchor backward still runs and produces finite but meaningless `G_anchor`, which is worse than a crash. | Keep `[comm_eff][EXP-18][anchor-load] loaded X/Y` and assert `X == Y` for every refresh. Prefer a stable load map built once from canonical names. |
| `verl/workers/comm_eff/anchor.py`, `verl/workers/engine/fsdp/transformer_impl.py` | Anchor loss must be clean PG, not the fast PPO ratio/clip loss. | The old logprobs may come from a compressed old-policy recompute. Reusing fast-path PPO loss in the unmasked stale anchor pass makes ratio/clipping corrupt `G_anchor`, so `M_anchor` is not the clean gradient. | Keep `anchor_pg_loss` isolated to the anchor path, keep `old_log_probs` ignored there, and keep a scale comparison against the dense fast-path clean gradient in the on-box probe. |
| `verl/workers/comm_eff/powersgd_activation.py`, `verl/workers/engine_workers.py` | EXP-25 wants anchor-owned `Q`, but the current PowerSGD implementation still updates `Q` on the fast path via `maybe_update_basis`. | If anchor-owned `Q` is added without hard-disabling the fast writer, the codebook has two writers. The fast net can silently overwrite the anchor broadcast or update between refreshes, invalidating the stale-anchor correction experiment. | Add an explicit `anchor_owns_q` mode where fast sketch consumption and `maybe_update_basis` are gated off. Assert fast-side `Q` changes only by receiving an anchor broadcast. |
| `verl/workers/comm_eff/powersgd_activation.py`, `verl/workers/engine/fsdp/transformer_impl.py` | EXP-25 requires broadcast receipt for anchor-owned `Q` and `M`. | A dropped broadcast, wrong process group, stale cached tensor, or same-value copy can leave fast ranks using cold-start or old values while logs merely show that the source computed something. | Log source and received checksums for both `Q` and `M`; assert every DP rank receives the source value and that `changed=true` when the source changed. |
| `verl/workers/config/comm_eff.py`, `verl/workers/comm_eff/spectral_filter.py` | EXP-25 `signed_ema` merger is now the LIVE correction; `correction_mode` accepts `signed_ema` (default), `inject`, `blend`. The dead `reweight`/SVD/Tikhonov/seeded path was REMOVED. | A typo or no-op hook could make the run look like prior inert anchor/spectral attempts; an unknown/removed mode (e.g. `reweight`) now raises a `ValueError` in both validation and the dispatch, so it fails fast rather than silently no-op'ing. | Assert per-step correction count from the `[comm_eff][merger]` line: `G_correct = alpha*G_noisy + (1-alpha)*abs(G_noisy)*sign(M)` fires on every fast step with finite grads (`merger_coldM_fallbacks==0` after warm-up). |
| `verl/workers/comm_eff/spectral_filter.py` | **`signed_ema` with a COLD/zero `M` silently ZEROS the gradient at α=0.** `sign(0)=0`, so `G_correct = α·G_noisy + (1−α)·|G_noisy|·0 = α·G_noisy`; at the SFT-validated **α=0** that is `0`. Hits the **first `delay_K` steps** (M not yet refreshed) and **any matrix `M` fails to cover**. No crash — the run just stops learning those matrices. `blend_matrix:520` already guards this (`if anc_norm<=eps: return g_mask`); a naive `signed_ema` transcription WITHOUT the same guard passes every other invariant and quietly kills learning. | `signed_ema` MUST replicate the cold-anchor fallback: when `‖M[name]‖<=eps`, return `G_noisy` unchanged (α=1 behaviour for that matrix). Emit `merger_coldM_fallbacks=N`; assert on step 1 it equals the target count AND those grads are byte-equal to pre-merger `G_noisy` (not zeroed), then →0 after warm-up. |
| `verl/workers/config/comm_eff.py` | **RESOLVED (EXP-25): `seed_anchor_cache` removed.** Historically defaulted `true`, pre-seeding `M_anchor` with a FAKE deterministic PSD basis so `sign(fake_seed)` drove the merger until the live anchor overwrote it. | No longer reachable: the seeded-anchor cache + SVD/reweight path was deleted, so `M` ALWAYS cold-starts at zeros and the flag does not exist (the structured config rejects it). The cold-`M` guard (`merger_coldM_fallbacks`) handles the unwarmed window. | None — structurally fixed. Do NOT pass `spectral.seed_anchor_cache`; the live anchor owns `M` end-to-end. |
| `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`, `verl/workers/config/comm_eff.py` | **`anchor.delay_K` defaults to 20** in both config (`delay_K:int=20`) and launcher (`COMM_EFF_ANCHOR_DELAY_K:-20`); the plan pins only `cadence=5`. | An unpinned arm silently runs at **4× the intended staleness** (delay_K 20 vs ~5). The issue itself notes the K-sweep degrades 0.79→0.67, so this is a large, invisible confound — the run looks fine and produces a curve, just for the wrong K. | PIN `anchor.delay_K` explicitly per arm (probe delay_K=1 so the anchor fires in ≤2 steps; arms delay_K=5 to match cadence) and assert it from `resolved_params.txt` as a controlled variable. Separate `cadence` (fire frequency) from `delay_K` (staleness) in all wording. |
| `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`, `verl/workers/comm_eff/state.py` | **Launcher defaults do NOT resolve to PowerSGD r=77.** Defaults are `COMPRESSION_TYPE=dense`, `MASK_ENABLED=true`, `POWERSGD_RANK=102`; `dense`+enabled mask resolves to `prf_mask`, not PowerSGD. | "Override only the new flags" silently leaves the codec as `prf_mask r=102` (or dense), so both the experimental arms AND the off-path **PowerSGD r=77 parity** check would run on the WRONG codec — the parity check would compare against the wrong baseline and pass/fail meaninglessly. | Explicitly set `COMPRESSION_TYPE=powersgd`, `POWERSGD_RANK=77`, mask OFF; verify `resolved_params.txt` shows codec=powersgd r=77 before trusting any arm or the parity gate. |

## Test Scaffolding We Need Because Of These Risks

| Test file | What it catches | Why it matters |
|---|---|---|
| `tests/workers/comm_eff/test_activation_mask.py` | PRF determinism, path eligibility including `mask_recompute`, hook lifecycle, per-path counters, checkpoint-state contamination. | Prevents wrong-path masking and catches cases where `old_logprob` is compressed outside the explicit recompute mode. |
| `tests/workers/comm_eff/test_mask_rescale.py` | Inverted-dropout and RMS-style mask rescale plumbing. | Keeps the settled mask magnitude contract from drifting while changing hook code. |
| `tests/workers/comm_eff/test_anchor_pg_loss.py` | Anchor clean-PG loss ignores `old_log_probs` and avoids PPO ratio/clip. | Prevents the anchor EMA from being fed a clipped/masked-denominator gradient. |
| `tests/workers/comm_eff/test_anchor_queue.py` | Staleness queue, raw EMA feed, no rollout/reward/optimizer side effects, simulated FSDP1 post-backward hook collision, anchor clone isolation. | Proves anchor backward does not touch live FSDP/optimizer params and exercises the CPU-checkable anchor invariants. |
| `tests/workers/comm_eff/test_grad_correction_hook.py` | Flat-param silent skip, near-zero-grad correction, numeric-only metrics, stable seeded anchors. | Catches cases where correction silently does nothing or crashes metric reduction. |
| `tests/workers/comm_eff/test_spectral_filter.py` | FSDP infix name-key consistency and correction math. | Prevents anchor EMA writes and reads from using different keys. |
| `tests/workers/comm_eff/test_powersgd_activation.py` | Hook lifecycle, Q frozen within a step, current fast-side sketch gating, fixed boundary iteration, single-rank sync-basis behavior. | Prevents wrong-path projection, double-counted sketches, and rank-order issues. It must grow new anchor-owned-Q sole-writer/broadcast tests when EXP-25 lands. |
| `tests/workers/config/test_comm_eff_config.py` | Registered config keys and disabled no-op behavior. | Prevents launcher/config drift from silently selecting the wrong codec or rejecting shared launcher args. |

## Multi-GPU Probes CPU Tests Cannot Replace

These must be checked on a real 4-8 GPU GRPO probe, not with unit tests or
mocked tensors:

1. `M_anchor` is DP-reduced over the actor DP group and at the correct mean
   scale, not just cross-rank identical.
2. `M_anchor` target coverage equals the merger's correction target set. For
   Qwen2.5-1.5B that should be every 2D q/k/v/o/gate/up/down matrix across all
   decoder layers, not the PowerSGD boundary layer set.
3. Anchor DP all-reduce walks the same sorted target order on every rank and
   contributes shaped zeros for missing targets.
4. Anchor clone load reports `loaded == total` and `||dM_anchor|| > 0` across
   consecutive refreshes.
5. Every load-bearing function that should fire has a nonzero counter:
   `anchor_backwards`, target count, `||dM_anchor||`, grad-correction count,
   and, after EXP-25 R2, Q/M broadcast receipt with `changed=true`.
6. Off-path parity is tested with new EXP-25 flags off while `comm_eff.enabled=true`
   and `compression_type=powersgd` remain on. Turning all comm-eff off tests the
   dense path, not the PowerSGD r=77 parity path.
7. `signed_ema` cold-`M` fallback fires: on step 1 (M not yet warmed) every targeted
   matrix takes the `‖M‖<=eps → return G_noisy` path (`merger_coldM_fallbacks` ==
   target count) and the corrected grad is byte-equal to `G_noisy` (NOT zeroed);
   after the first refresh the fallback count drops to ~0. This is the check that
   `α=0` is not silently zeroing early/uncovered gradients.
8. Provenance confirmed from `resolved_params.txt`, not assumed from launcher
   defaults: codec is `powersgd r=77` (not `prf_mask`/`dense`/`r=102`),
   `anchor.cadence=5`, `anchor.delay_K=5` (not the default 20),
   `correction_mode=signed_ema`, `max_targets=-1`. A run that trains "fine" on the
   wrong codec or 4× staleness is the most expensive silent failure here.
   (NB the `seed_anchor_cache` landmine is now structurally impossible: the
   seeded-anchor cache + SVD/reweight path was removed in EXP-25, so `M` always
   cold-starts at zeros and the flag no longer exists.)

## Cleanup Direction

1. Replace runtime forward-hook registration with explicit compressed-forward
   wrappers where possible.
2. Remove `self.module` swapping from the anchor path.
3. Split mask and PowerSGD activation state instead of sharing `mask_active`.
4. Move anchor target selection and grad correction target selection to one
   shared selector with set-equality assertions.
5. Put anchor DP reduction and future Q/M broadcasts behind a collective helper
   that validates target order, shape, process group, and receipt checksums.
6. Keep discovery/logging out of metric dicts unless a typed non-reduced channel
   exists.
7. Keep every guard loud until the corresponding global hook or FSDP workaround
   is removed.
