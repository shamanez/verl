# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Asynchronous unmasked anchor circuit.

The anchor circuit produces a CLEAN per-target gradient ``G_anchor`` that the
spectral filter consumes into its anchor-gradient EMA ``M_anchor``. "Clean"
means three things:

* **Same loss as the fast path, UNMASKED.** The anchor reuses the GRPO
  actor-loss over the rollout-expanded batch (``responses``, ``response_mask``,
  ``old_log_probs``, ``advantages``, optional ``ref_log_prob``) — exactly the
  fast path's ``ppo_loss`` — but with the activation masker DISABLED even though
  it runs on the actor-train path (``mask_active=False`` ⇒
  ``anchor_mask_applications == 0``). It is NOT a supervised next-token loss; it
  does NOT generate rollouts; it does NOT recompute rewards.

* **K-stale snapshot, no optimizer step.** The anchor forwards from a
  ``delay_K``-stale weight snapshot taken OFF the optimizer's parameter group
  (so the optimizer never sees it and no accidental step occurs), and takes NO
  ``optimizer.step()`` of its own (``anchor_optimizer_steps == 0``).

* **Raw gradient into the EMA, before any correction.** ``G_anchor`` is read
  RAW per target and fed to ``SpectralFilter.update_anchor`` BEFORE any
  fast-path corrector (``signed_ema_matrix`` / ``inject_matrix`` /
  ``blend_matrix``) runs (``anchor_grad_corrected == 0``). The anchor
  gradient is never the input to the correction.

This module owns the FSDP-AGNOSTIC pieces so they are unit-testable on CPU with
no distributed runtime:

* :class:`AnchorStalenessQueue` — a bounded ring of weight snapshots keyed by
  trainer step, returning the ``t - delay_K`` snapshot (or the oldest available
  while the queue is still warming up).
* :func:`snapshot_named_params` — detached CPU/GPU clones of the model's named
  parameters, explicitly DECOUPLED from the optimizer's param group.
* :func:`extract_target_grads` — the raw per-target 2D gradient extraction
  protocol (mirrors the spectral hook's iteration), returning
  ``{name: full_2d_grad}`` with NO correction applied.
* :func:`feed_anchor_grads_into_ema` — wires the raw grads into
  ``SpectralFilter.update_anchor`` and reports ``||ΔM_anchor||`` per target so
  the engine can log EMA evolution.

The actual unmasked forward/backward on the (possibly sharded) FSDP module lives
in the engine (``FSDPEngine._maybe_comm_eff_anchor_refresh``), which calls these
helpers; keeping the fwd/bwd there is what lets the pure logic above stay
CPU-testable.
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "AnchorStalenessQueue",
    "AnchorReplayRing",
    "snapshot_named_params",
    "extract_target_grads",
    "feed_anchor_grads_into_ema",
    "anchor_should_fire",
    "anchor_pg_loss",
    "build_anchor_module",
    "assert_anchor_module_isolated",
    "capture_anchor_tensors",
    "clone_batch_for_replay",
    "maybe_build_replay_ring",
    "snapshot_canary",
    "verify_canary_on_module",
]


def _canon(name: str) -> str:
    """Strip the FSDP per-layer-wrap infix from a parameter name.

    Mirrors ``spectral_filter._canon`` (kept local so this CPU-testable module
    has no cross-module import dependency). When ``build_anchor_module`` falls
    back to a config-rebuild, the rebuilt clone is a PLAIN module whose
    ``named_parameters()`` names lack the ``._fsdp_wrapped_module.`` infix the
    live (per-layer FSDP-wrapped) ``inner_module`` may carry. Matching the
    fallback param/buffer copy by canonical key ensures the clone is seeded with
    the REAL live weights rather than keeping its random init — otherwise
    ``G_anchor`` is computed from garbage.
    """
    name = name.replace("._fsdp_wrapped_module", "")
    if name.startswith("_fsdp_wrapped_module."):
        name = name[len("_fsdp_wrapped_module."):]
    return name


def anchor_should_fire(step: int, cadence: int, enabled: bool) -> bool:
    """True iff the anchor refresh fires on trainer ``step``.

    Pure predicate (no side effects) so the cadence policy is unit-testable.
    ``step`` is 1-based (the engine advances it before the check). The anchor
    fires when ``enabled`` and ``(step % cadence) == 0`` — so ``cadence=1`` fires
    every step and ``cadence=20`` fires on steps 20, 40, ...
    """
    if not enabled or cadence < 1:
        return False
    return (step % cadence) == 0


def anchor_pg_loss(config, model_output, data, dp_group=None):
    """CLEAN policy-gradient loss for the anchor pass.

    **Why this replaces the fast-path ``ppo_loss`` for the anchor refresh.**
    The anchor circuit does ONE forward/backward per refresh, so the PPO
    *importance ratio* ``exp(logπ_new − old_log_probs)`` is not just unnecessary,
    it is actively *wrong* here: the batch's ``old_log_probs`` were produced by
    the MASKED fast path, while the anchor re-forwards UNMASKED at the
    ``delay_K``-stale weights. That mismatch drives the ratio away from 1, the
    PPO clip then mangles the per-token loss, and the resulting ``G_anchor`` is
    NOT the clean unmasked policy gradient that ``M_anchor`` should represent.

    **What this computes instead — the clean true gradient at θ_{t-K}.**
    With ratio ≡ 1 (no ``old_log_probs``, no clip), ``compute_policy_loss_vanilla``
    provably reduces to the per-token loss ``-advantages · logπ`` aggregated by
    ``agg_loss``; its gradient is ``-(A · ∇logπ_unmasked)`` — exactly "the clean
    step's gradient, evaluated at the stale weights". We reuse the SAME log-prob
    extraction (``no_padding_2_padding``), the SAME field selection/padding, and
    the SAME ``agg_loss`` + ``global_batch_info`` normalization as ``ppo_loss``
    so ``M_anchor`` lands at the identical scale as the fast-path clean gradient
    (under the default ``token-mean`` this equals the spec's
    ``-(A·logπ·mask).sum()/mask.sum()``; for other agg modes it stays faithful
    to the fast path).

    Signature mirrors ``verl.workers.utils.losses.ppo_loss`` so it can be bound
    with ``functools.partial(anchor_pg_loss, config=actor_config)`` and dropped
    into ``_forward_backward_batch_inner`` in place of the fast-path loss. It is
    used ONLY by ``FSDPEngine._maybe_comm_eff_anchor_refresh`` (the anchor pass);
    the fast path's real PPO ratio/clip loss is left completely untouched.

    Args:
        config: the actor ``ActorConfig`` (carries ``loss_agg_mode``,
            ``loss_scale_factor``, ``global_batch_info``). Bound via ``partial``.
        model_output: dict with ``log_probs`` (per-token log-probs of the
            response, possibly nested) exactly as ``ppo_loss`` consumes.
        data: the rollout-expanded ``TensorDict`` (carries ``response_mask`` and
            ``advantages``; ``old_log_probs`` is deliberately IGNORED).
        dp_group: data-parallel process group (unused here; kept for signature
            parity with ``ppo_loss``).

    Returns:
        ``(loss, metrics)`` — ``metrics`` carries ``actor/anchor_pg_loss`` plus
        ``actor/anchor_ratio_mean`` (≡ 1.0 by construction) under MEAN
        aggregation.
    """
    # Lazy imports: keep module import cheap + CPU-testable; the engine path and
    # the CPU tests both have these available.
    from verl.utils.metric import AggregationType, Metric
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.workers.utils.padding import no_padding_2_padding

    # Per-token log-probs of the response, padded to (bsz, max_response_len) —
    # IDENTICAL extraction to ppo_loss.
    log_prob = no_padding_2_padding(model_output["log_probs"], data)

    # Mirror ppo_loss's global-batch bookkeeping so agg_loss normalizes exactly
    # like the fast path (loss invariant to FSDP sharding).
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metric_aggregation = AggregationType.SUM

    # Select ONLY the fields the clean PG needs — NOT old_log_probs. This is the
    # whole point: no importance ratio, so old_log_probs never enters.
    selected = data.select("response_mask", "advantages").to_padded_tensor()
    response_mask = selected["response_mask"].to(bool)
    advantages = selected["advantages"]

    loss_agg_mode = config.loss_agg_mode

    # ratio ≡ 1 (no clip): the per-token PPO loss collapses to -A·logπ, whose
    # gradient is the clean unmasked policy gradient -(A·∇logπ). agg_loss applies
    # the response_mask and the same normalization the fast path uses.
    per_token_pg = -advantages * log_prob
    pg_loss = agg_loss(
        loss_mat=per_token_pg,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    metrics = {
        "actor/anchor_pg_loss": Metric(value=pg_loss, aggregation=metric_aggregation),
        # ratio is identically 1 here (no old_log_probs / no clip).
        "actor/anchor_ratio_mean": Metric(value=1.0, aggregation=AggregationType.MEAN),
    }
    return pg_loss, metrics


class AnchorStalenessQueue:
    """Bounded ring of weight snapshots keyed by trainer step (the t-K buffer).

    ``push(step, snapshot)`` records a snapshot; ``get_stale(step, delay_K)``
    returns the snapshot taken at step ``step - delay_K`` if present, else the
    OLDEST retained snapshot (the queue is still warming up — at ``delay_K=1``
    the very first refresh has only the current snapshot, which is the documented
    "step-1 sees an empty/near-current anchor" behaviour, finite by construction).

    The queue retains at most ``delay_K + 1`` snapshots (enough to serve any
    ``t - delay_K`` lookup) so memory does not grow with the run length. A
    snapshot is a ``dict[name -> detached tensor clone]`` produced by
    :func:`snapshot_named_params` — explicitly NOT optimizer state.
    """

    def __init__(self, delay_K: int):
        assert delay_K >= 0, f"delay_K must be >= 0, got {delay_K}"
        self.delay_K = int(delay_K)
        # OrderedDict[step -> snapshot]; insertion order == step order.
        self._snapshots: "OrderedDict[int, dict]" = OrderedDict()
        self._maxlen = self.delay_K + 1

    def push(self, step: int, snapshot: dict) -> None:
        """Record ``snapshot`` taken at trainer ``step`` and evict stale entries."""
        self._snapshots[int(step)] = snapshot
        # Evict oldest beyond the retention window.
        while len(self._snapshots) > self._maxlen:
            self._snapshots.popitem(last=False)

    def get_stale(self, step: int, delay_K: Optional[int] = None) -> Optional[dict]:
        """Return the snapshot from step ``step - delay_K``.

        Falls back to the oldest retained snapshot while warming up (fewer than
        ``delay_K`` snapshots seen). Returns ``None`` only if the queue is empty.
        """
        if not self._snapshots:
            return None
        k = self.delay_K if delay_K is None else int(delay_K)
        target_step = int(step) - k
        snap = self._snapshots.get(target_step)
        if snap is not None:
            return snap
        # Warming up (or target evicted): use the oldest available snapshot.
        oldest_step = next(iter(self._snapshots))
        return self._snapshots[oldest_step]

    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def steps(self) -> list:
        return list(self._snapshots.keys())


def _clone_tensor_for_replay(t: torch.Tensor, device=None) -> torch.Tensor:
    """Detached deep clone of one batch leaf, NJT (jagged) safe.

    The fast clone path is ``detach().clone()`` (the production NJT hot path).
    If a torch version mishandles clone/device-move on a jagged leaf, fall back
    to decomposing ``(values, offsets, _ragged_idx)`` and rebuilding via
    ``torch.nested.nested_tensor_from_jagged`` — the same API
    ``verl.utils.tensordict_utils.nested_tensor_from_tensor_list`` uses.
    """
    if getattr(t, "is_nested", False):
        try:
            out = t.detach().clone()
            if device is not None:
                out = out.to(device)
            return out
        except Exception:
            values = t.values().detach().clone()
            offsets = t.offsets().detach().clone()
            if device is not None:
                values = values.to(device)
                offsets = offsets.to(device)
            nt = torch.nested.nested_tensor_from_jagged(values=values, offsets=offsets)
            try:
                nt._ragged_idx = t._ragged_idx
            except AttributeError:
                pass
            return nt
    out = t.detach().clone()
    if device is not None:
        out = out.to(device)
    return out


def clone_batch_for_replay(data, device=None):
    """Deep clone of a train_batch TensorDict for the anchor replay ring.

    A deep clone at STORE time is required: ``_forward_backward_batch_inner``
    mutates the live batch in place (``tu.assign_non_tensor``), and the masked
    fast path consumes the same TensorDict right after the anchor hook. The
    clone (a) shallow-copies the key->value mapping (so later key assignment on
    the live batch never touches the stored copy) and (b) deep-clones every
    tensor leaf, detached, optionally moved to ``device`` (``"cpu"`` keeps the
    ring off HBM). Non-tensor entries ride along by reference — they are
    replaced (not mutated) by ``assign_non_tensor``, so the mapping copy
    isolates them.
    """
    out = data.copy() if hasattr(data, "copy") else copy.copy(data)
    for key in list(out.keys()):
        val = out.get(key)
        if isinstance(val, torch.Tensor):
            out[key] = _clone_tensor_for_replay(val, device=device)
    return out


def snapshot_canary(snapshot: dict, target_substrs=None, n: int = 2) -> dict:
    """Record fp32-on-CPU ``(norm, sum)`` fingerprints of ``n`` canary matrices.

    Deterministic target choice: the first and last sorted 2D names matching
    ``target_substrs`` (falling back to all keys if none match). Both record
    and verify cast ``bf16 -> cpu -> fp32`` (an exact widening) and reduce on
    CPU, so an unchanged byte payload reproduces the values BITWISE — the
    value-level staleness check for the CPU-resident snapshot ring.
    """
    names = sorted(
        k
        for k, v in snapshot.items()
        if getattr(v, "dim", None) is not None
        and v.dim() == 2
        and (target_substrs is None or any(s in k for s in target_substrs))
    )
    if not names:
        names = sorted(snapshot.keys())
    picked = [names[0]]
    if len(names) > 1 and n > 1:
        picked.append(names[-1])
    out = {}
    for name in picked:
        t = snapshot[name].detach().to("cpu", torch.float32)
        out[name] = (float(torch.linalg.norm(t).item()), float(t.sum().item()))
    return out


def verify_canary_on_module(module: torch.nn.Module, canary: dict, canon: Optional[Callable] = None):
    """Recompute the canary off ``module``'s params; bitwise-match the record.

    Returns ``(ok, results)`` where ``results`` maps each canary name to the
    recomputed ``(norm, sum)`` (or ``(None, None)`` if the param is missing).
    The caller hard-asserts ``ok`` — a mismatch means the clone did NOT receive
    the recorded historical weights (storage corruption, a lossy device round
    trip, or a load that silently skipped the param).
    """
    canon = canon or (lambda s: s)
    params = {canon(p_name): p for p_name, p in module.named_parameters()}
    results = {}
    ok = True
    for name, (ref_norm, ref_sum) in canary.items():
        p = params.get(canon(name))
        if p is None:
            results[name] = (None, None)
            ok = False
            continue
        t = p.detach().to("cpu", torch.float32)
        got = (float(torch.linalg.norm(t).item()), float(t.sum().item()))
        results[name] = got
        if got != (float(ref_norm), float(ref_sum)):
            ok = False
    return ok, results


class AnchorReplayRing:
    """Paired ``(batch, generator-weights)`` replay ring for the anchor refresh.

    EXP-29: the anchor's stale weights must be paired with the trajectories
    those SAME weights generated. A snapshot taken at the FIRST ``train_batch``
    tick of global step ``G`` (before any optimizer tick of ``G``) is exactly
    the weights vLLM held when it generated step ``G``'s rollouts; per-tick
    batch clones then give exact ``(batch[t-K], gen_snapshot)`` pairs, warmup
    included. Data staleness is ``delay_K`` ticks post-warmup; realized WEIGHT
    staleness alternates ``K``/``K+1`` ticks by construction (the snapshot sits
    at the first tick of the batch's global step) and is reported so the engine
    can log it.

    Holds:
      * ``tick -> (batch_clone, gs)`` — bounded to ``delay_K + 1`` entries;
      * ``gs -> (snapshot, canary, push_tick)`` — one snapshot per global step
        (``push_snapshot`` is idempotent per ``gs``); snapshots are evicted as
        soon as no retained batch references them.

    Pure container — no collectives, no RNG, CPU-testable.
    """

    def __init__(self, delay_K: int):
        assert delay_K >= 0, f"delay_K must be >= 0, got {delay_K}"
        self.delay_K = int(delay_K)
        self._maxlen = self.delay_K + 1
        self._batches: "OrderedDict[int, tuple]" = OrderedDict()
        self._snapshots: "OrderedDict[int, tuple]" = OrderedDict()

    def has_snapshot(self, gs: int) -> bool:
        return int(gs) in self._snapshots

    def push_snapshot(self, gs: int, snapshot: dict, canary: Optional[dict] = None, tick: int = -1) -> bool:
        """Record the generator snapshot for global step ``gs`` (first-tick wins).

        Returns True iff this call stored the snapshot (False = ``gs`` already
        had one — the caller is on a later tick of the same global step).
        """
        gs = int(gs)
        if gs in self._snapshots:
            return False
        self._snapshots[gs] = (snapshot, dict(canary or {}), int(tick))
        return True

    def push_batch(self, tick: int, batch, gs: int) -> None:
        """Record this tick's deep-cloned batch, paired with its generator gs."""
        gs = int(gs)
        assert gs in self._snapshots, (
            f"AnchorReplayRing.push_batch(tick={tick}, gs={gs}) before push_snapshot for that gs — "
            "the generator snapshot must be recorded at the FIRST tick of the global step."
        )
        self._batches[int(tick)] = (batch, gs)
        while len(self._batches) > self._maxlen:
            self._batches.popitem(last=False)
        # Evict snapshots no retained batch references (bounded memory).
        live_gs = {g for (_b, g) in self._batches.values()}
        for g in [g for g in self._snapshots if g not in live_gs]:
            del self._snapshots[g]

    def get_replay(self, tick: int, delay_K: Optional[int] = None):
        """Return ``(used_tick, batch, gs, snapshot, canary, snap_tick, warmup_fallback)``.

        Post-warmup (``tick > delay_K``) the ``tick - delay_K`` batch MUST be
        retained (push runs every tick; the ring keeps ``delay_K + 1``); during
        warmup we fall back to the OLDEST retained batch — its pairing with its
        own generator snapshot stays exact (warmup included). ``None`` only if
        the ring is empty.
        """
        if not self._batches:
            return None
        k = self.delay_K if delay_K is None else int(delay_K)
        req = int(tick) - k
        if req in self._batches:
            used, fallback = req, False
        else:
            used, fallback = next(iter(self._batches)), True
        batch, gs = self._batches[used]
        snapshot, canary, snap_tick = self._snapshots[gs]
        return used, batch, gs, snapshot, canary, snap_tick, fallback

    def __len__(self) -> int:
        return len(self._batches)

    @property
    def batch_ticks(self) -> list:
        return list(self._batches.keys())

    @property
    def snapshot_steps(self) -> list:
        return list(self._snapshots.keys())


def maybe_build_replay_ring(state, anchor_cfg, delay_K: int) -> Optional[AnchorReplayRing]:
    """Build (once, on the state) and return the replay ring iff
    ``anchor.replay_paired_batch`` is true; return ``None`` otherwise.

    The OFF path constructs NOTHING (no ring, no buffers) — the flag-OFF parity
    invariant, CPU-testable: ``maybe_build_replay_ring(state, cfg_off, K) is
    None`` and leaves no ``_anchor_replay_ring`` attribute behind.
    """
    if not bool(getattr(anchor_cfg, "replay_paired_batch", False)):
        return None
    ring = getattr(state, "_anchor_replay_ring", None)
    if ring is None:
        ring = AnchorReplayRing(delay_K=delay_K)
        setattr(state, "_anchor_replay_ring", ring)
    return ring


def snapshot_named_params(
    named_params,
    *,
    target_substrs=None,
    device: Optional[torch.device] = None,
    detach: bool = True,
) -> dict:
    """Detached clones of the model's named parameters (the anchor snapshot).

    The returned tensors are plain detached clones that
    the optimizer NEVER sees — they are not registered in any param group, so no
    optimizer step can be taken on them. The anchor restores these into the live
    module for its forward, runs backward to populate ``.grad`` on the LIVE
    params (read raw), then the caller restores the live weights.

    Args:
        named_params: iterator of ``(name, param-or-tensor)``.
        target_substrs: if given, only snapshot params whose name contains one of
            these substrings (the 2D decoder matrices the anchor cares about);
            ``None`` snapshots everything (needed to faithfully restore the model
            after the anchor forward).
        device: optional device to place the clones on (e.g. ``"cpu"`` to keep
            the snapshot off HBM for a large model); ``None`` keeps each on its
            param's device.
        detach: clone with ``.detach()`` so no autograd history is retained.

    Returns:
        ``dict[name -> tensor]``.
    """
    snap = {}
    for name, p in named_params:
        if target_substrs is not None and not any(s in name for s in target_substrs):
            continue
        t = p.detach() if detach else p
        t = t.clone()
        if device is not None:
            t = t.to(device)
        snap[name] = t
    return snap


def extract_target_grads(
    named_params,
    *,
    target_substrs,
    max_targets: int,
    full_grad_of: Callable,
) -> dict:
    """Extract the RAW per-target 2D gradient ``G_anchor`` for each target matrix.

    This mirrors the iteration/selection of the spectral hook's
    ``apply_spectral_correction_to_params`` — same target substrings, same 2D
    filter, same ``max_targets`` cap — but applies NO correction: it returns the
    raw full 2D gradients exactly as backward produced them. The engine feeds
    these straight into ``SpectralFilter.update_anchor``
    (the EMA) before any fast-path corrector is ever called.

    Args:
        named_params: iterator of ``(name, param)`` whose ``.grad`` is the
            anchor backward's gradient (full logical 2D after FSDP unshard, via
            ``full_grad_of``).
        target_substrs: substrings selecting the targeted 2D matrices.
        max_targets: cap on the number of targets (``-1`` ⇒ no cap).
        full_grad_of: ``grad -> (full_2d_tensor, meta)`` — the FSDP unshard
            callable (identity for plain CPU/non-FSDP tensors). Same contract as
            the spectral hook so the engine reuses one implementation.

    Returns:
        ``dict[name -> full_2d_grad]`` (detached clones; the caller may zero the
        live grads afterwards without disturbing the EMA inputs).
    """
    grads = {}
    for name, p in named_params:
        grad = getattr(p, "grad", None)
        if grad is None:
            continue
        if not any(s in name for s in target_substrs):
            continue
        if max_targets >= 0 and len(grads) >= max_targets:
            break
        full, _meta = full_grad_of(grad)
        if full.dim() != 2:
            continue
        # Detached clone so a later optimizer_zero_grad on the live grads does not
        # alias/mutate what we already fed to the EMA.
        grads[name] = full.detach().clone()
    return grads


def build_anchor_module(inner_module: torch.nn.Module) -> torch.nn.Module:
    """Return a deep-cloned ``nn.Module`` for the anchor's backward that is
    **fully detached from any FSDP wrapping / post-backward hooks**
    registered on the live actor module.

    Why this exists:
        FSDP1's ``_post_backward_hook`` calls
        ``_check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)``
        on the registered ``FlatParameter``s. ``_saved_grad_shard`` is only set
        up between the fast-path's pre-backward and the optimizer step, so the
        anchor backward (outside that window) hit
        ``AttributeError: 'NoneType' object has no attribute 'shape'``. The fix
        fix is to break the autograd-hook chain entirely — the anchor pass must NOT
        be allowed to fire any hook registered on the live FSDP params.

    Mechanism:
        ``copy.deepcopy`` of the underlying ``nn.Module`` creates fresh
        ``Parameter`` objects with empty ``_backward_hooks`` /
        ``post_accumulate_grad_hooks`` registries. The clone is a plain
        non-FSDP ``nn.Module`` — there is no ``_handles`` attribute, no
        ``_post_backward_hooks``, and crucially no ``FlatParameter`` /
        ``_saved_grad_shard`` machinery. Backward through the clone fires
        ONLY the autograd graph of the clone itself, not the live actor's
        FSDP hooks.

    The caller is responsible for:
        - Loading the K-stale snapshot weights into the clone BEFORE calling
          fwd/bwd (this function returns a clone of the LIVE weights; the
          engine then ``copy_``'s the staleness-queue snapshot into it).
        - Reading ``p.grad`` off the clone's named_parameters() for
          extract_target_grads, never the live module.
        - Discarding the clone after the refresh (it carries no state across
          refreshes by design).

    Args:
        inner_module: the **unwrapped** underlying ``nn.Module`` (e.g.
            ``self.module._fsdp_wrapped_module`` for FSDP1). For FSDP1 the
            caller MUST be inside a ``FSDP.summon_full_params(...,
            with_grads=False, writeback=False)`` block when calling this, so
            the parameters are unsharded full tensors at the moment of
            ``copy.deepcopy``. Failure to summon will deep-copy a sharded
            local fragment.

    Returns:
        A plain ``nn.Module`` with the same architecture and a fresh copy of
        the weights; NO FSDP wrapping, NO hook registrations.
    """
    try:
        clone = copy.deepcopy(inner_module)
    except TypeError as exc:
        # verl/HF monkey-patches can install function attributes holding Python
        # module references on the model class, which are not picklable. Fall
        # back to config-rebuild + state_dict load.
        if "cannot pickle" not in str(exc):
            raise
        cfg = getattr(inner_module, "config", None)
        if cfg is None:
            raise RuntimeError(
                "build_anchor_module: copy.deepcopy failed AND inner_module "
                "has no .config attribute — cannot rebuild via the HuggingFace "
                "config path. Original error: " + repr(exc)
            ) from exc
        ModelClass = type(inner_module)
        clone = ModelClass(cfg)
        # Caller is inside FSDP.summon_full_params, so inner_module is
        # currently un-sharded; state_dict() returns full logical tensors.
        # strict=False guards against monkey-patched buffers that may not
        # be present on the freshly-instantiated clone.
        # FSDP1+use_orig_params may return DTensor entries even inside
        # summon_full_params, while the rebuilt clone has plain Tensor params.
        # Copy each param/buffer manually after DTensor -> Tensor materialization.
        import torch as _torch
        def _to_plain(t):
            # DTensor -> full unsharded plain Tensor (works inside summon).
            if hasattr(t, "full_tensor"):
                try:
                    return t.full_tensor()
                except Exception:
                    pass
            if hasattr(t, "to_local"):
                try:
                    return t.to_local()
                except Exception:
                    pass
            return t
        # Match by canonical (FSDP-infix-stripped) key. The rebuilt clone
        # has NON-infixed names; the live inner_module's summoned names may carry
        # the `._fsdp_wrapped_module.` infix (per-layer wrapping). A raw `n in
        # src_params` lookup then misses for every layer and the clone keeps its
        # RANDOM init weights → G_anchor is garbage. Canon-keying both sides
        # gives the clone the real live weights even before the snapshot-load.
        with _torch.no_grad():
            src_params = {_canon(n): p for n, p in inner_module.named_parameters()}
            for n, p_dst in clone.named_parameters():
                s = src_params.get(_canon(n))
                if s is not None:
                    s = _to_plain(s.detach())
                    p_dst.data.copy_(s.to(p_dst.device, p_dst.dtype))
            src_buffers = {_canon(n): b for n, b in inner_module.named_buffers()}
            for n, b_dst in clone.named_buffers():
                s = src_buffers.get(_canon(n))
                if s is not None:
                    s = _to_plain(s)
                    b_dst.copy_(s.to(b_dst.device, b_dst.dtype))
    # Belt-and-braces: explicitly clear any post-accumulate-grad hooks the
    # deepcopy might have transferred via a custom __deepcopy__ on a parent
    # class (defensive — torch.nn.Parameter.__deepcopy__ does NOT transfer
    # _backward_hooks today, but a future torch version might). Walk the
    # clone's params and zero out any Python-level hook dict if present.
    for _, p in clone.named_parameters():
        # Drop the FSDP1 sentinel attributes if a future deepcopy ever
        # propagated them; the clone must look like a plain nn.Module to
        # autograd's hook machinery.
        for sentinel in ("_saved_grad_shard", "_handles", "_post_backward_hooks", "_fsdp1_hook_fired"):
            if hasattr(p, sentinel):
                try:
                    delattr(p, sentinel)
                except (AttributeError, TypeError):
                    pass
    return clone


def assert_anchor_module_isolated(
    clone: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    fsdp_module: Optional[torch.nn.Module] = None,
) -> None:
    """Sanity-assert the anchor clone is fully off the live optimizer and FSDP.

    The clone's parameters must NOT appear in any optimizer param_group and must
    NOT be registered with any FSDP ``_handles`` / ``_fsdp_wrapped_module``
    instance on the live actor. Cheap to run; called from
    ``_maybe_comm_eff_anchor_refresh`` once per refresh.

    Raises:
        AssertionError if any clone param aliases or shares ``id()`` with a
        param the live optimizer or live FSDP module owns.
    """
    clone_ids = {id(p) for _, p in clone.named_parameters()}

    if optimizer is not None:
        opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        intersect = clone_ids & opt_ids
        assert not intersect, (
            f"anchor clone shares {len(intersect)} param object(s) with the live "
            "optimizer's param_groups — criterion 7 at risk "
            "(snapshot MUST be OFF the optimizer's parameter group)."
        )

    if fsdp_module is not None:
        live_ids = {id(p) for _, p in fsdp_module.named_parameters()}
        intersect = clone_ids & live_ids
        assert not intersect, (
            f"anchor clone shares {len(intersect)} param object(s) with the live "
            "FSDP module — criterion 13 at risk (clone must be a fresh deep-copy, "
            "not an alias)."
        )


def capture_anchor_tensors(
    *,
    writer,
    role: str,
    grads: dict,
    global_step: int,
    optimizer_tick: int,
) -> int:
    """EXP-26 Step A: dump a ``{name: tensor}`` map under ``role`` (detached/fp32).

    Pure I/O — used for the K-stale ``G_anchor`` map (role ``"G_anchor"``), the
    anchor EMA ``M`` map (role ``"M"``), and the ``delay_K=0`` fresh-anchor
    measurement grad (role ``"G_fresh_anchor"``). The writer detaches + clones, so
    this NEVER feeds the optimizer or the EMA (the measurement-only invariant).
    No-op (returns 0) when ``writer is None``. Returns the number of tensors
    written.
    """
    if writer is None or not grads:
        return 0
    n = 0
    for name, t in grads.items():
        if t is None:
            continue
        if writer.dump(
            role=role, target_name=name, tensor=t,
            global_step=global_step, optimizer_tick=optimizer_tick,
        ):
            n += 1
    return n


def feed_anchor_grads_into_ema(grads: dict, spectral, *, state=None) -> dict:
    """Feed RAW ``{name: G_anchor}`` into the spectral filter's anchor EMA.

    This calls ``spectral.update_anchor`` (the RAW EMA blend
    ``M_anchor <- beta*M_anchor + (1-beta)*G_anchor``) — NEVER ``correct_matrix``
    — so the anchor gradient is never spectrally corrected. Returns
    ``{name: ||ΔM_anchor||}`` (Frobenius norm of the EMA change this refresh) so
    the engine can log EMA evolution.

    If ``state`` is given, ``state.anchor_grad_corrected`` is left untouched by
    this function by design — it stays 0 unless a corrected tensor is
    (defensively) ever routed here, which it must not be.
    """
    deltas = {}
    for name, g_anchor in grads.items():
        before = spectral._anchor.get(name)
        before = before.detach().to(torch.float32).clone() if before is not None else None
        after = spectral.update_anchor(name, g_anchor)
        if before is None:
            # First sight: ΔM relative to the (zero/seed) initial state.
            delta = torch.linalg.norm(after.detach().to(torch.float32)).item()
        else:
            # update_anchor may have moved devices; compare on a common device.
            a = after.detach().to(torch.float32)
            b = before.to(a.device)
            delta = torch.linalg.norm(a - b).item()
        deltas[name] = delta
    return deltas
