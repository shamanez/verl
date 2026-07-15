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

"""Delayed paired dense-anchor helpers for communication-efficient GRPO.

The anchor replays a generator-consistent GRPO batch at a delayed or RELEX-
projected checkpoint, evaluates the clean ratio-one policy-gradient objective,
and feeds the raw dense gradient into the signed-EMA reference ``M``. It runs
on an isolated model clone and never takes an optimizer step.

This module contains the FSDP-agnostic queue, replay, snapshot, loss, gradient
extraction, and EMA-update pieces. The FSDP forward/backward integration lives
in ``FSDPEngine._maybe_comm_eff_anchor_refresh``.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Callable, Optional

import torch

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
    "clone_batch_for_replay",
    "select_anchor_batch_for_scope",
    "maybe_build_replay_ring",
    "snapshot_canary",
    "verify_canary_on_module",
]


def _canon(name: str) -> str:
    """Strip the FSDP per-layer-wrap infix from a parameter name.

    Mirrors ``spectral_filter._canon`` and stays local to avoid a cross-module
    import dependency. When ``build_anchor_module`` falls
    back to a config-rebuild, the rebuilt clone is a PLAIN module whose
    ``named_parameters()`` names lack the ``._fsdp_wrapped_module.`` infix the
    live (per-layer FSDP-wrapped) ``inner_module`` may carry. Matching the
    fallback param/buffer copy by canonical key ensures the clone is seeded with
    the REAL live weights rather than keeping its random init — otherwise
    ``G_anchor`` is computed from the wrong weights.
    """
    name = name.replace("._fsdp_wrapped_module", "")
    if name.startswith("_fsdp_wrapped_module."):
        name = name[len("_fsdp_wrapped_module.") :]
    return name


def anchor_should_fire(step: int, cadence: int, enabled: bool) -> bool:
    """True iff the anchor refresh fires on trainer ``step``.

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
    it is actively *wrong* here: the anchor re-forwards at the
    ``delay_K``-stale weights. That mismatch drives the ratio away from 1, the
    PPO clip then mangles the per-token loss, and the resulting ``G_anchor`` is
    NOT the clean dense policy gradient that ``M_anchor`` should represent.

    **What this computes instead — the clean ratio-one objective at θ_{t-K}.**
    With ratio ≡ 1 (no ``old_log_probs``, no clip), ``compute_policy_loss_vanilla``
    provably reduces to the per-token loss ``-advantages · logπ`` aggregated by
    ``agg_loss``; its gradient is ``-(A · ∇logπ)`` — exactly "the clean
    step's gradient, evaluated at the stale weights". We reuse the SAME log-prob
    extraction (``no_padding_2_padding``), the SAME field selection/padding, and
    the SAME ``agg_loss`` + ``global_batch_info`` normalization as ``ppo_loss``
    so ``M_anchor`` lands at the identical scale as the fast-path clean gradient
    (under the default ``token-mean`` this equals the spec's
    ``-(A·logπ·mask).sum()/mask.sum()``; for other agg modes it stays faithful
    to the fast path). Rollout importance weights, entropy regularization, and
    reference-policy KL are mirrored from the same resolved actor config. This
    is the objective-parity contract: every additive term that steers the fast
    actor also steers ``M``. The only intentional differences are the
    compressed-policy ``old_log_probs`` / PPO importance ratio and clipping,
    which are invalid for this stale or projected uncompressed forward and are
    therefore replaced by ratio one with no clipping.

    Signature mirrors ``verl.workers.utils.losses.ppo_loss`` so it can be bound
    with ``functools.partial(anchor_pg_loss, config=actor_config)`` and dropped
    into ``_forward_backward_batch_inner`` in place of the fast-path loss. It is
    used ONLY by ``FSDPEngine._maybe_comm_eff_anchor_refresh`` (the anchor pass);
    the fast path's real PPO ratio/clip loss is left completely untouched.

    Args:
        config: the actor ``ActorConfig`` (carries ``loss_agg_mode``,
            ``loss_scale_factor``, ``global_batch_info``). Bound via ``partial``.
        model_output: dict with ``log_probs`` (per-token log-probs of the
            response, possibly nested) exactly as ``ppo_loss`` consumes, plus
            ``entropy`` whenever ``entropy_coeff`` is nonzero.
        data: the rollout-expanded ``TensorDict`` (carries ``response_mask`` and
            ``advantages``, optional ``rollout_is_weights``, and
            ``ref_log_prob`` when KL loss is enabled; ``old_log_probs`` is
            deliberately IGNORED).
        dp_group: data-parallel process group (unused here; kept for signature
            parity with ``ppo_loss``).

    Returns:
        ``(loss, metrics)`` — ``metrics`` carries ``actor/anchor_pg_loss`` plus
        ``actor/anchor_ratio_mean`` (≡ 1.0 by construction) under MEAN
        aggregation.
    """
    # Lazy imports keep this module cheap to import; the engine path provides
    # these dependencies when the anchor objective runs.
    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    policy_loss = getattr(config, "policy_loss", None)
    if policy_loss is None:
        loss_mode = "vanilla"
    elif hasattr(policy_loss, "get"):
        loss_mode = policy_loss.get("loss_mode", "vanilla")
    else:
        loss_mode = getattr(policy_loss, "loss_mode", "vanilla")
    if loss_mode != "vanilla":
        raise ValueError(
            "comm_eff anchor objective parity currently supports "
            "actor.policy_loss.loss_mode='vanilla' only; "
            f"got {loss_mode!r}. Implement and test an explicit ratio-one "
            "anchor mapping before enabling this policy loss."
        )

    entropy_coeff = float(getattr(config, "entropy_coeff", 0.0))
    if entropy_coeff != 0.0 and model_output.get("entropy") is None:
        raise KeyError(
            "entropy: comm_eff anchor objective parity requires model_output['entropy'] "
            f"when actor.entropy_coeff={entropy_coeff}"
        )

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

    # Select the ratio-one objective fields — never old_log_probs. The optional
    # rollout importance weights and reference log-probability mirror ppo_loss.
    # Missing ref_log_prob fails closed through TensorDict.select whenever the
    # configured fast objective enables KL loss.
    fields = ["response_mask", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if config.use_kl_loss:
        fields.append("ref_log_prob")
    selected = data.select(*fields).to_padded_tensor()
    response_mask = selected["response_mask"].to(bool)
    advantages = selected["advantages"]

    # One resolved contract line per actor-config object makes objective drift
    # visible in paid-run logs. Emit PASS only after every required objective
    # input has been selected successfully; a missing KL/entropy input must not
    # leave a misleading success marker behind. The private marker lives on the
    # config so multiple actors in one process remain independent.
    if not getattr(config, "_comm_eff_anchor_objective_contract_logged", False):
        print(
            "[comm_eff][anchor-objective] parity=PASS "
            f"fast_policy_loss={loss_mode} anchor_surrogate=ratio_one_pg "
            f"use_kl_loss={str(bool(config.use_kl_loss)).lower()} "
            f"kl_type={getattr(config, 'kl_loss_type', 'n/a')} "
            f"kl_coef={float(getattr(config, 'kl_loss_coef', 0.0)):.12g} "
            f"entropy_coef={entropy_coeff:.12g} "
            f"rollout_is_weights={str('rollout_is_weights' in selected).lower()} "
            f"loss_agg={config.loss_agg_mode} "
            "exceptions=old_log_probs,ppo_ratio,ppo_clip",
            flush=True,
        )
        # BaseConfig is frozen after construction. This telemetry-only marker is
        # private and cannot alter any scientific setting.
        object.__setattr__(config, "_comm_eff_anchor_objective_contract_logged", True)

    loss_agg_mode = config.loss_agg_mode

    # ratio ≡ 1 (no clip): the per-token PPO loss collapses to -A·logπ, whose
    # gradient is the clean dense policy gradient -(A·∇logπ). agg_loss applies
    # the response_mask and the same normalization the fast path uses.
    per_token_pg = -advantages * log_prob
    if "rollout_is_weights" in selected:
        per_token_pg = per_token_pg * selected["rollout_is_weights"]
    pg_loss = agg_loss(
        loss_mat=per_token_pg,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    objective_loss = pg_loss
    metrics = {
        "actor/anchor_pg_loss": Metric(value=pg_loss, aggregation=metric_aggregation),
        # ratio is identically 1 here (no old_log_probs / no clip).
        "actor/anchor_ratio_mean": Metric(value=1.0, aggregation=AggregationType.MEAN),
    }
    entropy = model_output.get("entropy")
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        objective_loss = objective_loss - entropy_coeff * entropy_loss
        metrics["actor/anchor_entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)
        metrics["actor/anchor_entropy_coef"] = entropy_coeff
    if config.use_kl_loss:
        ref_log_prob = selected["ref_log_prob"]
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        objective_loss = objective_loss + kl_loss * config.kl_loss_coef
        metrics["actor/anchor_kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["actor/anchor_kl_coef"] = config.kl_loss_coef

    metrics["actor/anchor_total_loss"] = Metric(value=objective_loss, aggregation=metric_aggregation)
    return objective_loss, metrics


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
        self._snapshots: OrderedDict[int, dict] = OrderedDict()
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
    mutates the live batch in place (``tu.assign_non_tensor``), and the
    compressed fast path consumes the same TensorDict right after the anchor hook. The
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


def select_anchor_batch_for_scope(batch_scope: str, current_batch, rollout_batch=None):
    """Resolve the anchor's current data source without a silent fallback.

    ``rollout_batch`` is the complete worker-local actor update retained before
    PPO splitting. Requesting it outside that context is a correctness error;
    falling back to ``current_batch`` would silently turn a 512-prompt request
    back into the historical 256-prompt scope.
    """
    if batch_scope == "ppo_minibatch":
        return current_batch
    if batch_scope == "rollout_batch":
        if rollout_batch is None:
            raise RuntimeError(
                "comm_eff anchor batch_scope=rollout_batch requires the pre-split "
                "train_mini_batch context, but no full update batch is available"
            )
        return rollout_batch
    raise RuntimeError(f"unsupported comm_eff anchor batch_scope={batch_scope!r}")


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
    the recorded weights (storage corruption, a lossy device round trip, or a
    load that skipped the param).
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


def verify_canary_on_snapshot(snapshot: dict, canary: dict):
    """Bitwise-verify ``canary`` against a raw snapshot DICT (pre-clone-load).

    Same fp32-on-CPU ``(norm, sum)`` reduction as
    :func:`verify_canary_on_module`, but off the snapshot mapping itself. Used
    by the look-ahead (weight-projection) path: the clone there is loaded with
    the PROJECTED ``theta_hat`` — the clone no longer holds the recorded stale
    weights verbatim, so the value-level staleness guard must run against the
    SOURCE snapshot instead. Returns ``(ok, results)``; the caller hard-asserts
    ``ok``.
    """
    results = {}
    ok = True
    for name, (ref_norm, ref_sum) in canary.items():
        t = snapshot.get(name)
        if t is None:
            results[name] = (None, None)
            ok = False
            continue
        tt = t.detach().to("cpu", torch.float32)
        got = (float(torch.linalg.norm(tt).item()), float(tt.sum().item()))
        results[name] = got
        if got != (float(ref_norm), float(ref_sum)):
            ok = False
    return ok, results


class AnchorReplayRing:
    """Paired ``(batch, generator-weights)`` replay ring for the anchor refresh.

    The anchor's stale weights can be paired with the trajectories those same
    weights generated. A snapshot taken at the FIRST ``train_batch``
    tick of global step ``G`` (before any optimizer tick of ``G``) is exactly
    the weights vLLM held when it generated step ``G``'s rollouts; per-tick
    batch clones then give exact ``(batch[t-K], gen_snapshot)`` pairs, warmup
    included. Data staleness is ``delay_K`` ticks post-warmup; realized WEIGHT
    staleness alternates ``K``/``K+1`` ticks by construction (the snapshot sits
    at the first tick of the batch's global step) and is reported so the engine
    can log it.

    **Fire-aware retention.** The anchor only fires on ticks ``t ≡ 0 (mod
    cadence)`` and a fire at ``t`` only ever consumes tick ``t − delay_K``, so
    the only ticks worth storing satisfy ``tick ≡ (−delay_K) mod cadence``
    (:meth:`tick_retained`). Everything else is rejected at push time; the engine
    also skips the deep clone for those ticks. Bounds are asserted on every push:

      * batches:   ``delay_K // cadence + 1`` entries (== ``delay_K + 1`` at
        cadence=1);
      * snapshots: one per global step still referenced by a retained batch,
        plus the current (newest) gs awaiting its batches — ``maxlen + 1``.

    Holds ``tick -> (batch_clone, gs)`` and ``gs -> (snapshot, canary,
    push_tick)`` (``push_snapshot`` is idempotent per ``gs``). The container
    performs no collectives and draws no RNG.
    """

    def __init__(self, delay_K: int, cadence: int = 1):
        assert delay_K >= 0, f"delay_K must be >= 0, got {delay_K}"
        self.delay_K = int(delay_K)
        self.cadence = max(1, int(cadence))
        # Fire ticks are multiples of cadence; a fire at t requests t - delay_K,
        # so only ticks of this residue class can ever be replayed.
        self._keep_residue = (-self.delay_K) % self.cadence
        self._maxlen = self.delay_K // self.cadence + 1
        self._batches: OrderedDict[int, tuple] = OrderedDict()
        self._snapshots: OrderedDict[int, tuple] = OrderedDict()

    def tick_retained(self, tick: int) -> bool:
        """True iff a future anchor fire can ever request ``tick``'s batch.

        The engine consults this BEFORE deep-cloning the batch, so non-replayable
        ticks cost neither the clone nor ring space.
        """
        return (int(tick) % self.cadence) == self._keep_residue

    def has_snapshot(self, gs: int) -> bool:
        return int(gs) in self._snapshots

    def snapshot_tick(self, gs: int) -> int:
        """Tick at which global step ``gs``'s generator snapshot was taken.

        That is the FIRST train_batch tick of ``gs`` — the exact weight point
        vLLM sampled this step's rollouts from. The look-ahead projector uses it
        as the projection TARGET for the current step (projecting to the fire
        tick instead would overshoot the generator by the fire's within-step
        tick offset and de-pair the ratio-1 anchor loss from its batch).
        Returns ``-1`` when ``gs`` has no retained snapshot.
        """
        entry = self._snapshots.get(int(gs))
        return int(entry[2]) if entry is not None else -1

    def push_snapshot(self, gs: int, snapshot: dict, canary: Optional[dict] = None, tick: int = -1) -> bool:
        """Record the generator snapshot for global step ``gs`` (first-tick wins).

        Returns True iff this call stored the snapshot (False = ``gs`` already
        had one — the caller is on a later tick of the same global step). A new
        gs beginning also evicts OLDER gs snapshots no retained batch references
        (their batches can never arrive again — gs is monotonic).
        """
        gs = int(gs)
        if gs in self._snapshots:
            return False
        live_gs = {g for (_b, g) in self._batches.values()}
        for g in [g for g in self._snapshots if g < gs and g not in live_gs]:
            del self._snapshots[g]
        self._snapshots[gs] = (snapshot, dict(canary or {}), int(tick))
        assert len(self._snapshots) <= self._maxlen + 1, (
            f"AnchorReplayRing snapshot retention blew its bound: "
            f"{len(self._snapshots)} > maxlen+1={self._maxlen + 1} "
            f"(delay_K={self.delay_K} cadence={self.cadence}) — eviction regressed."
        )
        return True

    def push_batch(self, tick: int, batch, gs: int) -> bool:
        """Record this tick's deep-cloned batch, paired with its generator gs.

        Returns False (storing nothing) for a tick no future fire can request —
        the cadence-filtered retention. The caller should pre-check
        :meth:`tick_retained` to also skip the deep-clone cost.
        """
        if not self.tick_retained(tick):
            return False
        gs = int(gs)
        assert gs in self._snapshots, (
            f"AnchorReplayRing.push_batch(tick={tick}, gs={gs}) before push_snapshot for that gs — "
            "the generator snapshot must be recorded at the FIRST tick of the global step."
        )
        self._batches[int(tick)] = (batch, gs)
        while len(self._batches) > self._maxlen:
            self._batches.popitem(last=False)
        # Evict snapshots no retained batch references (bounded memory). The
        # newest gs is always kept — later ticks of it may still be retained.
        live_gs = {g for (_b, g) in self._batches.values()}
        newest_gs = max(self._snapshots) if self._snapshots else None
        for g in [g for g in self._snapshots if g not in live_gs and g != newest_gs]:
            del self._snapshots[g]
        assert len(self._batches) <= self._maxlen, (
            f"AnchorReplayRing batch retention blew its bound: {len(self._batches)} > "
            f"maxlen={self._maxlen} (delay_K={self.delay_K} cadence={self.cadence})."
        )
        return True

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


def maybe_build_replay_ring(state, anchor_cfg, delay_K: int, cadence: int = 1) -> Optional[AnchorReplayRing]:
    """Build (once, on the state) and return the replay ring iff
    ``anchor.replay_paired_batch`` is true; return ``None`` otherwise.

    ``cadence`` is the anchor fire cadence — it keys the ring's fire-aware
    retention (only ticks a future fire can request are stored). The OFF path
    constructs no ring or buffers and leaves no ``_anchor_replay_ring``
    attribute behind.
    """
    if not bool(getattr(anchor_cfg, "replay_paired_batch", False)):
        return None
    ring = getattr(state, "_anchor_replay_ring", None)
    if ring is None:
        ring = AnchorReplayRing(delay_K=delay_K, cadence=cadence)
        state._anchor_replay_ring = ring
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
    target_scope: str = "decoder_matrices",
) -> dict:
    """Extract the RAW gradient ``G_anchor`` for each configured target tensor.

    This mirrors the iteration/selection of the spectral hook's
    ``apply_spectral_correction_to_params`` — same scope, substrings, and
    ``max_targets`` cap — but applies NO correction: it returns the raw full
    gradients exactly as backward produced them. The engine feeds
    these straight into ``SpectralFilter.update_anchor``
    (the EMA) before any fast-path corrector is ever called.

    Args:
        named_params: iterator of ``(name, param)`` whose ``.grad`` is the
            anchor backward's gradient (full logical 2D after FSDP unshard, via
            ``full_grad_of``).
        target_substrs: substrings used by ``decoder_matrices``.
        target_scope: ``decoder_matrices`` (substring-matched 2-D tensors) or
            ``all_floating`` (every unique floating parameter with a gradient).
        max_targets: cap on the number of targets (``-1`` ⇒ no cap).
        full_grad_of: ``grad -> (full_2d_tensor, meta)`` — the FSDP unshard
            callable (identity for plain CPU/non-FSDP tensors). Same contract as
            the spectral hook so the engine reuses one implementation.

    Returns:
        ``dict[name -> full_2d_grad]`` (detached clones; the caller may zero the
        live grads afterwards without disturbing the EMA inputs).
    """
    grads = {}
    # Deferred import keeps anchor.py's light-weight utility import path free of
    # spectral-filter construction and, critically, shares one selector with the
    # fast merger so anchor coverage cannot drift from writeback coverage.
    from verl.workers.comm_eff.spectral_filter import is_spectral_target

    for name, p in named_params:
        grad = getattr(p, "grad", None)
        if grad is None:
            continue
        if not is_spectral_target(name, p, target_substrs=target_substrs, target_scope=target_scope):
            continue
        if max_targets >= 0 and len(grads) >= max_targets:
            break
        full, _meta = full_grad_of(grad)
        if not full.is_floating_point():
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
        # initial weights. Canon-keying both sides gives the clone the real live
        # weights even before the snapshot-load.
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
