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

"""Asynchronous unmasked anchor circuit (M2, the *second circuit*) — EXP-8.

The anchor circuit produces a CLEAN per-target gradient ``G_anchor`` that the
spectral filter (EXP-7) consumes into its anchor-gradient EMA ``M_anchor``.
"Clean" means three things, all of which are load-bearing falsifiers
(``tests/workers/comm_eff/test_anchor_queue.py`` and the on-box counters):

* **Same loss as the fast path, UNMASKED.** The anchor reuses the GRPO
  actor-loss over the rollout-expanded batch (``responses``, ``response_mask``,
  ``old_log_probs``, ``advantages``, optional ``ref_log_prob``) — exactly the
  fast path's ``ppo_loss`` — but with the activation masker DISABLED even though
  it runs on the actor-train path (GUARD 5: ``mask_active=False`` ⇒
  ``anchor_mask_applications == 0``). It is NOT a supervised next-token loss; it
  does NOT generate rollouts; it does NOT recompute rewards.

* **K-stale snapshot, no optimizer step.** The anchor forwards from a
  ``delay_K``-stale weight snapshot taken OFF the optimizer's parameter group
  (so the optimizer never sees it and no accidental step occurs), and takes NO
  ``optimizer.step()`` of its own (``anchor_optimizer_steps == 0``).

* **Raw gradient into the EMA, before any correction.** ``G_anchor`` is read
  RAW per target and fed to ``SpectralFilter.update_anchor`` BEFORE any
  ``correct_matrix`` call (GUARD 6: ``anchor_grad_corrected == 0``). The anchor
  gradient is never the input to the spectral projection.

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
    "snapshot_named_params",
    "extract_target_grads",
    "feed_anchor_grads_into_ema",
    "anchor_should_fire",
    "build_anchor_module",
    "assert_anchor_module_isolated",
]


def anchor_should_fire(step: int, cadence: int, enabled: bool) -> bool:
    """True iff the anchor refresh fires on trainer ``step``.

    Pure predicate (no side effects) so the cadence policy is unit-testable.
    ``step`` is 1-based (the engine advances it before the check). The anchor
    fires when ``enabled`` and ``(step % cadence) == 0`` — so ``cadence=1`` fires
    every step (smoke), ``cadence=20`` fires on steps 20, 40, ... (paper).
    """
    if not enabled or cadence < 1:
        return False
    return (step % cadence) == 0


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


def snapshot_named_params(
    named_params,
    *,
    target_substrs=None,
    device: Optional[torch.device] = None,
    detach: bool = True,
) -> dict:
    """Detached clones of the model's named parameters (the anchor snapshot).

    CRITICAL (criterion 7): the returned tensors are plain detached clones that
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
    raw full 2D gradients exactly as backward produced them. This is the GUARD-6
    seam: the engine feeds these straight into ``SpectralFilter.update_anchor``
    (the EMA) before any ``correct_matrix`` is ever called on the fast path.

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
    """EXP-12 fix — return a deep-cloned ``nn.Module`` for the anchor's backward
    that is **fully detached from any FSDP wrapping / post-backward hooks**
    registered on the live actor module.

    Why this exists (the EXP-8 collision):
        EXP-8 ran the anchor's ``loss.backward()`` on the FSDP1-wrapped actor.
        FSDP1's ``_post_backward_hook`` calls
        ``_check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)``
        on the registered ``FlatParameter``s. ``_saved_grad_shard`` is only set
        up between the fast-path's pre-backward and the optimizer step, so the
        anchor backward (outside that window) hit
        ``AttributeError: 'NoneType' object has no attribute 'shape'``. The fix
        is to break the autograd-hook chain entirely — the anchor pass must NOT
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
        # EXP-12 iter02: verl/HF monkey-patches install function attributes
        # holding Python module references on the model class, which are not
        # picklable. deepcopy uses pickle reductors, so it fails. Fall back
        # to config-rebuild + state_dict load: structurally identical clone,
        # no monkey-patched closures travel through the rebuild.
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
        # EXP-12 iter03: state_dict() under FSDP1+use_orig_params returns DTensor
        # entries (even inside summon_full_params), but the freshly-instantiated
        # clone has plain torch.Tensor params. load_state_dict()'s copy_ guard
        # rejects mixed DTensor/Tensor combos with
        # "aten.copy_.default got mixed torch.Tensor and DTensor". Bypass
        # load_state_dict and copy each param/buffer manually with DTensor->
        # plain-Tensor materialization via .full_tensor() / .to_local().
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
        with _torch.no_grad():
            src_params = dict(inner_module.named_parameters())
            for n, p_dst in clone.named_parameters():
                if n in src_params:
                    s = _to_plain(src_params[n].detach())
                    p_dst.data.copy_(s.to(p_dst.device, p_dst.dtype))
            src_buffers = dict(inner_module.named_buffers())
            for n, b_dst in clone.named_buffers():
                if n in src_buffers:
                    s = _to_plain(src_buffers[n])
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

    Belt-and-braces (next_actions §anchor_optimizer_param_group + EXP-12
    criterion 13): the clone's parameters must NOT appear in any optimizer
    param_group (criterion 7) and must NOT be registered with any FSDP
    ``_handles`` / ``_fsdp_wrapped_module`` instance on the live actor
    (criterion 13). Cheap to run; called from
    ``_maybe_comm_eff_anchor_refresh`` once per refresh as a runtime guard.

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

    GUARD 6: this calls ``spectral.update_anchor`` (the RAW EMA blend
    ``M_anchor <- beta*M_anchor + (1-beta)*G_anchor``) — NEVER ``correct_matrix``
    — so the anchor gradient is never spectrally corrected. Returns
    ``{name: ||ΔM_anchor||}`` (Frobenius norm of the EMA change this refresh) so
    the engine can log EMA evolution (``||ΔM_anchor|| > 0`` across >= 2
    refreshes is the criterion-3 falsifier).

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
