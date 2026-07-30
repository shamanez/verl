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
"""Layer-rotation GRPO: block-coordinate descent over decoder depth (issue #95).

This module is DELIBERATELY independent of the communication-efficient circuits.
It imports exactly one thing from ``comm_eff`` -- the pure, stateless helper
``find_decoder_layers`` -- and nothing else. No PowerSGD, no PRF mask, no anchor,
no ``CommEffState``. A layer-rotation run has ``comm_eff.enabled=false``.

What it provides
----------------
* ``parse_layer_schedule`` / ``schedule_from_env``: the ``LAYER_SCHEDULE`` grammar.
* ``apply_active_set``: the issue #64 ``apply_block_freeze`` generalised from a
  contiguous ``(lo, hi)`` block to an arbitrary index set, plus the ``LAYER_OTHER``
  switch for the embedding / tied-head / final-norm "root" group.
* ``RotationSchedule``: the pure cyclic visit calendar (deterministic in the
  trainer's global step, so visit accounting can be verified offline).
* Adam park / unpark helpers (``ROTATE_ADAM=persist_park``).
* Byte-accounting telemetry for the memory money gate.
* ``LayerRotationController``: the object the FSDP engine owns.

Money-gate visibility (the #64 lesson, hard requirement)
--------------------------------------------------------
Issue #64 lost a full investigation because its freeze gate logged through
``logger.info``, which the vast launchers do not capture. EVERY gate line in this
module goes through :func:`gate_print`, i.e. ``print(..., flush=True)``. This
module must never import or use a ``logging`` logger.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn

# The ONLY comm_eff import: a pure helper that locates the decoder-block
# ``nn.ModuleList`` of a (possibly FSDP-wrapped) HF causal LM. It carries no
# comm_eff state, so importing it does not couple layer rotation to any codec.
from verl.workers.comm_eff.activation_mask import find_decoder_layers

__all__ = [
    "GATE_PREFIX",
    "LayerRotationController",
    "LayerSchedule",
    "RotationSchedule",
    "apply_active_set",
    "decoder_layer_params",
    "gate_print",
    "grad_bytes",
    "one_layer_opt_bytes",
    "optimizer_state_bytes_split",
    "park_optimizer_state",
    "parse_layer_schedule",
    "root_params",
    "schedule_from_env",
    "unpark_optimizer_state",
]

# ---------------------------------------------------------------------------#
# Env knob names (issue #95 plan, "One knob, no ambiguity").                  #
# ---------------------------------------------------------------------------#
ENV_SCHEDULE = "LAYER_SCHEDULE"
ENV_ROTATE_EVERY = "ROTATE_EVERY"
ENV_ROTATE_ADAM = "ROTATE_ADAM"
ENV_STATE_DEVICE = "ROTATE_STATE_DEVICE"
ENV_LAYER_OTHER = "LAYER_OTHER"
# Issue #96. Width (gamma) = how many decoder layers are active AT ONCE, and order =
# how the cycle walks them. These exist because updates-per-layer is
# ``u = total_steps * width / len(indices)`` and ROTATE_EVERY does NOT appear in it:
# rotate_every only controls whether a layer's updates are CONSECUTIVE. Issue #95
# measured C = 0.115 + 0.165*ln(u) over u in [5.4, 150], so at width=1 a 28-layer
# rotation needs about 3300 steps to reach C=0.90, while width=4 reaches u=86 in 600.
ENV_ROTATE_WIDTH = "ROTATE_WIDTH"
ENV_ROTATE_ORDER = "ROTATE_ORDER"
ENV_ROTATE_SEED = "ROTATE_SEED"
# Escape hatch only. The shipped mechanism is decided by the laptop FSDP CPU gate
# (tests/workers/test_layer_rotation.py gate 3) and recorded in
# research/runs/95-layer-rotation-grpo/mechanism.txt.
ENV_MECHANISM = "ROTATE_MECHANISM"

#: Mechanism that ships. P1 = true post-wrap ``requires_grad`` toggle on the
#: original params AND the owning FSDP handle's ``flat_param``. P2 = grad masking
#: before the clip (``p.grad = None`` outside the active set). Set by the CPU gate.
DEFAULT_MECHANISM = "p1"

VALID_MECHANISMS = ("p1", "p2")
VALID_ADAM_POLICIES = ("persist_park", "reset")
VALID_LAYER_OTHER = ("freeze", "train")
#: ``cycle`` walks a fixed partition of ``indices`` in order (issue #95 behaviour, and
#: the default so every #95 arm reproduces bit-for-bit). ``shuffle`` re-permutes
#: ``indices`` once per full cycle with a seeded RNG and chunks the permutation, i.e.
#: uniform sampling WITHOUT replacement. That is deliberately not i.i.d. uniform: with
#: 120 selections over 28 layers, i.i.d. leaves a layer completely untrained with
#: probability 1.3% each, so about a 30% chance at least one layer never trains at all.
#: Sampling without replacement gives every layer an equal, deterministic visit count
#: for the same cost while still removing any fixed-order artefact.
VALID_ROTATE_ORDERS = ("cycle", "shuffle")

GATE_PREFIX = "[layer_rotation]"

#: fp32 Adam keeps two moments per parameter element.
ADAM_MOMENTS = 2
ADAM_MOMENT_BYTES = 4


def gate_print(msg: str) -> None:
    """Emit one money-gate line to stdout.

    ``print(..., flush=True)`` and NOTHING else. The vast launchers tee stdout
    into ``train.log`` but drop ``logger.info``; issue #64 proved that the hard
    way, so a gate line that is not visible in ``train.log`` counts as a missing
    gate.

    The prefix is written as a LITERAL here (not interpolated from ``GATE_PREFIX``)
    so CPU gate 6 can find it by static analysis and prove that every
    ``[layer_rotation]`` message leaves the process through ``print(flush=True)``.
    """
    print(f"[layer_rotation] {msg}", flush=True)


# ---------------------------------------------------------------------------#
# Schedule grammar.                                                           #
# ---------------------------------------------------------------------------#
@dataclass(frozen=True)
class LayerSchedule:
    """A resolved ``LAYER_SCHEDULE`` plus its support knobs.

    ``mode`` is one of ``dense`` / ``static`` / ``rotate``. ``indices`` is the
    resolved, sorted, 0-indexed decoder-layer set (empty for ``dense``).
    """

    mode: str
    indices: tuple[int, ...]
    spec: str = ""
    rotate_every: int = 1
    adam_policy: str = "persist_park"
    state_device: str = "cpu"
    layer_other: str = "freeze"
    mechanism: str = DEFAULT_MECHANISM
    rotate_width: int = 1
    rotate_order: str = "cycle"
    rotate_seed: int = 0

    @property
    def is_dense(self) -> bool:
        return self.mode == "dense"

    @property
    def is_rotating(self) -> bool:
        return self.mode == "rotate"

    def describe(self) -> str:
        if self.is_dense:
            return "dense (every parameter trainable)"
        return (
            f"{self.mode}:{_fmt_indices(self.indices)} | rotate_every={self.rotate_every} "
            f"| width={self.rotate_width} | order={self.rotate_order} | seed={self.rotate_seed} "
            f"| adam={self.adam_policy} | state_device={self.state_device} "
            f"| layer_other={self.layer_other} | mechanism={self.mechanism}"
        )


def _fmt_indices(indices: Sequence[int]) -> str:
    if not indices:
        return "-"
    if len(indices) == 1:
        return str(indices[0])
    contiguous = list(indices) == list(range(indices[0], indices[-1] + 1))
    return f"{indices[0]}-{indices[-1]}" if contiguous else ",".join(str(i) for i in indices)


def _parse_index_range(body: str, num_layers: int, spec: str) -> tuple[int, ...]:
    """Parse ``a`` or ``a-b`` (inclusive, 0-indexed) into a sorted index tuple."""
    body = body.strip()
    if body == "":
        raise ValueError(f"{ENV_SCHEDULE}={spec!r}: missing layer index or range after the mode prefix")
    try:
        if "-" in body:
            lo_str, hi_str = body.split("-", 1)
            lo, hi = int(lo_str.strip()), int(hi_str.strip())
        else:
            lo = hi = int(body)
    except ValueError as exc:
        raise ValueError(f"{ENV_SCHEDULE}={spec!r} is not an int or an inclusive 'lo-hi' range") from exc
    if not (0 <= lo <= hi < num_layers):
        raise ValueError(
            f"{ENV_SCHEDULE}={spec!r} -> (lo={lo}, hi={hi}) is out of range for num_layers={num_layers}; "
            f"require 0 <= lo <= hi < num_layers"
        )
    return tuple(range(lo, hi + 1))


def _validate_int(value: Any, name: str, minimum: int) -> int:
    try:
        out = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}={value!r} is not an integer") from exc
    if out < minimum:
        raise ValueError(f"{name}={value!r} must be >= {minimum}")
    return out


def _validate_choice(value: Any, name: str, choices: Sequence[str]) -> str:
    out = str(value).strip().lower()
    if out not in choices:
        raise ValueError(f"{name}={value!r} is invalid; expected one of {tuple(choices)}")
    return out


def _validate_device(value: Any, name: str) -> str:
    out = str(value).strip()
    if out == "":
        raise ValueError(f"{name} must name a torch device (e.g. 'cpu')")
    try:
        torch.device(out)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}={value!r} is not a valid torch device") from exc
    return out


def parse_layer_schedule(
    spec: Optional[str],
    num_layers: int,
    *,
    rotate_every: Any = 1,
    adam_policy: Any = "persist_park",
    state_device: Any = "cpu",
    layer_other: Any = "freeze",
    mechanism: Any = DEFAULT_MECHANISM,
    rotate_width: Any = 1,
    rotate_order: Any = "cycle",
    rotate_seed: Any = 0,
) -> LayerSchedule:
    """Resolve a ``LAYER_SCHEDULE`` spec into a :class:`LayerSchedule`.

    Grammar (0-indexed, inclusive ranges)::

        unset / ""      -> dense: every parameter trainable
        static:14       -> freeze all but decoder layer 14
        static:11-15    -> freeze all but decoder layers 11..15
        rotate:11-15    -> one active layer at a time, cyclic over 11..15
        rotate:0-27     -> one active layer at a time, cyclic over all 28

    A malformed or out-of-range spec (or a malformed support knob) raises
    ``ValueError`` at build time. A mis-parse must abort the build, never quietly
    train the wrong surface and burn the whole spend.
    """
    if num_layers is None or int(num_layers) <= 0:
        raise ValueError(f"num_layers must be a positive int, got {num_layers!r}")
    num_layers = int(num_layers)

    raw = "" if spec is None else str(spec).strip()
    every = _validate_int(rotate_every, ENV_ROTATE_EVERY, 1)
    adam = _validate_choice(adam_policy, ENV_ROTATE_ADAM, VALID_ADAM_POLICIES)
    device = _validate_device(state_device, ENV_STATE_DEVICE)
    other = _validate_choice(layer_other, ENV_LAYER_OTHER, VALID_LAYER_OTHER)
    mech = _validate_choice(mechanism, ENV_MECHANISM, VALID_MECHANISMS)
    width = _validate_int(rotate_width, ENV_ROTATE_WIDTH, 1)
    order = _validate_choice(rotate_order, ENV_ROTATE_ORDER, VALID_ROTATE_ORDERS)
    seed = _validate_int(rotate_seed, ENV_ROTATE_SEED, 0)

    if raw == "":
        return LayerSchedule(
            mode="dense",
            indices=(),
            spec="",
            rotate_every=every,
            adam_policy=adam,
            state_device=device,
            layer_other=other,
            mechanism=mech,
            rotate_width=width,
            rotate_order=order,
            rotate_seed=seed,
        )

    if ":" not in raw:
        raise ValueError(
            f"{ENV_SCHEDULE}={spec!r} is missing a mode prefix; expected 'static:<idx|lo-hi>' or "
            f"'rotate:<idx|lo-hi>' (unset/empty means dense)"
        )
    mode, body = raw.split(":", 1)
    mode = mode.strip().lower()
    if mode not in ("static", "rotate"):
        raise ValueError(f"{ENV_SCHEDULE}={spec!r}: unknown mode {mode!r}; expected 'static' or 'rotate'")
    indices = _parse_index_range(body, num_layers, raw)
    # Width is only meaningful for a rotation, and a width at or above the set size
    # would make "rotation" a static arm wearing a rotate label, which would silently
    # void the memory claim. Refuse it at build time rather than train the wrong
    # surface for 16 hours.
    if mode == "rotate" and width > len(indices):
        raise ValueError(
            f"{ENV_ROTATE_WIDTH}={width} exceeds the {len(indices)} layers in "
            f"{ENV_SCHEDULE}={spec!r}; width must be <= the rotating set size"
        )
    if mode == "static" and width != 1:
        raise ValueError(
            f"{ENV_ROTATE_WIDTH}={width} is meaningless for a static schedule "
            f"({ENV_SCHEDULE}={spec!r}); every listed layer is always active"
        )
    return LayerSchedule(
        mode=mode,
        indices=indices,
        spec=raw,
        rotate_every=every,
        adam_policy=adam,
        state_device=device,
        layer_other=other,
        mechanism=mech,
        rotate_width=width,
        rotate_order=order,
        rotate_seed=seed,
    )


def schedule_from_env(num_layers: int, env: Optional[dict] = None) -> LayerSchedule:
    """Resolve the schedule from the process environment (env reaches ray workers
    by inheritance, which issue #64 proved on this exact launcher family)."""
    src = os.environ if env is None else env
    return parse_layer_schedule(
        src.get(ENV_SCHEDULE, ""),
        num_layers,
        rotate_every=src.get(ENV_ROTATE_EVERY, 1) or 1,
        adam_policy=src.get(ENV_ROTATE_ADAM, "persist_park") or "persist_park",
        state_device=src.get(ENV_STATE_DEVICE, "cpu") or "cpu",
        layer_other=src.get(ENV_LAYER_OTHER, "freeze") or "freeze",
        mechanism=src.get(ENV_MECHANISM, DEFAULT_MECHANISM) or DEFAULT_MECHANISM,
        rotate_width=src.get(ENV_ROTATE_WIDTH, 1) or 1,
        rotate_order=src.get(ENV_ROTATE_ORDER, "cycle") or "cycle",
        rotate_seed=src.get(ENV_ROTATE_SEED, 0) or 0,
    )


# ---------------------------------------------------------------------------#
# Cyclic visit calendar (pure, so visit accounting is verifiable offline).     #
# ---------------------------------------------------------------------------#
class RotationSchedule:
    """Cyclic block-coordinate calendar over ``indices``.

    Steps are the trainer's 1-indexed ``global_steps``. Visit index is
    ``(step - 1) // rotate_every`` and the active layer is
    ``indices[visit_index % len(indices)]``, so step 1 activates ``indices[0]``.
    Pure: no hidden state, so the same step always yields the same layer and a
    repeated / replayed step is idempotent.
    """

    def __init__(
        self,
        indices: Sequence[int],
        rotate_every: int = 1,
        *,
        width: int = 1,
        order: str = "cycle",
        seed: int = 0,
    ):
        if not indices:
            raise ValueError("RotationSchedule needs at least one decoder-layer index")
        if int(rotate_every) < 1:
            raise ValueError(f"rotate_every must be >= 1, got {rotate_every!r}")
        if int(width) < 1:
            raise ValueError(f"width must be >= 1, got {width!r}")
        if int(width) > len(indices):
            raise ValueError(f"width {width} exceeds the {len(indices)} rotating layers")
        if str(order) not in VALID_ROTATE_ORDERS:
            raise ValueError(f"order must be one of {VALID_ROTATE_ORDERS}, got {order!r}")
        self.indices: tuple[int, ...] = tuple(int(i) for i in indices)
        self.rotate_every = int(rotate_every)
        self.width = int(width)
        self.order = str(order)
        self.seed = int(seed)
        # Groups per full cycle. The last group is short when width does not divide
        # the set evenly (28 / 4 = 7 exact, but 28 / 5 leaves a group of 3).
        self._n_groups = math.ceil(len(self.indices) / self.width)

    @property
    def cycle_len(self) -> int:
        """Number of GROUPS in one full cycle, i.e. selections needed to cover the set."""
        return self._n_groups

    def visit_index(self, step: int) -> int:
        return (max(int(step), 1) - 1) // self.rotate_every

    def _cycle_order(self, cycle: int) -> tuple[int, ...]:
        """The layer order used for one full cycle.

        ``cycle`` walks the natural order every time. ``shuffle`` draws a fresh
        permutation per cycle from a seeded RNG, so it is uniform WITHOUT replacement:
        every layer is visited exactly once per cycle, and the order carries no
        positional artefact. Deriving the RNG from ``(seed, cycle)`` keeps the whole
        calendar pure, so replaying a step always yields the same group and a
        resumed or re-run job reproduces bit-for-bit.
        """
        if self.order == "cycle":
            return self.indices
        # Explicit arithmetic, not tuple hashing: hash() is only guaranteed stable
        # for ints, and relying on that subtlety in a calendar that must reproduce
        # across processes and resumes is not worth the cleverness.
        rng = random.Random(self.seed * 1_000_003 + int(cycle))
        perm = list(self.indices)
        rng.shuffle(perm)
        return tuple(perm)

    def active_layers(self, step: int) -> tuple[int, ...]:
        """The ``width`` decoder layers active at ``step``, sorted."""
        v = self.visit_index(step)
        cycle, pos = divmod(v, self._n_groups)
        order = self._cycle_order(cycle)
        return tuple(sorted(order[pos * self.width : (pos + 1) * self.width]))

    def active_layer(self, step: int) -> int:
        """Back-compat single-layer view: the lowest active index at ``step``."""
        return self.active_layers(step)[0]

    def visits_of_active_layer(self, step: int) -> int:
        """How many times the current group's layers have been visited, inclusive.

        With sampling without replacement every layer is visited exactly once per
        cycle, so this is the cycle number regardless of width or order.
        """
        return self.visit_index(step) // self._n_groups + 1

    def visit_counts(self, last_step: int) -> dict[int, int]:
        """Per-layer visit counts over steps ``1..last_step`` (inclusive)."""
        counts = {i: 0 for i in self.indices}
        n_visits = self.visit_index(int(last_step)) + 1 if int(last_step) >= 1 else 0
        for v in range(n_visits):
            cycle, pos = divmod(v, self._n_groups)
            order = self._cycle_order(cycle)
            for layer in order[pos * self.width : (pos + 1) * self.width]:
                counts[layer] += 1
        return counts


# ---------------------------------------------------------------------------#
# Param bookkeeping + the generalised freeze.                                 #
# ---------------------------------------------------------------------------#
def _decoder_layers_or_raise(module: nn.Module) -> nn.ModuleList:
    layers = find_decoder_layers(module)
    if layers is None:
        raise RuntimeError(
            "layer_rotation: could not locate the decoder-layer ModuleList "
            "(find_decoder_layers returned None) -- refusing to freeze blindly"
        )
    return layers


def decoder_layer_params(module: nn.Module) -> dict[int, list[nn.Parameter]]:
    """Map decoder-layer index -> its parameter objects (deduplicated, in order).

    Keyed off module objects rather than parameter NAMES: FSDP1 rewrites names
    with ``_fsdp_wrapped_module`` prefixes, so a name filter is fragile while
    ``layers[i].parameters()`` is not.
    """
    layers = _decoder_layers_or_raise(module)
    out: dict[int, list[nn.Parameter]] = {}
    seen: set[int] = set()
    for i, layer in enumerate(layers):
        bucket: list[nn.Parameter] = []
        for p in layer.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            bucket.append(p)
        out[i] = bucket
    return out


def root_params(module: nn.Module) -> list[nn.Parameter]:
    """Every parameter that is NOT inside a decoder layer.

    On Qwen2.5-Math-1.5B (``tie_word_embeddings=true``) this is the ONE
    151936 x 1536 tied matrix that serves as both input embedding and output head,
    plus the final RMSNorm vector. ``nn.Module.parameters()`` deduplicates the tied
    pair, so the matrix appears exactly once.
    """
    inside = {id(p) for bucket in decoder_layer_params(module).values() for p in bucket}
    out: list[nn.Parameter] = []
    seen: set[int] = set()
    for p in module.parameters():
        if id(p) in inside or id(p) in seen:
            continue
        seen.add(id(p))
        out.append(p)
    return out


def tied_root_params(module: nn.Module) -> list[nn.Parameter]:
    """Root params that are SHARED, i.e. reachable under more than one name.

    On Qwen2.5-Math-1.5B (``tie_word_embeddings=true``) the 151936 x 1536 matrix is
    reachable as both ``model.embed_tokens.weight`` and ``lm_head.weight``, which is
    exactly the condition that puts it in an FSDP1 handle's ``_shared_param_infos``.
    Detected by object identity across the NON-deduplicated name list, so it holds
    for any HF causal LM and never keys on the string ``lm_head``.
    """
    counts: dict[int, int] = {}
    for _name, p in module.named_parameters(remove_duplicate=False):
        counts[id(p)] = counts.get(id(p), 0) + 1
    return [p for p in root_params(module) if counts.get(id(p), 0) > 1]


def apply_active_set(
    module: nn.Module,
    active_layers: Iterable[int],
    *,
    layer_other: str = "freeze",
    root_requires_grad: Optional[bool] = None,
) -> dict:
    """Freeze everything except decoder layers ``active_layers`` (and, optionally, root).

    The issue #64 ``apply_block_freeze`` generalised from a contiguous ``(lo, hi)``
    block to an arbitrary index set:

    1. ``requires_grad = False`` on ALL params;
    2. ``requires_grad = True`` on the params of every decoder layer in
       ``active_layers``;
    3. with ``layer_other="train"`` (the ``rotate-band5-embhead`` rider only), the
       root group -- token embeddings, the tied LM head (same tensor) and the final
       norm -- is trainable AND optimized.

    ``root_requires_grad`` (only meaningful when ``layer_other="freeze"``) sets the
    FSDP-level flag on the root group. ``None`` means AUTO, and AUTO resolves to
    ``True`` when the root group holds a TIED tensor. That is not cosmetic:

        FSDP1 with ``use_orig_params=True`` CANNOT run a frozen flat_param that owns a
        shared (tied) parameter. Measured on the CPU gate, torch 2.12: the first
        forward/backward succeeds and the SECOND raises
        ``AssertionError: as_params=True type(prim_param)=<class 'torch.Tensor'>``
        from ``FlatParamHandle._use_unsharded_views``, then
        ``NotImplementedError: Changing shared parameters is not supported yet``.
        A non-tied model, or a tied root left trainable, runs fine. The trigger is
        the frozen root unit, NOT this module's rotation logic (verified by a build
        with no rotation at all).

    So under AUTO the root group keeps ``requires_grad=True`` and is frozen the
    OTHER way instead: it is excluded from the optimizer entirely and its grads are
    dropped before the clip every step. The parameter trajectory is identical to a
    true freeze (bit-identical weights, zero optimizer state, no weight decay); the
    only cost is that a dense root gradient buffer is still allocated during the
    backward. Callers get ``root_optimized=False`` either way, and the honest number
    to report is ``optimized_params``, not the ``requires_grad`` count.

    An empty ``active_layers`` with ``layer_other="freeze"`` is refused: it would
    leave nothing to train. Returns a report dict; raises ``ValueError`` on an
    out-of-range index and ``RuntimeError`` if the decoder list cannot be found.
    """
    layer_other = _validate_choice(layer_other, ENV_LAYER_OTHER, VALID_LAYER_OTHER)
    layers = _decoder_layers_or_raise(module)
    num_layers = len(layers)
    active = tuple(sorted({int(i) for i in active_layers}))
    for i in active:
        if not (0 <= i < num_layers):
            raise ValueError(f"apply_active_set: layer index {i} out of range for {num_layers} decoder layers")
    if not active and layer_other == "freeze":
        raise ValueError("apply_active_set: empty active set with LAYER_OTHER=freeze would freeze the whole model")

    for param in module.parameters():
        param.requires_grad_(False)

    by_index = decoder_layer_params(module)
    active_names: list[str] = []
    for i in active:
        for p in by_index[i]:
            p.requires_grad_(True)
        active_names.append(f"{type(layers[i]).__name__}[{i}]")

    roots = root_params(module)
    tied = tied_root_params(module)
    if layer_other == "train":
        root_flag = True
    elif root_requires_grad is None:
        root_flag = bool(tied)  # AUTO: tying forces the FSDP1 flag on
    else:
        root_flag = bool(root_requires_grad)
    for p in roots:
        p.requires_grad_(root_flag)

    root_numel = int(sum(p.numel() for p in roots))
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    optimized = sum(p.numel() for i in active for p in by_index[i]) + (root_numel if layer_other == "train" else 0)
    return {
        "num_decoder_layers": num_layers,
        "active_layers": active,
        "active_layer_names": active_names,
        "layer_other": layer_other,
        "root_optimized": layer_other == "train",
        "root_requires_grad": root_flag,
        "root_grad_masked": layer_other == "freeze" and root_flag,
        "root_tied": bool(tied),
        "root_params": len(roots),
        "root_numel": root_numel,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_frac": (float(trainable) / float(total)) if total else 0.0,
        "optimized_params": int(optimized),
        "optimized_frac": (float(optimized) / float(total)) if total else 0.0,
    }


# ---------------------------------------------------------------------------#
# FSDP handle plumbing (mechanism P1).                                        #
# ---------------------------------------------------------------------------#
def _flat_params_of(fsdp_module: nn.Module) -> list[torch.Tensor]:
    """Every ``flat_param`` owned directly by ``fsdp_module`` (FSDP1) if any.

    Returns ``[]`` for a plain ``nn.Module`` (no FSDP wrap), which is what the
    non-distributed unit tests and the ``use_orig_params=false`` path see.
    """
    out: list[torch.Tensor] = []
    handle = getattr(fsdp_module, "_handle", None)
    handles = getattr(fsdp_module, "_handles", None) or ([] if handle is None else [handle])
    for h in handles:
        fp = getattr(h, "flat_param", None)
        if fp is not None:
            out.append(fp)
    if not out:
        fp = getattr(fsdp_module, "_flat_param", None)
        if fp is not None:
            out.append(fp)
    return out


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Local shard of a possibly-DTensor tensor."""
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        try:
            return to_local()
        except Exception:
            return tensor
    return tensor


def _local_numel(tensor: torch.Tensor) -> int:
    return int(_local(tensor).numel())


# ---------------------------------------------------------------------------#
# Adam park / unpark + byte accounting.                                       #
# ---------------------------------------------------------------------------#
def _state_tensors(state: dict) -> list[torch.Tensor]:
    return [v for v in state.values() if torch.is_tensor(v)]


def park_optimizer_state(optimizer, params: Iterable[nn.Parameter], device: str = "cpu") -> int:
    """Move the Adam moments of ``params`` to ``device``; return the bytes moved.

    Torch creates Adam state lazily on the first step in which a param has a
    grad, so a layer that has never been visited holds nothing and this is a
    no-op for it. Parked params are never stepped (their grad is None under both
    mechanisms), so the foreach AdamW path never touches a parked tensor.
    """
    target = torch.device(device)
    moved = 0
    for p in params:
        st = optimizer.state.get(p)
        if not st:
            continue
        for key, value in list(st.items()):
            if torch.is_tensor(value) and value.device != target:
                st[key] = value.to(target, non_blocking=False)
                moved += st[key].numel() * st[key].element_size()
    return moved


def unpark_optimizer_state(optimizer, params: Iterable[nn.Parameter], device) -> int:
    """Move the Adam moments of ``params`` back onto ``device``; return bytes moved.

    ``ROTATE_ADAM=persist_park`` keeps Adam bias correction continuous across
    visits, which is the true block-coordinate-descent analogue.
    """
    target = torch.device(device)
    moved = 0
    for p in params:
        st = optimizer.state.get(p)
        if not st:
            continue
        for key, value in list(st.items()):
            if torch.is_tensor(value) and value.device != target:
                st[key] = value.to(target, non_blocking=False)
                moved += st[key].numel() * st[key].element_size()
    return moved


def reset_optimizer_state(optimizer, params: Iterable[nn.Parameter]) -> int:
    """Drop the Adam state of ``params`` entirely (``ROTATE_ADAM=reset``)."""
    dropped = 0
    for p in params:
        st = optimizer.state.pop(p, None)
        if st:
            dropped += sum(t.numel() * t.element_size() for t in _state_tensors(st))
    return dropped


def optimizer_state_bytes_split(
    optimizer,
    parked_params: Iterable[nn.Parameter],
    *,
    park_device: str = "cpu",
    compute_device: Optional[str] = None,
) -> tuple[int, int]:
    """Split optimizer-state bytes into (compute-resident, parked).

    Residency is the park bookkeeping, which is the ground truth of
    :func:`park_optimizer_state`. When the compute device genuinely differs from
    the park device (a GPU box) the bookkeeping is CROSS-CHECKED against the real
    tensor devices and a mismatch RAISES, so the accounting cannot silently drift
    from reality where it matters. On a CPU-only host the two devices coincide and
    the device cross-check is vacuous, which is why the laptop gate additionally
    checks the analytic byte count.
    """
    park = torch.device(park_device)
    parked_ids = {id(p) for p in parked_params}
    cross_check = compute_device is not None and torch.device(compute_device).type != park.type

    resident = 0
    parked = 0
    for p, st in optimizer.state.items():
        tensors = _state_tensors(st)
        if not tensors:
            continue
        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        is_parked = id(p) in parked_ids
        if cross_check:
            on_park = all(t.device.type == park.type for t in tensors)
            if is_parked != on_park:
                raise RuntimeError(
                    f"layer_rotation: park bookkeeping disagrees with tensor devices "
                    f"(bookkeeping parked={is_parked}, tensors on {park.type}={on_park}); "
                    f"refusing to report a memory number we cannot trust"
                )
        if is_parked:
            parked += nbytes
        else:
            resident += nbytes
    return resident, parked


def grad_bytes(module: nn.Module) -> int:
    """Bytes held by gradients on ``module`` right now (params carrying a grad)."""
    total = 0
    seen: set[int] = set()
    for p in module.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        g = p.grad
        if g is None:
            continue
        gl = _local(g)
        total += gl.numel() * gl.element_size()
    return total


def one_layer_opt_bytes(module: nn.Module, layer_index: int) -> int:
    """Analytic Adam-state size of ONE decoder layer on this rank.

    ``2 moments x 4 bytes x param count``. This is the DENOMINATOR of the memory
    money gate, logged online so the ratio needs no offline guessing.
    """
    by_index = decoder_layer_params(module)
    if layer_index not in by_index:
        raise ValueError(f"one_layer_opt_bytes: layer {layer_index} not found ({len(by_index)} decoder layers)")
    numel = sum(_local_numel(p) for p in by_index[layer_index])
    return ADAM_MOMENTS * ADAM_MOMENT_BYTES * numel


# ---------------------------------------------------------------------------#
# The controller the FSDP engine owns.                                        #
# ---------------------------------------------------------------------------#
@dataclass
class _Immutability:
    fingerprint: Optional[dict] = None
    checks: int = 0
    max_checks: int = 3


class LayerRotationController:
    """Owns the active set, the rotation clock, Adam parking and the money gates.

    Lifecycle, mirroring the engine's build order:

    1. ``apply_pre_wrap(raw_module)`` -- BEFORE ``_build_fsdp_module``. Static arms
       get the full issue #64 freeze here (uniform ``requires_grad`` per FSDP unit,
       so the optimizer only ever receives trainable params). Rotating arms get the
       FIRST active layer here, so the wrap and the optimizer are built consistent
       with step 1.
    2. ``bind(wrapped_module, optimizer)`` -- AFTER the wrap and the optimizer build.
    3. ``advance(step)`` -- at the TOP of ``update_actor``, before any optimizer
       tick of that global step, so every tick inside one step trains one layer.
    4. ``pre_step()`` / ``post_step()`` -- inside ``optimizer_step``, around the
       update. ``pre_step`` runs BEFORE the grad clip, so the reported grad norm is
       the norm over the ACTIVE parameters only in every arm.
    """

    def __init__(self, schedule: LayerSchedule, num_decoder_layers: int, *, rank: int = 0):
        self.schedule = schedule
        self.num_decoder_layers = int(num_decoder_layers)
        self.rank = int(rank)
        self.mechanism = schedule.mechanism
        self.rotation: Optional[RotationSchedule] = (
            RotationSchedule(
                schedule.indices,
                schedule.rotate_every,
                width=schedule.rotate_width,
                order=schedule.rotate_order,
                seed=schedule.rotate_seed,
            )
            if schedule.is_rotating
            else None
        )
        # Seed with step 1's GROUP (not indices[:1]): under width > 1 or a shuffled
        # order the first group is neither a single layer nor necessarily the lowest,
        # and the pre-wrap active set must match step 1 exactly or the FSDP wrap is
        # built against a different surface than the first optimizer tick trains.
        self.active: tuple[int, ...] = (
            self.rotation.active_layers(1) if self.rotation is not None else tuple(schedule.indices)
        )
        self.step: int = 1
        self.visit_index: int = 0
        self.visits_of_active: int = 1
        self.rotations: int = 0
        self.module: Optional[nn.Module] = None
        self.optimizer = None
        self._by_index: dict[int, list[nn.Parameter]] = {}
        self._roots: list[nn.Parameter] = []
        self._visited: set[int] = set(self.active)
        self._parked: set[int] = set()
        self._pre_wrap_report: Optional[dict] = None
        self._root_grad_masked = False
        self._grad_checked = False
        self._immut = _Immutability()
        self._grad_bytes_at_step = 0
        self._park_bytes_moved = 0
        self._unpark_bytes_moved = 0
        self._compute_device: Optional[str] = None
        self._telemetry_extra: dict[str, float] = {}

    # -- properties ------------------------------------------------------- #
    @property
    def active_layer(self) -> int:
        return self.active[0] if len(self.active) == 1 else -1

    @property
    def is_rotating(self) -> bool:
        return self.rotation is not None

    @property
    def root_trainable(self) -> bool:
        return self.schedule.layer_other == "train"

    # -- stage 1: pre-wrap ------------------------------------------------ #
    def apply_pre_wrap(self, module: nn.Module) -> dict:
        """Apply the initial active set to the RAW (pre-FSDP) module.

        Static arms: exactly the issue #64 mechanism, so each FSDP unit is built
        with uniform ``requires_grad`` and both the optimizer-state and the
        gradient-buffer savings are real.

        Rotating arms under P1: the same, seeded with ``indices[0]`` (step 1's
        layer), so the wrap is consistent with the first step and the runtime
        toggle only ever moves between whole units.

        Rotating arms under P2: the build path stays byte-identical to dense for
        the decoder layers (they all keep ``requires_grad=True``); only the root
        group is frozen when ``LAYER_OTHER=freeze``, which is static and therefore
        safe to do pre-wrap. Masking happens at ``pre_step``.
        """
        if self.schedule.is_dense:
            raise RuntimeError("apply_pre_wrap called for a dense schedule")

        if self.is_rotating and self.mechanism == "p2":
            # P2: keep every decoder layer trainable at build; the active set is
            # enforced by grad masking at pre_step.
            report = apply_active_set(module, range(self.num_decoder_layers), layer_other=self.schedule.layer_other)
            report["p2_build"] = "all decoder layers trainable at build; grad masking at pre_step"
        else:
            report = apply_active_set(module, self.active, layer_other=self.schedule.layer_other)

        self._pre_wrap_report = report
        # Under LAYER_OTHER=freeze with a TIED root tensor, the root group keeps
        # requires_grad=True (FSDP1 cannot run a frozen flat_param that owns a shared
        # param) and is frozen by exclusion from the optimizer plus a grad mask. Record
        # it so pre_step masks it and so post_step watches it for immobility.
        self._root_grad_masked = bool(report["root_grad_masked"])
        frac = report["optimized_frac"]
        if not (0.0 < frac <= 1.0):
            raise RuntimeError(f"{GATE_PREFIX} freeze sanity FAILED: optimized_frac={frac:.6f}; report={report}")
        if self.rank == 0:
            gate_print(f"resolved schedule: {self.schedule.describe()}")
            gate_print(
                f"active set: layers={list(self.active)} of {report['num_decoder_layers']} decoder layers "
                f"| modules={report['active_layer_names']} | layer_other={report['layer_other']} "
                f"(root_optimized={report['root_optimized']}, root_tied={report['root_tied']}, "
                f"root_requires_grad={report['root_requires_grad']}, root_grad_masked={report['root_grad_masked']}, "
                f"root_numel={report['root_numel']})"
            )
            gate_print(
                f"trainable={report['optimized_params']}/{report['total_params']} ({frac:.6f}) at build "
                f"(pre-wrap, mechanism={self.mechanism}; requires_grad numel="
                f"{report['trainable_params']} incl. the grad-masked root group)"
            )
            if report["root_grad_masked"]:
                gate_print(
                    "root group (tied embedding + final norm) keeps requires_grad=True because FSDP1 "
                    "use_orig_params cannot run a frozen flat_param that owns a TIED tensor; it is frozen "
                    "instead by exclusion from the optimizer + a grad mask before the clip, so its weights "
                    "are bit-identical and it holds ZERO optimizer state"
                )
        return report

    # -- stage 2: post-wrap ----------------------------------------------- #
    def build_param_groups(self, module: nn.Module) -> list[dict]:
        """One param group per decoder layer plus one ``other`` group (ROTATING arms).

        Identical lr / betas / weight decay in every group (they are the
        optimizer's own defaults); the split exists purely so a rotation can move
        the active set without rebuilding the optimizer. Every DECODER param is
        handed over, including currently-inactive ones: torch never allocates state
        for a param whose grad stays None, so this costs nothing.

        The ``other`` group carries the root params ONLY for the rider
        (``LAYER_OTHER=train``). Under ``LAYER_OTHER=freeze`` it is deliberately
        EMPTY -- the optimizer must never see the tied embedding, so it cannot step
        it or decay it even if a grad mask were ever skipped. The empty group is
        still appended so the param-group structure is identical across both knob
        settings.
        """
        by_index = decoder_layer_params(module)
        groups: list[dict] = []
        for i in sorted(by_index):
            groups.append({"params": by_index[i], "layer_rotation_group": f"layer{i}", "layer_rotation_index": i})
        others = root_params(module) if self.schedule.layer_other == "train" else []
        groups.append({"params": others, "layer_rotation_group": "other", "layer_rotation_index": -1})
        return groups

    def optimizer_input(self, module: nn.Module):
        """What ``_build_optimizer`` should hand to ``build_optimizer``.

        Rotating arms get per-layer param GROUPS. Static arms get the issue #64
        trainable-only LIST, minus any grad-masked root param, so the optimizer
        never sees a parameter that must not move.
        """
        if self.is_rotating:
            return self.build_param_groups(module)
        masked = {id(p) for p in (root_params(module) if self.schedule.layer_other != "train" else [])}
        return [p for p in module.parameters() if p.requires_grad and id(p) not in masked]

    def bind(self, module: nn.Module, optimizer=None, *, compute_device: Optional[str] = None) -> None:
        self.module = module
        self.optimizer = optimizer
        self._by_index = decoder_layer_params(module)
        self._roots = root_params(module)
        self._compute_device = compute_device
        if len(self._by_index) != self.num_decoder_layers:
            raise RuntimeError(
                f"{GATE_PREFIX} bind FAILED: found {len(self._by_index)} decoder layers post-wrap, "
                f"expected {self.num_decoder_layers}"
            )
        if self.rank == 0:
            optimized = sum(p.numel() for p in self.optimized_params())
            total = sum(p.numel() for p in module.parameters())
            gate_print(
                f"bound post-wrap: trainable={optimized}/{total} "
                f"({(optimized / total) if total else 0.0:.6f}) | active={list(self.active)} "
                f"| one_layer_opt_bytes={self.one_layer_opt_bytes()}"
            )

    def optimized_params(self) -> list[nn.Parameter]:
        """The params the optimizer can actually update this step.

        This is the honest denominator for every "trainable" number: the active
        decoder layer plus, for the rider only, the root group. A grad-masked root
        group is NOT counted even though its ``requires_grad`` is True, because it
        cannot move.
        """
        out = list(self.params_of(self.active))
        if self.root_trainable:
            out.extend(self._roots)
        return out

    # -- stage 3: rotation ------------------------------------------------ #
    def params_of(self, layers: Iterable[int]) -> list[nn.Parameter]:
        out: list[nn.Parameter] = []
        for i in layers:
            out.extend(self._by_index.get(int(i), ()))
        return out

    def _set_requires_grad(self, layers: Iterable[int], flag: bool) -> None:
        """P1: toggle ``requires_grad`` on the ORIGINAL params AND on the owning
        FSDP handle's ``flat_param``.

        Both halves matter. Without the ``flat_param`` half, FSDP1 still registers a
        post-backward hook for the unit and still allocates a grad shard for it, so
        the memory claim would be false even though the parameter trajectory looked
        right.
        """
        layers = [int(i) for i in layers]
        decoder = _decoder_layers_or_raise(self.module)
        for i in layers:
            for p in self._by_index.get(i, ()):
                p.requires_grad_(flag)
                if not flag:
                    p.grad = None
            for fp in _flat_params_of(decoder[i]):
                fp.requires_grad_(flag)
                if not flag:
                    fp.grad = None

    def advance(self, step: int) -> Optional[dict]:
        """Advance the cyclic schedule to the trainer's ``step``.

        Called at the TOP of ``update_actor``, before ``train_mini_batch``, so
        every optimizer tick inside one global step trains the same layer. Pure in
        ``step``: replaying the same step is a no-op.
        """
        if self.rotation is None:
            return None
        step = int(step)
        new_visit = self.rotation.visit_index(step)
        new_active = self.rotation.active_layers(step)
        self.step = step
        self.visits_of_active = self.rotation.visits_of_active_layer(step)
        if new_visit == self.visit_index and new_active == self.active:
            return None

        outgoing = tuple(self.active)
        self.visit_index = new_visit
        self.active = new_active
        self.rotations += 1
        self._visited.update(new_active)

        # Set differences, not "everything except the one new layer". Under a shuffled
        # order consecutive groups can overlap, and toggling a layer that stays active
        # off and on again would drop its grads and needlessly park then unpark its
        # Adam state in the same rotation.
        stay = set(outgoing) & set(new_active)
        leaving = [i for i in outgoing if i not in stay]
        entering = [i for i in new_active if i not in stay]

        if self.module is not None and self.mechanism == "p1":
            self._set_requires_grad(leaving, False)
            self._set_requires_grad(entering, True)

        park_moved = unparked = 0
        if self.optimizer is not None:
            if self.schedule.adam_policy == "reset":
                # ROTATE_ADAM=reset: drop the outgoing layers' moments outright, so
                # nothing is parked and nothing survives the visit.
                outgoing_params = self.params_of(leaving)
                reset_optimizer_state(self.optimizer, outgoing_params)
                for p in outgoing_params:
                    self._parked.discard(id(p))
            else:
                park_moved = park_optimizer_state(self.optimizer, self.params_of(leaving), self.schedule.state_device)
                for p in self.params_of(leaving):
                    if self.optimizer.state.get(p):
                        self._parked.add(id(p))
                entering_params = self.params_of(entering)
                if entering_params:
                    target = self._compute_device or _infer_param_device(entering_params)
                    unparked = unpark_optimizer_state(self.optimizer, entering_params, target)
                    for p in entering_params:
                        self._parked.discard(id(p))
        self._park_bytes_moved = park_moved
        self._unpark_bytes_moved = unparked

        if self.rank == 0:
            gate_print(
                f"rotation #{self.rotations} at step {step}: {list(outgoing)} -> {list(self.active)} "
                f"| visit_index={self.visit_index} | visits_of_active={self.visits_of_active} "
                f"| adam={self.schedule.adam_policy} parked={park_moved}B unparked={unparked}B"
            )
        return {
            "step": step,
            "outgoing": outgoing,
            "active": self.active,
            "visit_index": self.visit_index,
            "parked_bytes_moved": park_moved,
            "unparked_bytes_moved": unparked,
        }

    # -- stage 4: around the optimizer update ----------------------------- #
    def _active_param_ids(self) -> set[int]:
        return {id(p) for p in self.optimized_params()}

    def mask_inactive_grads(self, *, decoder: bool = True) -> int:
        """Drop gradients outside the active set BEFORE the clip.

        ``torch.optim.AdamW`` skips a param whose grad is None (including its
        decoupled weight decay) and allocates no state for it, so lazy state and
        the optimizer-state saving are exact.

        ``decoder=False`` masks ONLY the grad-masked root group, which is what P1
        needs: P1 already froze the inactive decoder layers via ``requires_grad``, so
        they carry no grad, but the root group is deliberately left
        ``requires_grad=True`` for FSDP1's tied-param path and must be masked here.

        Grad buffers stay dense-sized for whatever is masked: that is the
        mechanism's stated reach, not a bug. Every cell of issue #95 runs a single
        GPU, i.e. FSDP world size 1, i.e. ``NO_SHARD``, where
        ``FSDP.clip_grad_norm_`` delegates to ``torch.nn.utils.clip_grad_norm_``
        over the original params and therefore also sees the mask. Under a sharded
        strategy the flat-param grad would still enter the clip norm; the update
        itself stays exact because AdamW keys off the original params.
        """
        keep = self._active_param_ids()
        pool = self.module.parameters() if decoder else self._roots
        dropped = 0
        for p in pool:
            if id(p) in keep or p.grad is None:
                continue
            p.grad = None
            dropped += 1
        return dropped

    def grad_flow_assert(self) -> None:
        """After the first backward: grads reach the active set and nothing else."""
        leaked: list[str] = []
        n_active_with_grad = 0
        keep = self._active_param_ids()
        for name, p in self.module.named_parameters():
            g = p.grad
            if id(p) in keep:
                if g is not None:
                    n_active_with_grad += 1
                continue
            if g is None:
                continue
            gl = _local(g)
            try:
                nonzero = bool(torch.count_nonzero(gl).item() > 0)
            except Exception:
                nonzero = True  # cannot prove it is zero => treat as a leak (fail-safe)
            if nonzero:
                leaked.append(name)
        if leaked:
            raise RuntimeError(
                f"{GATE_PREFIX} grad-flow FAILED: {len(leaked)} INACTIVE params carry nonzero grads "
                f"(the active set leaked) -- e.g. {leaked[:5]}"
            )
        if n_active_with_grad == 0:
            raise RuntimeError(
                f"{GATE_PREFIX} grad-flow FAILED: NO active param received a gradient "
                f"(layers {list(self.active)} are not training)"
            )
        if self.root_trainable:
            root_with_grad = sum(1 for p in self._roots if p.grad is not None)
            if root_with_grad == 0:
                raise RuntimeError(
                    f"{GATE_PREFIX} rider anti-clobber FAILED: LAYER_OTHER=train but no root param "
                    f"(tied embedding / final norm) carries a gradient"
                )
        if self.rank == 0:
            gate_print(
                f"grad-flow OK (step {self.step}): 0 inactive params with grad, "
                f"{n_active_with_grad} active param tensors with grad"
            )

    def rider_assert(self) -> None:
        """Anti-clobber (the fake-null trap): rotation must never touch the root group."""
        if not self.root_trainable:
            return
        frozen = [
            name
            for name, p in self.module.named_parameters()
            if any(p is r for r in self._roots) and not p.requires_grad
        ]
        if frozen:
            raise RuntimeError(
                f"{GATE_PREFIX} rider anti-clobber FAILED at step {self.step}: root params lost "
                f"requires_grad -- e.g. {frozen[:3]}"
            )

    def _l2sq(self, tensor: torch.Tensor) -> tuple[float, int]:
        t = _local(tensor.detach())
        return float(t.float().pow(2).sum().item()), int(t.numel())

    def _sampled_immutable(self) -> list[tuple[str, nn.Parameter]]:
        """A small, deterministic sample of params that must NEVER move.

        Under a rotating schedule "frozen" is a moving target, so the sample is
        restricted to layers that the schedule can never activate, plus the root
        group when ``LAYER_OTHER=freeze``. Those are immutable for the whole run,
        which makes the fingerprint check meaningful rather than flaky.
        """
        never_active = [i for i in range(self.num_decoder_layers) if i not in set(self.schedule.indices)]
        watch: list[nn.Parameter] = []
        for i in never_active[:2]:
            watch.extend(self._by_index.get(i, ())[:2])
        if not self.root_trainable:
            watch.extend(self._roots[:2])
        ids = {id(p) for p in watch}
        return [(name, p) for name, p in self.module.named_parameters() if id(p) in ids]

    def pre_step(self) -> None:
        """Before the clip and the update: mask (P2), assert, fingerprint, measure."""
        if self.module is None:
            return
        if self.is_rotating and self.mechanism == "p2":
            self.mask_inactive_grads(decoder=True)
        elif self._root_grad_masked:
            # P1 and the static arms: only the tied root group needs a mask.
            self.mask_inactive_grads(decoder=False)
        self.rider_assert()
        self._grad_bytes_at_step = grad_bytes(self.module)
        if not self._grad_checked:
            self.grad_flow_assert()
            self._grad_checked = True
        if self._immut.fingerprint is None:
            self._immut.fingerprint = {name: self._l2sq(p.data) for name, p in self._sampled_immutable()}

    def post_step(self) -> None:
        """After the update: sampled never-active params are bit-stable."""
        if self.module is None or self._immut.fingerprint is None:
            return
        if self._immut.checks >= self._immut.max_checks:
            return
        self._immut.checks += 1
        drifted = []
        for name, p in self._sampled_immutable():
            base = self._immut.fingerprint.get(name)
            if base is None:
                continue
            cur = self._l2sq(p.data)
            if abs(cur[0] - base[0]) > 1e-6 * (abs(base[0]) + 1.0):
                drifted.append((name, base[0], cur[0]))
        if drifted:
            raise RuntimeError(
                f"{GATE_PREFIX} immutability FAILED after update {self._immut.checks}: "
                f"{len(drifted)} never-active tensors changed -- e.g. {drifted[:3]}"
            )
        if self.rank == 0:
            gate_print(
                f"immutability OK after update {self._immut.checks}: "
                f"{len(self._immut.fingerprint)} sampled never-active tensors unchanged"
            )

    # -- telemetry -------------------------------------------------------- #
    def one_layer_opt_bytes(self) -> int:
        if self.module is None:
            return 0
        index = self.active[0] if self.active else next(iter(sorted(self._by_index)))
        return one_layer_opt_bytes(self.module, index)

    def parked_params(self) -> list[nn.Parameter]:
        out: list[nn.Parameter] = []
        for bucket in self._by_index.values():
            out.extend(p for p in bucket if id(p) in self._parked)
        return out

    def telemetry(self) -> dict[str, float]:
        """The ``layer_rotation/*`` keys, merged into ``final_metrics`` next to
        ``perf/max_memory_allocated_gb`` so they ride the normal metric path."""
        if self.module is None:
            return {}
        resident = parked = 0
        if self.optimizer is not None:
            resident, parked = optimizer_state_bytes_split(
                self.optimizer,
                self.parked_params(),
                park_device=self.schedule.state_device,
                compute_device=self._compute_device,
            )
        total = sum(p.numel() for p in self.module.parameters())
        # The honest number: what the optimizer can actually update. A grad-masked
        # root group has requires_grad=True but cannot move, so counting it would
        # report 0.18 for rotate-band5 instead of the true 0.030.
        trainable = sum(p.numel() for p in self.optimized_params())
        out = {
            "layer_rotation/active_layer": float(self.active_layer),
            # active_layer is -1 whenever more than one layer is active, which is
            # EVERY step once width > 1, so it cannot carry the visit accounting on
            # its own. These two do: first + count identify the group, and summing
            # active_layers_n over steps recovers total layer-updates.
            "layer_rotation/active_layer_first": float(self.active[0]) if self.active else -1.0,
            "layer_rotation/active_layers_n": float(len(self.active)),
            "layer_rotation/width": float(self.schedule.rotate_width),
            "layer_rotation/visit_index": float(self.visit_index),
            "layer_rotation/visits_of_active_layer": float(self.visits_of_active),
            "layer_rotation/opt_state_bytes_gpu": float(resident),
            "layer_rotation/opt_state_bytes_cpu": float(parked),
            "layer_rotation/grad_bytes_gpu": float(self._grad_bytes_at_step),
            "layer_rotation/one_layer_opt_bytes": float(self.one_layer_opt_bytes()),
            "layer_rotation/trainable_params": float(trainable),
            "layer_rotation/trainable_frac": float(trainable) / float(total) if total else 0.0,
        }
        out.update(self._telemetry_extra)
        return out


def _infer_param_device(params: Sequence[nn.Parameter]) -> str:
    for p in params:
        return str(_local(p.data).device)
    return "cpu"


# ---------------------------------------------------------------------------#
# Engine-side entry point.                                                    #
# ---------------------------------------------------------------------------#
def build_controller(
    module: nn.Module,
    *,
    rank: int = 0,
    env: Optional[dict] = None,
    forward_only: bool = False,
) -> Optional[LayerRotationController]:
    """Resolve ``LAYER_SCHEDULE`` and apply the pre-wrap active set.

    Returns ``None`` for the dense path (``LAYER_SCHEDULE`` unset/empty) and for
    ``forward_only`` engines (the reference and rollout engines are never frozen,
    so their full-model forward is unchanged). Called on the RAW module, AFTER
    ``_build_module()`` and BEFORE ``_build_fsdp_module()``.

    A non-empty ``LAYER_SCHEDULE`` that fails to resolve RAISES: a rotating cell
    whose schedule did not reach the worker must abort the build rather than
    quietly train dense and mislabel the whole spend.
    """
    if forward_only:
        return None
    layers = find_decoder_layers(module)
    if layers is None:
        raw = (env or os.environ).get(ENV_SCHEDULE, "")
        if str(raw).strip() == "":
            return None
        raise RuntimeError(
            f"{GATE_PREFIX} build FAILED: {ENV_SCHEDULE}={raw!r} was requested but the decoder-layer "
            f"ModuleList could not be located"
        )
    schedule = schedule_from_env(len(layers), env=env)
    if schedule.is_dense:
        # Off-path parity: nothing frozen => every param trainable => byte-identical
        # to the dense control. Assert it, and say so loudly, so a schedule that
        # failed to reach this worker is impossible to miss in train.log.
        if not all(p.requires_grad for p in module.parameters()):
            raise RuntimeError(
                f"{GATE_PREFIX} off-path parity FAILED: {ENV_SCHEDULE} is unset but some param has requires_grad=False"
            )
        if rank == 0:
            gate_print(
                f"{ENV_SCHEDULE} unset/empty on a TRAINING engine -- every param is trainable (DENSE). "
                f"If a layer schedule was intended, the env var did NOT reach this worker; STOP before "
                f"spending on a mislabeled dense run."
            )
        return None
    controller = LayerRotationController(schedule, len(layers), rank=rank)
    controller.apply_pre_wrap(module)
    return controller
