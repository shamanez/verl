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

"""Look-ahead (linear weight extrapolation) anchor projector — M4.

The anchor circuit computes a CLEAN per-target gradient ``G_anchor`` at a
``delay_K``-stale weight point ``theta[t-K]``. At high latency (cadence/delay_K
= 20/20) that stale point has rotated ~orthogonal to the live weights (the
"k-collapse"). This module replaces the stale weight point fed to the anchor
backward with a LINEARLY EXTRAPOLATED look-ahead point::

    theta_hat[t] = a1 * theta[t-K] + a2 * theta[t-2K] + a3 * theta[t-3K]

The fixed-linear seed (AsyncPP / arXiv:2505.01099) is ``[a1=2, a2=-1, a3=0]``::

    theta_hat[t] = 2 * theta[t-K] - theta[t-2K]

i.e. a one-step linear (Nesterov-style) weight extrapolation along the recent
training trajectory. The RLVR-linearity paper (arXiv:2601.04537) licenses this:
RLVR weights move ~linearly (R^2 ~ 0.9, linear extrapolation holds ~600 steps;
our staleness K ~ 10-20 ticks).

**One code path for fixed and learned.** The projector always evaluates the
SAME per-block affine combination of the retained snapshots. ``fixed_linear``
holds the coefficients FROZEN at the seed ``[2, -1, 0]`` (so it is exactly the
AsyncPP extrapolation); ``learned_linear_with_fixed_linear_cold_start`` starts
at the SAME seed and unfreezes a small per-block residual that is trained ONLY
from retrospective residuals ``theta_true[t_prev] - theta_hat[t_prev]`` at a
PRIOR fire (never the current weights — the no-peek invariant). With the
residual frozen at zero the learned mode is byte-identical to fixed-linear, so
the first learned fire equals the fixed-linear prediction.

**LayerNorm / embedding / lm_head excluded.** Per the linearity paper (Fig 8 /
App A.2) norm + embedding layers have LOW linearity; extrapolating them injects
error. Excluded params take the raw ``theta[t-K]`` (no projection). The decoder
weight-matrix scope reuses the existing ``spectral.target_substr`` selector
(``comm_eff.py:228-238``) so the exclusion set is exactly the merger's
non-target set.

**Cross-rank determinism.** ``theta_hat`` is a pure per-element function of the
DP-identical FSDP snapshots, so fixed-linear is trivially identical on every DP
rank. The learned per-block residual is updated only from DP-identical
retrospective residuals (no rank/device-local RNG), so it is also identical;
the engine PROVES this by emitting a cross-rank max-rel-dev scalar.

This module owns the FSDP-AGNOSTIC pieces (pure tensor math + a snapshot ring)
so they are unit-testable on CPU with no distributed runtime; the engine
(``FSDPEngine._maybe_comm_eff_anchor_refresh``) summons the full weights, takes
the snapshots, and loads ``theta_hat`` into the isolated anchor clone.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "LOOKAHEAD_MODES",
    "FIXED_LINEAR_COEFFS",
    "lookahead_enabled",
    "lookahead_num_source_points",
    "lookahead_learns",
    "is_lookahead_target",
    "compute_theta_hat",
    "LookaheadSnapshotRing",
    "LookaheadProjector",
    "cross_rank_max_rel_dev",
]


def _canon(name: str) -> str:
    """Strip the FSDP per-layer-wrap infix from a parameter name.

    Mirrors ``anchor._canon`` / ``spectral_filter._canon`` (kept local so this
    CPU-testable module has no cross-module import dependency). The look-ahead
    ring + projector key snapshots by the canonical (infix-stripped) name so a
    fallback config-rebuilt clone (non-infixed names) and the live FSDP module
    (infixed names) agree.
    """
    name = name.replace("._fsdp_wrapped_module", "")
    if name.startswith("_fsdp_wrapped_module."):
        name = name[len("_fsdp_wrapped_module.") :]
    return name


# The three look-ahead modes. ``disabled`` (default) is a strict no-op — the
# anchor forwards from the raw stale ``theta[t-K]`` exactly as today.
# ``fixed_linear`` uses the FROZEN AsyncPP seed. The learned mode unfreezes a
# per-block residual but cold-starts at the fixed-linear seed.
LOOKAHEAD_MODES = ("disabled", "fixed_linear", "learned_linear_with_fixed_linear_cold_start")

# The fixed-linear (AsyncPP) coefficients for [theta[t-K], theta[t-2K], theta[t-3K]]:
# theta_hat = 2*theta[t-K] - theta[t-2K] + 0*theta[t-3K]. a3=0 means the third
# snapshot is unused in fixed-linear (so fixed-linear needs only 2 source points).
FIXED_LINEAR_COEFFS = (2.0, -1.0, 0.0)


def lookahead_enabled(anchor_cfg) -> bool:
    """True iff the look-ahead extrapolation is active for this anchor config.

    Gated by BOTH the ``lookahead_anchor`` master flag AND a non-``disabled``
    ``lookahead_mode`` — so a stray ``lookahead_anchor=true`` with
    ``lookahead_mode=disabled`` (or vice-versa) is inert, matching the
    dataclass ``__post_init__`` validation. Pure predicate, no side effects.
    """
    if anchor_cfg is None:
        return False
    if not bool(getattr(anchor_cfg, "lookahead_anchor", False)):
        return False
    mode = str(getattr(anchor_cfg, "lookahead_mode", "disabled"))
    return mode in ("fixed_linear", "learned_linear_with_fixed_linear_cold_start")


def lookahead_learns(anchor_cfg) -> bool:
    """True iff the projector trains a per-block residual (learned mode)."""
    if not lookahead_enabled(anchor_cfg):
        return False
    return str(getattr(anchor_cfg, "lookahead_mode", "disabled")) == (
        "learned_linear_with_fixed_linear_cold_start"
    )


def lookahead_num_source_points(anchor_cfg) -> int:
    """Number of fire-aligned source snapshots the projector consumes.

    ``fixed_linear`` uses 2 (``theta[t-K]``, ``theta[t-2K]``); the learned mode
    uses 3 (``+ theta[t-3K]``) so the residual has a third basis point. The
    NEW look-ahead ring is sized from this (plus the retrospective-residual
    ``theta_hat_prev`` slot the learned mode keeps separately). Returns 0 when
    look-ahead is disabled.
    """
    if not lookahead_enabled(anchor_cfg):
        return 0
    return 3 if lookahead_learns(anchor_cfg) else 2


def lookahead_strength(anchor_cfg) -> float:
    """Projection horizon strength alpha (M4).

    ``theta_hat = (1+alpha)*theta[t-K] - alpha*theta[t-2K]`` ⇒ ``alpha`` is the
    fraction of the staleness ``K`` extrapolated FORWARD. ``alpha=1.0`` (default)
    reproduces the frozen AsyncPP seed ``(2, -1, 0)`` = full catch-up to the
    current step; ``alpha<1`` projects a SHORTER horizon (a gentler, less-sharp
    anchor); ``alpha=0`` = the raw stale ``theta[t-K]`` (no projection). Read from
    the anchor config; defaults to ``1.0`` so every prior (non-look-ahead) path is byte-identical.
    """
    if anchor_cfg is None:
        return 1.0
    try:
        return float(getattr(anchor_cfg, "lookahead_strength", 1.0))
    except (TypeError, ValueError):
        return 1.0


def is_lookahead_target(name: str, target_substrs) -> bool:
    """True iff ``name`` is a decoder weight matrix the projector extrapolates.

    Reuses the merger's ``target_substr`` selector (the decoder
    attention/MLP projection matrices: q/k/v/o_proj, gate/up/down_proj). Norms,
    biases, embeddings and the lm head do NOT contain these substrings, so they
    are NOT targets → they take the raw ``theta[t-K]`` (the LayerNorm/embedding
    exclusion). 2D-ness is enforced separately by the caller against the actual
    tensor shape.
    """
    if not target_substrs:
        return False
    return any(s in name for s in target_substrs)


def compute_theta_hat(
    sources: list,
    coeffs,
    *,
    target_substrs,
    residual: Optional[dict] = None,
) -> tuple:
    """Materialize the per-target extrapolated weights ``theta_hat`` (CPU/full).

    Args:
        sources: list ``[S0, S1, (S2)]`` of snapshot dicts ``{canon_name ->
            full param tensor}`` ordered NEWEST-first — ``S0 = theta[t-K]``,
            ``S1 = theta[t-2K]``, optional ``S2 = theta[t-3K]``. All keyed by
            CANONICAL name. Every dict covers the same key set (full-param
            snapshots).
        coeffs: the affine combination ``(a1, a2, a3)`` applied to
            ``(S0, S1, S2)``. ``a3`` is ignored when only 2 sources are given.
        target_substrs: the decoder-matrix selector. A param is extrapolated iff
            it is a target AND 2D; every OTHER param takes ``S0`` (the raw stale
            weight) unchanged — the LayerNorm/embedding exclusion.
        residual: optional ``{canon_name -> per-block scalar residual delta}``
            ADDED to the targeted ``theta_hat`` in the learned mode. ``None`` (or
            an all-zero residual) ⇒ pure fixed-linear. The residual is a
            cross-rank-identical per-target scalar (broadcast over the matrix),
            so it cannot diverge ranks.

    Returns:
        ``(theta_hat, excluded_names)`` — ``theta_hat`` is ``{canon_name ->
        tensor}`` (every key of ``S0``, targets extrapolated, others raw);
        ``excluded_names`` is the sorted list of NON-extrapolated keys (logged).

    The arithmetic is a plain per-element op done in fp32 on whatever device the
    snapshots live on (CPU for ``snapshot_device=cpu``), cast back to ``S0``'s
    dtype, so it composes with bf16 snapshots with no cross-shard ambiguity.
    """
    assert len(sources) >= 2, f"compute_theta_hat needs >= 2 source snapshots, got {len(sources)}"
    s0 = sources[0]
    s1 = sources[1]
    s2 = sources[2] if len(sources) >= 3 else None
    a1, a2, a3 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2] if len(coeffs) >= 3 else 0.0)

    theta_hat: dict = {}
    excluded: list = []
    for name, p0 in s0.items():
        is_target = is_lookahead_target(name, target_substrs) and getattr(p0, "dim", lambda: 0)() == 2
        if not is_target:
            # LayerNorm / embedding / lm_head / bias / 1-D: NO extrapolation.
            theta_hat[name] = p0
            excluded.append(name)
            continue
        p1 = s1.get(name)
        if p1 is None or p1.shape != p0.shape:
            # Defensive: a missing/mismatched source for a target → fall back to
            # the raw stale weight rather than fabricate an extrapolation.
            theta_hat[name] = p0
            excluded.append(name)
            continue
        dtype = p0.dtype
        acc = a1 * p0.to(torch.float32) + a2 * p1.to(torch.float32)
        if s2 is not None and a3 != 0.0:
            p2 = s2.get(name)
            if p2 is not None and p2.shape == p0.shape:
                acc = acc + a3 * p2.to(torch.float32)
        if residual is not None:
            r = residual.get(name)
            if r is not None:
                acc = acc + float(r)
        theta_hat[name] = acc.to(dtype)
    return theta_hat, sorted(excluded)


class LookaheadSnapshotRing:
    """NEW fire-aligned snapshot ring for the look-ahead source points.

    **Why a new ring (not the AnchorReplayRing).** With ``replay_paired_batch``
    the live anchor path uses :class:`anchor.AnchorReplayRing`, which retains
    only ``delay_K // cadence + 1 = 2`` slots at 20/20 and aggressively evicts
    every global step no retained batch references — so ``theta[t-2K]`` is gone
    before the next fire. The legacy :class:`anchor.AnchorStalenessQueue` holds
    ``theta[t-K]..theta[t]`` but not ``theta[t-2K]``, and is not even built in
    replay mode. So the look-ahead history is its OWN ring, pushed at every
    anchor FIRE (not every tick) and keyed by the fire tick.

    Holds the last ``n_points`` fire-aligned snapshots (2 fixed / 3 learned),
    NEWEST-first via :meth:`sources`. Optionally keeps ONE prior ``theta_hat``
    (the learned mode's retrospective-residual target) — counted in the bound.
    Pure container: no collectives, no RNG, CPU-testable.
    """

    def __init__(self, n_points: int, keep_theta_hat: bool = False):
        assert n_points >= 2, f"look-ahead ring needs >= 2 source points, got {n_points}"
        self.n_points = int(n_points)
        self.keep_theta_hat = bool(keep_theta_hat)
        # OrderedDict[fire_tick -> snapshot dict]; insertion order == tick order.
        self._snaps: OrderedDict[int, dict] = OrderedDict()
        # One prior theta_hat for the retrospective residual (learned mode).
        self._prev_theta_hat: Optional[dict] = None
        self._prev_theta_hat_tick: int = -1
        # Bound: n_points fire snapshots. (theta_hat lives in its own single slot,
        # asserted separately.) Track the peak for the bounded-memory report.
        self._maxlen = self.n_points
        self.peak_retained = 0

    def push(self, fire_tick: int, snapshot: dict) -> None:
        """Record the full-param ``snapshot`` taken at anchor fire ``fire_tick``.

        Evicts the oldest beyond ``n_points``. Asserts the bound on every push
        (mirrors ``AnchorReplayRing.push_snapshot`` :457-461) so a regressed
        eviction is a loud failure, not a silent leak.
        """
        self._snaps[int(fire_tick)] = snapshot
        while len(self._snaps) > self._maxlen:
            self._snaps.popitem(last=False)
        self.peak_retained = max(self.peak_retained, self.total_retained())
        assert len(self._snaps) <= self._maxlen, (
            f"LookaheadSnapshotRing retention blew its bound: {len(self._snaps)} > "
            f"maxlen={self._maxlen} (n_points={self.n_points}) — eviction regressed."
        )

    def ready(self) -> bool:
        """True iff enough source snapshots are retained to extrapolate."""
        return len(self._snaps) >= self.n_points

    def sources(self):
        """Return ``[S0, S1, (S2)]`` NEWEST-first with their fire ticks.

        Returns ``(snaps, ticks)`` where ``snaps[0]`` is the newest retained
        (``theta[t-K]``), ``snaps[1]`` the next (``theta[t-2K]``), etc., and
        ``ticks`` the matching fire-tick indices (newest-first). Returns
        ``(None, None)`` until :meth:`ready`.
        """
        if not self.ready():
            return None, None
        items = list(self._snaps.items())  # oldest-first
        items = items[-self.n_points :]  # the n_points newest
        items.reverse()  # newest-first
        ticks = [int(t) for t, _s in items]
        snaps = [s for _t, s in items]
        return snaps, ticks

    def set_prev_theta_hat(self, fire_tick: int, theta_hat: dict) -> None:
        """Stash the PRIOR fire's ``theta_hat`` (learned-mode residual target).

        Only one is retained (the most recent). Counted in the memory report.
        """
        if not self.keep_theta_hat:
            return
        self._prev_theta_hat = theta_hat
        self._prev_theta_hat_tick = int(fire_tick)
        self.peak_retained = max(self.peak_retained, self.total_retained())

    def prev_theta_hat(self):
        """Return ``(theta_hat, tick)`` of the prior fire, or ``(None, -1)``."""
        return self._prev_theta_hat, self._prev_theta_hat_tick

    def total_retained(self) -> int:
        """Total full-param snapshots held (sources + optional prior theta_hat)."""
        return len(self._snaps) + (1 if self._prev_theta_hat is not None else 0)

    @property
    def ticks(self) -> list:
        return list(self._snaps.keys())


class LookaheadProjector:
    """Per-block linear weight projector (fixed-linear + learned residual).

    Stateless for ``fixed_linear`` (the coefficients are the FROZEN seed). For
    the learned mode it carries a per-block scalar residual ``r[name]`` (init
    zero ⇒ identical to fixed-linear on the first fire), updated ONLY from a
    retrospective residual ``theta_true[t_prev] - theta_hat[t_prev]`` projected
    onto the per-block update direction — no current weights, no peek.

    The learned residual is a single scalar per target matrix (a per-block
    correction to the extrapolation magnitude), updated by a small step from
    the DP-identical retrospective residual, so it is cross-rank-identical by
    construction. The engine emits a cross-rank max-rel-dev to PROVE it.
    """

    def __init__(self, anchor_cfg, target_substrs):
        self.target_substrs = tuple(target_substrs or ())
        self.learns = lookahead_learns(anchor_cfg)
        self.n_points = lookahead_num_source_points(anchor_cfg)
        # Projection horizon: coeffs = (1+alpha, -alpha, 0). alpha=1.0 reproduces
        # the frozen AsyncPP seed (2,-1,0) = full catch-up; alpha<1 = a shorter
        # look-ahead horizon (M4 horizon sweep). The learned mode cold-
        # starts here and then adapts the per-block residual on top of these coeffs.
        self.strength = lookahead_strength(anchor_cfg)
        self.coeffs = [1.0 + self.strength, -self.strength, 0.0]
        # Per-block scalar residual (learned mode only); zero ⇒ fixed-linear.
        self._residual: dict = {}
        # Learning rate for the residual update from the retrospective error.
        # Small + bounded; the residual is a magnitude nudge on the extrapolation,
        # not a free parameter. Cross-rank-identical (no RNG, no device-local state).
        self._residual_lr = 0.1
        # Bound the absolute residual so a pathological retrospective error cannot
        # blow the extrapolation; keeps the learned mode a CORRECTION to fixed-linear.
        self._residual_clip = 1.0e-3

    def project(self, sources: list):
        """Compute ``theta_hat`` from the source snapshots.

        Returns ``(theta_hat, excluded_names)``. ``residual`` is folded in only
        in the learned mode (it is ``{}`` until the first retrospective update).
        """
        residual = self._residual if self.learns else None
        return compute_theta_hat(
            sources,
            self.coeffs,
            target_substrs=self.target_substrs,
            residual=residual,
        )

    def update_from_retrospective(self, theta_true_prev: dict, theta_hat_prev: dict) -> dict:
        """Update the learned residual from the PRIOR fire's true-vs-predicted error.

        ``theta_true_prev`` is the FSDP live weights summoned at THIS fire (the
        true weights that the prior fire's ``theta_hat`` was predicting); read
        DP-identical. ``theta_hat_prev`` is the prior fire's extrapolation. The
        per-block residual moves a small step toward closing the mean error
        ``mean(theta_true_prev - theta_hat_prev)`` per target — a scalar nudge,
        cross-rank-identical (both inputs are DP-mean / DP-identical). No-op for
        ``fixed_linear``. Returns the updated residual dict (for telemetry).
        """
        if not self.learns or theta_true_prev is None or theta_hat_prev is None:
            return dict(self._residual)
        for name, t_true in theta_true_prev.items():
            cname = _canon(name)
            if not is_lookahead_target(cname, self.target_substrs):
                continue
            t_hat = theta_hat_prev.get(cname) if cname in theta_hat_prev else theta_hat_prev.get(name)
            if t_hat is None or t_hat.shape != t_true.shape:
                continue
            err = (t_true.to(torch.float32) - t_hat.to(torch.float32)).mean().item()
            prev = float(self._residual.get(cname, 0.0))
            new = prev + self._residual_lr * float(err)
            # Clip to keep the learned residual a bounded correction.
            new = max(-self._residual_clip, min(self._residual_clip, new))
            self._residual[cname] = new
        return dict(self._residual)

    def residual_vector(self):
        """Return the per-block residual as a sorted-by-name fp32 1-D tensor.

        Used by the engine to emit a cross-rank max-rel-dev (proving the learned
        residual is DP-identical). Empty tensor when fixed-linear / not yet
        updated.
        """
        if not self._residual:
            return torch.zeros(0, dtype=torch.float32)
        names = sorted(self._residual.keys())
        return torch.tensor([float(self._residual[n]) for n in names], dtype=torch.float32)


def cross_rank_max_rel_dev(local_vec: torch.Tensor, dp_group=None) -> Optional[float]:
    """Max relative cross-rank deviation of a 1-D vector (determinism probe).

    Mirrors the ``_powersgd_q_agreement_dev`` pattern: all-reduce(MAX) of
    ``|local - mean| / (|mean| + eps)`` over the DP group. Returns 0.0 when the
    vector is identical on every rank (the expected look-ahead invariant), a
    positive number if ranks diverged, and ``None`` when there is no distributed
    runtime / the vector is empty (nothing to check).

    The mean is computed via all-reduce(SUM)/world; both reductions are over the
    SAME fixed-length vector on every rank (the residual is keyed by the
    architecture-identical target set), so the collective sequence is symmetric.
    """
    if local_vec is None or local_vec.numel() == 0:
        return None
    if not torch.distributed.is_initialized():
        # Single process: trivially identical.
        return 0.0
    try:
        world = torch.distributed.get_world_size(group=dp_group)
    except Exception:
        world = 1
    if world <= 1:
        return 0.0
    v = local_vec.detach().to(torch.float32)
    dev = v.device if v.is_cuda else None
    # Move to GPU for the collective if available (NCCL); else keep on CPU (gloo).
    if torch.cuda.is_available():
        v = v.cuda()
    summed = v.clone()
    torch.distributed.all_reduce(summed, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    mean = summed / float(world)
    rel = (v - mean).abs() / (mean.abs() + 1.0e-12)
    rmax = rel.max()
    torch.distributed.all_reduce(rmax, op=torch.distributed.ReduceOp.MAX, group=dp_group)
    out = float(rmax.item())
    del v, summed, mean, rel, rmax
    if dev is None and torch.cuda.is_available():
        pass
    return out
