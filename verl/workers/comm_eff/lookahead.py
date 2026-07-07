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

**Generalized horizon (the seed alone is NOT enough).** The source snapshots
are spaced by the anchor FIRE cadence (a real recorded gap of ``g`` ticks:
``g = tick(S0) - tick(S1)``), while the projection target is the CURRENT fire
tick ``t``, which sits ``h = t - tick(S0)`` ticks ahead of ``S0`` (``h >=
delay_K``). The seed coefficients ``(1+alpha, -alpha)`` extrapolate exactly ONE
inter-snapshot gap forward, so they land on ``t`` only when ``h == g`` (i.e.
cadence == delay_K — true at the 20/20 and 5/5 operating points, false
otherwise). :meth:`LookaheadProjector.project` therefore consumes the ring's
REAL recorded ticks and uses::

    theta_hat = S0 + alpha * (h/g) * (S0 - S1)
              = (1 + alpha*h/g) * S0 - (alpha*h/g) * S1

which reduces EXACTLY to the frozen seed at ``h == g`` (behavior at the
operating points unchanged) and projects to the true fire tick otherwise.

**One code path for fixed and learned.** The projector always evaluates the
SAME per-block affine combination of the retained snapshots. ``fixed_linear``
holds the coefficients FROZEN at the seed ``[2, -1, 0]`` (so it is exactly the
AsyncPP extrapolation); ``learned_linear_with_fixed_linear_cold_start`` starts
at the SAME seed and unfreezes a small per-block residual that is trained ONLY
from retrospective residuals ``theta_true[t_prev] - theta_hat[t_prev]`` at a
PRIOR fire (never the current weights — the no-peek invariant). With the
residual frozen at zero the learned mode is byte-identical to fixed-linear, so
the first learned fire equals the fixed-linear prediction.

**Third mode — ``learned_step_scale_with_fixed_linear_cold_start`` (the adaptive
projector).** The DC-offset residual above corrects only the MEAN of the
per-block extrapolation error, but a linear-extrapolation error is ~zero-mean
across a weight matrix (``mean(theta_true - theta_hat) ~ 0``), so that signal is
nearly vacuous — the residual is learning the coefficient on the all-ones basis
vector ``1`` when the error actually lives along the trajectory direction
``d = S0 - S1``. This mode fixes the BASIS: it learns a per-block SCALE
``beta[name]`` (init ``1.0``) on the extrapolation STEP —

    theta_hat = S0 + beta * base_scale * (S0 - S1)
              = (1 + beta*base_scale) * S0 - (beta*base_scale) * S1

with ``base_scale = alpha * h/g`` the same realized-horizon step the other modes
use. ``beta`` is updated retrospectively by projecting the prior fire's error
onto the STEP it actually took (:meth:`LookaheadProjector._update_step_scale`):
``rho = <e, step> / (||step||^2)`` is the fractional over/under-shoot
(``rho > 0`` ⇒ the truth lay FURTHER along the step ⇒ we under-projected ⇒ grow
``beta``; ``rho < 0`` ⇒ over-projected ⇒ shrink), and ``beta <- clip(beta +
lr*rho, beta_min, beta_max)``. The ``[beta_min, beta_max] = [0, 2]`` clip is a
per-block trust region: ``beta -> 0`` degrades gracefully to the raw stale
weight (the safe fallback in blocks where the trajectory is NOT locally linear —
important for the harder Big-Math dynamics), ``beta`` caps over-projection at
``2x`` the linear step. Every reduction is an INNER PRODUCT of DP-identical
stale tensors (no mean-collapse of the signal, no RNG), so ``beta`` is
cross-rank-identical by construction just like the offset residual. At
``beta == 1`` (the cold start) the step equals the fixed-linear step EXACTLY, so
the first fire is byte-identical to ``fixed_linear``.

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
    "MODE_DISABLED",
    "MODE_FIXED_LINEAR",
    "MODE_LEARNED_OFFSET",
    "MODE_LEARNED_STEP_SCALE",
    "LOOKAHEAD_ROLLOUT_SOURCES",
    "FIXED_LINEAR_COEFFS",
    "lookahead_enabled",
    "lookahead_num_source_points",
    "lookahead_min_points",
    "lookahead_learns",
    "resolve_lookahead_rollout_source",
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


# The look-ahead modes. ``disabled`` (default) is a strict no-op — the anchor
# forwards from the raw stale ``theta[t-K]`` exactly as today. ``fixed_linear``
# uses the FROZEN AsyncPP seed. The two learned modes cold-start at the
# fixed-linear seed and unfreeze a per-block state trained ONLY from
# retrospective (no-peek) errors: the OFFSET mode learns an additive DC residual,
# the STEP_SCALE mode learns a multiplicative scale on the extrapolation step
# (the better-conditioned basis — see the module docstring). Named constants are
# used for internal dispatch; the whitelist tuple is the config single-source.
MODE_DISABLED = "disabled"
MODE_FIXED_LINEAR = "fixed_linear"
MODE_LEARNED_OFFSET = "learned_linear_with_fixed_linear_cold_start"
MODE_LEARNED_STEP_SCALE = "learned_step_scale_with_fixed_linear_cold_start"
LOOKAHEAD_MODES = (MODE_DISABLED, MODE_FIXED_LINEAR, MODE_LEARNED_OFFSET, MODE_LEARNED_STEP_SCALE)
# The modes that train per-block state (need a retained prior theta_hat + a
# retrospective update). Both learned modes; fixed_linear/disabled do not.
_LEARNED_MODES = (MODE_LEARNED_OFFSET, MODE_LEARNED_STEP_SCALE)

# The fixed-linear (AsyncPP) coefficients for [theta[t-K], theta[t-2K], theta[t-3K]]:
# theta_hat = 2*theta[t-K] - theta[t-2K] + 0*theta[t-3K]. a3=0 means the third
# snapshot is unused in fixed-linear (so fixed-linear needs only 2 source points).
# This is the documented SEED — the exact coefficients at alpha=1 AND horizon
# h == gap g (cadence == delay_K). The live projection generalizes to
# (1 + alpha*h/g, -alpha*h/g, 0) from the ring's REAL recorded ticks; see
# LookaheadProjector.project.
FIXED_LINEAR_COEFFS = (2.0, -1.0, 0.0)

# Which rollouts the anchor consumes when the look-ahead projector is on.
#   "auto"          -> resolves to "current_step" when look-ahead is enabled,
#                      else "stale_paired" (matching rollouts are THE DEFAULT
#                      whenever weight projection is ON; zero effect when OFF).
#   "stale_paired"  -> today's exact behavior: the replayed t-delay_K batch in
#                      replay mode. (Legacy non-replay mode ALREADY consumes the
#                      current tick's batch, so this option only changes
#                      behavior in replay mode.)
#   "current_step"  -> replay mode consumes a copy of the CURRENT tick's batch
#                      (the step-t rollouts that the projected theta_hat[t]
#                      weights correspond to). Config-validated to require the
#                      projector ON (stale-weights + fresh-rollouts is an
#                      unsupported ablation).
#   "self_generate" -> RESERVED seam for a future "anchor generates its own
#                      rollouts" option; REJECTED at config validation.
LOOKAHEAD_ROLLOUT_SOURCES = ("auto", "stale_paired", "current_step", "self_generate")


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
    mode = str(getattr(anchor_cfg, "lookahead_mode", MODE_DISABLED))
    return mode in (MODE_FIXED_LINEAR, MODE_LEARNED_OFFSET, MODE_LEARNED_STEP_SCALE)


def lookahead_learns(anchor_cfg) -> bool:
    """True iff the projector trains per-block state (either learned mode).

    Gates the engine's three learned-mode-only actions — retaining a prior
    ``theta_hat`` (``keep_theta_hat``), running the retrospective update, and
    stashing this fire's ``theta_hat`` for the next update — so it must be True
    for BOTH the offset (:data:`MODE_LEARNED_OFFSET`) and step-scale
    (:data:`MODE_LEARNED_STEP_SCALE`) modes. The *specific* per-block quantity
    each mode trains (additive residual vs. multiplicative ``beta``) is dispatched
    inside :class:`LookaheadProjector`, not here.
    """
    if not lookahead_enabled(anchor_cfg):
        return False
    return str(getattr(anchor_cfg, "lookahead_mode", MODE_DISABLED)) in _LEARNED_MODES


def lookahead_num_source_points(anchor_cfg) -> int:
    """Number of fire-aligned source snapshots the projector consumes.

    ``fixed_linear`` and ``learned_step_scale_with_fixed_linear_cold_start`` are
    FIRST-ORDER (``S0 = theta[t-K]``, ``S1 = theta[t-2K]``) so they use 2; the
    offset ``learned_linear_with_fixed_linear_cold_start`` uses 3 (``+
    theta[t-3K]``) so its residual has a third basis point. The look-ahead ring
    is sized from this (plus the retrospective ``theta_hat_prev`` slot both
    learned modes keep separately — the step-scale mode reconstructs its prior
    STEP from ``S1``, which the 2-point ring already retains). Returns 0 when
    look-ahead is disabled.
    """
    if not lookahead_enabled(anchor_cfg):
        return 0
    return 3 if str(getattr(anchor_cfg, "lookahead_mode", MODE_DISABLED)) == MODE_LEARNED_OFFSET else 2


def lookahead_min_points(anchor_cfg) -> int:
    """Ring snapshots required before the projector engages (E3).

    Reads ``anchor.lookahead_min_snapshots``: ``-1`` (default) ⇒ the mode's full
    source count :func:`lookahead_num_source_points` (2 for fixed_linear /
    step-scale, 3 for the offset learned mode) — today's behavior. A concrete
    value (config-validated to ``[2, n_points]``) lets the projector engage at
    the earliest mathematically-legal fire (2 = fire 2). This is the SINGLE
    readiness threshold the ring keys :meth:`ready` on, so the no_correct skip
    gate and the projected-vs-fallback decision share it (a second hardcoded
    ``n_points`` check would silently extend the skip window).
    Returns 0 when look-ahead is disabled.
    """
    n = lookahead_num_source_points(anchor_cfg)
    if n == 0:
        return 0
    raw = int(getattr(anchor_cfg, "lookahead_min_snapshots", -1))
    if raw == -1:
        return n
    return raw


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


def resolve_lookahead_rollout_source(anchor_cfg) -> str:
    """Resolve ``anchor.lookahead_rollout_source`` to a concrete source.

    Pure function (unit-testable, no side effects). ``"auto"`` (the default)
    resolves to ``"current_step"`` iff the look-ahead projector is enabled
    (:func:`lookahead_enabled`) and to ``"stale_paired"`` otherwise — so the
    matching-rollouts behavior is THE DEFAULT whenever weight projection is ON
    and has ZERO effect when it is OFF. Explicit values pass through unchanged;
    the invalid combinations (``current_step`` without the projector,
    ``self_generate``) are rejected up front by the config ``__post_init__``,
    not here.
    """
    src = str(getattr(anchor_cfg, "lookahead_rollout_source", "auto")) if anchor_cfg is not None else "auto"
    if src == "auto":
        return "current_step" if lookahead_enabled(anchor_cfg) else "stale_paired"
    return src


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
    per_block_scale: Optional[dict] = None,
    base_scale: Optional[float] = None,
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
            Ignored entirely when ``per_block_scale`` is given.
        target_substrs: the decoder-matrix selector. A param is extrapolated iff
            it is a target AND 2D; every OTHER param takes ``S0`` (the raw stale
            weight) unchanged — the LayerNorm/embedding exclusion.
        residual: optional ``{canon_name -> per-block scalar residual delta}``
            ADDED to the targeted ``theta_hat`` in the OFFSET learned mode.
            ``None`` (or an all-zero residual) ⇒ pure fixed-linear. The residual
            is a cross-rank-identical per-target scalar (broadcast over the
            matrix), so it cannot diverge ranks. Mutually exclusive with
            ``per_block_scale``.
        per_block_scale: optional ``{canon_name -> per-block scale ``beta``}`` for
            the STEP-SCALE learned mode. When given, the coefficients are IGNORED
            and each target uses ``(1 + beta*base_scale, -beta*base_scale)`` with
            ``beta`` defaulting to ``1.0`` for any un-updated block — so an
            all-default ``beta`` reproduces the fixed-linear step EXACTLY. Requires
            ``base_scale``. Cross-rank-identical (a scalar per target).
        base_scale: the realized linear step ``alpha*h/g`` (a scalar), used only
            with ``per_block_scale``. ``beta*base_scale`` is the effective step.

    Returns:
        ``(theta_hat, excluded_names)`` — ``theta_hat`` is ``{canon_name ->
        tensor}`` (every key of ``S0``, targets extrapolated, others raw);
        ``excluded_names`` is the sorted list of NON-extrapolated keys (logged).

    The arithmetic is a plain per-element op done in fp32 on whatever device the
    snapshots live on (CPU for ``snapshot_device=cpu``), cast back to ``S0``'s
    dtype, so it composes with bf16 snapshots with no cross-shard ambiguity.
    """
    assert len(sources) >= 2, f"compute_theta_hat needs >= 2 source snapshots, got {len(sources)}"
    assert not (per_block_scale is not None and residual is not None), (
        "compute_theta_hat: per_block_scale (step-scale mode) and residual (offset mode) are "
        "mutually exclusive — a mode should pass exactly one."
    )
    assert per_block_scale is None or base_scale is not None, (
        "compute_theta_hat: per_block_scale requires base_scale (the realized alpha*h/g step)."
    )
    s0 = sources[0]
    s1 = sources[1]
    s2 = sources[2] if len(sources) >= 3 else None
    a1, a2, a3 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2] if len(coeffs) >= 3 else 0.0)
    _base = float(base_scale) if base_scale is not None else 0.0

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
        if per_block_scale is not None:
            # STEP-SCALE mode: eff coeffs (1 + beta*base_scale, -beta*base_scale).
            # beta defaults to 1.0 for an un-updated block, so eff == base_scale
            # and this is the fixed-linear step EXACTLY (byte-identical cold
            # start). The additive residual / third snapshot are unused here.
            beta = float(per_block_scale.get(name, 1.0))
            eff = beta * _base
            acc = (1.0 + eff) * p0.to(torch.float32) + (-eff) * p1.to(torch.float32)
        else:
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
    anchor FIRE (not every tick) and keyed by the snapshot's TRUE tick — the
    tick whose weights the snapshot actually holds (a fire at tick ``t``
    replays ``theta[t-K]``, so the key is ``t-K``, NOT ``t``). True-tick keying
    is what lets :meth:`LookaheadProjector.project` compute the REAL horizon
    ``h`` and gap ``g`` instead of assuming ``cadence == delay_K``. Pushes store
    REFERENCES (no tensor copies) — retention here keeps replay-ring-evicted
    snapshot generations alive by refcount.

    Holds the last ``n_points`` fire-aligned snapshots (2 fixed / 3 learned),
    NEWEST-first via :meth:`sources`. Optionally keeps ONE prior ``theta_hat``
    (the learned mode's retrospective-residual target) — counted in the bound.
    Pure container: no collectives, no RNG, CPU-testable.
    """

    def __init__(self, n_points: int, keep_theta_hat: bool = False, min_points: Optional[int] = None):
        assert n_points >= 2, f"look-ahead ring needs >= 2 source points, got {n_points}"
        self.n_points = int(n_points)
        # Readiness threshold (E3). Defaults to n_points (today's behavior);
        # a smaller min_points lets :meth:`ready` engage the projector at the
        # earliest legal fire while retention still keeps n_points (the learned
        # residual's 3rd point arrives later). Must be in [2, n_points].
        self.min_points = int(min_points) if min_points is not None else self.n_points
        assert 2 <= self.min_points <= self.n_points, (
            f"look-ahead ring min_points must be in [2, n_points={self.n_points}]; got {self.min_points}"
        )
        self.keep_theta_hat = bool(keep_theta_hat)
        # OrderedDict[true_tick -> snapshot dict]; insertion order == tick order.
        self._snaps: OrderedDict[int, dict] = OrderedDict()
        # One prior theta_hat for the retrospective residual (learned mode).
        self._prev_theta_hat: Optional[dict] = None
        self._prev_theta_hat_tick: int = -1
        # Bound: n_points fire snapshots. (theta_hat lives in its own single slot,
        # asserted separately.) Track the peak for the bounded-memory report.
        self._maxlen = self.n_points
        self.peak_retained = 0

    def push(self, tick: int, snapshot: dict) -> None:
        """Record the full-param ``snapshot`` whose weights belong to ``tick``.

        ``tick`` is the snapshot's TRUE tick (``t - delay_K`` at a fire tick
        ``t``), not the fire tick — see the class docstring. Re-pushing the
        same tick (a warmup fallback replaying one snapshot twice) overwrites
        in place. Evicts the oldest beyond ``n_points``. Asserts the bound on
        every push (mirrors ``AnchorReplayRing.push_snapshot`` :457-461) so a
        regressed eviction is a loud failure, not a silent leak.
        """
        self._snaps[int(tick)] = snapshot
        while len(self._snaps) > self._maxlen:
            self._snaps.popitem(last=False)
        self.peak_retained = max(self.peak_retained, self.total_retained())
        assert len(self._snaps) <= self._maxlen, (
            f"LookaheadSnapshotRing retention blew its bound: {len(self._snaps)} > "
            f"maxlen={self._maxlen} (n_points={self.n_points}) — eviction regressed."
        )

    def ready(self) -> bool:
        """True iff enough source snapshots are retained to extrapolate.

        Keys on ``min_points`` (not ``n_points``) so the projector can engage at
        the earliest legal fire when ``lookahead_min_snapshots`` relaxes it. This
        is the SINGLE readiness gate — the engine's no_correct skip and the
        projected-vs-fallback branch both derive from it.
        """
        return len(self._snaps) >= self.min_points

    def sources(self):
        """Return ``[S0, S1, (S2)]`` NEWEST-first with their TRUE ticks.

        Returns ``(snaps, ticks)`` where ``snaps[0]`` is the newest retained
        (``theta[t-K]``), ``snaps[1]`` the next (``theta[t-K-cadence]``), etc.,
        and ``ticks`` the matching TRUE snapshot ticks (newest-first) — exactly
        the inputs :meth:`LookaheadProjector.project` needs to compute the real
        horizon. Returns ``(None, None)`` until :meth:`ready`. Returns the
        ``min(len, n_points)`` newest — when readiness is relaxed to
        ``min_points`` the ring may hold only 2 of a learned mode's 3 points at
        the first projected fire; ``compute_theta_hat`` handles ``s2=None`` (the
        seed's ``a3=0``), so a 2-source projection is legal.
        """
        if not self.ready():
            return None, None
        items = list(self._snaps.items())  # oldest-first
        items = items[-self.n_points :]  # the n_points newest (clamps to len)
        items.reverse()  # newest-first
        ticks = [int(t) for t, _s in items]
        snaps = [s for _t, s in items]
        return snaps, ticks

    def set_prev_theta_hat(self, fire_tick: int, theta_hat: dict) -> None:
        """Stash the PRIOR fire's ``theta_hat`` (learned-mode residual target).

        ``fire_tick`` is the PROJECTION-TARGET tick — the tick the
        extrapolation was projected TO (in replay mode the current step's
        generator tick, not the literal fire tick), so its ground truth is
        ``theta[fire_tick]`` and, at cadence == delay_K, a LATER fire's source
        snapshot carries exactly that true tick (:meth:`get` then serves the
        retrospective update). Only one is retained (the most recent). Counted
        in the memory report.
        """
        if not self.keep_theta_hat:
            return
        self._prev_theta_hat = theta_hat
        self._prev_theta_hat_tick = int(fire_tick)
        self.peak_retained = max(self.peak_retained, self.total_retained())

    def get(self, tick: int):
        """Return the retained snapshot whose TRUE tick is ``tick`` (or None).

        Used by the learned mode's retrospective update: the prior fire's
        ``theta_hat`` predicted ``theta[t_prev_fire]``, whose exact ground
        truth is retained here iff some source snapshot's true tick equals
        ``t_prev_fire`` (at cadence == delay_K that is precisely this fire's
        freshly-pushed S0). Exact-match only — never an approximation.
        """
        return self._snaps.get(int(tick))

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
        self.mode = str(getattr(anchor_cfg, "lookahead_mode", MODE_DISABLED))
        # learns == "trains per-block state" (both learned modes; gates the ring's
        # keep_theta_hat + the engine's retrospective block). The SPECIFIC learned
        # quantity is dispatched by the two mode flags below — exactly one is True
        # in a learned mode, both False for fixed_linear.
        self.learns = lookahead_learns(anchor_cfg)
        self.learns_offset = self.mode == MODE_LEARNED_OFFSET
        self.learns_scale = self.mode == MODE_LEARNED_STEP_SCALE
        self.n_points = lookahead_num_source_points(anchor_cfg)
        # SEED coefficients: (1+alpha, -alpha, 0). alpha=1.0 reproduces the
        # frozen AsyncPP seed (2,-1,0) = full catch-up; alpha<1 = a shorter
        # look-ahead horizon (M4 horizon sweep). These are exact ONLY when the
        # realized horizon h equals the inter-source gap g (cadence == delay_K);
        # :meth:`project` generalizes to (1 + alpha*h/g, -alpha*h/g, 0) from the
        # ring's REAL ticks and falls back to this seed when ticks are omitted.
        # Both learned modes cold-start here and then adapt their per-block state
        # on top of the effective step.
        self.strength = lookahead_strength(anchor_cfg)
        self.coeffs = [1.0 + self.strength, -self.strength, 0.0]
        # --- OFFSET mode state ---
        # Per-block scalar residual (offset mode only); zero ⇒ fixed-linear.
        self._residual: dict = {}
        # Learning rate for the residual update from the retrospective error.
        # Small + bounded; the residual is a magnitude nudge on the extrapolation,
        # not a free parameter. Cross-rank-identical (no RNG, no device-local state).
        self._residual_lr = 0.1
        # Bound the absolute residual so a pathological retrospective error cannot
        # blow the extrapolation; keeps the learned mode a CORRECTION to fixed-linear.
        self._residual_clip = 1.0e-3
        # --- STEP-SCALE mode state ---
        # Per-block multiplicative scale beta on the extrapolation STEP; default
        # 1.0 for an un-updated block (via .get(name, 1.0)) ⇒ fixed-linear. Keyed
        # by canonical target name; updated by :meth:`_update_step_scale`.
        self._beta: dict = {}
        # Learning rate for the beta update from the fractional over/under-shoot
        # rho (a per-block integral controller, mirroring the offset residual's
        # small-gain design). rho is dimensionless, so lr is a direct fraction.
        self._beta_lr = 0.2
        # Per-block trust region: beta_min=0 degrades a block gracefully to the
        # raw stale weight (safe where the trajectory is NOT locally linear);
        # beta_max=2 caps over-projection at 2x the linear step.
        self._beta_min = 0.0
        self._beta_max = 2.0
        # Clip a SINGLE fire's over/under-shoot signal so one pathological
        # retrospective error cannot swing beta by more than beta_lr*rho_clip.
        self._rho_clip = 1.0
        # Guard the division by ||step||^2 (a block that took no step this fire).
        self._beta_eps = 1.0e-12

    def project(self, sources: list, ticks: Optional[list] = None, fire_tick: Optional[int] = None):
        """Compute ``theta_hat`` from the source snapshots.

        ``ticks`` (TRUE snapshot ticks, newest-first, aligned with ``sources``)
        and ``fire_tick`` (the tick being projected TO) generalize the seed to
        the REAL horizon. With ``g = ticks[0] - ticks[1]`` (inter-source gap,
        = cadence) and ``h = fire_tick - ticks[0]`` (realized staleness of S0,
        >= delay_K), the effective coefficients are::

            (1 + alpha*h/g, -alpha*h/g, 0)

        — the linear extrapolation that lands ON ``fire_tick``. At ``h == g``
        (cadence == delay_K, the 20/20 and 5/5 operating points) this reduces
        EXACTLY to the frozen seed, so prior behavior is unchanged there. The
        seed alone lands short/long whenever ``h != g`` — that was the original
        naive-linear bug this signature fixes. Omitting ``ticks``/``fire_tick``
        (unit tests, seed-only callers) falls back to the seed coefficients.

        Returns ``(theta_hat, excluded_names, proj_info)`` where ``proj_info``
        carries the realized ``{"h", "g", "coeffs"}`` for fire logging (``h``/
        ``g`` are ``None`` on the seed fallback; ``coeffs`` is the beta==1
        baseline step in step-scale mode). The offset ``residual`` (offset mode)
        or the per-block ``beta`` (step-scale mode) is folded in — each is empty
        until the first retrospective update, so the first fire equals fixed-linear.
        """
        coeffs = list(self.coeffs)
        base_scale = self.strength  # tick-less fallback: effective step == alpha
        h = g = None
        if ticks is not None and fire_tick is not None and len(ticks) >= 2:
            g = int(ticks[0]) - int(ticks[1])
            h = int(fire_tick) - int(ticks[0])
            assert g > 0, (
                f"look-ahead source ticks must be strictly decreasing newest-first; "
                f"got gap g={g} from ticks={ticks} — the ring's true-tick keying regressed."
            )
            assert h >= 0, (
                f"look-ahead fire_tick={fire_tick} precedes the newest source tick "
                f"{ticks[0]} (h={h}) — the projection would run BACKWARD; the caller "
                f"passed a stale fire tick or the ring keying regressed."
            )
            base_scale = self.strength * (float(h) / float(g))
            coeffs = [1.0 + base_scale, -base_scale, 0.0]
        if self.learns_scale:
            # STEP-SCALE mode: per-block beta multiplies base_scale; coeffs are
            # ignored downstream (kept in proj_info as the beta==1 baseline).
            theta_hat, excluded = compute_theta_hat(
                sources,
                coeffs,
                target_substrs=self.target_substrs,
                per_block_scale=self._beta,
                base_scale=base_scale,
            )
        else:
            residual = self._residual if self.learns_offset else None
            theta_hat, excluded = compute_theta_hat(
                sources,
                coeffs,
                target_substrs=self.target_substrs,
                residual=residual,
            )
        return theta_hat, excluded, {"h": h, "g": g, "coeffs": tuple(coeffs)}

    def update_from_retrospective(
        self, theta_true_prev: dict, theta_hat_prev: dict, s0_prev: Optional[dict] = None
    ) -> dict:
        """Update the learned per-block state from the PRIOR fire's error.

        ``theta_true_prev`` is the now-aged-in stale snapshot whose true tick
        equals the tick the prior fire projected TO (retrieved from the ring at
        THIS fire — NOT the live weights, preserving the no-peek invariant).
        ``theta_hat_prev`` is the prior fire's extrapolation. Dispatches on the
        learned mode:

        * OFFSET (:data:`MODE_LEARNED_OFFSET`): the per-block residual moves a
          small step toward closing the MEAN error ``mean(theta_true - theta_hat)``
          per target — a scalar nudge on the DC component.
        * STEP-SCALE (:data:`MODE_LEARNED_STEP_SCALE`): see
          :meth:`_update_step_scale` — needs ``s0_prev`` (the prior fire's ``S0``,
          = THIS fire's ``S1`` at cadence == delay_K) to reconstruct the step.

        Both reductions are over DP-identical stale tensors (no RNG), so the
        learned state is cross-rank-identical. No-op for ``fixed_linear``. Returns
        the updated learned-state dict (for telemetry).
        """
        if self.learns_scale:
            return self._update_step_scale(theta_true_prev, theta_hat_prev, s0_prev)
        if not self.learns_offset or theta_true_prev is None or theta_hat_prev is None:
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

    def _update_step_scale(
        self, theta_true_prev: dict, theta_hat_prev: dict, s0_prev: Optional[dict]
    ) -> dict:
        """Update per-block ``beta`` from the prior fire's over/under-shoot.

        Reconstructs the STEP the prior fire actually took, ``step = theta_hat_prev
        - S0_prev`` (parallel to the trajectory ``S0-S1``), and the residual error
        ``e = theta_true_prev - theta_hat_prev``. The fractional over/under-shoot
        projected onto the step is::

            rho = <e, step> / (||step||^2)

        ``rho > 0`` ⇒ the truth lay FURTHER along the step than we went
        (under-projected) ⇒ grow ``beta``; ``rho < 0`` ⇒ over-projected ⇒ shrink.
        ``beta <- clip(beta + beta_lr * clip(rho, +-rho_clip), beta_min, beta_max)``.
        Unlike the offset mode's ``mean(e)`` (≈0 on zero-mean weight-update
        matrices), ``<e, step>`` is a genuine inner product — an informative,
        cross-rank-identical scalar (both ``e`` and ``step`` are differences of
        DP-identical stale snapshots). No-op unless ``s0_prev`` is provided.
        """
        if theta_true_prev is None or theta_hat_prev is None or s0_prev is None:
            return dict(self._beta)
        for name, t_true in theta_true_prev.items():
            cname = _canon(name)
            if not is_lookahead_target(cname, self.target_substrs):
                continue
            t_hat = theta_hat_prev.get(cname) if cname in theta_hat_prev else theta_hat_prev.get(name)
            s0p = s0_prev.get(cname) if cname in s0_prev else s0_prev.get(name)
            if t_hat is None or s0p is None:
                continue
            if t_hat.shape != t_true.shape or s0p.shape != t_true.shape:
                continue
            e = t_true.to(torch.float32) - t_hat.to(torch.float32)
            step = t_hat.to(torch.float32) - s0p.to(torch.float32)
            step_sq = float((step * step).sum().item())
            if step_sq <= self._beta_eps:
                # No step taken this fire (h=0, or beta already collapsed to 0):
                # the over/under-shoot is undefined — leave beta unchanged.
                continue
            rho = float((e * step).sum().item()) / (step_sq + self._beta_eps)
            rho = max(-self._rho_clip, min(self._rho_clip, rho))
            prev = float(self._beta.get(cname, 1.0))
            new = prev + self._beta_lr * rho
            new = max(self._beta_min, min(self._beta_max, new))
            self._beta[cname] = new
        return dict(self._beta)

    def residual_vector(self):
        """Return the learned per-block state as a sorted-by-name fp32 1-D tensor.

        Offset mode → the residual deltas; step-scale mode → the per-block
        ``beta``. Used by the engine to emit a cross-rank max-rel-dev (proving the
        learned state is DP-identical). Empty tensor when fixed-linear / not yet
        updated.
        """
        state = self._beta if self.learns_scale else self._residual
        if not state:
            return torch.zeros(0, dtype=torch.float32)
        names = sorted(state.keys())
        return torch.tensor([float(state[n]) for n in names], dtype=torch.float32)


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
