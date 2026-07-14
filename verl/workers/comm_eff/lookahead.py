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

    theta_hat[t] = a1 * theta[t-K] + a2 * theta[t-2K]

The fixed-linear seed (AsyncPP / arXiv:2505.01099) is ``[a1=2, a2=-1]``::

    theta_hat[t] = 2 * theta[t-K] - theta[t-2K]

i.e. a one-step linear (Nesterov-style) weight extrapolation along the recent
training trajectory. The RLVR-linearity paper (arXiv:2601.04537) licenses this:
RLVR weights move ~linearly (per-weight R^2 > 0.7, concentrated near 0.8, with
linear extrapolation holding hundreds of steps; our staleness K ~ 10-20 ticks).

**Correspondence to the RLVR-linearity paper (arXiv:2601.04537, Eq 4).** That
paper's weight-space extrapolation predicts a future step ``t'`` from two prior
checkpoints ``t0 < t1`` as ``W_t' = W_t0 + beta * (W_t1 - W_t0)`` with
``beta = (t' - t0) / (t1 - t0) > 1``. Mapping its NEWER checkpoint
``W_t1 -> S0 = theta[t-K]`` and OLDER ``W_t0 -> S1 = theta[t-2K]`` (ticks
``t1 -> tick(S0)``, ``t0 -> tick(S1)``, target ``t' -> fire_tick``) gives
``beta = (h + g) / g = 1 + h/g`` (with ``h``/``g`` the realized horizon and gap
defined below), so their Eq 4 expands to
``theta_hat = (1 + h/g) * S0 - (h/g) * S1`` — EXACTLY the generalized projection
this module evaluates at ``alpha == 1`` (``alpha`` is an added horizon-strength
knob absent from the paper; ``alpha=1`` is the default, so the default IS Eq 4).
The frozen AsyncPP seed ``(2, -1)`` is the ``h == g`` special case
(``beta = 2``). So ``fixed_linear`` IS the paper's weight-space extrapolation,
not merely licensed by its linearity finding — the paper reports it matches
standard-RL performance at a ~6.1x training speedup (its Sec 6, Fig 5).

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

**Mode-specific tensor coverage.** ``fixed_linear`` retains the linearity
paper's decoder-matrix-only scope: norms, biases, embeddings and the LM head
take the raw ``theta[t-K]``. ``rank1_relex`` instead follows RELEX Algorithms
1--2 (arXiv:2605.21468): it finds and extrapolates one rank-1 checkpoint
trajectory independently for every unique floating named parameter tensor. Tied
parameters naturally appear once in ``named_parameters()``; untied LM heads
remain independent tensors. This weight-projection coverage is deliberately
separate from ``spectral.target_substr``, which still scopes only the downstream
2-D decoder-gradient correction.

**Cross-rank determinism.** ``theta_hat`` is a pure per-element function of the
DP-identical FSDP snapshots, so it is trivially identical on every DP rank —
there is no learned per-block state and no rank/device-local RNG.

This module owns the FSDP-AGNOSTIC pieces (pure tensor math + a snapshot ring)
so they are unit-testable on CPU with no distributed runtime; the engine
(``FSDPEngine._maybe_comm_eff_anchor_refresh``) summons the full weights, takes
the snapshots, and loads ``theta_hat`` into the isolated anchor clone.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from numbers import Integral
from typing import Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "LOOKAHEAD_MODES",
    "MODE_DISABLED",
    "MODE_FIXED_LINEAR",
    "MODE_RANK1_RELEX",
    "LOOKAHEAD_ROLLOUT_SOURCES",
    "FIXED_LINEAR_COEFFS",
    "DEFAULT_RANK1_CHUNK_NUMEL",
    "Rank1ProjectionError",
    "lookahead_enabled",
    "rank1_relex_enabled",
    "lookahead_num_source_points",
    "lookahead_min_points",
    "resolve_lookahead_rollout_source",
    "is_lookahead_target",
    "compute_theta_hat",
    "LookaheadSnapshotRing",
    "LookaheadProjector",
    "Rank1SnapshotHistory",
    "project_rank1_tensor",
    "Rank1RelexProjector",
    "validate_rank1_broadcast_receipts",
]


# The look-ahead modes. ``disabled`` (default) is a strict no-op — the anchor
# forwards from the raw stale ``theta[t-K]`` exactly as today. ``fixed_linear``
# uses the FROZEN AsyncPP seed (= the RLVR-linearity paper's Eq 4 weight-space
# extrapolation). Named constants are used for internal dispatch; the whitelist
# tuple is the config single-source.
MODE_DISABLED = "disabled"
MODE_FIXED_LINEAR = "fixed_linear"
MODE_RANK1_RELEX = "rank1_relex"
LOOKAHEAD_MODES = (MODE_DISABLED, MODE_FIXED_LINEAR, MODE_RANK1_RELEX)

# Bound the largest temporary trajectory slab to W * 4 MiB at the default W4
# window. The live projector makes two passes over each tensor (Gram, then
# right-vector reconstruction) and never retains a model-sized direction.
DEFAULT_RANK1_CHUNK_NUMEL = 1 << 20


class Rank1ProjectionError(RuntimeError):
    """Fail-closed error raised before a rank-1 anchor/M update can run."""


# The fixed-linear (AsyncPP) coefficients for [theta[t-K], theta[t-2K]]:
# theta_hat = 2*theta[t-K] - theta[t-2K]. This is the documented SEED — the
# exact coefficients at alpha=1 AND horizon h == gap g (cadence == delay_K).
# The live projection generalizes to (1 + alpha*h/g, -alpha*h/g) from the ring's
# REAL recorded ticks; see LookaheadProjector.project. (The trailing 0.0 keeps
# proj_info["coeffs"] a stable 3-tuple for the fire log line.)
FIXED_LINEAR_COEFFS = (2.0, -1.0, 0.0)

# Which rollouts the anchor consumes when the look-ahead projector is on.
#   "auto"          -> resolves to "current_step" when look-ahead is enabled,
#                      else "stale_paired" (matching rollouts are THE DEFAULT
#                      whenever weight projection is ON; zero effect when OFF).
#   "stale_paired"  -> today's exact behavior: the replayed t-delay_K batch in
#                      replay mode. (Legacy non-replay mode ALREADY consumes the
#                      current tick's batch, so this option only changes
#                      behavior in replay mode.)
#   "current_step"  -> replay mode consumes a copy of the CURRENT tick's batch.
#                      These trajectories are time-aligned with the forecast
#                      target tick, but were generated by the live fast actor,
#                      not by projected theta_hat[t]. Config-validated to require
#                      the projector ON (stale-weights + fresh-rollouts is an
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
    return str(getattr(anchor_cfg, "lookahead_mode", MODE_DISABLED)) in (
        MODE_FIXED_LINEAR,
        MODE_RANK1_RELEX,
    )


def rank1_relex_enabled(anchor_cfg) -> bool:
    """True iff the pure sliding rank-1 RELEX projector is active."""
    return lookahead_enabled(anchor_cfg) and (
        str(getattr(anchor_cfg, "lookahead_mode", MODE_DISABLED)) == MODE_RANK1_RELEX
    )


def lookahead_num_source_points(anchor_cfg) -> int:
    """Number of fire-aligned source snapshots the projector consumes.

    ``fixed_linear`` is FIRST-ORDER (``S0 = theta[t-K]``, ``S1 = theta[t-2K]``)
    so it uses 2. ``rank1_relex`` uses its complete configured sliding window,
    including the oldest base checkpoint. Returns 0 when look-ahead is disabled.
    """
    if not lookahead_enabled(anchor_cfg):
        return 0
    if rank1_relex_enabled(anchor_cfg):
        return int(getattr(anchor_cfg, "lookahead_window_snapshots", 4))
    return 2


def lookahead_min_points(anchor_cfg) -> int:
    """Ring snapshots required before the projector engages (E3).

    Reads ``anchor.lookahead_min_snapshots``: ``-1`` (default) ⇒ the mode's full
    source count :func:`lookahead_num_source_points` (2 for fixed_linear) —
    today's behavior. A concrete value (config-validated to ``[2, n_points]``)
    lets the projector engage at the earliest mathematically-legal fire (2 =
    fire 2). For ``rank1_relex``, the history continues growing and then sliding
    up to its configured complete window after this threshold is reached. This
    is the SINGLE readiness threshold the ring keys :meth:`ready` on, so the
    no_correct skip gate and the projected-vs-fallback decision share it (a
    second hardcoded ``n_points`` check would silently extend the skip window).
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
    reproduces the frozen AsyncPP seed ``(2, -1)`` = full catch-up to the
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


def compute_theta_hat(sources: list, coeffs, *, target_substrs) -> tuple:
    """Materialize the per-target extrapolated weights ``theta_hat`` (CPU/full).

    Args:
        sources: list ``[S0, S1]`` of snapshot dicts ``{canon_name -> full param
            tensor}`` ordered NEWEST-first — ``S0 = theta[t-K]``,
            ``S1 = theta[t-2K]`` — keyed by CANONICAL name. Both dicts cover the
            same key set (full-param snapshots).
        coeffs: the affine combination ``(a1, a2, ...)`` applied to
            ``(S0, S1)``. Only the first two entries are used (fixed_linear is
            first-order); any trailing entries are ignored.
        target_substrs: the decoder-matrix selector. A param is extrapolated iff
            it is a target AND 2D; every OTHER param takes ``S0`` (the raw stale
            weight) unchanged — the LayerNorm/embedding exclusion.

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
    a1, a2 = float(coeffs[0]), float(coeffs[1])

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
        acc = a1 * p0.to(torch.float32) + a2 * p1.to(torch.float32)
        theta_hat[name] = acc.to(p0.dtype)
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

    Holds the last ``n_points`` fire-aligned snapshots (2 for fixed_linear),
    NEWEST-first via :meth:`sources`. Pure container: no collectives, no RNG,
    CPU-testable.
    """

    def __init__(self, n_points: int, min_points: Optional[int] = None):
        assert n_points >= 2, f"look-ahead ring needs >= 2 source points, got {n_points}"
        self.n_points = int(n_points)
        # Readiness threshold (E3). Defaults to n_points (today's behavior);
        # a smaller min_points lets :meth:`ready` engage the projector at the
        # earliest legal fire while retention still keeps n_points. Must be in
        # [2, n_points].
        self.min_points = int(min_points) if min_points is not None else self.n_points
        assert 2 <= self.min_points <= self.n_points, (
            f"look-ahead ring min_points must be in [2, n_points={self.n_points}]; got {self.min_points}"
        )
        # OrderedDict[true_tick -> snapshot dict]; insertion order == tick order.
        self._snaps: OrderedDict[int, dict] = OrderedDict()
        # Bound: n_points fire snapshots. Track the peak for the bounded-memory report.
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
        """Return ``[S0, S1]`` NEWEST-first with their TRUE ticks.

        Returns ``(snaps, ticks)`` where ``snaps[0]`` is the newest retained
        (``theta[t-K]``), ``snaps[1]`` the next (``theta[t-K-cadence]``), and
        ``ticks`` the matching TRUE snapshot ticks (newest-first) — exactly
        the inputs :meth:`LookaheadProjector.project` needs to compute the real
        horizon. Returns ``(None, None)`` until :meth:`ready`. Returns the
        ``min(len, n_points)`` newest.
        """
        if not self.ready():
            return None, None
        items = list(self._snaps.items())  # oldest-first
        items = items[-self.n_points :]  # the n_points newest (clamps to len)
        items.reverse()  # newest-first
        ticks = [int(t) for t, _s in items]
        snaps = [s for _t, s in items]
        return snaps, ticks

    def get(self, tick: int):
        """Return the retained snapshot whose TRUE tick is ``tick`` (or None).

        Exact-match only — never an approximation. Used to assert eviction /
        retention (the ring is a plain true-tick-keyed container).
        """
        return self._snaps.get(int(tick))

    def total_retained(self) -> int:
        """Total full-param source snapshots held."""
        return len(self._snaps)

    @property
    def ticks(self) -> list:
        return list(self._snaps.keys())


class LookaheadProjector:
    """Per-block linear weight projector (fixed-linear).

    Stateless: the coefficients are the FROZEN AsyncPP seed, generalized to the
    ring's REAL recorded horizon in :meth:`project`. ``theta_hat`` is a pure
    per-element function of the DP-identical stale snapshots, so it is trivially
    cross-rank-identical — there is no learned per-block state and no RNG.
    """

    def __init__(self, anchor_cfg, target_substrs):
        self.target_substrs = tuple(target_substrs or ())
        self.mode = str(getattr(anchor_cfg, "lookahead_mode", MODE_DISABLED))
        self.n_points = lookahead_num_source_points(anchor_cfg)
        # SEED coefficients: (1+alpha, -alpha). alpha=1.0 reproduces the frozen
        # AsyncPP seed (2,-1) = full catch-up; alpha<1 = a shorter look-ahead
        # horizon (M4 horizon sweep). These are exact ONLY when the realized
        # horizon h equals the inter-source gap g (cadence == delay_K);
        # :meth:`project` generalizes to (1 + alpha*h/g, -alpha*h/g) from the
        # ring's REAL ticks and falls back to this seed when ticks are omitted.
        self.strength = lookahead_strength(anchor_cfg)
        self.coeffs = [1.0 + self.strength, -self.strength, 0.0]

    def project(self, sources: list, ticks: Optional[list] = None, fire_tick: Optional[int] = None):
        """Compute ``theta_hat`` from the source snapshots.

        ``ticks`` (TRUE snapshot ticks, newest-first, aligned with ``sources``)
        and ``fire_tick`` (the tick being projected TO) generalize the seed to
        the REAL horizon. With ``g = ticks[0] - ticks[1]`` (inter-source gap,
        = cadence) and ``h = fire_tick - ticks[0]`` (realized staleness of S0,
        >= delay_K), the effective coefficients are::

            (1 + alpha*h/g, -alpha*h/g)

        — the linear extrapolation that lands ON ``fire_tick`` (the paper's
        Eq 4 at ``alpha == 1``; see the module docstring). At ``h == g``
        (cadence == delay_K, the 20/20 and 5/5 operating points) this reduces
        EXACTLY to the frozen seed, so prior behavior is unchanged there. The
        seed alone lands short/long whenever ``h != g`` — that was the original
        naive-linear bug this signature fixes. Omitting ``ticks``/``fire_tick``
        (unit tests, seed-only callers) falls back to the seed coefficients.

        Returns ``(theta_hat, excluded_names, proj_info)`` where ``proj_info``
        carries the realized ``{"h", "g", "coeffs"}`` for fire logging (``h``/
        ``g`` are ``None`` on the seed fallback).
        """
        coeffs = list(self.coeffs)
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
        theta_hat, excluded = compute_theta_hat(sources, coeffs, target_substrs=self.target_substrs)
        return theta_hat, excluded, {"h": h, "g": g, "coeffs": tuple(coeffs)}


def _rank1_tick(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Rank1ProjectionError(f"rank1_relex {label} must be an integer optimizer tick; got {value!r}")
    tick = int(value)
    if tick < 0:
        raise Rank1ProjectionError(f"rank1_relex {label} must be >= 0; got {tick}")
    return tick


def _validate_rank1_timeline(ticks, target_tick) -> tuple[list[int], int]:
    clean = [_rank1_tick(t, f"history tick[{i}]") for i, t in enumerate(ticks)]
    if len(clean) < 2:
        raise Rank1ProjectionError(f"rank1_relex needs at least 2 checkpoints (base + 1 delta); got {len(clean)}")
    if any(b <= a for a, b in zip(clean, clean[1:], strict=False)):
        raise Rank1ProjectionError(f"rank1_relex history ticks must be unique and strictly increasing; got {clean}")
    target = _rank1_tick(target_tick, "target_tick")
    if target <= clean[-1]:
        raise Rank1ProjectionError(f"rank1_relex target_tick={target} must be newer than latest exact tick={clean[-1]}")
    return clean, target


class Rank1SnapshotHistory:
    """Strict sliding history for exact generator checkpoints.

    The first pre-update generator snapshot is retained by reference as the
    local base. Later entries are admitted only by the engine's exact delayed
    transfer path. Duplicate and out-of-order transfers are excluded without
    mutating the window; fixed-linear's permissive overwrite ring is untouched.
    ``min_snapshots`` controls readiness only: retained history continues to
    grow and then slide at ``window_snapshots``.
    """

    def __init__(self, window_snapshots: int = 4, *, min_snapshots: Optional[int] = None):
        if isinstance(window_snapshots, bool) or not isinstance(window_snapshots, Integral):
            raise Rank1ProjectionError(
                f"rank1_relex window_snapshots must be an integer >= 2; got {window_snapshots!r}"
            )
        self.window_snapshots = int(window_snapshots)
        if self.window_snapshots < 2:
            raise Rank1ProjectionError(f"rank1_relex window_snapshots must be >= 2; got {self.window_snapshots}")
        if min_snapshots is None:
            min_snapshots = self.window_snapshots
        if isinstance(min_snapshots, bool) or not isinstance(min_snapshots, Integral):
            raise Rank1ProjectionError(
                f"rank1_relex min_snapshots must be an integer in [2, {self.window_snapshots}]; got {min_snapshots!r}"
            )
        self.min_snapshots = int(min_snapshots)
        if not 2 <= self.min_snapshots <= self.window_snapshots:
            raise Rank1ProjectionError(
                f"rank1_relex min_snapshots must be in [2, {self.window_snapshots}]; got {self.min_snapshots}"
            )
        self._snaps: OrderedDict[int, dict] = OrderedDict()
        # Lightweight schema only: retaining base tensor references here would
        # keep the first full-model checkpoint alive after the window slides.
        self._schema: Optional[dict[str, tuple[tuple[int, ...], torch.dtype, torch.device]]] = None
        self.peak_retained = 0

    def _validate_snapshot(
        self,
        snapshot,
        *,
        checkpoint: int,
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype, torch.device]]:
        """Validate an exact checkpoint before it can drive Q or history.

        Warmup Q refreshes consume raw exact transfers before the projector is
        ready. Validation therefore belongs at history admission,
        not only in :meth:`Rank1RelexProjector.project`: otherwise malformed
        weights could mutate Q several fires before the projector fails.
        """
        if not isinstance(snapshot, dict) or not snapshot:
            raise Rank1ProjectionError("rank1_relex checkpoint snapshot must be a non-empty dict")

        keys = set(snapshot)
        if self._schema is not None and keys != set(self._schema):
            expected_keys = set(self._schema)
            missing = sorted(expected_keys - keys)[:5]
            extra = sorted(keys - expected_keys)[:5]
            raise Rank1ProjectionError(
                f"rank1_relex checkpoint {checkpoint} key mismatch: missing={missing} extra={extra}"
            )

        schema: dict[str, tuple[tuple[int, ...], torch.dtype, torch.device]] = {}
        for name in sorted(keys):
            tensor = _validate_rank1_checkpoint_tensor(
                snapshot[name],
                name=name,
                checkpoint=checkpoint,
                expected=None,
                chunk_numel=DEFAULT_RANK1_CHUNK_NUMEL,
            )
            metadata = (tuple(tensor.shape), tensor.dtype, tensor.device)
            if self._schema is not None:
                expected_shape, expected_dtype, expected_device = self._schema[name]
                if metadata[0] != expected_shape:
                    raise Rank1ProjectionError(
                        f"rank1_relex checkpoint {checkpoint} parameter {name!r} shape mismatch: "
                        f"expected {expected_shape}, got {metadata[0]}"
                    )
                if metadata[1] != expected_dtype:
                    raise Rank1ProjectionError(
                        f"rank1_relex checkpoint {checkpoint} parameter {name!r} dtype mismatch: "
                        f"expected {expected_dtype}, got {metadata[1]}"
                    )
                if metadata[2] != expected_device:
                    raise Rank1ProjectionError(
                        f"rank1_relex checkpoint {checkpoint} parameter {name!r} device mismatch: "
                        f"expected {expected_device}, got {metadata[2]}"
                    )
            schema[name] = metadata
        return schema

    def seed_base(self, tick: int, snapshot: dict) -> bool:
        """Retain the first generator snapshot by reference; first call wins."""
        tick = _rank1_tick(tick, "base tick")
        if self._snaps:
            return False
        schema = self._validate_snapshot(snapshot, checkpoint=tick)
        self._schema = schema
        self._snaps[tick] = snapshot
        self.peak_retained = 1
        return True

    def admit_exact(self, tick: int, snapshot: dict) -> bool:
        """Admit a strictly newer exact delayed transfer and slide immediately."""
        tick = _rank1_tick(tick, "exact transfer tick")
        if not self._snaps:
            raise Rank1ProjectionError("rank1_relex exact transfer arrived before the local base was seeded")
        latest_tick = next(reversed(self._snaps))
        if tick <= latest_tick:
            if self.ready():
                raise Rank1ProjectionError(
                    f"rank1_relex ready history requires a strictly newer exact transfer; "
                    f"got tick={tick} after latest={latest_tick}"
                )
            # Pre-ready duplicate/out-of-order transfers still drive the raw,
            # paired Q-only forward, so validate them before returning false.
            self._validate_snapshot(snapshot, checkpoint=tick)
            return False
        self._validate_snapshot(snapshot, checkpoint=tick)
        self._snaps[tick] = snapshot
        while len(self._snaps) > self.window_snapshots:
            self._snaps.popitem(last=False)
        self.peak_retained = max(self.peak_retained, len(self._snaps))
        assert len(self._snaps) <= self.window_snapshots
        return True

    def ready(self) -> bool:
        return len(self._snaps) >= self.min_snapshots

    def sources(self):
        """Return all retained ready snapshots and ticks, both oldest-first."""
        if not self.ready():
            return None, None
        items = list(self._snaps.items())
        return [snapshot for _tick, snapshot in items], [int(tick) for tick, _snapshot in items]

    def latest(self):
        if not self._snaps:
            return None, None
        tick = next(reversed(self._snaps))
        return self._snaps[tick], int(tick)

    def total_retained(self) -> int:
        return len(self._snaps)

    @property
    def ticks(self) -> list[int]:
        return list(self._snaps.keys())


def _rank1_delta_chunk(flat_snapshots, start: int, end: int) -> torch.Tensor:
    base = flat_snapshots[0][start:end].to(torch.float32)
    return torch.stack(
        [snapshot[start:end].to(torch.float32) - base for snapshot in flat_snapshots[1:]],
        dim=0,
    )


def _validate_rank1_checkpoint_tensor(
    tensor,
    *,
    name: str,
    checkpoint: int,
    expected: Optional[torch.Tensor],
    chunk_numel: int,
) -> torch.Tensor:
    """Validate one exact-checkpoint value with bounded finite checks."""
    label = f"rank1_relex checkpoint {checkpoint} parameter {name!r}"
    if not isinstance(tensor, torch.Tensor):
        raise Rank1ProjectionError(f"{label} must be a torch.Tensor; got {type(tensor).__name__}")
    if expected is not None:
        if tensor.shape != expected.shape:
            raise Rank1ProjectionError(
                f"{label} shape mismatch: expected {tuple(expected.shape)}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected.dtype:
            raise Rank1ProjectionError(f"{label} dtype mismatch: expected {expected.dtype}, got {tensor.dtype}")
        if tensor.device != expected.device:
            raise Rank1ProjectionError(f"{label} device mismatch: expected {expected.device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise Rank1ProjectionError(f"{label} is non-contiguous; refusing an unbounded flatten copy")
    flat = tensor.view(-1)
    for start in range(0, flat.numel(), chunk_numel):
        end = min(start + chunk_numel, flat.numel())
        if not bool(torch.isfinite(flat[start:end]).all()):
            raise Rank1ProjectionError(f"{label} contains a non-finite value in flattened chunk [{start}:{end}]")
    return tensor


def _rank1_ols(ticks: list[int], coefficients: torch.Tensor) -> tuple[float, float, float]:
    t = torch.tensor(ticks, dtype=torch.float64, device=coefficients.device)
    c = coefficients.to(torch.float64)
    t_centered = t - t.mean()
    denom = torch.sum(t_centered.square())
    if not bool(torch.isfinite(denom)) or float(denom.item()) <= 0.0:
        raise Rank1ProjectionError(f"rank1_relex OLS timestamps are degenerate: {ticks}")
    c_mean = c.mean()
    slope_t = torch.sum(t_centered * (c - c_mean)) / denom
    intercept_t = c_mean - slope_t * t.mean()
    fitted = slope_t * t + intercept_t
    sse = torch.sum((c - fitted).square())
    sst = torch.sum((c - c_mean).square())
    r2_t = torch.ones((), dtype=torch.float64, device=c.device) if float(sst.item()) == 0.0 else 1.0 - sse / sst
    if not bool(torch.isfinite(slope_t)) or not bool(torch.isfinite(intercept_t)) or not bool(torch.isfinite(r2_t)):
        raise Rank1ProjectionError("rank1_relex OLS produced a non-finite slope, intercept, or R^2")
    return float(slope_t.item()), float(intercept_t.item()), float(r2_t.item())


def project_rank1_tensor(
    snapshots: list[torch.Tensor],
    ticks,
    target_tick,
    *,
    strength: float = 1.0,
    chunk_numel: int = DEFAULT_RANK1_CHUNK_NUMEL,
) -> tuple[torch.Tensor, dict]:
    """Project one tensor via a chunked rank-1 checkpoint trajectory.

    ``snapshots[0]`` is the window base and ``snapshots[-1]`` is the newest
    exact checkpoint. The zero base row is excluded, so W checkpoints produce
    W-1 cumulative deltas. Gram construction and right-vector recovery are
    bounded fp32 passes; coefficient OLS uses actual ticks in fp64. The final
    increment is pinned to the newest exact tensor, preserving its off-subspace
    residual.
    """
    clean_ticks, target = _validate_rank1_timeline(ticks, target_tick)
    if len(snapshots) != len(clean_ticks):
        raise Rank1ProjectionError(
            f"rank1_relex snapshot/tick count mismatch: {len(snapshots)} snapshots vs {len(clean_ticks)} ticks"
        )
    try:
        strength = float(strength)
    except (TypeError, ValueError) as exc:
        raise Rank1ProjectionError(f"rank1_relex strength must be finite and >= 0; got {strength!r}") from exc
    if not math.isfinite(strength) or strength < 0.0:
        raise Rank1ProjectionError(f"rank1_relex strength must be finite and >= 0; got {strength!r}")
    if isinstance(chunk_numel, bool) or not isinstance(chunk_numel, Integral) or int(chunk_numel) < 1:
        raise Rank1ProjectionError(f"rank1_relex chunk_numel must be an integer >= 1; got {chunk_numel!r}")
    chunk_numel = int(chunk_numel)

    if not snapshots or not all(isinstance(t, torch.Tensor) for t in snapshots):
        raise Rank1ProjectionError("rank1_relex tensor history must contain only torch.Tensor values")
    shape = snapshots[0].shape
    dtype = snapshots[0].dtype
    device = snapshots[0].device
    if not torch.is_floating_point(snapshots[0]):
        raise Rank1ProjectionError(f"rank1_relex tensor history must be floating point; got {dtype}")
    for i, tensor in enumerate(snapshots):
        if tensor.shape != shape:
            raise Rank1ProjectionError(
                f"rank1_relex tensor shape mismatch at checkpoint {i}: expected {tuple(shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device != device:
            raise Rank1ProjectionError(
                f"rank1_relex tensor device mismatch at checkpoint {i}: expected {device}, got {tensor.device}"
            )
        if tensor.dtype != dtype:
            raise Rank1ProjectionError(
                f"rank1_relex tensor dtype mismatch at checkpoint {i}: expected {dtype}, got {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise Rank1ProjectionError(
                f"rank1_relex tensor at checkpoint {i} is non-contiguous; refusing an unbounded flatten copy"
            )

    flat_snapshots = [tensor.view(-1) for tensor in snapshots]
    numel = flat_snapshots[0].numel()
    n_deltas = len(snapshots) - 1
    gram = torch.zeros((n_deltas, n_deltas), dtype=torch.float32, device=device)
    for start in range(0, numel, chunk_numel):
        end = min(start + chunk_numel, numel)
        deltas = _rank1_delta_chunk(flat_snapshots, start, end)
        if not bool(torch.isfinite(deltas).all()):
            raise Rank1ProjectionError(
                f"rank1_relex encountered a non-finite checkpoint delta in flattened chunk [{start}:{end}]"
            )
        gram.add_(deltas @ deltas.transpose(0, 1))

    energy = float(torch.trace(gram).item())
    if not math.isfinite(energy):
        raise Rank1ProjectionError("rank1_relex Gram energy is non-finite")
    latest = snapshots[-1]
    horizon = target - clean_ticks[-1]
    if energy <= 0.0:
        return latest, {
            "sigma": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "evr": 0.0,
            "r2": 1.0,
            "zero_motion": True,
            "delta_count": n_deltas,
            "prediction_horizon": horizon,
        }

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    except Exception as exc:
        raise Rank1ProjectionError(f"rank1_relex Gram eigensolver failed: {exc}") from exc
    if not bool(torch.isfinite(eigenvalues).all()) or not bool(torch.isfinite(eigenvectors).all()):
        raise Rank1ProjectionError("rank1_relex Gram eigensolver produced non-finite values")
    positive = eigenvalues.clamp_min(0.0)
    total_positive = float(positive.sum().item())
    lambda1 = float(positive[-1].item())
    if not math.isfinite(total_positive) or not math.isfinite(lambda1):
        raise Rank1ProjectionError("rank1_relex eigenvalue energy is non-finite")
    if total_positive <= 0.0 or lambda1 <= 0.0:
        return latest, {
            "sigma": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "evr": 0.0,
            "r2": 1.0,
            "zero_motion": True,
            "delta_count": n_deltas,
            "prediction_horizon": horizon,
            "fit_kind": "two_checkpoint_secant" if n_deltas == 1 else "rank1_ols",
        }

    sigma = math.sqrt(lambda1)
    if sigma <= max(sigma * 1e-6, 1e-12):
        return latest, {
            "sigma": sigma,
            "slope": 0.0,
            "intercept": 0.0,
            "evr": lambda1 / total_positive,
            "r2": 1.0,
            "zero_motion": True,
            "delta_count": n_deltas,
            "prediction_horizon": horizon,
            "fit_kind": "two_checkpoint_secant" if n_deltas == 1 else "rank1_ols",
        }

    u1 = eigenvectors[:, -1]
    coefficients = u1 * sigma
    if n_deltas == 1:
        # A literal two-checkpoint window has one nonzero cumulative delta, so
        # the paper's free-intercept fit over ``clean_ticks[1:]`` cannot identify
        # a slope.  Use the known base coordinate c(t_base)=0 as the second
        # point.  This is exactly the per-tensor secant
        #
        #   latest + alpha * horizon/gap * (latest - base),
        #
        # i.e. naive two-point linear extrapolation.  Keep this fallback local
        # to W=2: adding the zero-base point to W>=3 would change the established
        # RELEX/W4 fit and invalidate the completed reference run.
        zero = torch.zeros(1, dtype=coefficients.dtype, device=coefficients.device)
        fit_coefficients = torch.cat((zero, coefficients))
        fit_ticks = clean_ticks
        fit_kind = "two_checkpoint_secant"
    else:
        fit_coefficients = coefficients
        fit_ticks = clean_ticks[1:]
        fit_kind = "rank1_ols"
    slope, intercept, r2 = _rank1_ols(fit_ticks, fit_coefficients)
    scale = strength * slope * float(horizon)
    if not math.isfinite(scale):
        raise Rank1ProjectionError("rank1_relex pinned prediction scale is non-finite")
    if scale == 0.0:
        projected = latest
    else:
        projected = torch.empty_like(latest, memory_format=torch.preserve_format)
        projected_flat = projected.view(-1)
        latest_flat = latest.view(-1)
        for start in range(0, numel, chunk_numel):
            end = min(start + chunk_numel, numel)
            deltas = _rank1_delta_chunk(flat_snapshots, start, end)
            v1 = torch.sum(u1[:, None] * deltas, dim=0) / sigma
            out = latest_flat[start:end].to(torch.float32) + scale * v1
            if not bool(torch.isfinite(out).all()):
                raise Rank1ProjectionError(
                    f"rank1_relex produced a non-finite projected tensor in flattened chunk [{start}:{end}]"
                )
            projected_flat[start:end].copy_(out.to(latest.dtype))

    return projected, {
        "sigma": sigma,
        "slope": slope,
        "intercept": intercept,
        "evr": lambda1 / total_positive,
        "r2": r2,
        "zero_motion": False,
        "delta_count": n_deltas,
        "prediction_horizon": horizon,
        "fit_kind": fit_kind,
    }


class Rank1RelexProjector:
    """Pure sliding RELEX projector over every exact parameter trajectory.

    RELEX fits one rank-1 temporal subspace *per floating parameter tensor*, not
    only for decoder matrices. The constructor deliberately has no target
    selector: fixed-linear and spectral correction retain their decoder-only
    selectors, but those selectors must never narrow rank1_relex coverage.
    """

    def __init__(
        self,
        anchor_cfg,
        *,
        min_snapshots: Optional[int] = None,
        chunk_numel: int = DEFAULT_RANK1_CHUNK_NUMEL,
    ):
        self.window_snapshots = int(getattr(anchor_cfg, "lookahead_window_snapshots", 4))
        if min_snapshots is None:
            configured_min = int(getattr(anchor_cfg, "lookahead_min_snapshots", -1))
            min_snapshots = self.window_snapshots if configured_min == -1 else configured_min
        if isinstance(min_snapshots, bool) or not isinstance(min_snapshots, Integral):
            raise Rank1ProjectionError(
                f"rank1_relex min_snapshots must be an integer in [2, {self.window_snapshots}]; got {min_snapshots!r}"
            )
        self.min_snapshots = int(min_snapshots)
        if not 2 <= self.min_snapshots <= self.window_snapshots:
            raise Rank1ProjectionError(
                f"rank1_relex min_snapshots must be in [2, {self.window_snapshots}]; got {self.min_snapshots}"
            )
        self.strength = lookahead_strength(anchor_cfg)
        self.chunk_numel = int(chunk_numel)

    def project(self, sources: list[dict], ticks, target_tick: int):
        clean_ticks, target = _validate_rank1_timeline(ticks, target_tick)
        if len(sources) != len(clean_ticks):
            raise Rank1ProjectionError(
                f"rank1_relex snapshot/tick count mismatch: got {len(sources)} snapshots and {len(clean_ticks)} ticks"
            )
        if not self.min_snapshots <= len(clean_ticks) <= self.window_snapshots:
            raise Rank1ProjectionError(
                f"rank1_relex requires between min={self.min_snapshots} and "
                f"W={self.window_snapshots} checkpoints; got {len(clean_ticks)}"
            )
        if not sources or not all(isinstance(snapshot, dict) and snapshot for snapshot in sources):
            raise Rank1ProjectionError("rank1_relex history must contain non-empty checkpoint dicts")

        base_keys = set(sources[0])
        for i, snapshot in enumerate(sources[1:], start=1):
            keys = set(snapshot)
            if keys != base_keys:
                missing = sorted(base_keys - keys)[:5]
                extra = sorted(keys - base_keys)[:5]
                raise Rank1ProjectionError(f"rank1_relex checkpoint {i} key mismatch: missing={missing} extra={extra}")

        # Validate all retained exact checkpoints before constructing any
        # projected tensor. Every unique named parameter, including norms,
        # biases, embeddings, and an untied LM head, is a RELEX trajectory.
        # A malformed tensor must fail before the clone or M/Q is touched.
        for name in sorted(base_keys):
            expected = _validate_rank1_checkpoint_tensor(
                sources[0][name],
                name=name,
                checkpoint=0,
                expected=None,
                chunk_numel=self.chunk_numel,
            )
            for i, snapshot in enumerate(sources[1:], start=1):
                _validate_rank1_checkpoint_tensor(
                    snapshot[name],
                    name=name,
                    checkpoint=i,
                    expected=expected,
                    chunk_numel=self.chunk_numel,
                )
        latest = sources[-1]
        target_names = sorted(name for name, tensor in latest.items() if torch.is_floating_point(tensor))
        passthrough_names = sorted(set(latest) - set(target_names))
        if not target_names:
            raise Rank1ProjectionError("rank1_relex found no floating parameter tensors")

        theta_hat = dict(latest)
        stats = []
        for name in target_names:
            tensor_history = []
            expected_shape = latest[name].shape
            for i, snapshot in enumerate(sources):
                tensor = snapshot.get(name)
                if not isinstance(tensor, torch.Tensor):
                    raise Rank1ProjectionError(
                        f"rank1_relex parameter {name!r} is missing or non-tensor at checkpoint {i}"
                    )
                if tensor.shape != expected_shape:
                    raise Rank1ProjectionError(
                        f"rank1_relex parameter {name!r} shape mismatch at checkpoint {i}: "
                        f"expected {tuple(expected_shape)}, got {tuple(tensor.shape)}"
                    )
                tensor_history.append(tensor)
            try:
                projected, tensor_stats = project_rank1_tensor(
                    tensor_history,
                    clean_ticks,
                    target,
                    strength=self.strength,
                    chunk_numel=self.chunk_numel,
                )
            except Rank1ProjectionError as exc:
                raise Rank1ProjectionError(f"rank1_relex parameter {name!r} failed: {exc}") from exc
            theta_hat[name] = projected
            stats.append(tensor_stats)

        evrs = [float(s["evr"]) for s in stats]
        r2s = [float(s["r2"]) for s in stats]
        zero_motion = sum(bool(s["zero_motion"]) for s in stats)
        info = {
            "history_ticks": tuple(clean_ticks),
            "checkpoint_count": len(clean_ticks),
            "delta_count": len(clean_ticks) - 1,
            "window_span": clean_ticks[-1] - clean_ticks[0],
            "target_tick": target,
            "prediction_horizon": target - clean_ticks[-1],
            "targets_projected": len(stats),
            "nonfloating_tensors_passthrough": len(passthrough_names),
            "zero_motion_tensors": int(zero_motion),
            "evr_mean": sum(evrs) / len(evrs),
            "evr_min": min(evrs),
            "r2_mean": sum(r2s) / len(r2s),
            "r2_min": min(r2s),
            "fit_kind": "two_checkpoint_secant" if len(clean_ticks) == 2 else "rank1_ols",
        }
        for key, value in info.items():
            if key != "history_ticks" and isinstance(value, float) and not math.isfinite(value):
                raise Rank1ProjectionError(f"rank1_relex aggregate {key} is non-finite")
        return theta_hat, info


def validate_rank1_broadcast_receipts(
    *,
    q_only: bool,
    dp_multi: bool,
    q_receipts,
    m_receipts,
    spectral_enabled: bool,
) -> None:
    """Enforce Q-only versus full-rank1 distributed broadcast contracts."""
    if q_only and m_receipts is not None:
        raise RuntimeError("rank1_relex q_only warmup must not broadcast M_anchor")
    if not dp_multi:
        return
    if not q_receipts:
        raise RuntimeError("rank1_relex anchor-owned Q broadcast returned no multi-rank receipts")
    if not q_only and spectral_enabled and not m_receipts:
        raise RuntimeError("rank1_relex full anchor M broadcast returned no multi-rank receipts")
