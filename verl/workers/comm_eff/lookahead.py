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

**LayerNorm / embedding / lm_head excluded.** Per the linearity paper (Fig 8 /
App A.2) norm + embedding layers have LOW linearity; extrapolating them injects
error. Excluded params take the raw ``theta[t-K]`` (no projection). The decoder
weight-matrix scope reuses the existing ``spectral.target_substr`` selector
(``comm_eff.py:228-238``) so the exclusion set is exactly the merger's
non-target set.

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
from collections import OrderedDict
from typing import Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "LOOKAHEAD_MODES",
    "MODE_DISABLED",
    "MODE_FIXED_LINEAR",
    "LOOKAHEAD_ROLLOUT_SOURCES",
    "FIXED_LINEAR_COEFFS",
    "lookahead_enabled",
    "lookahead_num_source_points",
    "lookahead_min_points",
    "resolve_lookahead_rollout_source",
    "is_lookahead_target",
    "compute_theta_hat",
    "LookaheadSnapshotRing",
    "LookaheadProjector",
]


# The look-ahead modes. ``disabled`` (default) is a strict no-op — the anchor
# forwards from the raw stale ``theta[t-K]`` exactly as today. ``fixed_linear``
# uses the FROZEN AsyncPP seed (= the RLVR-linearity paper's Eq 4 weight-space
# extrapolation). Named constants are used for internal dispatch; the whitelist
# tuple is the config single-source.
MODE_DISABLED = "disabled"
MODE_FIXED_LINEAR = "fixed_linear"
LOOKAHEAD_MODES = (MODE_DISABLED, MODE_FIXED_LINEAR)

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
    return str(getattr(anchor_cfg, "lookahead_mode", MODE_DISABLED)) == MODE_FIXED_LINEAR


def lookahead_num_source_points(anchor_cfg) -> int:
    """Number of fire-aligned source snapshots the projector consumes.

    ``fixed_linear`` is FIRST-ORDER (``S0 = theta[t-K]``, ``S1 = theta[t-2K]``)
    so it uses 2. Returns 0 when look-ahead is disabled.
    """
    return 2 if lookahead_enabled(anchor_cfg) else 0


def lookahead_min_points(anchor_cfg) -> int:
    """Ring snapshots required before the projector engages (E3).

    Reads ``anchor.lookahead_min_snapshots``: ``-1`` (default) ⇒ the mode's full
    source count :func:`lookahead_num_source_points` (2 for fixed_linear) —
    today's behavior. A concrete value (config-validated to ``[2, n_points]``)
    lets the projector engage at the earliest mathematically-legal fire (2 =
    fire 2). This is the SINGLE readiness threshold the ring keys :meth:`ready`
    on, so the no_correct skip gate and the projected-vs-fallback decision share
    it (a second hardcoded ``n_points`` check would silently extend the skip
    window). Returns 0 when look-ahead is disabled.
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
