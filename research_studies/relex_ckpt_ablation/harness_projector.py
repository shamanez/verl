"""Faithful, CPU-only port of the two-circuit anchor weight projector.

This is a *self-contained* reproduction of the exact math the live harness runs
in ``verl/workers/comm_eff/lookahead.py`` (branch ``exp/rank1-relax``):

  * ``project_rank1_tensor``      -> :func:`project_rank1_tensor`  (rank1_relex mode)
  * ``Rank1RelexProjector``       -> :func:`rank1_relex_project`
  * ``LookaheadProjector`` /
    ``compute_theta_hat``          -> :func:`fixed_linear_project`  (fixed_linear mode)

It also implements the *contrast* baselines the ablation compares against:

  * ``relex_from_base_project``   -> the ORIGINAL RELEX paper reconstruction
                                     (theta_0 + c_pred @ V_r.T), i.e. rebuild the
                                     whole delta in the rank-r subspace from the
                                     window base. Supports rank r >= 1.
  * ``stale_baseline``            -> the "no projection" control: just reuse the
                                     newest exact checkpoint (theta[t-K]).

WHY A PORT (not an import): the study runs on a laptop with no distributed
runtime and no FSDP. The math in ``project_rank1_tensor`` is pure per-tensor
tensor algebra, so it ports exactly. :func:`assert_matches_live_harness` (opt-in,
run inside the ``exp/rank1-relax`` worktree with torch available) proves this port
is numerically identical to the live function on random tensors, so results are
attributable to the real system, not a re-derivation.

KEY SEMANTICS (must match the harness exactly):
  - Deltas are CUMULATIVE vs the WINDOW BASE ``snapshots[0]`` (NOT consecutive):
    ``delta_i = snapshots[i] - snapshots[0]``, i = 1..W-1  =>  W checkpoints give
    W-1 deltas; the zero base row is excluded.
  - Rank-1 via the Gram trick: ``G = D D^T`` [n_deltas, n_deltas], top eigenpair.
  - Temporal coefficients ``c = u1 * sigma`` [n_deltas].
  - W == 2 (one delta) SPECIAL CASE: a single delta cannot identify an OLS slope
    over ``ticks[1:]``, so the KNOWN base coordinate ``c(t_base)=0`` is added as a
    second fit point -> exact two-point per-tensor SECANT
    ``latest + alpha*(h/g)*(latest - base)``. (fit_kind = "two_checkpoint_secant")
  - W >= 3: OLS ``c_i ~ slope*tick_i + intercept`` over ``ticks[1:]`` (base row
    NOT added). (fit_kind = "rank1_ols")
  - PREDICTION IS PINNED TO ``latest`` (the newest exact checkpoint), adding only
    the incremental rank-1 motion:
        theta_hat = latest + (alpha * slope * horizon) * v1
    where v1 = unit rank-1 spatial direction, horizon = target - ticks[-1].
    This PRESERVES the newest checkpoint's off-subspace residual — the crucial
    difference from the paper's rebuild-from-base reconstruction.

Source of truth: verl/workers/comm_eff/lookahead.py:731-1044 (project_rank1_tensor,
Rank1RelexProjector) and :282-329 (compute_theta_hat).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = [
    "project_rank1_tensor",
    "rank1_relex_project",
    "fixed_linear_project",
    "relex_from_base_project",
    "stale_baseline",
    "assert_matches_live_harness",
]


# ----------------------------------------------------------------------------- #
# rank1_relex mode  (the harness's primary weight projector)
# ----------------------------------------------------------------------------- #
def project_rank1_tensor(
    snapshots: list[torch.Tensor],
    ticks: list[int],
    target_tick: int,
    *,
    strength: float = 1.0,
    rank: int = 1,
) -> tuple[torch.Tensor, dict]:
    """Faithful port of verl.workers.comm_eff.lookahead.project_rank1_tensor.

    ``snapshots[0]`` = window base, ``snapshots[-1]`` = newest exact checkpoint.
    ``ticks`` are the checkpoint time indices (strictly increasing), ``target_tick``
    is the step being predicted (> ticks[-1]).

    ``rank`` is an ABLATION EXTENSION not in the live code (which is hard rank-1):
    rank=1 is byte-identical to the harness; rank>1 keeps the top-r subspace and
    fits/extrapolates each temporal component, then pins the sum to ``latest``.
    Use rank=1 for the faithful comparison; rank>1 answers "is rank-1 enough?".
    """
    clean_ticks = [int(t) for t in ticks]
    assert len(clean_ticks) >= 2, "need >= 2 checkpoints (base + 1 delta)"
    assert all(b > a for a, b in zip(clean_ticks, clean_ticks[1:], strict=False)), "ticks must strictly increase"
    target = int(target_tick)
    assert target > clean_ticks[-1], "target must be newer than newest exact tick"
    assert len(snapshots) == len(clean_ticks)

    latest = snapshots[-1]
    dtype = latest.dtype
    horizon = target - clean_ticks[-1]
    n_deltas = len(snapshots) - 1
    rank = min(int(rank), n_deltas)

    flat = [t.reshape(-1).to(torch.float32) for t in snapshots]
    base = flat[0]
    D = torch.stack([flat[i] - base for i in range(1, len(flat))], dim=0)  # [n_deltas, d]

    gram = D @ D.transpose(0, 1)  # [n_deltas, n_deltas]
    energy = float(torch.trace(gram).item())
    zero_stats = dict(
        sigma=0.0,
        slope=0.0,
        intercept=0.0,
        evr=0.0,
        r2=1.0,
        zero_motion=True,
        delta_count=n_deltas,
        prediction_horizon=horizon,
        fit_kind="two_checkpoint_secant" if n_deltas == 1 else "rank1_ols",
    )
    if energy <= 0.0:
        return latest.clone(), zero_stats

    eigvals, eigvecs = torch.linalg.eigh(gram)  # ascending
    eigvals = eigvals.clamp_min(0.0)
    total_positive = float(eigvals.sum().item())
    lambda1 = float(eigvals[-1].item())
    if total_positive <= 0.0 or lambda1 <= 0.0:
        return latest.clone(), zero_stats
    sigma = math.sqrt(lambda1)
    if sigma <= max(sigma * 1e-6, 1e-12):
        s = dict(zero_stats)
        s["sigma"] = sigma
        s["evr"] = lambda1 / total_positive
        return latest.clone(), s

    # --- rank-1 (harness default) --------------------------------------------
    if rank <= 1:
        u1 = eigvecs[:, -1]  # [n_deltas]
        coefficients = u1 * sigma  # temporal coeffs c_i
        if n_deltas == 1:
            # two-checkpoint secant: add the known base coordinate c(t_base)=0.
            fit_coeffs = torch.cat((torch.zeros(1, dtype=coefficients.dtype), coefficients))
            fit_ticks = clean_ticks
            fit_kind = "two_checkpoint_secant"
        else:
            fit_coeffs = coefficients
            fit_ticks = clean_ticks[1:]
            fit_kind = "rank1_ols"
        slope, intercept, r2 = _ols(fit_ticks, fit_coeffs)
        scale = strength * slope * float(horizon)
        if scale == 0.0:
            projected = latest.clone()
        else:
            v1 = (u1[:, None] * D).sum(dim=0) / sigma  # [d] unit spatial dir
            out = latest.reshape(-1).to(torch.float32) + scale * v1
            projected = out.reshape(latest.shape).to(dtype)
        return projected, dict(
            sigma=sigma,
            slope=slope,
            intercept=intercept,
            evr=lambda1 / total_positive,
            r2=r2,
            zero_motion=False,
            delta_count=n_deltas,
            prediction_horizon=horizon,
            fit_kind=fit_kind,
        )

    # --- rank>1 ablation extension -------------------------------------------
    # Keep the top-r eigvecs; each temporal component c_k(t) gets its own OLS.
    idx = list(range(eigvals.numel() - 1, eigvals.numel() - 1 - rank, -1))
    increment = torch.zeros_like(base)
    r2s, evr_num = [], 0.0
    for k in idx:
        lam_k = float(eigvals[k].item())
        if lam_k <= 0.0:
            continue
        sig_k = math.sqrt(lam_k)
        u_k = eigvecs[:, k]
        c_k = u_k * sig_k
        if n_deltas == 1:
            fit_coeffs = torch.cat((torch.zeros(1, dtype=c_k.dtype), c_k))
            fit_ticks = clean_ticks
        else:
            fit_coeffs = c_k
            fit_ticks = clean_ticks[1:]
        slope_k, _, r2_k = _ols(fit_ticks, fit_coeffs)
        v_k = (u_k[:, None] * D).sum(dim=0) / sig_k
        increment = increment + (strength * slope_k * float(horizon)) * v_k
        r2s.append(r2_k)
        evr_num += lam_k
    projected = (latest.reshape(-1).to(torch.float32) + increment).reshape(latest.shape).to(dtype)
    return projected, dict(
        sigma=sigma,
        slope=float("nan"),
        intercept=float("nan"),
        evr=evr_num / total_positive,
        r2=(sum(r2s) / len(r2s) if r2s else 1.0),
        zero_motion=False,
        delta_count=n_deltas,
        prediction_horizon=horizon,
        fit_kind=f"rank{rank}_ols",
    )


def _ols(ticks, coefficients) -> tuple[float, float, float]:
    """OLS slope/intercept/R^2 of coefficients vs ticks (fp64), matching _rank1_ols."""
    t = torch.tensor([float(x) for x in ticks], dtype=torch.float64)
    c = coefficients.to(torch.float64)
    tc = t - t.mean()
    denom = float(torch.sum(tc * tc).item())
    assert denom > 0.0, f"degenerate ticks {ticks}"
    c_mean = c.mean()
    slope = torch.sum(tc * (c - c_mean)) / torch.sum(tc * tc)
    intercept = c_mean - slope * t.mean()
    fitted = slope * t + intercept
    sse = torch.sum((c - fitted) ** 2)
    sst = torch.sum((c - c_mean) ** 2)
    r2 = 1.0 if float(sst.item()) == 0.0 else float(1.0 - sse / sst)
    return float(slope.item()), float(intercept.item()), r2


def rank1_relex_project(sources: list[dict], ticks, target_tick, *, strength=1.0, rank=1):
    """Port of Rank1RelexProjector.project: project EVERY floating tensor.

    ``sources`` = list of {param_name -> tensor} snapshots, oldest-first; the last
    is the newest exact. Non-floating tensors pass through as the newest exact.
    Returns (theta_hat_dict, per_tensor_stats_dict).
    """
    latest = sources[-1]
    names = sorted(n for n, t in latest.items() if torch.is_floating_point(t))
    theta_hat = dict(latest)
    stats = {}
    for name in names:
        history = [s[name] for s in sources]
        proj, st = project_rank1_tensor(history, ticks, target_tick, strength=strength, rank=rank)
        theta_hat[name] = proj
        stats[name] = st
    return theta_hat, stats


# ----------------------------------------------------------------------------- #
# fixed_linear mode  (decoder-matrix-only 2-point extrapolation = AsyncPP / Eq.4)
# ----------------------------------------------------------------------------- #
def fixed_linear_project(s0: torch.Tensor, s1: torch.Tensor, *, h: int, g: int, strength=1.0):
    """theta_hat = (1 + alpha*h/g) * s0 - (alpha*h/g) * s1  (S0=theta[t-K], S1=theta[t-2K]).

    Port of compute_theta_hat's per-target arithmetic. At h==g, alpha==1 this is
    the frozen (2, -1) seed. This is the *decoder-matrix-only* projector; in the
    study apply it per 2-D decoder-matrix tensor and take S0 (stale) for the rest.
    """
    base_scale = float(strength) * (float(h) / float(g))
    a1, a2 = 1.0 + base_scale, -base_scale
    out = a1 * s0.to(torch.float32) + a2 * s1.to(torch.float32)
    return out.to(s0.dtype)


# ----------------------------------------------------------------------------- #
# CONTRAST baselines
# ----------------------------------------------------------------------------- #
def relex_from_base_project(snapshots, ticks, target_tick, *, rank=1, strength=1.0):
    """ORIGINAL RELEX reconstruction: theta_hat = base + (c_pred @ V_r.T).

    Rebuilds the whole predicted delta in the rank-r subspace from the WINDOW BASE
    (snapshots[0]), discarding the newest checkpoint's off-subspace residual. This
    is the paper's ``extrapolate`` mode (svd_extrapolation.py). Contrast with
    :func:`project_rank1_tensor` which pins to ``latest``.
    """
    clean_ticks = [int(t) for t in ticks]
    target = int(target_tick)
    base = snapshots[0].reshape(-1).to(torch.float32)
    flat = [t.reshape(-1).to(torch.float32) for t in snapshots]
    D = torch.stack([flat[i] - base for i in range(1, len(flat))], dim=0)  # [T-1, d]; row t = delta at ticks[t+1]
    n = D.shape[0]
    rank = min(int(rank), n)
    gram = D @ D.transpose(0, 1)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals = eigvals.clamp_min(0.0)
    order = list(range(eigvals.numel() - 1, -1, -1))[:rank]
    delta_pred = torch.zeros_like(base)
    # RELEX-consistent temporal fit: the base checkpoint (theta_0) has coefficient
    # 0 in the from-base subspace (its delta is 0). Include it so the OLS is
    # well-defined for every W (at W=2 this reduces the from-base predictor to the
    # SAME two-point secant as the pinned-to-latest projector).
    fit_ticks = clean_ticks
    for k in order:
        lam = float(eigvals[k].item())
        if lam <= 0:
            continue
        sig = math.sqrt(lam)
        u = eigvecs[:, k]
        c = u * sig  # temporal coeff per delta (non-base)
        v = (u[:, None] * D).sum(dim=0) / sig  # spatial dir
        c_full = torch.cat((torch.zeros(1, dtype=c.dtype), c))  # prepend base coeff = 0
        slope, intercept, _ = _ols(fit_ticks, c_full)
        c_pred = strength * (slope * target + intercept)
        delta_pred = delta_pred + c_pred * v
    out = (base + delta_pred).reshape(snapshots[0].shape).to(snapshots[0].dtype)
    return out, {"rank": rank}


def stale_baseline(snapshots):
    """No-projection control: reuse the newest exact checkpoint theta[t-K]."""
    return snapshots[-1].clone()


# ----------------------------------------------------------------------------- #
# Equivalence proof against the live harness (opt-in)
# ----------------------------------------------------------------------------- #
def assert_matches_live_harness(seed: int = 0, atol: float = 1e-5, lookahead_path: Optional[str] = None):
    """Prove this port == verl.workers.comm_eff.lookahead.project_rank1_tensor.

    Loads the leaf ``lookahead.py`` DIRECTLY (importlib) so it does NOT trigger
    the heavy ``verl`` package ``__init__`` dependency chain — only torch is
    needed. Pass ``lookahead_path`` or run from the worktree root so the default
    relative path resolves. Skips gracefully if the file is not found.
    """
    import importlib.util
    import os

    candidates = [lookahead_path] if lookahead_path else []
    candidates += [
        "verl/workers/comm_eff/lookahead.py",
        os.path.join(os.path.dirname(__file__), "..", "..", "verl", "workers", "comm_eff", "lookahead.py"),
    ]
    src = next((p for p in candidates if p and os.path.exists(p)), None)
    if src is None:
        print(f"[skip] live lookahead.py not found (tried {candidates}); run from the worktree root.")
        return
    spec = importlib.util.spec_from_file_location("lookahead_live", src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    live = m.project_rank1_tensor
    print(f"[live] loaded {os.path.realpath(src)}")
    g = torch.Generator().manual_seed(seed)
    for W in (2, 3, 4, 6):
        ticks = [10 * (i + 1) for i in range(W)]
        snaps = [torch.randn(37, 41, generator=g, dtype=torch.float32) for _ in range(W)]
        target = ticks[-1] + 10
        mine, ms = project_rank1_tensor(snaps, ticks, target, strength=1.0, rank=1)
        theirs, ts = live(snaps, ticks, target, strength=1.0)
        assert torch.allclose(mine, theirs, atol=atol), (
            f"W={W}: tensor mismatch (max {float((mine - theirs).abs().max())})"
        )
        assert ms["fit_kind"] == ts["fit_kind"], f"W={W}: fit_kind {ms['fit_kind']} != {ts['fit_kind']}"
        print(f"[ok] W={W}: port == live  (fit_kind={ms['fit_kind']}, slope~{ms['slope']:.4g})")
    print("[ok] harness_projector matches the live rank1_relex projector.")


if __name__ == "__main__":
    assert_matches_live_harness()
