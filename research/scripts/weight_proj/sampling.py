#!/usr/bin/env python3
"""weight_proj/sampling.py — SOLE OWNER of the fast-mode coordinate-sampling math.

Ports the per-tensor uniform coordinate subsampling + two-stage trajectory filter
of Wang et al. arXiv:2601.04537 (analysis/weight/weight_linearity.py:
SAMPLE_PERCENTAGE=0.001, MIN_SAMPLES_THRESHOLD=50, MIN_ABS_CHANGE=1e-4,
MIN_UNIQUE_VALUES=4) into a deterministic, reader-agnostic sample plan the MOAT
scorecard gathers ONCE per trace ("the panel").

Determinism contract (deliberately stronger than the paper repo, whose draws
depend on `accelerate` process count and layer iteration order): every matrix's
index set depends ONLY on (seed, matrix name) via a sha256-derived per-matrix RNG
— never on process count, matrix ordering, tick count, or numpy global state.

Filter-scope contract (the load-bearing decision, mirrored in moat_scorecard):
`trajectory_filters` shapes ONLY the per-scalar linearity-R² population (the
paper's "exclude constant weights" semantics). The Gram/ratio/prediction
population stays the unfiltered sample so the hold_stale identity and
cross-fidelity ratio comparability survive.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

# machine-checkable string tag of the sampling contract version (pinned by the
# scorecard next to METRIC_CONTRACT_EXPECTED)
SAMPLING_CONTRACT = "weight-proj-sampling-v1"

# Statistical-soundness floor for strips mode: a matrix sampled as ONE contiguous
# strip is a single correlated draw — under low-rank update structure its
# per-matrix ratio/median estimates are fragile (observed ~9% median / 29% p90
# relative error at 1 strip vs ~1.5% for scatter at identical k). plan_matrix
# therefore SHRINKS the strip length per matrix so every strips-mode plan has at
# least this many independent clusters (IO stays page-local: n_strips pages/tick).
MIN_STRIPS_PER_MATRIX = 32

# The paper's MIN_ABS_CHANGE=1e-4 was calibrated to bf16 checkpoints (~bf16 ULP at
# typical weight magnitude). On an fp32 per-step trace the quantization floor is
# eps(fp32)/eps(bf16) = 2^-23 / 2^-7 = 2^-16 times smaller; transplanting 1e-4
# unchanged guts the population (typical per-coordinate range on the EXP-57 fp32
# 80-step window is ~3e-5). resolve_min_abs_change scales the threshold by the
# dump dtype unless the caller passes an explicit float.
PAPER_MIN_ABS_CHANGE = 1e-4          # bf16-calibrated paper value (App. B.1)
FP32_BF16_EPS_RATIO = 2.0 ** -16     # eps(fp32)=2^-23 over eps(bf16)=2^-7


def resolve_min_abs_change(value, dump_dtype: str):
    """Resolve the range-filter threshold: 'auto' scales the paper's bf16 value by
    the machine-epsilon ratio of the trace dtype; anything else parses as an
    explicit float (paper-verbatim reproduction: pass 1e-4). Returns
    (threshold, mode) where mode in {'explicit', 'auto:bf16-paper',
    'auto:fp32-ulp-scaled'}."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        dt = (dump_dtype or "").lower()
        if "bf16" in dt or "bfloat16" in dt:
            return PAPER_MIN_ABS_CHANGE, "auto:bf16-paper"
        return PAPER_MIN_ABS_CHANGE * FP32_BF16_EPS_RATIO, "auto:fp32-ulp-scaled"
    return float(value), "explicit"


def derive_seed(seed: int, name: str) -> int:
    """Per-matrix RNG seed: first 8 bytes of sha256(f"{seed}:{name}"), little-endian."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8],
                          "little")


def runs_from(idx: np.ndarray) -> list[tuple[int, int]]:
    """Maximal contiguous [a, b) runs covering a SORTED ascending index array."""
    if idx.size == 0:
        return []
    brk = np.nonzero(np.diff(idx) != 1)[0]
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk, [idx.size - 1]])
    return [(int(idx[s]), int(idx[e]) + 1) for s, e in zip(starts, ends)]


@dataclass
class MatrixSamplePlan:
    name: str
    numel: int
    k_target: int
    k_actual: int
    mode: str                       # "all" | "scatter" | "strips"
    idx: np.ndarray                 # sorted ascending int64, len == k_actual
    runs: list[tuple[int, int]]     # maximal contiguous [a,b) runs covering idx


def plan_matrix(name: str, numel: int, frac: float, min_k: int, strip: int,
                small_full: int, scatter_cutoff: int, seed: int) -> MatrixSamplePlan:
    """One matrix's deterministic sample plan.

    k_target = min(max(min_k, int(numel*frac)), numel) — the paper's k rule
    verbatim. Mode routing: tiny tensors (numel <= small_full) are taken WHOLE
    (norms/biases exact — they are the noisiest under 50-sample estimates and
    cheap); mid-size tensors get a scattered randperm sample (paper-faithful);
    big tensors get whole aligned strips of AT MOST `strip` contiguous elements
    (IO-efficient page-local reads; the strip length shrinks per matrix so every
    plan has >= MIN_STRIPS_PER_MATRIX clusters; k_actual rounds UP to whole
    strips — recorded, never hidden). strip <= 1 restores pure scatter everywhere.
    """
    k_target = min(max(min_k, int(numel * frac)), numel)
    rng = np.random.default_rng(derive_seed(seed, name))
    if k_target >= numel or numel <= small_full:
        mode, idx = "all", np.arange(numel, dtype=np.int64)
    elif numel <= scatter_cutoff or strip <= 1:
        mode = "scatter"
        idx = np.sort(rng.choice(numel, size=k_target, replace=False)).astype(np.int64)
    else:
        mode = "strips"
        # cluster-count floor: shrink the strip length (never the sample) so the
        # plan has >= MIN_STRIPS_PER_MATRIX independent clusters — a small-k
        # matrix (e.g. k/v_proj at frac=0.001) would otherwise collapse to ONE
        # contiguous strip and its per-matrix statistics become fragile.
        strip_eff = min(strip, max(1, k_target // MIN_STRIPS_PER_MATRIX))
        n_slots = numel // strip_eff        # aligned slots; the tail is never sampled
        n_strips = min(math.ceil(k_target / strip_eff), n_slots)
        slots = np.sort(rng.choice(n_slots, size=n_strips, replace=False))
        idx = (slots[:, None] * strip_eff
               + np.arange(strip_eff)[None, :]).reshape(-1).astype(np.int64)
    return MatrixSamplePlan(name, numel, k_target, len(idx), mode, idx, runs_from(idx))


def build_sample_plan(name_dims: list[tuple[str, int]], frac: float, min_k: int,
                      strip: int, small_full: int, scatter_cutoff: int,
                      seed: int) -> dict[str, MatrixSamplePlan]:
    """{name: MatrixSamplePlan} — per-matrix independent, order-insensitive."""
    return {name: plan_matrix(name, d, frac, min_k, strip, small_full,
                              scatter_cutoff, seed)
            for name, d in name_dims}


def trajectory_filters(Y: np.ndarray, min_abs_change: float, min_unique: int):
    """Paper App. B.1 two-stage trajectory filter over a panel Y: [T, k] float64.

    Stage 1 (range): keep iff (Y.max(0) - Y.min(0)) > min_abs_change.
    Stage 2 (unique, on stage-1 survivors only, vectorized): sort each column,
    n_unique = 1 + (np.diff(Ys, axis=0) != 0).sum(0); keep iff n_unique >= min_unique.
    The sort-diff vectorization is exact-equivalent to the paper's per-coordinate
    torch.unique loop. Returns (keep_mask[k] bool, n_excluded_range, n_excluded_unique).
    """
    Y = np.asarray(Y, dtype=np.float64)
    keep = (Y.max(axis=0) - Y.min(axis=0)) > min_abs_change
    n_excluded_range = int(Y.shape[1] - keep.sum())
    survivors = np.where(keep)[0]
    n_excluded_unique = 0
    if survivors.size:
        Ys = np.sort(Y[:, survivors], axis=0)
        n_unique = 1 + (np.diff(Ys, axis=0) != 0).sum(axis=0)
        drop = n_unique < min_unique
        keep[survivors[drop]] = False
        n_excluded_unique = int(drop.sum())
    return keep, n_excluded_range, n_excluded_unique
