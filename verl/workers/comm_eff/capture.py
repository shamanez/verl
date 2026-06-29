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

"""Diagnostic tensor capture for communication-efficient gradient analysis.

This module owns optional fp32 tensor dumps for comm-eff diagnostics. It is a
strict no-op unless ``comm_eff.capture.enabled=true``. The design goals are:

1. **Zero numerical side effect.** Every captured tensor is ``.detach()``ed,
   cloned, moved to CPU and written to disk. The capture writer NEVER feeds a
   tensor back into the optimizer, the loss, or any in-graph buffer
   It is pure I/O.

2. **Keyed, self-describing dumps.** Every tensor is keyed by
   ``(global_step, optimizer_tick, role, target_name)`` and saved with its
   ``shape``, ``dtype`` and Frobenius ``norm`` recorded in a per-tick manifest
   row, so the analyst can recompute ``reconstruction_rel_error`` from the dumped
   ``A`` / ``Â`` and confirm it matches the logged scalar.

3. **Bounded volume.** The number of captured ticks is capped
   (``max_ticks``); the per-tick target set can be stratified to a few targets
   per matrix-type (``stratified_targets``). The chosen subset is recorded in
   the manifest.

Disk layout under ``capture_dir``::

    captures/
      manifest.jsonl                       # one row per (tick, role, target)
      tick_<gs>_<tick>/<role>/<sanitized_target>.pt   # the fp32 tensor

``role`` is one of the audit roles below (``A`` / ``A_hat`` / ``Q`` / ``G_comp``
/ ``G_corr`` / ``M`` / ``G_anchor`` / ``G_dense`` / ``G_fresh_anchor``).

The writer is process-local; under DP each rank writes to a ``rank<r>``
subdirectory so multi-rank dumps never collide. The analyst reads rank-0 by
default when the captured tensors are expected to be rank-identical; per-rank
``G_comp`` differs by shard.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "CaptureWriter",
    "CAPTURE_ROLES",
    "CountSketch",
    "WeightTrajObserver",
    "select_weight_traj_targets",
    "maybe_build_weight_traj_observer",
    "WEIGHT_TRAJ_DEFAULT_SUBSTRS",
]

# The capture roles. Each names a distinct diagnostic tensor.
#   A              -> the boundary activation M (N, H) at the projection hook
#   A_hat          -> the reconstruction (A@Q)Qᵀ at the projection hook
#   Q              -> the frozen PowerSGD basis (H, r)
#   G_comp         -> the fast compressed per-target gradient (merger INPUT)
#   G_corr         -> the post-merger pre-Adam per-target gradient (writeback)
#   M              -> the anchor-gradient EMA M_anchor per target
#   G_anchor       -> the RAW K-stale anchor gradient per target (DP-reduced)
#   G_dense        -> the parallel UNCOMPRESSED fast gradient (measurement probe)
#   G_fresh_anchor -> the delay_K=0 fresh-anchor gradient (measurement probe)
#   G_b            -> the boundary activation gradient dL_anchor/dh
#                     the operand the grad/tail/ticket family sketches are built from)
#   Q_<family>     -> passive-screen candidate basis per family
#                     (Q_act / Q_grad / Q_adv / Q_tail / Q_hybrid / Q_ticket), each
#                     an (H, r) orthonormal basis the analyst judges by update geometry
#   weights        -> a FULL current weight matrix theta_m[t] (the EXP-42 weight-
#                     trajectory role). Used only by the dump-only WeightTrajObserver
#                     (count-sketch + per-matrix mean), never by the optimizer/EMA/Q.
CAPTURE_ROLES = (
    "A",
    "A_hat",
    "Q",
    "G_comp",
    "G_corr",
    "M",
    "G_anchor",
    "G_dense",
    "G_fresh_anchor",
    "G_b",
    "Q_act",
    "Q_grad",
    "Q_adv",
    "Q_tail",
    "Q_hybrid",
    "Q_ticket",
    "weights",
)

# Matrix-type substrings used to stratify the per-tick target subset (one bucket
# per Qwen2.5 decoder projection matrix). Order is stable so the stratified pick
# is deterministic across ranks/ticks.
_MATRIX_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _canon(name: str) -> str:
    """Strip the FSDP per-layer-wrap infix so every role keys on the SAME name.

    Mirrors ``verl.workers.comm_eff.spectral_filter._canon`` (kept local to avoid a
    cross-module import). The parallel-clone dumps (G_dense / anchor roles) use
    NON-infixed names; the live-FSDP-module dumps (G_comp / G_corr) carry the
    ``._fsdp_wrapped_module.`` infix. Canonicalising the manifest ``target_name``
    here makes G_dense pair with G_comp in the audit without consumer-side fixes.
    """
    name = name.replace("._fsdp_wrapped_module", "")
    if name.startswith("_fsdp_wrapped_module."):
        name = name[len("_fsdp_wrapped_module.") :]
    return name


def _sanitize(name: str) -> str:
    """Make a parameter name safe as a filename (dots/slashes -> underscores)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name).replace(".", "_")


class CaptureWriter:
    """Process-local fp32 tensor-dump writer for comm-eff diagnostics.

    Constructed once per worker (only when ``capture.enabled``) and shared across
    the anchor / merger / projection hooks. Thread-safe append to the manifest.
    Caps the number of captured ticks and (optionally) stratifies the per-tick
    target set. A captured tick is the unit of the cap: the FIRST role written at
    a fresh ``(global_step, optimizer_tick)`` opens that tick; once ``max_ticks``
    distinct ticks have opened, later ticks are skipped (the writer returns False
    from :meth:`should_capture_tick`).
    """

    def __init__(
        self,
        *,
        capture_dir: str,
        max_ticks: int = 10,
        stratified_targets: int = 0,
        dump_dtype: str = "fp32",
        rank: Optional[int] = None,
        rank0_only: bool = True,
        min_tick: int = 0,
    ):
        self.base_dir = capture_dir or os.path.join(os.getcwd(), "captures")
        self.max_ticks = int(max_ticks)
        self.stratified_targets = int(stratified_targets)
        # Skip ticks below min_tick. This is useful when avoiding cold-start
        # captures before Q or anchor state has warmed.
        self.min_tick = int(min_tick)
        assert dump_dtype in ("fp32", "bf16"), dump_dtype
        self.dump_dtype = torch.float32 if dump_dtype == "fp32" else torch.bfloat16
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.rank = int(rank)
        # Disk-volume guard: by default capture only rank 0. Set
        # ``rank0_only=False`` only when per-rank dumps are needed.
        self.rank0_only = bool(rank0_only)
        self._inactive = self.rank0_only and self.rank != 0
        self.root = os.path.join(self.base_dir, f"rank{self.rank}")
        if not self._inactive:
            os.makedirs(self.root, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.jsonl")
        self._lock = threading.Lock()
        # Set of (global_step, optimizer_tick) tuples that have opened a tick dir.
        self._open_ticks: set = set()
        # Per (tick, role) → ordered list of target names already written, so the
        # stratified subset is decided once per (tick, role) and stable.
        self._strat_seen: dict = {}
        self._n_written = 0
        logger.info(
            "comm_eff.capture: writer @ %s (max_ticks=%s stratified_targets=%s dump_dtype=%s rank=%s)",
            self.root,
            self.max_ticks,
            self.stratified_targets,
            dump_dtype,
            self.rank,
        )
        print(
            f"[comm_eff][capture] writer root={self.root} max_ticks={self.max_ticks} "
            f"stratified_targets={self.stratified_targets} dump_dtype={dump_dtype} rank={self.rank} "
            f"rank0_only={self.rank0_only} min_tick={self.min_tick} inactive={self._inactive}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # tick gating
    # ------------------------------------------------------------------ #
    def should_capture_tick(self, global_step: int, optimizer_tick: int) -> bool:
        """True iff a dump at this ``(global_step, optimizer_tick)`` is in budget.

        A tick already open is always allowed (so all roles of an open tick land);
        a fresh tick opens only while fewer than ``max_ticks`` have opened AND the
        tick is at/above ``min_tick`` (post-Q-warm). Pure read of the open-set under
        the lock; opening happens lazily in :meth:`dump`. Always False on an
        inactive (non-rank-0) writer.
        """
        if self._inactive:
            return False
        if self.min_tick > 0 and int(optimizer_tick) < self.min_tick:
            return False  # pre-Q-warm tick — skip so the budget holds POST-warm ticks
        key = (int(global_step), int(optimizer_tick))
        with self._lock:
            if key in self._open_ticks:
                return True
            if self.max_ticks > 0 and len(self._open_ticks) >= self.max_ticks:
                return False
            return True

    def _strat_admit(self, tick_key: tuple, role: str, target_name: str) -> bool:
        """Stratified-subset gate: admit at most ``stratified_targets`` targets per
        matrix-type per (tick, role). ``stratified_targets<=0`` admits everything.

        The decision is deterministic and stable: the per-(tick, role, type) bucket
        fills in first-seen order, identical on every rank because the engine
        iterates ``named_parameters()`` in the same order.
        """
        if self.stratified_targets <= 0:
            return True
        mtype = next((t for t in _MATRIX_TYPES if t in target_name), "_other")
        bk = (tick_key, role, mtype)
        seen = self._strat_seen.setdefault(bk, [])
        if target_name in seen:
            return True
        if len(seen) >= self.stratified_targets:
            return False
        seen.append(target_name)
        return True

    # ------------------------------------------------------------------ #
    # dump
    # ------------------------------------------------------------------ #
    def dump(
        self,
        *,
        role: str,
        target_name: str,
        tensor: torch.Tensor,
        global_step: int,
        optimizer_tick: int,
        extra: Optional[dict] = None,
    ) -> bool:
        """Write ``tensor`` (detached/cloned) + append a manifest row. Returns True
        iff the dump was written (False if out of tick budget or stratified out).

        **Pure I/O — no autograd, no optimizer touch.** The tensor is detached,
        moved to CPU, cast to ``dump_dtype`` (fp32 by default) and saved. The
        Frobenius norm + shape + dtype are recorded so the analyst can verify the
        dump is the real fp32 tensor (the fidelity invariant) and recompute
        cosines / reconstruction error offline.
        """
        assert role in CAPTURE_ROLES, f"unknown capture role {role!r}"
        # Canonicalise the target name (strip the FSDP wrap-infix) so the SAME
        # logical matrix keys identically whether it came off the live FSDP module
        # (G_comp/G_corr) or the plain clone (G_dense / anchor roles) — else they
        # never pair in the audit. The raw name is preserved in the manifest as
        # ``target_name_raw`` for traceability.
        target_name_raw = target_name
        target_name = _canon(target_name)
        tick_key = (int(global_step), int(optimizer_tick))
        if not self.should_capture_tick(global_step, optimizer_tick):
            return False
        with self._lock:
            if not self._strat_admit(tick_key, role, target_name):
                return False
            # Open the tick (counts against max_ticks) lazily on first write.
            self._open_ticks.add(tick_key)
            tick_dir = os.path.join(self.root, f"tick_{tick_key[0]}_{tick_key[1]}", role)
            os.makedirs(tick_dir, exist_ok=True)
            fname = _sanitize(target_name) + ".pt"
            fpath = os.path.join(tick_dir, fname)
            # Detach + clone + CPU + cast. NEVER an in-graph tensor.
            t = tensor.detach().to(torch.float32)
            norm = float(torch.linalg.norm(t).item()) if t.numel() else 0.0
            t_out = t.to(self.dump_dtype).cpu().contiguous()
            torch.save(t_out, fpath)
            row = {
                "global_step": tick_key[0],
                "optimizer_tick": tick_key[1],
                "role": role,
                "target_name": target_name,
                "target_name_raw": target_name_raw,
                "shape": list(tensor.shape),
                "dtype": str(self.dump_dtype).replace("torch.", ""),
                "norm": norm,
                "path": os.path.relpath(fpath, self.root),
                "rank": self.rank,
            }
            if extra:
                row.update({k: v for k, v in extra.items()})
            with open(self.manifest_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            self._n_written += 1
        return True

    @property
    def n_written(self) -> int:
        return self._n_written


def maybe_build_capture_writer(config: Any, *, rank: Optional[int] = None) -> Optional[CaptureWriter]:
    """Construct a :class:`CaptureWriter` iff ``comm_eff.capture.enabled``, else None.

    The single gate so the disabled / non-capture path never creates a writer or
    touches the filesystem. ``config`` is the ``CommEffConfig`` (or a node with a
    ``capture`` sub-config).
    """
    cap = getattr(config, "capture", None)
    if cap is None or not bool(getattr(cap, "enabled", False)):
        return None
    return CaptureWriter(
        capture_dir=str(getattr(cap, "capture_dir", "") or ""),
        max_ticks=int(getattr(cap, "max_ticks", 10)),
        stratified_targets=int(getattr(cap, "stratified_targets", 0)),
        dump_dtype=str(getattr(cap, "dump_dtype", "fp32")),
        rank=rank,
        rank0_only=bool(getattr(cap, "rank0_only", True)),
        min_tick=int(getattr(cap, "min_tick", 0)),
    )


# ====================================================================== #
# EXP-42 weight-trajectory sketch instrument
# ====================================================================== #
#
# A dump-only per-tick weight-trajectory recorder used by the look-ahead
# weight-projection-accuracy study (research/.claude/plans/42.md). At every
# optimizer tick it summons the FULL current decoder weight matrices and writes
# a COMPACT count-sketch (+ per-matrix mean) instead of the ~5 GB full snapshot,
# plus optional EXACT fp32 headline scalars from a bounded CPU snapshot ring. It
# is strictly telemetry: it reads (never writes) the live weights, runs only on
# the actor-train path, and feeds nothing back into the optimizer, EMA, sketch V
# or Q.
#
# Design notes anchored to current behaviour:
#   * The metrics the study reports — weight_proj_ratio = ||θ̂−target||/||θ_stale
#     −target|| and dir_cos — are all functions of weight-DIFFERENCE vectors. A
#     count-sketch is LINEAR, so sketch(θ_t)−sketch(θ_s) == sketch(θ_t−θ_s): the
#     full horizon×method×spacing sweep is reconstructable OFFLINE from the saved
#     per-tick sketches (rel. std ≈ 1/√k). The per-matrix MEAN is kept (fp32) so
#     the learned-residual rule, whose update is mean(θ_now−θ̂_prev), replays
#     offline too.
#   * The sketch is stored fp32 (not fp16). The RLVR per-tick weight update is
#     O(1e-6)/element — ~1e-3 relative to the weights — so an fp16-quantised
#     ABSOLUTE sketch (≈1e-3 relative error) would swamp the very difference
#     signal the study measures. fp32 keeps the differenced sketch accurate to
#     ~1e-4 relative; the disk cost (≈3.2 MB/tick → ~0.5 GB/regime) is still
#     negligible vs full snapshots (~5 GB/tick). ``dump_dtype`` exposes fp16 for
#     callers who only need absolute magnitudes — do NOT use it for differencing.

# The decoder weight-matrix selector (matches the spectral merger / look-ahead
# projector target set: q/k/v/o_proj + gate/up/down_proj, 2-D only). LayerNorm,
# embeddings, lm_head and biases never contain these substrings, so they are
# excluded exactly as the projector excludes them.
WEIGHT_TRAJ_DEFAULT_SUBSTRS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def select_weight_traj_targets(named_params, target_substrs=None) -> list:
    """Return ``[(canon_name, tensor), ...]`` for the decoder weight matrices.

    Pure selection — no device moves, no clones, no FSDP. A parameter is selected
    iff its name contains one of ``target_substrs`` (default
    :data:`WEIGHT_TRAJ_DEFAULT_SUBSTRS`) AND its tensor is 2-D. Names are
    canonicalised (FSDP wrap-infix stripped) so the selection is identical off a
    summoned live module or a plain clone. On Qwen2.5-1.5B (28 layers) this is
    exactly the 196 decoder matrices the projector extrapolates.
    """
    substrs = tuple(target_substrs) if target_substrs else WEIGHT_TRAJ_DEFAULT_SUBSTRS
    out = []
    for name, p in named_params:
        if not any(s in name for s in substrs):
            continue
        if getattr(p, "dim", lambda: 0)() != 2:
            continue
        out.append((_canon(name), p))
    return out


class CountSketch:
    """Deterministic, cross-rank-identical count-sketch (a.k.a. feature hashing).

    For a length-``d`` vector ``x`` it computes ``s[b] = Σ_{i: bucket(i)=b}
    sign(i)·x[i]`` with ``bucket(i)∈[0,k)`` and ``sign(i)∈{±1}``. This is an
    unbiased AMS/JL sketch: ``E‖s‖² = ‖x‖²`` and ``E⟨s_x,s_y⟩ = ⟨x,y⟩`` with
    rel. std ≈ ``1/√k``, and it is LINEAR (``sketch(x−y) = sketch(x)−sketch(y)``)
    so weight-difference norms/cosines reconstruct from per-snapshot sketches.

    The hash/sign tables are drawn with **NumPy** ``default_rng(seed)`` (PCG64,
    bit-stable across NumPy versions) seeded purely from ``(d, k)`` — so the
    sketch is identical on every DP rank AND reproducible bit-for-bit by the
    MacBook analysis (``research/scripts/weight_proj_sweep.py`` re-draws the same
    tables). No rank/device/time input ⇒ no determinism leak.
    """

    def __init__(self, d: int, k: int):
        self.d = int(d)
        self.k = int(k)
        rng = np.random.default_rng([self.d, self.k])
        buckets = rng.integers(0, self.k, size=self.d, dtype=np.int64)
        signs = (rng.integers(0, 2, size=self.d, dtype=np.int8).astype(np.float32) * 2.0) - 1.0
        self._bucket = torch.from_numpy(buckets)
        self._sign = torch.from_numpy(signs)

    def sketch(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Return the length-``k`` fp32 sketch of the flat fp32 vector ``x_flat``."""
        x = x_flat.detach().reshape(-1).to(torch.float32)
        assert x.numel() == self.d, f"CountSketch dim mismatch: got {x.numel()} expected {self.d}"
        bucket = self._bucket.to(x.device)
        sign = self._sign.to(x.device)
        s = torch.zeros(self.k, dtype=torch.float32, device=x.device)
        s.scatter_add_(0, bucket, sign * x)
        return s


def _norm(t: torch.Tensor) -> float:
    return float(torch.linalg.norm(t.to(torch.float32)).item())


class WeightTrajObserver:
    """Per-tick dump-only weight-trajectory recorder (EXP-42).

    Constructed once per worker iff ``comm_eff.probe.weight_traj.enabled`` — and,
    crucially, INDEPENDENTLY of ``comm_eff.enabled`` so the plain-GRPO regime
    (codec OFF) is still instrumented. :meth:`observe` is the single entry point:
    the engine summons the full decoder matrices to CPU/fp32 and hands them here;
    this class writes the compact sketch + means and (sparsely) the EXACT fp32
    headline scalars. Pure I/O — it never mutates the tensors it is given.

    All storage/IO is on DP rank 0 by default (the summoned full params are
    DP-identical). Other ranks build an INACTIVE observer that no-ops, so the
    engine's summon collective stays symmetric across ranks.
    """

    def __init__(
        self,
        *,
        out_dir: str,
        k: int = 4096,
        dump_dtype: str = "fp32",
        target_substrs=None,
        calib_deltas=(10,),
        calib_horizons=(10,),
        calib_stride: int = 0,
        calib_max_snapshots: int = 6,
        rank: Optional[int] = None,
        rank0_only: bool = True,
    ):
        self.enabled = True
        self.out_dir = out_dir or os.path.join(os.getcwd(), "weights")
        self.k = int(k)
        assert dump_dtype in ("fp32", "fp16"), dump_dtype
        self._np_dtype = np.float32 if dump_dtype == "fp32" else np.float16
        self.dump_dtype = dump_dtype
        self.target_substrs = tuple(target_substrs) if target_substrs else WEIGHT_TRAJ_DEFAULT_SUBSTRS
        self.calib_deltas = tuple(int(x) for x in calib_deltas)
        self.calib_horizons = tuple(int(x) for x in calib_horizons)
        # Exact-calib grid = deltas × horizons (the on-box ground truth that the
        # offline sketch sweep is validated against). Operating point: Δ=h=10.
        self.calib_grid = [(d, h) for d in self.calib_deltas for h in self.calib_horizons]
        # New tripoint group opens every ``calib_stride`` ticks. Default = the
        # longest group lifetime (max Δ+h) so groups never overlap ⇒ ≤3 full
        # snapshots in flight per config (bounded CPU memory). Each full snapshot
        # is ~5 GB on Qwen2.5-1.5B, so the cap is the OOM guard.
        if calib_stride and calib_stride > 0:
            self.calib_stride = int(calib_stride)
        else:
            self.calib_stride = max((d + h for d, h in self.calib_grid), default=20)
        self.calib_max_snapshots = int(calib_max_snapshots)

        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.rank = int(rank)
        self.rank0_only = bool(rank0_only)
        self._inactive = self.rank0_only and self.rank != 0

        self._tick = 0  # monotonic optimizer-tick counter (own; comm_eff may be off)
        self._countsketches: dict = {}  # (d,k) -> CountSketch
        self._calib_groups: list = []  # open tripoint groups
        self._calib_skipped = 0
        self._n_sketched = 0

        if not self._inactive:
            os.makedirs(self.out_dir, exist_ok=True)
            self.manifest_path = os.path.join(self.out_dir, "manifest.jsonl")
            self.calib_path = os.path.join(self.out_dir, "calib.jsonl")
        print(
            f"[comm_eff][weight_traj] observer out_dir={self.out_dir} k={self.k} "
            f"dump_dtype={self.dump_dtype} calib_grid={self.calib_grid} "
            f"calib_stride={self.calib_stride} calib_max_snapshots={self.calib_max_snapshots} "
            f"rank={self.rank} rank0_only={self.rank0_only} inactive={self._inactive}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    def _cs_for(self, t: torch.Tensor) -> CountSketch:
        d = int(t.numel())
        cs = self._countsketches.get(d)
        if cs is None:
            cs = CountSketch(d, self.k)
            self._countsketches[d] = cs
        return cs

    def observe(self, weights: dict, global_step: int = -1) -> int:
        """Record this tick's weight matrices. Returns the tick index (or -1).

        ``weights`` is ``{canon_name -> 2-D CPU fp32 tensor}`` (the selected
        decoder matrices). Pure read: tensors are never mutated. No-op on an
        inactive (non-writer) rank.
        """
        if self._inactive:
            return -1
        tick = self._tick
        self._tick += 1

        # 1) per-matrix mean (fp32, computed in fp64 for accuracy — the mean is
        #    tiny and feeds the learned-residual replay) + the count-sketch.
        records = []
        sketch_arrays = {}
        for name in sorted(weights.keys()):
            w = weights[name]
            flat = w.detach().reshape(-1).to(torch.float32)
            mean = float(w.detach().to(torch.float64).mean().item())
            s = self._cs_for(flat).sketch(flat)
            san = _sanitize(name)
            sketch_arrays[san] = s.cpu().numpy().astype(self._np_dtype)
            rows = int(w.shape[0]) if w.dim() >= 1 else 0
            cols = int(w.shape[1]) if w.dim() >= 2 else 1
            records.append(
                {
                    "name": name,
                    "sanitized": san,
                    "rows": rows,
                    "cols": cols,
                    "d": int(flat.numel()),
                    "mean": mean,
                    "fro_norm": _norm(flat),
                }
            )

        sketch_path = os.path.join(self.out_dir, f"sketch_tick_{global_step}_{tick}.npz")
        np.savez(sketch_path, **sketch_arrays)
        with open(self.manifest_path, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "tick": tick,
                        "global_step": int(global_step),
                        "k": self.k,
                        "dump_dtype": self.dump_dtype,
                        "n_matrices": len(records),
                        "sketch_path": os.path.basename(sketch_path),
                        "matrices": records,
                    }
                )
                + "\n"
            )
        self._n_sketched += 1

        # 2) exact fp32 calib (sparse, bounded tripoint groups).
        self._update_calib(tick, int(global_step), weights)
        return tick

    # ------------------------------------------------------------------ #
    def _update_calib(self, tick: int, global_step: int, weights: dict) -> None:
        """Open/advance/close the bounded exact-calib tripoint groups.

        Each group for config ``(Δ, h)`` samples the full fp32 weights at three
        ticks — ``old = base``, ``stale = base+Δ``, ``target = base+Δ+h`` — and
        at the target tick computes the EXACT per-matrix ``weight_proj_ratio`` /
        ``dir_cos`` ground truth, then frees the snapshots. Total retained
        snapshots are capped at ``calib_max_snapshots`` (the CPU-OOM guard).
        """
        if not self.calib_grid:
            return

        def _retained() -> int:
            return sum(len(g["snaps"]) for g in self._calib_groups)

        # Open new groups (one per config) when the tick lands on the stride.
        if tick % self.calib_stride == 0:
            for (delta, h) in self.calib_grid:
                if _retained() + 1 > self.calib_max_snapshots:
                    self._calib_skipped += 1
                    continue
                self._calib_groups.append(
                    {"delta": delta, "h": h, "base": tick, "snaps": {}}
                )

        # Advance every open group: capture the sample if this tick is one of its
        # three sample points; clone the FULL fp32 weights (the calib needs exact
        # values, not the sketch).
        def _clone_full() -> dict:
            return {n: w.detach().to(torch.float32).clone() for n, w in weights.items()}

        closed = []
        snap_this_tick = None  # share one clone across groups sampling the same tick
        for g in self._calib_groups:
            base, delta, h = g["base"], g["delta"], g["h"]
            t_old, t_stale, t_target = base, base + delta, base + delta + h
            role = None
            if tick == t_old:
                role = "old"
            elif tick == t_stale:
                role = "stale"
            elif tick == t_target:
                role = "target"
            if role is None:
                continue
            if snap_this_tick is None:
                snap_this_tick = _clone_full()
            g["snaps"][role] = snap_this_tick
            if role == "target":
                if {"old", "stale", "target"} <= set(g["snaps"].keys()):
                    self._emit_calib_row(g, global_step)
                closed.append(g)
        if closed:
            self._calib_groups = [g for g in self._calib_groups if g not in closed]

    def _emit_calib_row(self, g: dict, global_step: int) -> None:
        delta, h, base = g["delta"], g["h"], g["base"]
        alpha = float(h) / float(delta)
        old, stale, target = g["snaps"]["old"], g["snaps"]["stale"], g["snaps"]["target"]
        w1, dcos, w3 = [], [], []
        for name in stale.keys():
            if name not in old or name not in target:
                continue
            th_s = stale[name].to(torch.float32)
            d_old = th_s - old[name].to(torch.float32)  # θ_stale − θ_old
            d_tgt = th_s - target[name].to(torch.float32)  # θ_stale − target
            proj_err = d_tgt + alpha * d_old  # θ̂_fix − target
            den = _norm(d_tgt)
            if den <= 0.0:
                continue
            w1.append(_norm(proj_err) / den)
            tgt_norm = _norm(target[name])
            if tgt_norm > 0.0:
                w3.append(_norm(proj_err) / tgt_norm)
            no, nt = _norm(d_old), den
            if no > 0.0 and nt > 0.0:
                ip = float(torch.sum(d_old * d_tgt).item())
                dcos.append(-ip / (no * nt))  # cos(θ_stale−θ_old, target−θ_stale)

        def _pct(a, q):
            return float(np.percentile(np.asarray(a, dtype=np.float64), q)) if a else None

        row = {
            "anchor_tick": base + delta,  # s = θ_stale's tick (matches the sketch sweep)
            "global_step": int(global_step),
            "delta": delta,
            "h": h,
            "alpha": alpha,
            "n_matrices": len(w1),
            "weight_proj_ratio_p10": _pct(w1, 10),
            "weight_proj_ratio_p50": _pct(w1, 50),
            "weight_proj_ratio_p90": _pct(w1, 90),
            "dir_cos_p50": _pct(dcos, 50),
            "weight_relerr_p50": _pct(w3, 50),
        }
        with open(self.calib_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"[comm_eff][weight_traj] calib anchor_tick={row['anchor_tick']} Δ={delta} h={h} "
            f"α={alpha:.3f} w1_p50={row['weight_proj_ratio_p50']} dir_cos_p50={row['dir_cos_p50']} "
            f"n={row['n_matrices']}",
            flush=True,
        )

    @property
    def n_sketched(self) -> int:
        return self._n_sketched


def maybe_build_weight_traj_observer(comm_eff_config: Any, *, rank: Optional[int] = None):
    """Construct a :class:`WeightTrajObserver` iff
    ``comm_eff.probe.weight_traj.enabled``, else ``None``.

    Read INDEPENDENTLY of ``comm_eff.enabled`` so the plain-GRPO regime (codec
    OFF) is still instrumented. The single gate that keeps the disabled path from
    ever touching the filesystem.
    """
    if comm_eff_config is None:
        return None
    probe = getattr(comm_eff_config, "probe", None)
    wt = getattr(probe, "weight_traj", None) if probe is not None else None
    if wt is None or not bool(getattr(wt, "enabled", False)):
        return None
    spectral = getattr(comm_eff_config, "spectral", None)
    substrs = getattr(spectral, "target_substr", None) if spectral is not None else None
    return WeightTrajObserver(
        out_dir=str(getattr(wt, "out_dir", "") or ""),
        k=int(getattr(wt, "k", 4096)),
        dump_dtype=str(getattr(wt, "dump_dtype", "fp32")),
        target_substrs=substrs,
        calib_deltas=tuple(getattr(wt, "calib_deltas", (10,)) or (10,)),
        calib_horizons=tuple(getattr(wt, "calib_horizons", (10,)) or (10,)),
        calib_stride=int(getattr(wt, "calib_stride", 0)),
        calib_max_snapshots=int(getattr(wt, "calib_max_snapshots", 6)),
        rank=rank,
        rank0_only=bool(getattr(wt, "rank0_only", True)),
    )
