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

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "CaptureWriter",
    "CAPTURE_ROLES",
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
#   weights        -> a FULL current weight matrix theta_m[t] (the weight-
#                     trajectory role). Used only by the dump-only WeightTrajObserver
#                     (full per-step weight snapshot), never by the optimizer/EMA/Q.
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
# Weight-trajectory FULL-weight instrument
# ====================================================================== #
#
# A dump-only weight-trajectory recorder used by the M4 weight-projection study
# (research/.claude/plans/43.md). The engine summons the FULL current weight
# matrices on every optimizer tick and hands them here; the observer saves the
# ACTUAL weight matrices to disk ONCE PER TRAINING STEP (deduped on global_step)
# as ``full/step_<gs>.pt`` (a ``torch.save`` state dict) + a ``full_manifest.jsonl``
# row. There is NO compression: the tensors saved ARE the weights (cast to
# ``dump_dtype``), so ANY offline analysis can be run on them directly. It is
# strictly telemetry: it reads (never writes) the live weights, runs only on the
# actor-train path, and feeds nothing back into the optimizer, EMA, sketch V or Q.
#
# Storage cost (Qwen2.5-1.5B, ~1.54B params): a bf16 full-model snapshot ≈ 3 GB;
# fp32 ≈ 6 GB. Dumping per training step (NOT per tick) keeps an 80-step bf16
# trajectory at ≈246 GB. ``dump_dtype=fp32`` doubles that and is needed only when
# the downstream analysis differences consecutive steps (the ~1e-3 per-step update
# would be swamped by bf16's ~4e-3 rounding); ``every_steps>1`` thins the
# trajectory to fit a smaller disk.
#
# NOTE: an earlier version of this instrument stored a lossy k-bucket COUNT-SKETCH
# of each matrix (non-invertible) plus a bounded exact-calibration ring. That was
# REMOVED (operator directive 2026-06-30): the study needs the raw weights, not a
# sketch. Recover the sketch implementation from git history if ever needed.

# The decoder weight-matrix selector (matches the spectral merger / look-ahead
# projector target set: q/k/v/o_proj + gate/up/down_proj, 2-D only). LayerNorm,
# embeddings, lm_head and biases never contain these substrings, so they are
# excluded exactly as the projector excludes them.
WEIGHT_TRAJ_DEFAULT_SUBSTRS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def select_weight_traj_targets(named_params, target_substrs=None, select_all: bool = False) -> list:
    """Return ``[(canon_name, tensor), ...]`` for the weight-trajectory matrices.

    Pure selection — no device moves, no clones, no FSDP. Names are canonicalised
    (FSDP wrap-infix stripped) so the selection is identical off a summoned live
    module or a plain clone.

    Two modes:

    * ``select_all=False`` (default — the projector's set): a parameter is
      selected iff its name contains one of ``target_substrs`` (default
      :data:`WEIGHT_TRAJ_DEFAULT_SUBSTRS`) AND its tensor is 2-D. On Qwen2.5-1.5B
      (28 layers) this is exactly the 196 decoder matrices the projector
      extrapolates.
    * ``select_all=True`` (EXP-42 completeness extension): select EVERY 1-D / 2-D
      float parameter — the 196 decoder linears PLUS the params the projector
      EXCLUDES (token embeddings, RMSNorm gains, attention biases). Lets the
      offline analysis measure what linear weight projection WOULD do on the
      excluded params (a direct test of the prior-work exclusion claim). The
      full-weight dump is shape-agnostic (the whole tensor is saved as-is), so
      1-D params are handled identically.
    """
    substrs = tuple(target_substrs) if target_substrs else WEIGHT_TRAJ_DEFAULT_SUBSTRS
    out = []
    for name, p in named_params:
        dim = getattr(p, "dim", lambda: 0)()
        if select_all:
            if dim in (1, 2):
                out.append((_canon(name), p))
            continue
        if not any(s in name for s in substrs):
            continue
        if dim != 2:
            continue
        out.append((_canon(name), p))
    return out


def _norm(t: torch.Tensor) -> float:
    return float(torch.linalg.norm(t.to(torch.float32)).item())


class WeightTrajObserver:
    """Per-step dump-only FULL-weight recorder.

    Constructed once per worker iff ``comm_eff.probe.weight_traj.enabled`` — and,
    crucially, INDEPENDENTLY of ``comm_eff.enabled`` so the plain-GRPO regime
    (codec OFF) is still instrumented. :meth:`observe` is the single entry point:
    the engine summons the full weight matrices to CPU/fp32 and hands them here on
    every optimizer tick; this class writes the **FULL weight matrices** to disk
    ONCE PER TRAINING STEP (deduped on ``global_step``, gated by ``every_steps``)
    as ``full/step_<gs>.pt`` (a ``torch.save`` state dict ``{canon_name -> tensor}``)
    plus a ``full_manifest.jsonl`` row. Pure I/O — it never mutates the tensors it
    is given, and feeds nothing back into the optimizer / EMA / Q.

    There is NO compression: the saved tensors ARE the weights (cast to
    ``dump_dtype``), so any offline analysis can run on them directly. Storage is
    on DP rank 0 by default (the summoned full params are DP-identical); other
    ranks build an INACTIVE observer that no-ops so the engine's summon collective
    stays symmetric across ranks.
    """

    def __init__(
        self,
        *,
        out_dir: str,
        dump_dtype: str = "bf16",
        target_substrs=None,
        select_all: bool = False,
        every_steps: int = 1,
        rank: Optional[int] = None,
        rank0_only: bool = True,
    ):
        self.enabled = True
        self.out_dir = out_dir or os.path.join(os.getcwd(), "weights")
        assert dump_dtype in ("bf16", "fp32"), dump_dtype
        self.dump_dtype = dump_dtype
        self._torch_dtype = torch.bfloat16 if dump_dtype == "bf16" else torch.float32
        self.target_substrs = tuple(target_substrs) if target_substrs else WEIGHT_TRAJ_DEFAULT_SUBSTRS
        # Completeness extension: when True the observer dumps EVERY 1-D/2-D param
        # (decoder linears + the projector-excluded embeddings / RMSNorm gains /
        # biases) ≈ the whole model. False = the 196-matrix projector set.
        self.select_all = bool(select_all)
        # Dump the FULL weights once per training step. The engine calls observe()
        # per optimizer TICK (>=1 tick/step); we dedup on global_step and gate on
        # ``every_steps`` so the on-disk trajectory is one snapshot per
        # ``every_steps`` training step(s) — the disk-volume control (a bf16
        # full-model snapshot is ~3 GB on Qwen2.5-1.5B).
        self.every_steps = max(1, int(every_steps))

        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.rank = int(rank)
        self.rank0_only = bool(rank0_only)
        self._inactive = self.rank0_only and self.rank != 0

        self._tick = 0  # monotonic optimizer-tick counter (own; comm_eff may be off)
        self._last_dumped_step = -1  # dedup: at most one full dump per global_step
        self._n_dumped = 0

        if not self._inactive:
            self.full_dir = os.path.join(self.out_dir, "full")
            os.makedirs(self.full_dir, exist_ok=True)
            self.manifest_path = os.path.join(self.out_dir, "full_manifest.jsonl")
        print(
            f"[comm_eff][weight_traj] FULL-weight observer out_dir={self.out_dir} "
            f"dump_dtype={self.dump_dtype} select_all={self.select_all} every_steps={self.every_steps} "
            f"rank={self.rank} rank0_only={self.rank0_only} inactive={self._inactive}",
            flush=True,
        )

    def observe(self, weights: dict, global_step: int = -1) -> int:
        """Record this tick. Dumps the FULL weights once per training step.

        ``weights`` is ``{canon_name -> CPU fp32 tensor}`` (the selected matrices).
        Pure read: tensors are never mutated. No-op on an inactive (non-writer)
        rank. The engine calls this per optimizer TICK; the full dump fires only on
        the FIRST tick of each new ``global_step`` (deduped), gated by
        ``every_steps``. Returns the tick index (or -1 when inactive).
        """
        if self._inactive:
            return -1
        tick = self._tick
        self._tick += 1

        gs = int(global_step)
        if gs >= 0 and gs != self._last_dumped_step and (gs % self.every_steps == 0):
            self._dump_full(gs, weights)
            self._last_dumped_step = gs
        return tick

    def _dump_full(self, global_step: int, weights: dict) -> None:
        """Write this step's FULL weight matrices to ``full/step_<gs>.pt``.

        No compression: each tensor is detached, cast to ``dump_dtype`` (bf16 by
        default), moved to CPU and stored AS-IS in a ``torch.save`` state dict
        ``{canon_name -> tensor}``. A ``full_manifest.jsonl`` row records per-matrix
        name / shape / exact-fp32 Frobenius norm so the analyst can verify the dump
        loads and the norms match the live weights within ``dump_dtype`` rounding.
        """
        records = []
        state = {}
        for name in sorted(weights.keys()):
            w = weights[name]
            t32 = w.detach().to(torch.float32)
            fro = float(torch.linalg.norm(t32).item()) if t32.numel() else 0.0
            state[name] = t32.to(self._torch_dtype).cpu().contiguous()
            records.append(
                {
                    "name": name,
                    "shape": list(w.shape),
                    "d": int(w.numel()),
                    "fro_norm": fro,
                }
            )
        fname = f"step_{global_step}.pt"
        fpath = os.path.join(self.full_dir, fname)
        torch.save(state, fpath)
        with open(self.manifest_path, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "global_step": int(global_step),
                        "dump_dtype": self.dump_dtype,
                        "n_matrices": len(records),
                        "path": os.path.join("full", fname),
                        "matrices": records,
                    }
                )
                + "\n"
            )
        self._n_dumped += 1
        print(
            f"[comm_eff][weight_traj] FULL dump step={global_step} n_matrices={len(records)} "
            f"dtype={self.dump_dtype} -> {os.path.join('full', fname)}",
            flush=True,
        )

    @property
    def n_dumped(self) -> int:
        return self._n_dumped


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
        select_all=bool(getattr(wt, "select_all", False)),
        dump_dtype=str(getattr(wt, "dump_dtype", "bf16")),
        every_steps=int(getattr(wt, "every_steps", 1)),
        target_substrs=substrs,
        rank=rank,
        rank0_only=bool(getattr(wt, "rank0_only", True)),
    )
