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

from verl.workers.comm_eff.r2_sink import maybe_build_r2_sink

logger = logging.getLogger(__name__)

__all__ = [
    "CaptureWriter",
    "CAPTURE_ROLES",
    "WeightTrajObserver",
    "select_weight_traj_targets",
    "maybe_build_weight_traj_observer",
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
        r2_sink=None,
    ):
        self.base_dir = capture_dir or os.path.join(os.getcwd(), "captures")
        # Optional R2 offload: upload each .pt then delete the local file after a
        # verified upload. None => keep .pt files local (byte-identical default).
        self.r2_sink = r2_sink
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
            # R2 offload (upload + delete-local after a verified upload). Raises on
            # failure and KEEPS the local file, so a bad upload never loses the
            # tensor; a misconfig (missing aws / creds / wrong bucket) fails loud.
            if self.r2_sink is not None:
                self.r2_sink.upload(
                    local_path=fpath,
                    key_suffix=f"{role}/tick_{tick_key[0]}_{tick_key[1]}/{fname}",
                    meta={
                        "role": role,
                        "global_step": tick_key[0],
                        "optimizer_tick": tick_key[1],
                        "target_name": target_name,
                    },
                )
        return True

    @property
    def n_written(self) -> int:
        return self._n_written

    def close(self) -> None:
        """Drain + join the (async) R2 upload pool and fail-loud on any failure.

        Idempotent. A no-op with no sink or a synchronous sink (nothing queued).
        Call at run end so the audit's final async uploads complete and any
        permanent failure surfaces.
        """
        if self.r2_sink is not None and hasattr(self.r2_sink, "close"):
            self.r2_sink.close()


def maybe_build_capture_writer(config: Any, *, rank: Optional[int] = None) -> Optional[CaptureWriter]:
    """Construct a :class:`CaptureWriter` iff ``comm_eff.capture.enabled``, else None.

    The single gate so the disabled / non-capture path never creates a writer or
    touches the filesystem. ``config`` is the ``CommEffConfig`` (or a node with a
    ``capture`` sub-config).
    """
    cap = getattr(config, "capture", None)
    if cap is None or not bool(getattr(cap, "enabled", False)):
        return None
    writer = CaptureWriter(
        capture_dir=str(getattr(cap, "capture_dir", "") or ""),
        max_ticks=int(getattr(cap, "max_ticks", 10)),
        stratified_targets=int(getattr(cap, "stratified_targets", 0)),
        dump_dtype=str(getattr(cap, "dump_dtype", "fp32")),
        rank=rank,
        rank0_only=bool(getattr(cap, "rank0_only", True)),
        min_tick=int(getattr(cap, "min_tick", 0)),
    )
    # R2 offload for the raw grad/activation dumps. Built only on the writer rank
    # (reusing the writer's resolved root for the local r2_manifest.jsonl). The
    # async knobs decouple uploads from compute (off => synchronous, as before).
    if bool(getattr(cap, "r2_enabled", False)) and not writer._inactive:
        writer.r2_sink = maybe_build_r2_sink(
            enabled=True,
            artifact_kind="grads",
            manifest_dir=writer.root,
            delete_local=bool(getattr(cap, "r2_delete_local", True)),
            async_mode=bool(getattr(cap, "r2_async", False)),
            upload_workers=int(getattr(cap, "r2_upload_workers", 4)),
            max_staged_gb=float(getattr(cap, "r2_max_staged_gb", 80.0)),
        )
    return writer


# ====================================================================== #
# Weight-trajectory FULL-weight instrument
# ====================================================================== #
#
# A dump-only weight-trajectory recorder used by the M4 weight-projection study
# (research/.claude/plans/43.md). The engine summons the FULL current weight
# matrices on every optimizer tick and hands them here; the observer saves the
# ACTUAL weight matrices to disk as ``full/<snapshot>.pt`` (a ``torch.save`` state
# dict) + a ``full_manifest.jsonl`` row. The cadence is set by ``per_tick``:
# per-tick dumps EVERY optimizer tick (``full/tick_<tick>.pt``), per-step dumps
# once per training step (``full/step_<gs>.pt``, deduped on global_step). There is
# NO compression and NO subset: the tensors saved ARE every floating param (cast
# to ``dump_dtype``), so ANY offline analysis can be run on them directly. It is
# strictly telemetry: it reads (never writes) the live weights, runs only on the
# actor-train path, and feeds nothing back into the optimizer, EMA, sketch V or Q.
#
# Storage cost (Qwen2.5-1.5B, ~1.54B params): a bf16 full-model snapshot ≈ 3 GB;
# fp32 ≈ 6 GB. A per-STEP bf16 80-step trajectory is ≈246 GB; the per-TICK variant
# (2 ticks/step) ≈492 GB. ``dump_dtype=fp32`` doubles that and is needed only when
# the downstream analysis differences consecutive snapshots (the ~1e-3 per-step
# update would be swamped by bf16's ~4e-3 rounding). These volumes do not fit the
# box / laptop, so set ``r2_enabled`` to upload each snapshot to R2 and delete the
# local ``.pt`` (see ``verl.workers.comm_eff.r2_sink``) — local disk becomes a
# few-GB staging area.
#
# NOTE: an earlier version of this instrument stored a lossy k-bucket COUNT-SKETCH
# of each matrix (non-invertible) plus a bounded exact-calibration ring. That was
# REMOVED (operator directive 2026-06-30): the study needs the raw weights, not a
# sketch. Recover the sketch implementation from git history if ever needed.


def select_weight_traj_targets(named_params) -> list:
    """Return ``[(canon_name, tensor), ...]`` for the FULL weight trajectory.

    Always selects EVERY floating-point parameter (the whole model: decoder
    linears + token embeddings + lm_head + RMSNorm gains + biases). There is no
    subset, no projector substring set, and no ``select_all`` toggle: the EXP-43
    deliverable is the raw weights of every trainable param. Pure selection — no
    device moves, no clones, no FSDP. Names are canonicalised (FSDP wrap-infix
    stripped) so the selection is identical off a summoned live module or a plain
    clone. Non-floating params (e.g. int buffers) are skipped because a weight
    trajectory is only defined for float tensors.
    """
    out = []
    for name, p in named_params:
        if not torch.is_floating_point(p):
            continue
        out.append((_canon(name), p))
    return out


def _norm(t: torch.Tensor) -> float:
    return float(torch.linalg.norm(t.to(torch.float32)).item())


class WeightTrajObserver:
    """Dump-only FULL-weight recorder (per-step or per-tick).

    Constructed once per worker iff ``comm_eff.probe.weight_traj.enabled`` — and,
    crucially, INDEPENDENTLY of ``comm_eff.enabled`` so the plain-GRPO regime
    (codec OFF) is still instrumented. :meth:`observe` is the single entry point:
    the engine summons ALL floating params to CPU/fp32 and hands them here on every
    optimizer tick; this class writes the **FULL weight matrices** to disk as
    ``full/<snapshot>.pt`` (a ``torch.save`` state dict ``{canon_name -> tensor}``)
    plus a ``full_manifest.jsonl`` row. The cadence is set by ``per_tick``:

    * ``per_tick=True`` — dump EVERY optimizer tick (``full/tick_<tick>.pt``). For
      batch128/mini64 that is 2 ticks/step ≈ 160 snapshots over 80 steps.
    * ``per_tick=False`` (default) — dump once per training step
      (``full/step_<gs>.pt``, deduped on ``global_step``, gated by ``every_steps``).

    Each manifest row carries both ``global_step`` and ``tick``, so a per-tick
    trajectory can be subsampled to the per-step one offline (the 80-point set is a
    subset of the 160-point set). Pure I/O — it never mutates the tensors it is
    given, and feeds nothing back into the optimizer / EMA / Q.

    There is NO compression and NO subset: the saved tensors ARE every floating
    param (cast to ``dump_dtype``). Storage is on DP rank 0 by default (the summoned
    full params are DP-identical); other ranks build an INACTIVE observer that
    no-ops so the engine's summon collective stays symmetric across ranks. When
    ``r2_enabled`` each snapshot is uploaded to R2 and the local ``.pt`` is deleted
    after a verified upload (see :mod:`verl.workers.comm_eff.r2_sink`).
    """

    def __init__(
        self,
        *,
        out_dir: str,
        dump_dtype: str = "bf16",
        per_tick: bool = False,
        every_steps: int = 1,
        rank: Optional[int] = None,
        rank0_only: bool = True,
        r2_enabled: bool = False,
        r2_delete_local: bool = True,
        r2_async: bool = False,
        r2_flush_every_steps: int = 10,
        r2_upload_workers: int = 4,
        r2_max_staged_gb: float = 80.0,
    ):
        self.enabled = True
        self.out_dir = out_dir or os.path.join(os.getcwd(), "weights")
        assert dump_dtype in ("bf16", "fp32"), dump_dtype
        self.dump_dtype = dump_dtype
        self._torch_dtype = torch.bfloat16 if dump_dtype == "bf16" else torch.float32
        # Snapshot cadence. per_tick dumps every observe() call (keyed by the
        # monotonic tick); otherwise dedup one full dump per global_step, gated by
        # every_steps. The observer always dumps ALL floating params — no subset.
        self.per_tick = bool(per_tick)
        self.every_steps = max(1, int(every_steps))
        # Async-upload flush cadence (only meaningful when the sink is async): the
        # observer drains the upload queue + checkpoints the manifest every N steps
        # so disk stays bounded and a failure surfaces promptly.
        self.r2_flush_every_steps = max(1, int(r2_flush_every_steps))
        self._last_flush_step = -1  # dedup the per-step flush barrier

        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.rank = int(rank)
        self.rank0_only = bool(rank0_only)
        self._inactive = self.rank0_only and self.rank != 0

        self._tick = 0  # monotonic optimizer-tick counter (own; comm_eff may be off)
        self._last_dumped_step = -1  # dedup: at most one full dump per global_step (per-step mode)
        self._n_dumped = 0
        self._closed = False

        self.r2_sink = None
        if not self._inactive:
            self.full_dir = os.path.join(self.out_dir, "full")
            os.makedirs(self.full_dir, exist_ok=True)
            self.manifest_path = os.path.join(self.out_dir, "full_manifest.jsonl")
            # R2 offload: upload each snapshot then delete the local .pt. Built only
            # on the writer rank; creds + bucket guard live in r2_sink. The async
            # knobs decouple uploads from compute (off => synchronous, as before).
            self.r2_sink = maybe_build_r2_sink(
                enabled=bool(r2_enabled),
                artifact_kind="weights",
                manifest_dir=self.out_dir,
                delete_local=bool(r2_delete_local),
                async_mode=bool(r2_async),
                upload_workers=int(r2_upload_workers),
                max_staged_gb=float(r2_max_staged_gb),
            )
        print(
            f"[comm_eff][weight_traj] FULL-weight observer out_dir={self.out_dir} "
            f"dump_dtype={self.dump_dtype} per_tick={self.per_tick} every_steps={self.every_steps} "
            f"rank={self.rank} rank0_only={self.rank0_only} inactive={self._inactive} "
            f"r2={'on' if self.r2_sink is not None else 'off'} "
            f"r2_async={bool(r2_async) and self.r2_sink is not None} flush_every={self.r2_flush_every_steps}",
            flush=True,
        )

    def observe(self, weights: dict, global_step: int = -1) -> int:
        """Record this tick; dump the FULL weights per the configured cadence.

        ``weights`` is ``{canon_name -> CPU fp32 tensor}`` (ALL floating params).
        Pure read: tensors are never mutated. No-op on an inactive (non-writer)
        rank. The engine calls this per optimizer TICK. When ``per_tick`` the full
        dump fires EVERY tick (``full/tick_<tick>.pt``); otherwise it fires once per
        ``global_step`` (deduped, gated by ``every_steps``, ``full/step_<gs>.pt``).
        Returns the tick index (or -1 when inactive).
        """
        if self._inactive:
            return -1
        tick = self._tick
        self._tick += 1

        gs = int(global_step)
        if self.per_tick:
            # One snapshot per optimizer tick, keyed by the monotonic tick index.
            self._dump_full(f"tick_{tick}", weights, global_step=gs, tick=tick)
        elif gs >= 0 and gs != self._last_dumped_step and (gs % self.every_steps == 0):
            self._dump_full(f"step_{gs}", weights, global_step=gs, tick=tick)
            self._last_dumped_step = gs
        # Async-upload flush barrier: every r2_flush_every_steps trainer steps,
        # drain the upload queue + checkpoint the manifest (no-op when the sink is
        # synchronous or None — flush() short-circuits). Bounds disk and surfaces a
        # failed upload promptly (fail-loud). Deduped on global_step so a per-tick
        # cadence flushes once per matching step, not once per tick.
        if (
            self.r2_sink is not None
            and gs >= 0
            and gs != self._last_flush_step
            and (gs % self.r2_flush_every_steps == 0)
        ):
            self.r2_sink.flush()
            self._last_flush_step = gs
        return tick

    def close(self) -> None:
        """Run-end barrier: flush + drain the R2 upload queue, fail-loud, join workers.

        Idempotent. A no-op on an inactive rank or when no sink is attached. MUST be
        called at run end so the final in-flight async uploads complete and any
        permanent failure surfaces (a silently-incomplete trajectory is forbidden).
        With a synchronous sink this is a cheap no-op (nothing is queued).
        """
        if self._closed:
            return
        self._closed = True
        if self.r2_sink is not None and hasattr(self.r2_sink, "close"):
            self.r2_sink.close()

    def _dump_full(self, snapshot_id: str, weights: dict, *, global_step: int, tick: int) -> None:
        """Write a FULL-weight snapshot to ``full/<snapshot_id>.pt`` (+ manifest row).

        No compression: each tensor is detached, cast to ``dump_dtype`` (bf16 by
        default), moved to CPU and stored AS-IS in a ``torch.save`` state dict
        ``{canon_name -> tensor}``. The ``full_manifest.jsonl`` row records the
        snapshot's ``global_step`` + ``tick`` (so a per-tick trajectory subsamples
        to per-step) and per-matrix name / shape / exact-fp32 Frobenius norm so the
        analyst can verify the dump loads and the norms match within ``dump_dtype``
        rounding. When an R2 sink is attached the saved ``.pt`` is uploaded and the
        local file is deleted after a verified upload.
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
        fname = f"{snapshot_id}.pt"
        fpath = os.path.join(self.full_dir, fname)
        torch.save(state, fpath)
        with open(self.manifest_path, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "global_step": int(global_step),
                        "tick": int(tick),
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
            f"[comm_eff][weight_traj] FULL dump {snapshot_id} (step={global_step} tick={tick}) "
            f"n_matrices={len(records)} dtype={self.dump_dtype} -> {os.path.join('full', fname)}",
            flush=True,
        )
        if self.r2_sink is not None:
            self.r2_sink.upload(
                local_path=fpath,
                key_suffix=f"full/{snapshot_id}/{fname}",
                meta={
                    "role": "weights",
                    "global_step": int(global_step),
                    "tick": int(tick),
                    "n_matrices": len(records),
                    "dump_dtype": self.dump_dtype,
                },
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
    return WeightTrajObserver(
        out_dir=str(getattr(wt, "out_dir", "") or ""),
        dump_dtype=str(getattr(wt, "dump_dtype", "bf16")),
        per_tick=bool(getattr(wt, "per_tick", False)),
        every_steps=int(getattr(wt, "every_steps", 1)),
        rank=rank,
        rank0_only=bool(getattr(wt, "rank0_only", True)),
        r2_enabled=bool(getattr(wt, "r2_enabled", False)),
        r2_delete_local=bool(getattr(wt, "r2_delete_local", True)),
        r2_async=bool(getattr(wt, "r2_async", False)),
        r2_flush_every_steps=int(getattr(wt, "r2_flush_every_steps", 10)),
        r2_upload_workers=int(getattr(wt, "r2_upload_workers", 4)),
        r2_max_staged_gb=float(getattr(wt, "r2_max_staged_gb", 80.0)),
    )
