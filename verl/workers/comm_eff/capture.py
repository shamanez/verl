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
   (Correctness invariant "measurement-only probes never feed the optimizer" /
   "off-path parity"). It is pure I/O.

2. **Keyed, self-describing dumps.** Every tensor is keyed by
   ``(global_step, optimizer_tick, role, target_name)`` and saved with its
   ``shape``, ``dtype`` and Frobenius ``norm`` recorded in a per-tick manifest
   row, so the analyst can recompute ``reconstruction_rel_error`` from the dumped
   ``A`` / ``Â`` and confirm it matches the logged scalar (the "fp32 dump
   fidelity" invariant).

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
default (the substrate makes Q / M bit-identical across ranks; per-rank
``G_comp`` differs by shard, which the audit accounts for).
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
