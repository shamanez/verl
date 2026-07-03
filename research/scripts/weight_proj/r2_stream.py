#!/usr/bin/env python3
"""weight_proj/r2_stream.py — snapshot readers for the weight trace (R2 stream + local disk).

TWO readers behind ONE `load(tick, names) -> {name: fp32 tensor}` contract:
  * R2SnapshotStream    — the bounded-footprint STREAMING reader (unchanged). Governs
                          the "pull from R2 one .pt at a time" path used for cheap,
                          few-snapshot analyses (e.g. the GPU-gated tier, #46).
  * LocalSnapshotSource — reads a PRE-DOWNLOADED trace off local disk (NO download,
                          NO delete, NO df guard, NO working-set cap). This is the
                          "download everything first, then analyse on a big-disk box"
                          mode (tasks 1 & 3) — the bounded-footprint constraint is
                          RELEASED for analysis here.
Callers pick by presence of a local trace root (weight_proj_sweep.py --trace-root).

HARD streaming contract for R2SnapshotStream ONLY (bounded-footprint single-streaming-pass R2 access discipline):
  * Drive downloads from the in-repo manifests, NOT a bucket-list.
  * Load each `.pt` snapshot exactly ONCE, extract the per-matrix slices the caller
    asked for, then DELETE the local `.pt` immediately.
  * NEVER `aws s3 cp --recursive` the prefix from THIS path. The staging dir holds at
    most a couple of in-flight snapshots (bounded working set); we assert `df` headroom.
    (The one-shot whole-trace fetch lives in weight_proj_fetch_trace.py, not here.)
The COLLECTION launcher (weight_traj_run_cell.sh) keeps its upload-then-delete
discipline unchanged — the released-footprint toggle is ANALYSIS-only.

R2 credential mapping (cribbed VERBATIM from verify_full_weight_dump.py /
verl/workers/comm_eff/r2_sink.py — do NOT reinvent):
  AWS_ACCESS_KEY_ID     = $R2_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY = $R2_SECRET_ACCESS_KEY
  AWS_DEFAULT_REGION    = auto
  --endpoint-url          $R2_ENDPOINT  (or https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com)
  bucket                  shamane-pluralis  (from $R2_BUCKET / manifest)
Secret VALUES are never logged.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

CANONICAL_BUCKET = "shamane-pluralis"


def canonical_prefix() -> str:
    """R2 key prefix for the trace's `full/` dir.

    Defaults to the EXP-43 layout (byte-identical to the original streaming path)
    but is overridable for other experiments — chiefly the fp32 EXP-57 trace — via
    either WP_R2_PREFIX (a full prefix) or WP_R2_EXPERIMENT (just the experiment id).
    Read LIVE (not frozen at import) so a caller can set the env then stream.
    """
    p = os.environ.get("WP_R2_PREFIX", "")
    if p:
        return p.rstrip("/")
    exp = os.environ.get("WP_R2_EXPERIMENT", "EXP-43")
    return f"verl-research/{exp}/regimeA/weights/full"


# Back-compat module constant (docstrings / stale imports reference it). Reflects the
# default experiment at import time; prefer canonical_prefix() for live reads.
CANONICAL_PREFIX = canonical_prefix()


def r2_endpoint() -> str:
    ep = os.environ.get("R2_ENDPOINT", "")
    if not ep and os.environ.get("R2_ACCOUNT_ID"):
        ep = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return ep


def r2_env() -> dict:
    return {
        **os.environ,
        "AWS_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": "auto",
    }


def r2_bucket() -> str:
    return os.environ.get("R2_BUCKET", CANONICAL_BUCKET) or CANONICAL_BUCKET


def load_full_manifest(manifest_path: str) -> list[dict]:
    """Rows: global_step, tick, dump_dtype, n_matrices, path, matrices[...]. Tick-ordered."""
    rows = [json.loads(l) for l in open(manifest_path) if l.strip()]
    rows.sort(key=lambda r: int(r["tick"]))
    return rows


def load_r2_manifest(manifest_path: str) -> dict[int, dict]:
    """tick -> row{key, uri, remote_bytes, verified, ...}. Empty if file absent."""
    if not os.path.exists(manifest_path):
        return {}
    out = {}
    for l in open(manifest_path):
        if not l.strip():
            continue
        r = json.loads(l)
        out[int(r["tick"])] = r
    return out


def tick_key(tick: int, prefix: str | None = None) -> str:
    """Canonical R2 key for a tick (matches r2_manifest: full/tick_<N>/tick_<N>.pt).

    Uses the live canonical_prefix() (experiment-overridable) unless an explicit
    prefix is passed. NOTE: _download prefers the verified key from r2_manifest and
    only falls back here when no r2 row exists — so on EXP-57 you must set
    WP_R2_EXPERIMENT/WP_R2_PREFIX (or supply an r2_manifest) or this returns EXP-43
    keys.
    """
    base = prefix if prefix is not None else canonical_prefix()
    return f"{base}/tick_{tick}/tick_{tick}.pt"


def _df_free_bytes(path: str) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def _reduce_state_dict(sd: dict, names: list[str] | None) -> dict:
    """Keep only `names` (all if None) and cast each kept tensor to a cpu fp32 tensor.

    Shared by R2SnapshotStream (stream-from-R2) and LocalSnapshotSource (read-from-disk)
    so the fp32-cast contract is IDENTICAL regardless of where the .pt came from
    (differencing is done in fp32 upstream). bf16->fp32 is exact; fp32->fp32 is a copy.
    """
    import torch
    keep = names if names is not None else list(sd.keys())
    out = {}
    for n in keep:
        if n in sd:
            out[n] = sd[n].detach().to("cpu").to(torch.float32)
    return out


class R2SnapshotStream:
    """Streams full-model .pt snapshots one at a time with a bounded staging dir.

    Usage (blocks-outside / ticks-inside is enforced by the CALLER; this class just
    guarantees a single load + immediate delete per snapshot and a footprint cap):

        with R2SnapshotStream(staging_dir, min_free_gb=8) as stream:
            for tick in ticks:
                sd = stream.load(tick, names)     # downloads, torch.loads slices
                ... use sd (dict name->fp32 tensor) ...
                # sd is auto-freed; the .pt is already deleted on disk
    """

    def __init__(self, staging_dir: str, min_free_gb: float = 8.0,
                 r2_rows: dict[int, dict] | None = None, verbose: bool = True):
        self.staging_dir = staging_dir
        self.min_free = min_free_gb * (1 << 30)
        self.r2_rows = r2_rows or {}
        self.verbose = verbose
        self.max_staged_observed = 0
        self.downloads = 0
        os.makedirs(staging_dir, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # leave nothing behind
        for f in os.listdir(self.staging_dir):
            try:
                os.remove(os.path.join(self.staging_dir, f))
            except OSError:
                pass
        return False

    def _staged_count(self) -> int:
        return len([f for f in os.listdir(self.staging_dir) if f.endswith(".pt")])

    def _download(self, tick: int, dst: str) -> None:
        # prefer the verified key from r2_manifest; fall back to canonical layout
        row = self.r2_rows.get(tick)
        key = row["key"] if (row and "key" in row) else tick_key(tick)
        bucket = row.get("bucket", r2_bucket()) if row else r2_bucket()
        free = _df_free_bytes(self.staging_dir)
        if free < self.min_free:
            raise RuntimeError(
                f"staging dir free {free/(1<<30):.1f}GB < min {self.min_free/(1<<30):.1f}GB "
                f"— refusing download to protect the disk (bounded-footprint contract)"
            )
        cp = subprocess.run(
            ["aws", "s3", "cp", f"s3://{bucket}/{key}", dst,
             "--endpoint-url", r2_endpoint(), "--no-progress"],
            env=r2_env(), capture_output=True, text=True,
        )
        if cp.returncode != 0:
            raise RuntimeError(
                f"R2_STREAM_FAIL: aws s3 cp s3://{bucket}/{key} rc={cp.returncode}: "
                f"{cp.stderr.strip()[:300]}"
            )
        self.downloads += 1

    def load(self, tick: int, names: list[str] | None = None) -> dict:
        """Download tick_<N>.pt, torch.load it, keep only `names` as fp32, delete the .pt.

        Returns {name: torch.float32 tensor}. If names is None, keeps all 338.
        Footprint: the .pt exists on disk only between download and the delete below.
        """
        import torch
        with tempfile.TemporaryDirectory(dir=self.staging_dir) as td:
            dst = os.path.join(td, f"tick_{tick}.pt")
            self._download(tick, dst)
            self.max_staged_observed = max(self.max_staged_observed, self._staged_count())
            try:
                sd = torch.load(dst, map_location="cpu", weights_only=False)
            finally:
                # delete IMMEDIATELY after load — before any per-matrix work
                if os.path.exists(dst):
                    os.remove(dst)
        # cast the requested slices to fp32 (differencing is done in fp32 upstream)
        out = _reduce_state_dict(sd, names)
        del sd
        return out

    def footprint_ok(self, cap: int = 2) -> bool:
        """Bounded working set: at most `cap` .pt on disk at any observed moment."""
        return self.max_staged_observed <= cap


class LocalSnapshotSource:
    """Reads a PRE-DOWNLOADED full-model trace straight off local disk.

    Drop-in for R2SnapshotStream behind the SAME `load(tick, names) -> {name: fp32
    tensor}` contract, for the "whole trace already on disk" analysis mode — the cheap,
    big-disk, GPU-free box the operator downloads everything to first (tasks 1 & 3).
    It performs NO download, NO delete, NO df-headroom guard, and NO working-set cap;
    the bounded-footprint streaming discipline is intentionally RELEASED here (it still
    governs COLLECTION and the R2 streaming path, both unchanged).

        with LocalSnapshotSource(trace_root) as src:
            for tick in ticks:
                sd = src.load(tick, names)   # torch.load off disk, keep only names as fp32

    SAFETY: __exit__ is a PURE NO-OP. This source never owns the files it reads, so it
    must NEVER remove them — a stray delete would wipe the ~1 TB local trace.

    Layout: <trace_root>/full/tick_<N>/tick_<N>.pt (mirrors the R2 key layout that
    weight_proj_fetch_trace.py writes); a flat <trace_root>/full/tick_<N>.pt is accepted
    as a fallback.
    """

    def __init__(self, trace_root: str, verbose: bool = True):
        self.trace_root = trace_root
        self.verbose = verbose
        # interface parity with R2SnapshotStream (read by logging / footprint checks)
        self.max_staged_observed = 0
        self.downloads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # PURE NO-OP — the trace is pre-downloaded and operator-owned; never delete it.
        return False

    def footprint_ok(self, cap: int = 2) -> bool:
        return True

    def _path(self, tick: int) -> str:
        nested = os.path.join(self.trace_root, "full", f"tick_{tick}", f"tick_{tick}.pt")
        if os.path.exists(nested):
            return nested
        # flat fallback (matches full_manifest `path` field: full/tick_<N>.pt)
        return os.path.join(self.trace_root, "full", f"tick_{tick}.pt")

    def load(self, tick: int, names: list[str] | None = None) -> dict:
        """Read tick_<N>.pt off local disk, keep only `names` as fp32. NEVER deletes."""
        import torch
        path = self._path(tick)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LOCAL_TRACE_MISSING: tick {tick} not found under {self.trace_root} "
                f"(looked for full/tick_{tick}/tick_{tick}.pt and full/tick_{tick}.pt) — "
                f"is the trace fully downloaded? (weight_proj_fetch_trace.py)"
            )
        sd = torch.load(path, map_location="cpu", weights_only=False)
        out = _reduce_state_dict(sd, names)
        del sd
        return out
