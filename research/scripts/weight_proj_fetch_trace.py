#!/usr/bin/env python3
"""weight_proj_fetch_trace.py — one-time whole-trace downloader (W3).

Pulls an ENTIRE weight trace from R2 to local disk so the analysis can run in the
"download everything first, then analyse" mode (tasks 1 & 3) on a cheap, big-disk,
GPU-free box. This is the ONE place a bulk pull is allowed — the streaming reader
(weight_proj/r2_stream.R2SnapshotStream) and the collection launcher keep their
bounded-footprint discipline unchanged.

Writes <dest>/full/tick_<N>/tick_<N>.pt — the EXACT nested layout that
weight_proj.r2_stream.LocalSnapshotSource (and --trace-root) expects. Resumable: a tick
whose local file already exists at the expected size is skipped, so a re-run only fetches
what's missing. Copies run in parallel (--jobs).

Storage (Qwen2.5-1.5B, 160 ticks):
  EXP-57 fp32 ~6.17 GB/snapshot -> ~987 GB (~1 TB) all-160  | ~494 GB per-step (~80)
  EXP-43 bf16 ~3.08 GB/snapshot -> ~492 GB all-160          | ~246 GB per-step
Provision the box disk accordingly (e.g. --disk-gb 1100 for the fp32 all-160 trace).

R2 creds: `set -a; . ~/.config/verl-research/secrets.env; set +a` first (maps R2_* -> AWS_*
via weight_proj.r2_stream). Bucket shamane-pluralis. Secret VALUES never printed.

Usage (from research/):
  # whole fp32 trace, all 160 ticks, 6-way parallel (the operator default):
  python scripts/weight_proj_fetch_trace.py --experiment EXP-57 --dest /workspace/trace/EXP-57
  # per-step subsample (~half the bytes):
  python scripts/weight_proj_fetch_trace.py --experiment EXP-57 --dest /workspace/trace/EXP-57 --cadence per-step
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_proj import r2_stream as RS  # noqa: E402
from weight_proj import tick_select as TS  # noqa: E402


def _load_size_by_tick(experiment: str, regime: str) -> dict[int, int]:
    """remote_bytes per tick from a local r2_manifest.jsonl, if present (for size verify)."""
    path = f"runs/{experiment}/{regime}/weights/r2_manifest.jsonl"
    rows = RS.load_r2_manifest(path)
    return {t: r["remote_bytes"] for t, r in rows.items()
            if isinstance(r.get("remote_bytes"), int)}


def _fetch_one(tick: int, bucket: str, dest: str, expected: int | None) -> tuple[int, str]:
    """Download one tick to <dest>/full/tick_<N>/tick_<N>.pt (skip if already complete).

    ATOMIC + self-healing: downloads to a `.part` sidecar and `os.replace`s onto the final
    path ONLY after a clean transfer (+ size check when known). So an interrupted download
    (SSH drop / disk pressure) never leaves a partial file at the FINAL path where a later
    resume would silently skip it — a stale `.part` is simply overwritten on the next run.
    """
    key = RS.tick_key(tick)
    local_dir = os.path.join(dest, "full", f"tick_{tick}")
    local = os.path.join(local_dir, f"tick_{tick}.pt")
    part = local + ".part"
    if os.path.exists(local):
        sz = os.path.getsize(local)
        if (expected is None and sz > 0) or (expected is not None and sz == expected):
            return tick, "skip (present)"
        # present but wrong size (only detectable with expected) -> re-fetch
    os.makedirs(local_dir, exist_ok=True)
    if os.path.exists(part):
        os.remove(part)                       # discard any stale partial before re-download
    cp = subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", part,
         "--endpoint-url", RS.r2_endpoint(), "--no-progress"],
        env=RS.r2_env(), capture_output=True, text=True,
    )
    if cp.returncode != 0:
        if os.path.exists(part):
            os.remove(part)                   # never leave a partial behind
        return tick, f"FAIL rc={cp.returncode}: {cp.stderr.strip()[:200]}"
    sz = os.path.getsize(part) if os.path.exists(part) else 0
    if expected is not None and sz != expected:
        if os.path.exists(part):
            os.remove(part)
        return tick, f"FAIL size {sz} != expected {expected}"
    os.replace(part, local)                   # atomic promote to the final path
    return tick, f"ok ({sz/(1<<30):.2f} GB)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk-download a whole weight trace to local disk")
    ap.add_argument("--experiment", default="EXP-57")
    ap.add_argument("--regime", default="regimeA")
    ap.add_argument("--dest", required=True, help="local trace root (writes <dest>/full/tick_N/tick_N.pt)")
    ap.add_argument("--bucket", default=RS.CANONICAL_BUCKET)
    ap.add_argument("--cadence", default="per-tick", choices=["per-tick", "per-step"],
                    help="per-tick = all 160 (operator default; download everything); "
                         "per-step = first tick of each global_step (~half the bytes)")
    ap.add_argument("--n-ticks", type=int, default=160, help="max ticks at the chosen cadence")
    ap.add_argument("--jobs", type=int, default=6, help="parallel aws s3 cp streams")
    ap.add_argument("--verify-sizes", action="store_true",
                    help="verify each download against runs/<exp>/.../r2_manifest.jsonl remote_bytes")
    args = ap.parse_args()

    # resolve tick_key() to THIS experiment
    os.environ["WP_R2_PREFIX"] = f"verl-research/{args.experiment}/{args.regime}/weights/full"

    ticks = TS.select_ticks(args.cadence, args.n_ticks)
    if args.cadence == "per-step":
        print("[fetch] WARNING: --cadence per-step fetches ONLY the even ticks "
              f"({len(ticks)} of 160). The synthesized full_manifest still advertises all 160 ticks, so "
              "per-tick analysis will FileNotFoundError on the missing odd ticks. Use the default "
              "(per-tick, all 160) for the download-everything flow; per-step is a deliberate subset.", flush=True)
    size_by_tick = _load_size_by_tick(args.experiment, args.regime) if args.verify_sizes else {}
    if args.verify_sizes and not size_by_tick:
        print("[fetch] --verify-sizes set but no r2_manifest with remote_bytes found — "
              "falling back to nonzero-size check", flush=True)
    os.makedirs(os.path.join(args.dest, "full"), exist_ok=True)
    print(f"[fetch] {args.experiment}/{args.regime} -> {args.dest} : {len(ticks)} ticks "
          f"({args.cadence}), jobs={args.jobs}, verify_sizes={bool(size_by_tick)}", flush=True)

    results, failures, skipped = [], [], 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_fetch_one, t, args.bucket, args.dest, size_by_tick.get(t)): t
                for t in ticks}
        for fut in cf.as_completed(futs):
            t = futs[fut]
            _, status = fut.result()
            if status.startswith("FAIL"):
                failures.append((t, status))
            elif status.startswith("skip"):
                skipped += 1
            results.append((t, status))
            print(f"[fetch] tick {t}: {status}", flush=True)

    print(f"[fetch] DONE: {len(results)} ticks, {skipped} skipped, {len(failures)} failed", flush=True)
    if failures:
        for t, s in sorted(failures)[:20]:
            print(f"  - tick {t}: {s}", flush=True)
        return 1
    print(f"[fetch] trace ready at {args.dest} — analyse with: "
          f"python scripts/weight_proj_sweep.py runs/{args.experiment}/{args.regime}/weights/"
          f"full_manifest.jsonl --trace-root {args.dest} ...", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
