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

Storage (Qwen2.5-1.5B):
  EXP-57 fp32 GSM8K  ~6.17 GB/snap, tick_0..159 -> ~987 GB (~1 TB) all-160 | ~494 GB per-step
  EXP-43 bf16 GSM8K  ~3.08 GB/snap, tick_0..159 -> ~492 GB all-160         | ~246 GB per-step
  EXP-58 fp32 Big-Math ~6.17 GB/snap, step_20..1000 (50 snaps, spacing 20) -> ~309 GB all-50
Provision the box disk accordingly (e.g. --disk-gb 1100 for the fp32 all-160 trace).

Layouts. EXP-57/EXP-43 store full/tick_<N>/tick_<N>.pt (contiguous). EXP-58 Big-Math stores
full/step_<N>/step_<N>.pt with non-contiguous N=20..1000; --src-layout step auto-discovers
those and NORMALIZES them to a contiguous local tick_0..tick_49 axis, so the analysis engine
(tick_<i>-only) consumes Big-Math unchanged. R2 is READ-ONLY here (this only downloads).

R2 creds: `set -a; . ~/.config/verl-research/secrets.env; set +a` first (maps R2_* -> AWS_*
via weight_proj.r2_stream). Bucket shamane-pluralis. Secret VALUES never printed.

Usage (from research/):
  # whole fp32 GSM8K trace, all 160 ticks, 6-way parallel (the operator default):
  python scripts/weight_proj_fetch_trace.py --experiment EXP-57 --dest /workspace/trace/EXP-57
  # per-step subsample (~half the bytes):
  python scripts/weight_proj_fetch_trace.py --experiment EXP-57 --dest /workspace/trace/EXP-57 --cadence per-step
  # Big-Math fp32 (EXP-58): auto-discover step_<N>, normalize -> local tick_0..tick_49:
  python scripts/weight_proj_fetch_trace.py --experiment EXP-58 --src-layout step --dest /workspace/trace/EXP-58
  #   then synth its manifest (50 ticks) and analyse:
  #   python scripts/synth_exp57_manifests.py --experiment EXP-58 --trace-root /workspace/trace/EXP-58 --n-ticks 50
  #   python scripts/moat_scorecard.py --experiment EXP-58 --trace-root /workspace/trace/EXP-58 --n-ticks 50 \\
  #     --manifest runs/EXP-58/regimeA/weights/full_manifest.jsonl --delta 5,10,20 --h 1,2,5,10,20 --operating-point 20,20
  # offline logic check (no network/creds):
  python scripts/weight_proj_fetch_trace.py --self-test
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


def _parse_pre_indices(ls_output: str, stem: str) -> list[int]:
    """Parse `aws s3 ls <prefix>/` output -> sorted ascending [N] for <stem>_<N>/ subdirs.

    Pure (no network) so the layout-discovery logic is unit-testable offline (--self-test).
    """
    out: list[int] = []
    pfx = f"{stem}_"
    for line in ls_output.splitlines():
        line = line.strip()
        if line.startswith("PRE ") and pfx in line:
            token = line[4:].strip().rstrip("/")
            if token.startswith(pfx):
                try:
                    out.append(int(token[len(pfx):]))
                except ValueError:
                    pass
    return sorted(set(out))


def _discover_source_indices(bucket: str, prefix: str, stem: str) -> list[int]:
    """List <prefix>/ (READ-ONLY) and parse <stem>_<N>/ subdirs -> sorted ascending [N].

    Lets the fetcher self-configure to a trace whose snapshots are NOT contiguous 0..n-1
    (e.g. EXP-58 Big-Math fp32: step_20..step_1000, spacing 20)."""
    cp = subprocess.run(
        ["aws", "s3", "ls", f"s3://{bucket}/{prefix}/", "--endpoint-url", RS.r2_endpoint()],
        env=RS.r2_env(), capture_output=True, text=True,
    )
    if cp.returncode != 0:
        return []
    return _parse_pre_indices(cp.stdout, stem)


def _fetch_one(src_index: int, local_index: int, stem: str, bucket: str,
               dest: str, expected: int | None) -> tuple[int, str]:
    """Download source snapshot <prefix>/<stem>_<src>/<stem>_<src>.pt to the CANONICAL local
    layout <dest>/full/tick_<local>/tick_<local>.pt (skip if already complete).

    For EXP-57 (stem='tick', contiguous) src==local, so the local layout is byte-identical
    to the original tick-only fetcher. For EXP-58 Big-Math (stem='step', N=20..1000) the
    source step index is NORMALIZED to a contiguous local tick index, so the analysis engine
    (which only knows full/tick_<i>/tick_<i>.pt) consumes the Big-Math trace unchanged.

    ATOMIC + self-healing: downloads to a `.part` sidecar and `os.replace`s onto the final
    path ONLY after a clean transfer (+ size check when known). So an interrupted download
    (SSH drop / disk pressure) never leaves a partial file at the FINAL path where a later
    resume would silently skip it — a stale `.part` is simply overwritten on the next run.
    """
    src_key = f"{RS.canonical_prefix()}/{stem}_{src_index}/{stem}_{src_index}.pt"
    local_dir = os.path.join(dest, "full", f"tick_{local_index}")
    local = os.path.join(local_dir, f"tick_{local_index}.pt")
    part = local + ".part"
    if os.path.exists(local):
        sz = os.path.getsize(local)
        if (expected is None and sz > 0) or (expected is not None and sz == expected):
            return local_index, "skip (present)"
        # present but wrong size (only detectable with expected) -> re-fetch
    os.makedirs(local_dir, exist_ok=True)
    if os.path.exists(part):
        os.remove(part)                       # discard any stale partial before re-download
    cp = subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{src_key}", part,
         "--endpoint-url", RS.r2_endpoint(), "--no-progress"],
        env=RS.r2_env(), capture_output=True, text=True,
    )
    if cp.returncode != 0:
        if os.path.exists(part):
            os.remove(part)                   # never leave a partial behind
        return local_index, f"FAIL rc={cp.returncode}: {cp.stderr.strip()[:200]}"
    sz = os.path.getsize(part) if os.path.exists(part) else 0
    if expected is not None and sz != expected:
        if os.path.exists(part):
            os.remove(part)
        return local_index, f"FAIL size {sz} != expected {expected}"
    os.replace(part, local)                   # atomic promote to the final path
    return local_index, f"ok ({sz/(1<<30):.2f} GB)"


def _selftest() -> int:
    """Offline (no-network) checks of the pure layout logic: discovery parse, step->tick
    normalization, and tick-layout identity (EXP-57 byte-identical)."""
    ls = ("                           PRE step_20/\n"
          "                           PRE step_1000/\n"
          "                           PRE step_100/\n"
          "                           PRE junk/\n"
          "2026-07-02 04:13:00 6174966359 step_100.pt\n")
    assert _parse_pre_indices(ls, "step") == [20, 100, 1000], "discovery parse"
    assert _parse_pre_indices(ls, "tick") == [], "wrong-stem parse must be empty"
    # EXP-58-like {20,40,...,1000} normalized to contiguous local ticks 0..49
    src = list(range(20, 1001, 20))
    pairs = [(s, i) for i, s in enumerate(src)]
    assert len(pairs) == 50 and pairs[0] == (20, 0) and pairs[-1] == (1000, 49), "normalize"
    # EXP-57 tick layout: local index == source index (no normalization)
    tick_pairs = [(s, s) for s in range(160)]
    assert tick_pairs[7] == (7, 7) and len(tick_pairs) == 160, "tick identity"
    print("[fetch] SELF-TEST GO — discovery parse + step->tick normalization + "
          "tick identity OK", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk-download a whole weight trace to local disk")
    ap.add_argument("--experiment", default="EXP-57")
    ap.add_argument("--regime", default="regimeA")
    ap.add_argument("--dest", default="", help="local trace root (writes <dest>/full/tick_N/tick_N.pt); required unless --self-test")
    ap.add_argument("--bucket", default=RS.CANONICAL_BUCKET)
    ap.add_argument("--cadence", default="per-tick", choices=["per-tick", "per-step"],
                    help="tick-layout only: per-tick = all 160 (operator default; download "
                         "everything); per-step = first tick of each global_step (~half the bytes)")
    ap.add_argument("--src-layout", default="tick", choices=["tick", "step"],
                    help="filename stem of the SOURCE snapshots in R2. 'tick' = EXP-57 fp32 GSM8K "
                         "(full/tick_<N>/tick_<N>.pt, contiguous 0..159; local layout byte-identical). "
                         "'step' = EXP-58 fp32 Big-Math (full/step_<N>/step_<N>.pt, N=20..1000 spacing "
                         "20) — the step indices are auto-DISCOVERED and NORMALIZED to a contiguous "
                         "local tick axis so the analysis engine (tick_<i>-only) consumes them unchanged.")
    ap.add_argument("--src-indices", default="",
                    help="explicit comma list of SOURCE snapshot indices (override auto-discovery); "
                         "normalized to contiguous local tick_0..tick_{n-1}")
    ap.add_argument("--n-ticks", type=int, default=160,
                    help="cap on how many snapshots to fetch (tick: at the chosen cadence; "
                         "step/explicit: the first n discovered indices)")
    ap.add_argument("--jobs", type=int, default=6, help="parallel aws s3 cp streams")
    ap.add_argument("--verify-sizes", action="store_true",
                    help="verify each download against runs/<exp>/.../r2_manifest.jsonl remote_bytes")
    ap.add_argument("--self-test", action="store_true",
                    help="offline (no-network) check of the layout logic, then exit")
    args = ap.parse_args()

    if args.self_test:
        return _selftest()
    if not args.dest:
        ap.error("--dest is required (unless --self-test)")

    stem = args.src_layout
    prefix = f"verl-research/{args.experiment}/{args.regime}/weights/full"
    os.environ["WP_R2_PREFIX"] = prefix        # resolves canonical_prefix()/tick_key()

    # ---- resolve SOURCE snapshot indices + their normalized LOCAL tick indices ----
    if stem == "tick" and not args.src_indices:
        # EXP-57 path (UNCHANGED): contiguous ticks at the requested cadence; local
        # index == source index, so full/tick_<N>/tick_<N>.pt is byte-identical.
        src = TS.select_ticks(args.cadence, args.n_ticks)
        pairs = [(s, s) for s in src]
        normalized = False
        if args.cadence == "per-step":
            print("[fetch] WARNING: --cadence per-step fetches ONLY the even ticks "
                  f"({len(src)} of 160). The synthesized full_manifest still advertises all 160 "
                  "ticks, so per-tick analysis will FileNotFoundError on the missing odd ticks. Use "
                  "the default (per-tick) for the download-everything flow; per-step is a subset.",
                  flush=True)
    else:
        # EXP-58 / arbitrary layout: discover (or take explicit) source indices and
        # NORMALIZE to a contiguous local tick axis 0..n-1 for the analysis engine.
        if args.src_indices:
            src = sorted(int(x) for x in args.src_indices.split(",") if x.strip())
        else:
            src = _discover_source_indices(args.bucket, prefix, stem)
        if not src:
            print(f"[fetch] FATAL: no {stem}_<N> snapshots discovered under "
                  f"s3://{args.bucket}/{prefix}/ — check --experiment/--src-layout", flush=True)
            return 2
        if args.n_ticks and 0 < args.n_ticks < len(src):
            src = src[:args.n_ticks]
        pairs = [(s, i) for i, s in enumerate(src)]     # normalize -> tick_0..tick_{n-1}
        normalized = True

    size_by_tick = _load_size_by_tick(args.experiment, args.regime) if args.verify_sizes else {}
    if args.verify_sizes and not size_by_tick:
        print("[fetch] --verify-sizes set but no r2_manifest with remote_bytes found — "
              "falling back to nonzero-size check", flush=True)
    os.makedirs(os.path.join(args.dest, "full"), exist_ok=True)
    map_note = (f" (NORMALIZED {stem}_<N> -> local tick_<i>; e.g. {stem}_{pairs[0][0]}->tick_0"
                f" .. {stem}_{pairs[-1][0]}->tick_{pairs[-1][1]})") if normalized else ""
    print(f"[fetch] {args.experiment}/{args.regime} [{stem}] -> {args.dest} : {len(pairs)} snapshots"
          f"{map_note}, jobs={args.jobs}, verify_sizes={bool(size_by_tick)}", flush=True)

    results, failures, skipped = [], [], 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_fetch_one, s, l, stem, args.bucket, args.dest,
                          size_by_tick.get(s)): (s, l)
                for s, l in pairs}
        for fut in cf.as_completed(futs):
            s, l = futs[fut]
            _, status = fut.result()
            if status.startswith("FAIL"):
                failures.append((l, status))
            elif status.startswith("skip"):
                skipped += 1
            results.append((l, status))
            print(f"[fetch] {stem}_{s} -> tick_{l}: {status}", flush=True)

    print(f"[fetch] DONE: {len(results)} snapshots, {skipped} skipped, {len(failures)} failed", flush=True)
    if failures:
        for l, s in sorted(failures)[:20]:
            print(f"  - tick_{l}: {s}", flush=True)
        return 1
    # NOTE: for a normalized (step) trace, synthesize the manifest with matching --n-ticks:
    #   python scripts/synth_exp57_manifests.py --experiment {exp} --trace-root {dest} --n-ticks {n}
    print(f"[fetch] trace ready at {args.dest} ({len(pairs)} local ticks) — synth its manifest with "
          f"--n-ticks {len(pairs)}, then analyse: python scripts/moat_scorecard.py --manifest "
          f"runs/{args.experiment}/{args.regime}/weights/full_manifest.jsonl "
          f"--trace-root {args.dest} --n-ticks {len(pairs)} ...", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
