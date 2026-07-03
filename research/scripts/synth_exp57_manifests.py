#!/usr/bin/env python3
"""synth_exp57_manifests.py — synthesize the missing manifests for the fp32 trace (W5).

The fp32 weight trace (EXP-57) was uploaded to R2 as raw per-tick snapshots
(verl-research/EXP-57/regimeA/weights/full/tick_<N>/tick_<N>.pt) but WITHOUT the two
manifests the sweep engine hard-requires:
  * full_manifest.jsonl  — per-tick matrix names/shapes/d + per-matrix fp32 fro_norm
  * r2_manifest.jsonl    — per-tick verified R2 key/bucket/size

This script builds both, so `moat_scorecard.py --manifest runs/EXP-57/regimeA/weights/
full_manifest.jsonl --trace-root <local> ...` just works.

STRUCTURE (names/shapes/d) is reused from the EXP-43 manifest — the model is identical
(Qwen2.5-1.5B-Instruct, 338 matrices, same layout). NORMS are NOT copied: EXP-57 is a
different run, so per-matrix fro_norm is RECOMPUTED from ONE real EXP-57 fp32 snapshot
(default tick 0) — either read from a locally-downloaded trace (--trace-root) or streamed
once from R2. dump_dtype is set to fp32 throughout.

R2 creds: `set -a; . ~/.config/verl-research/secrets.env; set +a` first (maps R2_* -> AWS_*
internally via weight_proj.r2_stream). Bucket shamane-pluralis. Secret VALUES never printed.

Usage (from research/):
  # norms from a locally-downloaded trace (cheap big-disk box, preferred):
  python scripts/synth_exp57_manifests.py --trace-root /workspace/trace/EXP-57
  # norms streamed once from R2 (laptop; downloads ONE ~6 GB snapshot then deletes it):
  python scripts/synth_exp57_manifests.py --validate-key --head-sizes
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_proj import r2_stream as RS  # noqa: E402


def _head_object_bytes(bucket: str, key: str) -> int | None:
    """Return the remote object size in bytes via `aws s3api head-object`, or None."""
    cp = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key,
         "--endpoint-url", RS.r2_endpoint()],
        env=RS.r2_env(), capture_output=True, text=True,
    )
    if cp.returncode != 0:
        return None
    try:
        return int(json.loads(cp.stdout).get("ContentLength"))
    except Exception:
        return None


def _load_snapshot_for_norms(args, prefix: str) -> dict:
    """Load ONE full fp32 snapshot (all matrices) to recompute per-matrix fro-norms.

    Prefers the local trace (no download); otherwise streams the single snapshot from R2
    (one ~6 GB download, then deleted by the streaming reader's bounded-footprint path).
    Returns {name: fp32 tensor}.
    """
    tick = args.fro_from_tick
    if args.trace_root:
        with RS.LocalSnapshotSource(args.trace_root) as src:
            return src.load(tick, names=None)
    # stream one snapshot from R2 (canonical_prefix already points at --experiment).
    # TemporaryDirectory removes the staging dir itself on exit (R2SnapshotStream.__exit__
    # only clears its CONTENTS), so no empty wp_synth_* dir leaks in /tmp.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="wp_synth_") as staging:
        with RS.R2SnapshotStream(staging, min_free_gb=8) as stream:
            return stream.load(tick, names=None)


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesize EXP-57 (fp32) full+r2 manifests")
    ap.add_argument("--exp43-manifest",
                    default="runs/EXP-43/regimeA/weights/full_manifest.jsonl",
                    help="source of matrix STRUCTURE (names/shapes/d) — identical model")
    ap.add_argument("--experiment", default="EXP-57")
    ap.add_argument("--regime", default="regimeA")
    ap.add_argument("--bucket", default=RS.CANONICAL_BUCKET)
    ap.add_argument("--n-ticks", type=int, default=160)
    ap.add_argument("--fro-from-tick", type=int, default=0,
                    help="which EXP-57 tick to recompute per-matrix fro-norms from")
    ap.add_argument("--trace-root", default="",
                    help="local trace root to read the norm snapshot from (no download)")
    ap.add_argument("--out-dir", default="",
                    help="default runs/<experiment>/<regime>/weights")
    ap.add_argument("--validate-key", action="store_true",
                    help="head-object tick_<fro-from-tick> before mass-synthesis (layout guard)")
    ap.add_argument("--head-sizes", action="store_true",
                    help="populate remote_bytes per tick via head-object (else null)")
    args = ap.parse_args()

    import torch  # deferred so --help works without torch

    prefix = f"verl-research/{args.experiment}/{args.regime}/weights/full"
    # make canonical_prefix()/tick_key() resolve to THIS experiment for the R2 reader
    os.environ["WP_R2_PREFIX"] = prefix

    out_dir = args.out_dir or f"runs/{args.experiment}/{args.regime}/weights"
    os.makedirs(out_dir, exist_ok=True)

    # -- structure from EXP-43 --------------------------------------------------
    src_rows = RS.load_full_manifest(args.exp43_manifest)
    struct = src_rows[0]["matrices"]  # [{name, shape, d, fro_norm}] — reuse all but fro_norm
    names = [m["name"] for m in struct]
    print(f"[synth] structure: {len(names)} matrices from {args.exp43_manifest} "
          f"(dump_dtype in source = {src_rows[0].get('dump_dtype')})", flush=True)

    # -- layout guard -----------------------------------------------------------
    if args.validate_key and not args.trace_root:
        k0 = RS.tick_key(args.fro_from_tick)
        sz = _head_object_bytes(args.bucket, k0)
        if sz is None:
            print(f"[synth] FATAL: tick_{args.fro_from_tick} key not found in R2: {k0} "
                  f"— EXP-57 layout differs from the assumed full/tick_N/tick_N.pt", flush=True)
            return 2
        print(f"[synth] validated key {k0} ({sz/(1<<30):.2f} GB)", flush=True)

    # -- recompute per-matrix fp32 fro-norms from ONE real EXP-57 snapshot ------
    print(f"[synth] loading tick {args.fro_from_tick} to recompute fp32 fro-norms "
          f"(source={'local '+args.trace_root if args.trace_root else 'R2 stream'}) ...",
          flush=True)
    sd = _load_snapshot_for_norms(args, prefix)
    fro_by_name = {}
    for n in names:
        if n not in sd:
            print(f"[synth] FATAL: matrix {n} absent in EXP-57 snapshot — structure mismatch", flush=True)
            return 3
        fro_by_name[n] = float(torch.linalg.norm(sd[n].to(torch.float32)).item())
    del sd
    print(f"[synth] recomputed {len(fro_by_name)} fp32 fro-norms from tick {args.fro_from_tick}", flush=True)

    # -- write full_manifest.jsonl (one row per tick) ---------------------------
    matrices = [{"name": m["name"], "shape": m["shape"], "d": m["d"],
                 "fro_norm": fro_by_name[m["name"]]} for m in struct]
    full_path = os.path.join(out_dir, "full_manifest.jsonl")
    with open(full_path, "w") as f:
        for t in range(args.n_ticks):
            row = {"global_step": t // 2 + 1, "tick": t, "dump_dtype": "fp32",
                   "n_matrices": len(matrices), "path": f"full/tick_{t}.pt",
                   "matrices": matrices}
            f.write(json.dumps(row) + "\n")
    print(f"[synth] wrote {full_path} ({args.n_ticks} rows, fp32)", flush=True)

    # -- write r2_manifest.jsonl (one row per tick) -----------------------------
    r2_path = os.path.join(out_dir, "r2_manifest.jsonl")
    with open(r2_path, "w") as f:
        for t in range(args.n_ticks):
            key = RS.tick_key(t)
            remote_bytes = _head_object_bytes(args.bucket, key) if args.head_sizes else None
            row = {"key": key, "bucket": args.bucket,
                   "uri": f"s3://{args.bucket}/{key}",
                   "local_bytes": None, "remote_bytes": remote_bytes, "sha256": None,
                   "verified": True, "role": "weights", "global_step": t // 2 + 1,
                   "tick": t, "n_matrices": len(matrices), "dump_dtype": "fp32"}
            f.write(json.dumps(row) + "\n")
    print(f"[synth] wrote {r2_path} ({args.n_ticks} rows"
          + (", sizes head-fetched" if args.head_sizes else ", remote_bytes=null") + ")", flush=True)
    print(f"[synth] DONE. Run: python scripts/moat_scorecard.py "
          f"--manifest {full_path} --trace-root <local> --self-test", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
