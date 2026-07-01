#!/usr/bin/env python3
"""Verify a FULL-weight trajectory dump (EXP-43+).

Asserts each ``full/<snapshot>.pt`` is a state dict of the real weight matrices
(true shapes, NOT a flattened/reduced vector) and that each matrix's stored
Frobenius norm matches ``full_manifest.jsonl`` within ``--tol`` relative. This is
the analyst's dump-integrity check: the dump keeps the raw weights (nothing
lossy), so there is nothing to reconstruct — only that the dump loads and its
norms match the live weights within dump-dtype rounding.

Two snapshot cadences are handled transparently: per-step (``full/step_<gs>.pt``)
and per-tick (``full/tick_<tick>.pt``). Snapshots are matched to manifest rows by
their ``path`` field, so naming does not matter.

R2 mode: when the run uploaded each snapshot to R2 and deleted the local ``.pt``
(``r2_enabled``), the heavy tensors are no longer on disk - only the small
``full_manifest.jsonl`` (norms) and ``r2_manifest.jsonl`` (keys, verified sizes)
remain. Pass ``--r2`` to sample ``--r2-sample`` snapshots, ``aws s3 cp`` each from
R2 to a temp file, and verify those. Requires the R2_* env vars + the ``aws`` CLI.

Exit 0 + prints PASS iff: >=1 snapshot checked, every checked snapshot loads,
n_matrices matches the manifest, every 2-D matrix kept its real shape, and the max
relative norm error <= tol. Exit 1 + FAIL otherwise.

Usage:
  # local staging (smoke run with r2_delete_local=false, or r2 disabled)
  python verify_full_weight_dump.py runs/EXP-43/regimeA/weights [--tol 0.01] [--expect 160]
  # R2 (full run, local .pt deleted): sample-download from R2 and verify
  python verify_full_weight_dump.py runs/EXP-43/regimeA/weights --r2 --r2-sample 5
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile


def _r2_endpoint() -> str:
    ep = os.environ.get("R2_ENDPOINT", "")
    if not ep and os.environ.get("R2_ACCOUNT_ID"):
        ep = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return ep


def _r2_env() -> dict:
    return {
        **os.environ,
        "AWS_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": "auto",
    }


def _r2_download(key: str, dst: str) -> None:
    bucket = os.environ.get("R2_BUCKET", "")
    cp = subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", dst, "--endpoint-url", _r2_endpoint()],
        env=_r2_env(),
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"aws s3 cp s3://{bucket}/{key} failed rc={cp.returncode}: {cp.stderr.strip()[:300]}")


def _verify_snapshot(torch, sd, mr, tol, label, report) -> float:
    """Verify one loaded state dict against its manifest row. Returns max rel-err."""
    max_rel = 0.0
    if not isinstance(sd, dict) or not sd:
        report["errors"].append(f"{label}: not a non-empty state dict")
        return max_rel
    twod = [k for k, v in sd.items() if hasattr(v, "dim") and v.dim() == 2]
    if not twod:
        report["errors"].append(f"{label}: no 2-D weight matrices (shape lost - is this a flattened/reduced dump?)")
    if int(mr.get("n_matrices", -1)) != len(sd):
        report["errors"].append(f"{label}: manifest n_matrices={mr.get('n_matrices')} != loaded {len(sd)}")
    for m in mr.get("matrices", []):
        name = m["name"]
        if name not in sd:
            report["errors"].append(f"{label}: manifest matrix {name} absent in snapshot")
            continue
        if list(sd[name].shape) != list(m.get("shape", [])):
            report["errors"].append(f"{label}: {name} shape {list(sd[name].shape)} != manifest {m.get('shape')}")
        fro_man = float(m.get("fro_norm", 0.0))
        if fro_man > 0.0:
            fro_now = float(torch.linalg.norm(sd[name].to(torch.float32)).item())
            rel = abs(fro_now - fro_man) / fro_man
            max_rel = max(max_rel, rel)
            if rel > tol:
                report["errors"].append(
                    f"{label}: {name} norm rel-err {rel:.4f} > tol {tol} "
                    f"(loaded {fro_now:.4f} vs manifest {fro_man:.4f})"
                )
    return max_rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights_dir")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--expect", type=int, default=0, help="min expected number of manifest snapshots")
    ap.add_argument("--r2", action="store_true", help="sample-download snapshots from R2 (local .pt deleted)")
    ap.add_argument("--r2-sample", type=int, default=5, help="how many snapshots to download+verify in --r2 mode")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import torch  # deferred so --help works without torch

    wd = args.weights_dir
    full_dir = os.path.join(wd, "full")
    manifest = os.path.join(wd, "full_manifest.jsonl")
    rows = [json.loads(l) for l in open(manifest)] if os.path.exists(manifest) else []

    report = {
        "weights_dir": wd,
        "n_manifest_rows": len(rows),
        "tol": args.tol,
        "mode": "r2" if args.r2 else "local",
        "errors": [],
        "max_rel_norm_err": 0.0,
        "checked": 0,
    }

    if not rows:
        report["errors"].append("no full_manifest.jsonl rows (the full-weight dump did not fire)")
        _emit(report, args)
        print("FAIL  no manifest rows")
        return 1
    if args.expect and len(rows) < args.expect:
        report["note"] = f"manifest rows {len(rows)} < expect {args.expect}"

    max_rel = 0.0
    if args.r2:
        # Sample evenly across the trajectory; download each from R2, load, verify.
        r2_manifest = os.path.join(wd, "r2_manifest.jsonl")
        r2_rows = [json.loads(l) for l in open(r2_manifest)] if os.path.exists(r2_manifest) else []
        key_by_base = {os.path.basename(r["key"]): r["key"] for r in r2_rows if "key" in r}
        n_bad_verified = sum(1 for r in r2_rows if not r.get("verified", False))
        if r2_rows and n_bad_verified:
            report["errors"].append(f"r2_manifest has {n_bad_verified} unverified row(s)")
        sample = rows if len(rows) <= args.r2_sample else [rows[i] for i in _even_idx(len(rows), args.r2_sample)]
        for mr in sample:
            base = os.path.basename(mr["path"])  # full/<id>.pt -> <id>.pt
            key = key_by_base.get(base)
            if key is None:
                report["errors"].append(f"{base}: no r2_manifest key (snapshot not uploaded?)")
                continue
            with tempfile.TemporaryDirectory() as td:
                dst = os.path.join(td, base)
                try:
                    _r2_download(key, dst)
                    sd = torch.load(dst, map_location="cpu")
                except Exception as e:  # noqa: BLE001
                    report["errors"].append(f"{base}: R2 download/load failed: {e}")
                    continue
                max_rel = max(max_rel, _verify_snapshot(torch, sd, mr, args.tol, base, report))
                report["checked"] += 1
    else:
        snaps = sorted(glob.glob(os.path.join(full_dir, "*.pt")))
        if not snaps:
            report["errors"].append(
                "no full/*.pt snapshots on disk. If r2_enabled, the local .pt were deleted after upload - "
                "rerun with --r2 to sample-verify from R2."
            )
            _emit(report, args)
            print("FAIL  no local snapshots (use --r2 if uploaded)")
            return 1
        man_by_base = {os.path.basename(r["path"]): r for r in rows if "path" in r}
        for p in snaps:
            base = os.path.basename(p)
            mr = man_by_base.get(base)
            if mr is None:
                report["errors"].append(f"{base}: on disk but absent from manifest")
                continue
            try:
                sd = torch.load(p, map_location="cpu")
            except Exception as e:  # noqa: BLE001
                report["errors"].append(f"{base}: torch.load failed: {e}")
                continue
            max_rel = max(max_rel, _verify_snapshot(torch, sd, mr, args.tol, base, report))
            report["checked"] += 1

    report["max_rel_norm_err"] = max_rel
    _emit(report, args)

    if report["checked"] >= 1 and not report["errors"]:
        print(f"PASS  mode={report['mode']} checked={report['checked']}/{len(rows)}  "
              f"max_rel_norm_err={max_rel:.4f} <= tol={args.tol}")
        return 0
    print(f"FAIL  {len(report['errors'])} error(s); checked={report['checked']}; max_rel_norm_err={max_rel:.4f}")
    for e in report["errors"][:20]:
        print("  -", e)
    return 1


def _even_idx(n: int, k: int) -> list:
    if k >= n:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]


def _emit(report: dict, args) -> None:
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
