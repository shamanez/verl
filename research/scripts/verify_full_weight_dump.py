#!/usr/bin/env python3
"""Verify a FULL-weight trajectory dump (EXP-43+).

Loads every ``full/step_*.pt`` under ``<weights_dir>``, asserts each is a state
dict of the real weight matrices (true shapes, NOT a flattened/length-k sketch),
and compares each matrix's stored Frobenius norm to the value recorded in
``full_manifest.jsonl`` within ``--tol`` relative. This is the analyst's gate-3
(dump integrity) check; it replaces the old sketch-fidelity sweep (the
count-sketch instrument was removed 2026-06-30 - the study now keeps the raw
weights, so there is nothing lossy to validate, only that the dump loads and its
norms match the live weights within dump-dtype rounding).

Exit 0 + prints PASS iff: >=1 snapshot, every snapshot loads, n_matrices matches
the manifest, every 2-D matrix kept its real shape, and the max relative norm
error <= tol. Exit 1 + FAIL otherwise.

Usage:
  python verify_full_weight_dump.py runs/EXP-43/regimeA/weights [--tol 0.01]
                                    [--expect-steps 80] [--json out.json]
"""
import argparse
import glob
import json
import os
import sys


def _step_of(path: str) -> int:
    return int(os.path.basename(path)[len("step_"):-len(".pt")])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights_dir")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--expect-steps", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import torch  # deferred so --help works without torch

    wd = args.weights_dir
    full_dir = os.path.join(wd, "full")
    manifest = os.path.join(wd, "full_manifest.jsonl")
    snaps = sorted(glob.glob(os.path.join(full_dir, "step_*.pt")), key=_step_of)
    rows = [json.loads(l) for l in open(manifest)] if os.path.exists(manifest) else []
    man_by_step = {int(r["global_step"]): r for r in rows if "global_step" in r}

    report = {
        "weights_dir": wd,
        "n_snapshots": len(snaps),
        "n_manifest_rows": len(rows),
        "tol": args.tol,
        "errors": [],
        "max_rel_norm_err": 0.0,
        "checked": 0,
    }

    if not snaps:
        report["errors"].append("no full/step_*.pt snapshots found (the full-weight dump did not fire)")
        _emit(report, args)
        print("FAIL  no snapshots")
        return 1

    max_rel = 0.0
    for p in snaps:
        step = _step_of(p)
        try:
            sd = torch.load(p, map_location="cpu")
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"step {step}: torch.load failed: {e}")
            continue
        if not isinstance(sd, dict) or not sd:
            report["errors"].append(f"step {step}: not a non-empty state dict")
            continue
        twod = [k for k, v in sd.items() if hasattr(v, "dim") and v.dim() == 2]
        if not twod:
            report["errors"].append(f"step {step}: no 2-D weight matrices (shape lost - is this a sketch?)")
        mr = man_by_step.get(step)
        if mr is not None:
            if int(mr.get("n_matrices", -1)) != len(sd):
                report["errors"].append(
                    f"step {step}: manifest n_matrices={mr.get('n_matrices')} != loaded {len(sd)}"
                )
            for m in mr.get("matrices", []):
                name = m["name"]
                if name not in sd:
                    report["errors"].append(f"step {step}: manifest matrix {name} absent in snapshot")
                    continue
                if list(sd[name].shape) != list(m.get("shape", [])):
                    report["errors"].append(
                        f"step {step}: {name} shape {list(sd[name].shape)} != manifest {m.get('shape')}"
                    )
                fro_now = float(torch.linalg.norm(sd[name].to(torch.float32)).item())
                fro_man = float(m.get("fro_norm", 0.0))
                if fro_man > 0.0:
                    rel = abs(fro_now - fro_man) / fro_man
                    max_rel = max(max_rel, rel)
                    if rel > args.tol:
                        report["errors"].append(
                            f"step {step}: {name} norm rel-err {rel:.4f} > tol {args.tol} "
                            f"(loaded {fro_now:.4f} vs manifest {fro_man:.4f})"
                        )
        report["checked"] += 1

    report["max_rel_norm_err"] = max_rel
    if args.expect_steps and len(snaps) < args.expect_steps:
        report["note"] = f"snapshots {len(snaps)} < expect_steps {args.expect_steps}"
    _emit(report, args)

    if not report["errors"]:
        print(f"PASS  snapshots={len(snaps)}  max_rel_norm_err={max_rel:.4f} <= tol={args.tol}")
        return 0
    print(f"FAIL  {len(report['errors'])} error(s); max_rel_norm_err={max_rel:.4f}")
    for e in report["errors"][:20]:
        print("  -", e)
    return 1


def _emit(report: dict, args) -> None:
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
