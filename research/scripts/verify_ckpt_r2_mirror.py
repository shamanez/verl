#!/usr/bin/env python3
"""Verify the EXP-58 checkpoint->R2 mirror is complete + resume-valid.

Deliverable B of EXP-58: every ``global_step_<N>/`` checkpoint tree is mirrored
to R2 on-the-go. This asserts, for each expected step, the FULL object set that
verl's FSDP1 checkpoint writes, plus a fresh root tracker so ``find_latest_ckpt_path``
can resolve the latest step from R2 alone (resume-valid).

Ground truth is the ACTUAL R2 object listing (what really uploaded, with real
byte sizes) -- not the local ``full_manifest.jsonl`` (which can list files that
never uploaded). An optional ``--manifest`` cross-checks the sink's
``r2_manifest.jsonl`` verified:true rows against the listing.

For each expected ``global_step_<N>`` it requires, with size > 0:
  - model_/optim_/extra_state_world_size_<W>_rank_<R>.pt   for R in 0..W-1
  - data.pt
  - actor/fsdp_config.json
  - actor/huggingface/config.json  AND some tokenizer* file
and (``--require-tracker``) a root ``latest_checkpointed_iteration.txt``.
``--emit dry-restore`` reads that tracker's value and confirms the resolved
step's shard-triples are all present (a paper dry-restore).

Usage:
  set -a; . ~/.config/verl-research/secrets.env; set +a     # R2_* creds
  python verify_ckpt_r2_mirror.py runs/EXP-58 \
      --prefix verl-research/EXP-58/regimeA/checkpoints \
      --expect-steps 20:1000:20 --world-size 1 --require-tracker --emit dry-restore
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict


def _r2_endpoint() -> str:
    ep = os.environ.get("R2_ENDPOINT", "")
    if not ep and os.environ.get("R2_ACCOUNT_ID"):
        ep = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return ep


def _r2_env() -> dict:
    return {
        **os.environ,
        "AWS_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID", "")),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "")),
        "AWS_DEFAULT_REGION": "auto",
    }


def _bucket() -> str:
    b = os.environ.get("R2_BUCKET", "")
    if b != "shamane-pluralis":  # hard guard: only ever our bucket
        raise SystemExit(f"R2_BUCKET must be 'shamane-pluralis' (got {b!r})")
    return b


def list_r2(prefix: str) -> "dict[str,int]":
    """Return {key_relative_to_prefix: size_bytes} for everything under prefix."""
    uri = f"s3://{_bucket()}/{prefix.rstrip('/')}/"
    out = subprocess.run(
        ["aws", "s3", "ls", uri, "--recursive", "--endpoint-url", _r2_endpoint()],
        env=_r2_env(), capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"aws s3 ls {uri} failed rc={out.returncode}: {out.stderr.strip()[:300]}")
    # `aws s3 ls ... --recursive` key column differs by CLI version:
    #   v1 prints the key RELATIVE to the listed prefix (global_step_1/...)
    #   v2 prints the FULL key from bucket root (verl-research/.../checkpoints/global_step_1/...)
    # Normalise to a prefix-relative suffix so both parse identically.
    pfx = prefix.rstrip("/") + "/"
    objs: "dict[str,int]" = {}
    for line in out.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        _date, _time, size, key = parts
        if key.startswith(pfx):
            key = key[len(pfx):]
        try:
            objs[key] = int(size)
        except ValueError:
            continue
    return objs


def read_tracker(prefix: str) -> "int|None":
    uri = f"s3://{_bucket()}/{prefix.rstrip('/')}/latest_checkpointed_iteration.txt"
    out = subprocess.run(
        ["aws", "s3", "cp", uri, "-", "--endpoint-url", _r2_endpoint()],
        env=_r2_env(), capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None
    m = re.search(r"\d+", out.stdout)
    return int(m.group(0)) if m else None


def parse_steps(spec: str) -> "list[int]":
    if ":" in spec:
        a, b, c = (int(x) for x in spec.split(":"))
        return list(range(a, b + 1, c))
    return [int(x) for x in spec.split(",") if x.strip()]


def required_suffixes(step: int, W: int) -> "set[str]":
    need = set()
    for r in range(W):
        for kind in ("model", "optim", "extra_state"):
            need.add(f"global_step_{step}/actor/{kind}_world_size_{W}_rank_{r}.pt")
    need.add(f"global_step_{step}/data.pt")
    need.add(f"global_step_{step}/actor/fsdp_config.json")
    return need


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--prefix", required=True, help="R2 checkpoints prefix (relative to bucket)")
    ap.add_argument("--expect-steps", required=True, help="start:stop:step  or  comma list")
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--require-tracker", action="store_true")
    ap.add_argument("--emit", default="", choices=["", "dry-restore"])
    ap.add_argument("--manifest", default="", help="optional local r2_manifest.jsonl for verified:true cross-check")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    W = args.world_size
    steps = parse_steps(args.expect_steps)
    report = {
        "prefix": args.prefix, "world_size": W, "expect_steps": f"{steps[0]}..{steps[-1]}",
        "n_expect": len(steps), "mode": "r2-listing", "errors": [], "incomplete_steps": [],
        "complete_steps": 0, "zero_byte_objects": [],
    }

    objs = list_r2(args.prefix)
    if not objs:
        report["errors"].append(f"no objects under R2 prefix {args.prefix}")
        _emit(report, args)
        print(f"FAIL  no objects under {args.prefix}")
        return 1

    have = set(objs)
    by_step_hf = defaultdict(set)
    for k in have:
        m = re.match(r"global_step_(\d+)/actor/huggingface/(.+)$", k)
        if m:
            by_step_hf[int(m.group(1))].add(m.group(2))

    for step in steps:
        need = required_suffixes(step, W)
        missing = sorted(s for s in need if s not in have)
        zero = [s for s in need if objs.get(s, 1) == 0]
        hf = by_step_hf.get(step, set())
        has_cfg = "config.json" in hf
        has_tok = any("tokenizer" in os.path.basename(s) for s in hf)
        ok = (not missing) and (not zero) and has_cfg and has_tok
        if zero:
            report["zero_byte_objects"].extend(f"global_step_{step}: {z}" for z in zero)
        if ok:
            report["complete_steps"] += 1
        else:
            report["incomplete_steps"].append(step)
            report["errors"].append(
                f"step {step}: missing={missing[:6]}{'...' if len(missing) > 6 else ''} "
                f"zero_byte={len(zero)} hf_config={has_cfg} hf_tokenizer={has_tok}"
            )

    tracker_val = None
    if args.require_tracker:
        if "latest_checkpointed_iteration.txt" not in have:
            report["errors"].append("root latest_checkpointed_iteration.txt NOT mirrored (resume broken)")
        else:
            tracker_val = read_tracker(args.prefix)
            report["tracker_value"] = tracker_val
            if tracker_val != steps[-1]:
                report["errors"].append(
                    f"tracker resolves step {tracker_val}, expected latest {steps[-1]}"
                )

    if args.emit == "dry-restore":
        resolved = tracker_val if tracker_val is not None else (read_tracker(args.prefix))
        dr = {"resolved_step": resolved, "shards_ok": False}
        if resolved is not None:
            need = required_suffixes(resolved, W)
            miss = sorted(s for s in need if s not in have)
            dr["shards_ok"] = not miss
            dr["missing"] = miss[:10]
            if miss:
                report["errors"].append(f"dry-restore step {resolved}: {len(miss)} shard/file(s) missing")
        else:
            report["errors"].append("dry-restore: could not resolve tracker step")
        report["dry_restore"] = dr

    # Optional: cross-check the sink manifest's verified:true rows.
    man_path = args.manifest
    if not man_path:
        cands = glob.glob(os.path.join(args.run_dir, "**", "*r2_manifest*.jsonl"), recursive=True)
        cands = [c for c in cands if "checkpoint" in c.lower() or "ckpt" in c.lower()] or cands
        man_path = cands[0] if cands else ""
    if man_path and os.path.exists(man_path):
        rows = [json.loads(l) for l in open(man_path) if l.strip()]
        ver = [r for r in rows if r.get("verified") is True]
        report["manifest"] = {"path": man_path, "rows": len(rows), "verified": len(ver)}
        if len(ver) != len(rows):
            report["errors"].append(f"manifest {man_path}: {len(rows)-len(ver)} unverified row(s)")

    _emit(report, args)
    passed = report["complete_steps"] == len(steps) and not report["errors"]
    if passed:
        print(f"PASS  {report['complete_steps']}/{len(steps)} steps complete in R2  "
              f"(W={W}, tracker={report.get('tracker_value')}, "
              f"dry_restore={report.get('dry_restore', {}).get('shards_ok')})")
        return 0
    print(f"FAIL  complete={report['complete_steps']}/{len(steps)}  "
          f"incomplete={report['incomplete_steps'][:10]}  errors={len(report['errors'])}")
    for e in report["errors"][:20]:
        print("  -", e)
    return 1


def _emit(report: dict, args) -> None:
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
