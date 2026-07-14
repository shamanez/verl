"""Tensor-major forecast-skill ablation - I/O-efficient rewrite of run_forecast_ablation.

Identical math and identical outputs (forecast_rows.csv, summary.json) as the
combo-major runner, but it reads each tensor's step history ONCE and evaluates every
combo for that tensor in memory. That turns total disk I/O from O(combos x model)
into O(model) - a single pass over the checkpoint set - which is the difference
between hours and a minute on a laptop whose page cache cannot hold 36 GB.

Combo enumeration, skip logic, metrics, and aggregation are imported from
run_forecast_ablation so the two runners cannot drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

import harness_projector as hp
import torch
from run_forecast_ablation import DECODER_2D, CkptStore, aggregate, tensor_type, whole_tensor_metrics


def build_combos(windows, horizons, ranks, strengths, methods, gap, anchors, avail_set):
    combos = []
    skipped = 0
    for method in methods:
        for W in windows:
            for h in horizons:
                for rank in ranks:
                    r_eff = 1 if method == "fixed_linear" else min(rank, W - 1)
                    for strength in strengths:
                        for anchor in anchors:
                            src = [anchor - (W - 1 - i) * gap for i in range(W)]
                            target = anchor + h * gap
                            if not (set(src) | {target}) <= avail_set:
                                skipped += 1
                                continue
                            combos.append(
                                dict(
                                    method=method,
                                    W=W,
                                    horizon=h,
                                    gap=gap,
                                    rank=r_eff,
                                    strength=strength,
                                    anchor=anchor,
                                    target=target,
                                    src=src,
                                )
                            )
    # de-duplicate combos that collapse to the same effective params (e.g. capped rank)
    seen, uniq = set(), []
    for c in combos:
        k = (c["method"], c["W"], c["horizon"], c["gap"], c["rank"], c["strength"], c["anchor"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return uniq, skipped


def eval_tensor(name, ttype, tens, combos, skip_embedding, numel_full=None, pool_factor=1.0):
    """Return list of (combo_index, row) for this tensor across all combos.

    ``tens`` may hold full or coordinate-subsampled 1-D vectors (all steps share the
    same coordinate set). ``pool_factor`` = numel_full / n_sampled rescales proj/stale
    SSE so cross-tensor energy pooling still reflects the true tensor size; per-tensor
    skill and cosine are ratios and are sampling-invariant.
    """
    out = []
    latest_is_float = None
    n_sampled = next(iter(tens.values())).numel()
    if numel_full is None:
        numel_full = n_sampled
    for ci, c in enumerate(combos):
        if skip_embedding and ttype == "embed":
            continue
        snaps = [tens[s] for s in c["src"]]
        latest = snaps[-1]
        if latest_is_float is None:
            latest_is_float = torch.is_floating_point(latest)
        actual = tens[c["target"]]
        method = c["method"]
        if method == "rank1_relex":
            if not latest_is_float:
                continue
            proj, st = hp.project_rank1_tensor(
                snaps, list(c["src"]), c["target"], strength=c["strength"], rank=c["rank"]
            )
            evr, r2, fit_kind = st["evr"], st["r2"], st["fit_kind"]
        elif method == "relex_from_base":
            proj, _ = hp.relex_from_base_project(
                snaps, list(c["src"]), c["target"], rank=c["rank"], strength=c["strength"]
            )
            evr, r2, fit_kind = float("nan"), float("nan"), f"from_base_r{c['rank']}"
        elif method == "fixed_linear":
            # decoder-matrix types only (matches harness scope); works elementwise so
            # it is valid on the flattened / subsampled vector too.
            if ttype in DECODER_2D and c["W"] >= 2:
                proj = hp.fixed_linear_project(
                    snaps[-1], snaps[-2], h=c["horizon"] * c["gap"], g=c["gap"], strength=c["strength"]
                )
            else:
                proj = hp.stale_baseline(snaps)
            evr, r2, fit_kind = float("nan"), float("nan"), "fixed_linear"
        else:
            raise ValueError(method)
        m = whole_tensor_metrics(proj, latest, actual)
        # energy-pooling weight: scale SSE to represent the full tensor
        m["proj_sse"] = m["proj_sse"] * pool_factor
        m["stale_sse"] = m["stale_sse"] * pool_factor
        row = dict(
            anchor=c["anchor"],
            W=c["W"],
            horizon=c["horizon"],
            gap=c["gap"],
            rank=c["rank"],
            strength=c["strength"],
            method=method,
            target=c["target"],
            history=";".join(map(str, c["src"])),
            tensor=name,
            ttype=ttype,
            numel=numel_full,
            n_sampled=n_sampled,
            evr=evr,
            r2=r2,
            fit_kind=fit_kind,
            **m,
        )
        out.append((ci, row))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--windows", default="2,3,4,5,6,8")
    ap.add_argument("--horizons", default="1,2,3")
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--anchors", default="")
    ap.add_argument("--ranks", default="1,2,3")
    ap.add_argument("--strengths", default="1.0")
    ap.add_argument("--methods", default="rank1_relex,relex_from_base,fixed_linear")
    ap.add_argument("--skip_embedding", action="store_true")
    ap.add_argument(
        "--sample_coords",
        type=int,
        default=250000,
        help="For tensors larger than this, evaluate on a fixed deterministic "
        "coordinate sample of this size (a large-sample estimate of the whole-tensor "
        "metric; the live probe used only 16). 0 = always full tensor.",
    )
    ap.add_argument(
        "--only_steps",
        default="",
        help="Comma list restricting the usable steps (e.g. 10,20,30,40,50,60,70 to "
        "drop the frozen/anomalous tail). Default = every step on disk.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    store = CkptStore(args.ckpt_dir)
    windows = [int(x) for x in args.windows.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]
    strengths = [float(x) for x in args.strengths.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    avail = sorted(
        int(m.group(1)) for m in (re.match(r"global_step_(\d+)$", d) for d in os.listdir(args.ckpt_dir)) if m
    )
    if args.only_steps:
        keep = {int(x) for x in args.only_steps.split(",")}
        avail = [s for s in avail if s in keep]
    avail_set = set(avail)
    anchors = [int(x) for x in args.anchors.split(",")] if args.anchors else list(avail)
    print(f"Available steps: {avail}")

    names = None
    for s in avail:
        names = store.names(s) if names is None else (names & store.names(s))
    if args.skip_embedding:
        # Exclude the tied embedding entirely so it is never even loaded (10 x 933 MB
        # fp32 would spike into swap on a 26 GB laptop). Skipping only the compute is
        # not enough - the load itself is the problem.
        names = {n for n in names if tensor_type(n) != "embed"}
    names = sorted(names)
    print(f"Tracking {len(names)} tensors{' (embedding excluded)' if args.skip_embedding else ''}.")

    combos, skipped = build_combos(windows, horizons, ranks, strengths, methods, args.gap, anchors, avail_set)
    print(f"{len(combos)} unique combos ({skipped} enumerations skipped for missing steps).")
    steps_needed = sorted({s for c in combos for s in (list(c["src"]) + [c["target"]])})
    print(f"Steps referenced: {steps_needed}")

    combo_rows = {i: [] for i in range(len(combos))}
    n_sampled_tensors = 0
    for ti, name in enumerate(names):
        ttype = tensor_type(name)
        tens = {s: store.get(s, name) for s in steps_needed}
        numel_full = next(iter(tens.values())).numel()
        pool_factor = 1.0
        if args.sample_coords and numel_full > args.sample_coords:
            # Deterministic coordinate sample (same coords for every step of this tensor).
            # Seeded per tensor index so it is reproducible run-to-run.
            g = torch.Generator().manual_seed(1234 + ti)
            idx = torch.randint(0, numel_full, (args.sample_coords,), generator=g)
            tens = {s: t.reshape(-1)[idx] for s, t in tens.items()}
            pool_factor = numel_full / args.sample_coords
            n_sampled_tensors += 1
        for ci, row in eval_tensor(
            name, ttype, tens, combos, args.skip_embedding, numel_full=numel_full, pool_factor=pool_factor
        ):
            combo_rows[ci].append(row)
        del tens
        if (ti + 1) % 40 == 0:
            print(f"  {ti + 1}/{len(names)} tensors processed", flush=True)
    print(f"{n_sampled_tensors} tensors were coordinate-sampled to {args.sample_coords}; the rest evaluated in full.")

    all_rows, summaries = [], []
    for ci in range(len(combos)):
        rows = combo_rows[ci]
        if not rows:
            continue
        all_rows.extend(rows)
        summaries.append(aggregate(rows))
    for s in summaries:
        ov = s["overall"]
        print(
            f"{s['method']:16s} W={s['W']} h={s['horizon']} r={s['rank']} anchor={s['anchor']} "
            f"-> pooled={ov['pooled_skill']:+.3f} macro={ov['macro_skill']:+.3f} "
            f"cos={ov['macro_direction_cos']:+.3f} win={ov['frac_tensors_win']:.2f}"
        )

    csv_path = os.path.join(args.out_dir, "forecast_rows.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "grid": vars(args),
                "available_steps": avail,
                "n_rows": len(all_rows),
                "n_combos": len(summaries),
                "combos_skipped_missing_steps": skipped,
                "summaries": summaries,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {len(all_rows)} rows -> {csv_path}")
    print(f"Wrote {len(summaries)} combo summaries.")


if __name__ == "__main__":
    main()
