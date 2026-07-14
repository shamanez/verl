"""Digest the ablation outputs into report-ready numbers.

Reads summary.json (per-combo, one per anchor) + forecast_rows.csv and prints:
  * per (method, W, horizon, rank) aggregates over anchor positions:
    pooled_skill, macro_skill, macro_cos, frac_win, evr, r2  (mean +/- std, n)
  * per-tensor-type skill at a chosen combo (H6)
  * a compact JSON blob (report_digest.json) for programmatic report filling.

Usage: python summarize_results.py --in_dir <outputs> [--out_dir <outputs>]
"""
# ruff: noqa: E501  (wide aligned table header line is intentional)

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None, 0)
    m = st.mean(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return (m, s, len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.in_dir

    summ = json.load(open(os.path.join(args.in_dir, "summary.json")))
    combos = summ["summaries"]
    rows = list(csv.DictReader(open(os.path.join(args.in_dir, "forecast_rows.csv"))))

    print(f"available_steps={summ['available_steps']}")
    print(f"n_rows={summ['n_rows']} n_combos={summ['n_combos']} skipped={summ['combos_skipped_missing_steps']}")

    # ---- aggregate over anchors -------------------------------------------- #
    # The runner iterates ranks 1..3 even where rank is capped (W=2, fixed_linear),
    # so the same (method,W,h,rank,anchor) can appear more than once with identical
    # numbers. Dedup by (key, anchor) so n_anchors and means are not inflated.
    buck = defaultdict(lambda: defaultdict(list))
    seen = set()
    for c in combos:
        k = (c["method"], int(c["W"]), int(c["horizon"]), int(c["rank"]))
        dk = (k, int(c["anchor"]))
        if dk in seen:
            continue
        seen.add(dk)
        ov = c["overall"]
        buck[k]["pooled_skill"].append(ov["pooled_skill"])
        buck[k]["macro_skill"].append(ov["macro_skill"])
        buck[k]["macro_cos"].append(ov["macro_direction_cos"])
        buck[k]["frac_win"].append(ov["frac_tensors_win"])
        buck[k]["evr"].append(ov["evr_mean"])
        buck[k]["r2"].append(ov["r2_mean"])
        buck[k]["n_anchors"].append(c["anchor"])

    digest = {}
    hdr = f"{'method':16s} {'W':>2} {'h':>2} {'r':>2} | {'pooled':>16} {'macro':>16} {'cos':>16} {'win':>14} {'evr':>10} {'r2':>10}  n"
    print("\n== per (method,W,horizon,rank) over anchors ==")
    print(hdr)
    for k in sorted(buck):
        b = buck[k]
        ps, pss, n = _ms(b["pooled_skill"])
        ma, mas, _ = _ms(b["macro_skill"])
        co, cos, _ = _ms(b["macro_cos"])
        fw, fws, _ = _ms(b["frac_win"])
        ev, _, _ = _ms(b["evr"])
        r2, _, _ = _ms(b["r2"])
        method, W, h, r = k

        def f(m, s):
            return f"{m:+.3f}+-{s:.3f}" if m is not None else "   n/a   "

        print(
            f"{method:16s} {W:>2} {h:>2} {r:>2} | {f(ps, pss):>16} {f(ma, mas):>16} "
            f"{f(co, cos):>16} {f(fw, fws):>14} "
            f"{(f'{ev:.3f}' if ev is not None else 'n/a'):>10} "
            f"{(f'{r2:.3f}' if r2 is not None else 'n/a'):>10}  {n}"
        )
        digest[f"{method}|W{W}|h{h}|r{r}"] = dict(
            method=method,
            W=W,
            horizon=h,
            rank=r,
            pooled_skill=ps,
            pooled_skill_std=pss,
            macro_skill=ma,
            macro_skill_std=mas,
            macro_cos=co,
            macro_cos_std=cos,
            frac_win=fw,
            frac_win_std=fws,
            evr=ev,
            r2=r2,
            n_anchors=n,
        )

    # ---- per-type skill at chosen combos (H6) ------------------------------ #
    def per_type(method, W, h, r):
        sel = [
            x
            for x in rows
            if x["method"] == method and int(x["W"]) == W and int(x["horizon"]) == h and int(x["rank"]) == r
        ]
        by = defaultdict(list)
        seen_rt = set()  # dedup duplicate (anchor,tensor) from capped-rank reruns
        for x in sel:
            dk = (int(x["anchor"]), x["tensor"])
            if dk in seen_rt:
                continue
            seen_rt.add(dk)
            by[x["ttype"]].append(float(x["skill"]))
        return {t: _ms(v) for t, v in by.items()}

    type_digest = {}
    print("\n== per-tensor-type skill (rank1_relex, W=2, h=1, r=1) ==")
    for t, (m, s, n) in sorted(per_type("rank1_relex", 2, 1, 1).items(), key=lambda kv: -(kv[1][0] or -9)):
        print(f"  {t:12s} skill={m:+.3f} +- {s:.3f}  (n_rows={n})")
        type_digest.setdefault("rank1_relex|W2|h1|r1", {})[t] = dict(skill=m, std=s, n=n)

    # also W=4 for the H6 mixed-policy discussion
    for combo in [("rank1_relex", 4, 1, 1)]:
        method, W, h, r = combo
        key = f"{method}|W{W}|h{h}|r{r}"
        for t, (m, s, n) in per_type(method, W, h, r).items():
            type_digest.setdefault(key, {})[t] = dict(skill=m, std=s, n=n)

    out = os.path.join(out_dir, "report_digest.json")
    json.dump(
        {
            "scalars": digest,
            "by_type": type_digest,
            "meta": {k: summ[k] for k in ("available_steps", "n_rows", "n_combos", "combos_skipped_missing_steps")},
        },
        open(out, "w"),
        indent=2,
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
