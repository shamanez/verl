#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Capability figure: dense against 95 percent compressed, in-domain and OOD.

Reads the TSV written by tabulate_arms.py and draws the paired bar chart the
project uses for this comparison: one group per benchmark, dense in blue and
the compressed arm in green, the in-domain benchmark boxed off from the OOD
suite by a dashed divider, legend along the bottom.

    python3 plot_dense_vs_compressed.py --results RESULTS_<run>.tsv
    python3 plot_dense_vs_compressed.py --results r.tsv --dense dense600 --compressed prf600

The two arm columns are auto-detected from the TSV header when not named: the
column starting with "dense" is the control and the other trained column is the
method. The "base" column is read but not drawn, and is reported in the caption
line printed to stdout so the figure and the text stay consistent.
"""

from __future__ import annotations

import argparse
import csv
import sys

# Benchmarks drawn left to right within the OOD block. Anything present in the
# TSV but missing here is appended in file order, so a new benchmark still
# appears rather than being silently dropped.
OOD_ORDER = [
    "math500",
    "gsm8k",
    "minerva",
    "olympiad",
    "amc23",
    "mmlu_stem",
    "aime24",
    "aime25",
    "aime26",
    "hmmt25",
]

DENSE_COLOR = "#3277c9"
COMPRESSED_COLOR = "#2e9e6b"


def load(path: str) -> tuple[list[str], dict[str, dict[str, float | None]]]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if not rows:
        sys.exit(f"{path} is empty")
    header = rows[0]
    if header[0] != "bench":
        sys.exit(f"{path} does not look like a tabulate_arms.py TSV (first column is {header[0]!r})")
    tags = header[1:]
    table: dict[str, dict[str, float | None]] = {}
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        table[r[0]] = {t: (float(v) if v else None) for t, v in zip(tags, r[1:], strict=False)}
    return tags, table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="RESULTS_<run>.tsv from tabulate_arms.py")
    ap.add_argument("--dense", default=None, help="dense column name (default: the one starting with 'dense')")
    ap.add_argument("--compressed", default=None, help="compressed column name (default: the other trained column)")
    ap.add_argument("--in-domain", default="deepscaler_indomain", help="which benchmark is the in-domain set")
    ap.add_argument("--out", default=None, help="output image (default: <results>.png)")
    ap.add_argument("--title", default=None, help="figure title")
    args = ap.parse_args()

    tags, table = load(args.results)

    dense = args.dense or next((t for t in tags if t.startswith("dense")), None)
    if dense is None:
        sys.exit(f"no dense column in {tags}; pass --dense")
    compressed = args.compressed or next((t for t in tags if t != "base" and t != dense), None)
    if compressed is None:
        sys.exit(f"no compressed column in {tags}; pass --compressed")

    indomain = args.in_domain if args.in_domain in table else None
    ood = [b for b in OOD_ORDER if b in table]
    ood += [b for b in table if b not in ood and b != indomain]
    benches = ([indomain] if indomain else []) + ood

    drawn = [b for b in benches if table[b].get(dense) is not None and table[b].get(compressed) is not None]
    if not drawn:
        sys.exit("no benchmark has BOTH arms scored; nothing to plot (re-run the eval driver to fill the gaps)")
    skipped = [b for b in benches if b not in drawn]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    n = len(drawn)
    width = 0.38
    xs = list(range(n))
    dvals = [table[b][dense] for b in drawn]
    cvals = [table[b][compressed] for b in drawn]

    fig, ax = plt.subplots(figsize=(max(8.0, 1.15 * n + 2.0), 4.8))
    ax.bar([x - width / 2 for x in xs], dvals, width, label="dense", color=DENSE_COLOR)
    ax.bar([x + width / 2 for x in xs], cvals, width, label="95 percent compressed", color=COMPRESSED_COLOR)

    for x, v in zip(xs, dvals, strict=False):
        ax.text(x - width / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#333333")
    for x, v in zip(xs, cvals, strict=False):
        ax.text(x + width / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#333333")

    # The in-domain block is boxed off from the OOD suite by a dashed divider.
    if indomain in drawn:
        ax.axvline(x=drawn.index(indomain) + 0.5, color="#888888", linestyle="--", linewidth=1.1)
        ax.text(0.0, 1.02, "in-domain", transform=ax.get_xaxis_transform(), fontsize=8, color="#555555")
        ax.text(
            drawn.index(indomain) + 0.7,
            1.02,
            "out of domain",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color="#555555",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(drawn, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, min(1.0, max(max(dvals), max(cvals)) * 1.22 + 0.05))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    if args.title:
        ax.set_title(args.title, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=9)

    fig.tight_layout()
    out = args.out or (args.results.rsplit(".", 1)[0] + ".png")
    fig.savefig(out, dpi=200, bbox_inches="tight")

    deltas = [c - d for c, d in zip(cvals, dvals, strict=False)]
    n_up = sum(1 for x in deltas if x > 0)
    print(f"wrote {out}")
    print(f"  arms      dense={dense}  compressed={compressed}")
    print(f"  drawn     {len(drawn)} benchmarks: {', '.join(drawn)}")
    if skipped:
        print(f"  SKIPPED   {len(skipped)} without both arms scored: {', '.join(skipped)}")
    print(
        f"  headline  every gap within {max(abs(x) for x in deltas):+.4f}, "
        f"mean |delta| = {sum(abs(x) for x in deltas) / len(deltas):.4f}"
    )
    print(f"            compressed higher on {n_up} of {len(deltas)}, dense on {len(deltas) - n_up}")
    base_vals = {b: table[b].get("base") for b in drawn}
    if any(v is not None for v in base_vals.values()):
        print("  base      " + "  ".join(f"{b}={v:.3f}" for b, v in base_vals.items() if v is not None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
