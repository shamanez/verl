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

"""Tabulate a multi-arm capability audit from the per-cell verl train logs.

Reads ``$OOD_EVAL_ROOT/<tag>/<bench>/train.log`` for every (tag, bench) named in
the environment, prints a human table to stdout, and writes a machine-readable
TSV to ``$RESULTS_TSV`` for the plotter.

Called by ckpt_eval.sh, which owns the roster; every input is an env var so the
driver has exactly one place that decides what the roster is:

    OOD_EVAL_ROOT   eval output root
    TAGS_CSV        comma-separated model tags, "base" first
    BENCH_CSV       comma-separated benchmark names, in-domain first
    IN_DOMAIN_BENCH which of those is the in-domain set (a rule is drawn under it)
    RUN_ID          run name, for the header
    RESULTS_TSV     where to write the TSV

Scores come from the log, never from WandB: the rc=1 atexit teardown race has
silently dropped final-step values before, and the local log is authoritative.
"""

from __future__ import annotations

import os
import re
import sys

# Prefer the val-core key, which is what ood_eval.sh itself reports, so a stray
# acc/mean@ line elsewhere in the log cannot be picked up as the score. Fall back
# to the loose pattern only if the tight one finds nothing.
TIGHT = re.compile(r"val-core/\S*?acc/mean@[0-9]+['\"]?[: ]+([0-9.]+)")
LOOSE = re.compile(r"acc/mean@[0-9]+['\"]?[: ]+([0-9.]+)")


def read_score(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    hits = TIGHT.findall(text) or LOOSE.findall(text)
    return float(hits[-1]) if hits else None


def main() -> int:
    root = os.environ["OOD_EVAL_ROOT"]
    tags = [t for t in os.environ["TAGS_CSV"].split(",") if t]
    benches = [b for b in os.environ["BENCH_CSV"].split(",") if b]
    indomain = os.environ.get("IN_DOMAIN_BENCH", "")
    run_id = os.environ.get("RUN_ID", "run")
    results_tsv = os.environ.get("RESULTS_TSV", "")

    tbl = {(t, b): read_score(os.path.join(root, t, b, "train.log")) for t in tags for b in benches}

    trained = [t for t in tags if t != "base"]
    # The headline comparison: the two arms at the same step. Recovered from the
    # tag names rather than passed in, so a roster with more than two arms still
    # produces every pairwise "X - dense" column against the dense reference.
    dense_tags = [t for t in trained if t.startswith("dense")]
    other_tags = [t for t in trained if not t.startswith("dense")]
    pairs: list[tuple[str, str]] = []
    for d in dense_tags:
        step = d[len("dense") :]
        for o in other_tags:
            if o.endswith(step):
                pairs.append((o, d))

    w = 11
    # Wide enough for the longest benchmark name, so a long in-domain slug does
    # not push its own row's columns out of line with every other row.
    nw = max(len("benchmark"), max((len(b) for b in benches), default=0)) + 2
    head = f"{'benchmark':<{nw}s}" + "".join(f"{t:>{w}s}" for t in tags)
    head += "".join(f"{t + '-base':>{w + 4}s}" for t in trained)
    head += "".join(f"{a + '-' + b:>{w + 6}s}" for a, b in pairs)

    print(f"run {run_id}: in-domain + OOD capability audit")
    print("")
    print(head)
    print("-" * len(head))

    def cell(v: float | None) -> str:
        return f"{v:{w}.4f}" if v is not None else f"{'.':>{w}s}"

    def delta(a: float | None, z: float | None, width: int) -> str:
        """Signed delta, right-aligned, or 'n/a' when either side is unscored."""
        s = f"{a - z:+.4f}" if (a is not None and z is not None) else "n/a"
        return f"{s:>{width}s}"

    for b in benches:
        line = f"{b:<{nw}s}" + "".join(cell(tbl[(t, b)]) for t in tags)
        for t in trained:
            line += delta(tbl[(t, b)], tbl[("base", b)], w + 4)
        for x, y in pairs:
            line += delta(tbl[(x, b)], tbl[(y, b)], w + 6)
        print(line)
        if b == indomain:
            print("-" * len(head))

    # Headline summary, in the same terms the reference figure is read in.
    for x, y in pairs:
        deltas = [tbl[(x, b)] - tbl[(y, b)] for b in benches if tbl[(x, b)] is not None and tbl[(y, b)] is not None]
        if not deltas:
            continue
        n_up = sum(1 for d in deltas if d > 0)
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        print("")
        print(f"{x} against {y}, over {len(deltas)} scored benchmarks:")
        print(f"  mean |delta| = {mean_abs:.4f}   largest |delta| = {max(abs(d) for d in deltas):.4f}")
        print(f"  {x} higher on {n_up} of {len(deltas)}, {y} higher on {len(deltas) - n_up}")

    print("")
    print(f"in-domain ({indomain}) is the training validation split, scored with verl's")
    print("own validation sampling (n=1, temperature 0), so it is a cross-check of the")
    print("in-training val at the same step rather than a second opinion.")
    print("")
    print("A dot means no result on disk for that cell. Re-running the driver resumes it.")

    if results_tsv:
        with open(results_tsv, "w", encoding="utf-8") as fh:
            fh.write("bench\t" + "\t".join(tags) + "\n")
            for b in benches:
                vals = ["" if tbl[(t, b)] is None else f"{tbl[(t, b)]:.6f}" for t in tags]
                fh.write(b + "\t" + "\t".join(vals) + "\n")
        print("")
        print(f"TSV: {results_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
