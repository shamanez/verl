#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Turn finished COMPASS ablation arms into the paper's table rows.

Reads each arm's ``train.log`` (the on-box log is authoritative: WandB drops
the final step to an atexit race) and emits, for each arm, the MATH validation
trajectory, the throughput decomposition and the per-arm falsifiers that prove
the arm ran the configuration it claims.

The falsifiers matter more than the scores. A no-anchor arm that quietly kept
its anchor, or a K=80 arm that quietly ran K=20, would produce a perfectly
plausible number.

Usage:
    # pull the logs first
    scp -i ~/.ssh/vast_ai -P 20927 -r \\
        root@<box>:/workspace/runs/compass-\\* research/runs/compass-ablations/

    python3 research/scripts/collect_compass_ablations.py \\
        research/runs/compass-ablations --latex
"""

import argparse
import os
import re
import sys

VAL_KEY = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1"
STEP_RE = re.compile(r"step:(\d+)\s")
BASE_MATH = 0.447  # the paper's base reading for the ablation table

# Arms, in the order the tables print them, with the delta each one applies.
NOANCHOR_ARMS = [
    ("base", "Compressed, full anchor ($\\alpha=0.25$)", "yes", "yes"),
    ("nosign", "Compressed, no sign ($\\alpha=1$)", "yes", "yes"),
    ("noanchor", "Compressed, anchor removed", "no", "no"),
    ("dense", "Dense control", "no", "n/a"),
]
K_ARMS = [
    ("k10", 10, 1.0),
    ("base", 20, 1.0),
    ("k40", 40, 1.0),
    ("k40gm", 40, 0.5),
    ("k80", 80, 1.0),
    ("k80gm", 80, 0.25),
]
READOUTS = (40, 80, 120, 160, 200, 400, 600)


def num(tok):
    try:
        return float(tok)
    except ValueError:
        return None


def parse(path):
    """Return per-step metrics plus the counters used as falsifiers."""
    steps, val_at, fires = {}, {}, []
    key_re = re.compile(r"([A-Za-z0-9_/@\-.]+):(-?[0-9.]+(?:[eE][-+]?\d+)?)")
    stale_re = re.compile(r"delay_K=(\d+)")
    with open(path, errors="replace") as fh:
        for line in fh:
            m = stale_re.search(line)
            if m and "stale-replay" in line:
                fires.append(int(m.group(1)))
            sm = STEP_RE.search(line)
            if not sm:
                continue
            step = int(sm.group(1))
            row = steps.setdefault(step, {})
            for k, v in key_re.findall(line):
                f = num(v)
                if f is not None:
                    row[k] = f
            if VAL_KEY in row:
                val_at[step] = row[VAL_KEY]
    return steps, val_at, fires


def summarise(arm, root):
    path = os.path.join(root, f"compass-{arm}", "train.log")
    if not os.path.isfile(path):
        return None
    steps, val_at, fires = parse(path)
    if not steps:
        return None
    last = max(steps)
    body = [s for s in sorted(steps) if s >= 1]

    def total(key):
        return sum(steps[s][key] for s in body if key in steps[s])

    def cum(key):
        vals = [steps[s][key] for s in sorted(steps) if key in steps[s]]
        return vals[-1] if vals else 0.0

    # Fire-step peak vs non-fire peak. max_memory_allocated is a high-water
    # mark that torch never resets, so take the FIRST value at or after the
    # first anchor firing as the fire-step peak, and the last value strictly
    # before it as the pre-anchor peak. A median over the series would just
    # report when the watermark jumped.
    mem_key = "actor/perf/max_memory_allocated_gb"
    ab_key = "actor/comm_eff/anchor_backwards"
    pre_peak = fire_peak = None
    prev_ab = None
    for st in sorted(steps):
        row = steps[st]
        if mem_key not in row:
            continue
        ab = row.get(ab_key)
        fired = prev_ab is not None and ab is not None and ab > prev_ab
        if ab is not None:
            prev_ab = ab
        if fired and fire_peak is None:
            fire_peak = row[mem_key]
        if fire_peak is None:
            pre_peak = row[mem_key]

    wall = total("timing_s/step")
    toks = total("perf/total_num_tokens")
    n = sum(1 for s in body if "timing_s/step" in steps[s])
    return {
        "arm": arm,
        "last_step": last,
        "val": val_at,
        "n_timed": n,
        "step_s": wall / n if n else float("nan"),
        "tps": toks / wall if wall else float("nan"),
        # falsifiers
        "delay_K": sorted(set(fires)),
        "anchor_backwards": cum("actor/comm_eff/anchor_backwards"),
        "spectral_corrections": cum("actor/comm_eff/spectral_corrections"),
        "mask_train": cum("actor/comm_eff/mask_applications/train"),
        "pre_anchor_peak_gb": pre_peak,
        "fire_peak_gb": fire_peak,
        "rank1_fires": cum("actor/comm_eff/rank1_fires"),
        "horizon": cum("actor/comm_eff/rank1_prediction_horizon"),
    }


def check(r, expect_anchor, expect_codec, expect_K):
    """Return the reasons this arm is NOT what it claims to be."""
    bad = []
    if expect_codec and r["mask_train"] <= 0:
        bad.append("codec never fired (mask_applications/train == 0)")
    if not expect_codec and r["mask_train"] > 0:
        bad.append("codec fired on an arm that should be dense")
    if expect_anchor:
        if r["anchor_backwards"] <= 0 and r["last_step"] >= 20:
            bad.append("anchor never fired past step 20")
        if expect_K is not None and r["delay_K"] and r["delay_K"] != [expect_K]:
            bad.append(f"delay_K reads {r['delay_K']}, expected [{expect_K}]")
    else:
        if r["anchor_backwards"] > 0:
            bad.append(f"anchor fired {r['anchor_backwards']:.0f} times on a no-anchor arm")
        if r["spectral_corrections"] > 0:
            bad.append("merger ran on a no-anchor arm")
    return bad


def cell(r, step):
    if r is None or step not in r["val"]:
        return "--"
    return f"${r['val'][step]:.3f}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory holding compass-<arm>/train.log")
    ap.add_argument("--latex", action="store_true", help="also emit table bodies")
    args = ap.parse_args()

    arms = {}
    for name in sorted({a[0] for a in NOANCHOR_ARMS} | {a[0] for a in K_ARMS}):
        arms[name] = summarise(name, args.root)

    print("=" * 78)
    print(f"{'arm':10s} {'last':>5s} {'step_s':>8s} {'tok/s':>8s} {'K':>8s} "
          f"{'anc_bwd':>8s} {'mask':>10s}  val trajectory")
    print("=" * 78)
    for name, r in arms.items():
        if r is None:
            print(f"{name:10s}   -- not present --")
            continue
        traj = " ".join(f"{s}:{v:.3f}" for s, v in sorted(r["val"].items()))
        print(f"{name:10s} {r['last_step']:5d} {r['step_s']:8.2f} {r['tps']:8.0f} "
              f"{str(r['delay_K'] or '-'):>8s} {r['anchor_backwards']:8.0f} "
              f"{r['mask_train']:10.0f}  {traj}")

    print("\n" + "=" * 78)
    print("FALSIFIERS  (an arm that fails one of these did not run what it claims)")
    print("=" * 78)
    spec = {
        "base": (True, True, 20), "dense": (False, False, None),
        "noanchor": (False, True, None), "nosign": (True, True, 20),
        "k10": (True, True, 10), "k40": (True, True, 40),
        "k40gm": (True, True, 40), "k80": (True, True, 80),
        "k80gm": (True, True, 80),
    }
    any_bad = False
    for name, r in arms.items():
        if r is None:
            continue
        bad = check(r, *spec[name])
        if bad:
            any_bad = True
            for b in bad:
                print(f"  FAIL {name:10s} {b}")
        else:
            print(f"  ok   {name:10s}")
    if any_bad:
        print("\n  At least one arm is misconfigured. Do not put it in the paper.")

    if not args.latex:
        return 0 if not any_bad else 1

    print("\n" + "=" * 78)
    print("TABLE: anchor necessity (app:rlvr-noanchor)")
    print("=" * 78)
    for name, label, anchor, dense_pass in NOANCHOR_ARMS:
        r = arms.get(name)
        cells = " & ".join(cell(r, s) for s in (120, 200, 600))
        print(f"{label} & {anchor} & {dense_pass} & {cells} \\\\")

    print("\n" + "=" * 78)
    print("TABLE: staleness sweep (app:rlvr-staleness)")
    print("=" * 78)
    for name, K, strength in K_ARMS:
        r = arms.get(name)
        reach = f"${K / 20 * strength:.2g}\\times$"
        snaps = K // 20 + 1
        fires = "--" if r is None else f"{r['rank1_fires']:.0f}"
        cells = " & ".join(cell(r, s) for s in (120, 200))
        print(f"${K}$ & ${strength}$ & {reach} & ${snaps}$ & {fires} & {cells} \\\\")

    print("\n" + "=" * 78)
    print("TABLE: peak GPU memory decomposition (app:rlvr-throughput)")
    print("=" * 78)
    print("Replaces the single compressed-vs-dense ratio, which rested on one early")
    print("high-water excursion. These separate the codec's cost from the anchor's.")
    d, na, b = arms.get("dense"), arms.get("noanchor"), arms.get("base")
    rows = [
        ("Dense control, no codec, no anchor", d, "pre_anchor_peak_gb"),
        ("Compressed, anchor removed", na, "pre_anchor_peak_gb"),
        ("Compressed with anchor, between firings", b, "pre_anchor_peak_gb"),
        ("Compressed with anchor, on a firing step", b, "fire_peak_gb"),
    ]
    for label, r, key in rows:
        v = r.get(key) if r else None
        print(f"{label} & {f'${v:.1f}$' if v else '--'} \\\\")
    if na and b and na.get("pre_anchor_peak_gb") and b.get("fire_peak_gb"):
        print(f"\n  codec alone costs {na['pre_anchor_peak_gb'] - (d['pre_anchor_peak_gb'] if d and d.get('pre_anchor_peak_gb') else float('nan')):+.2f} GB, "
              f"anchor firing adds {b['fire_peak_gb'] - b['pre_anchor_peak_gb']:+.2f} GB")

    print(f"\n(base MATH reading for these comparisons: {BASE_MATH})")
    return 0 if not any_bad else 1


if __name__ == "__main__":
    sys.exit(main())
