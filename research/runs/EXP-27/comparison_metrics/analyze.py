#!/usr/bin/env python3
"""Build aligned-step tables + ignition-precursor scorecard from the pulled CSVs.

Reads <label>.csv (one row per logged step), emits:
  - aligned_<metric>.csv  : rows=steps, cols=runs (for the key metrics)
  - scorecard.csv         : precursor scorecard per run at its step-50 state
  - prints markdown tables to stdout (captured into RUN_COMPARISON.md by hand)
"""
import csv
import math
import os

RUNS = [
    ("dense", "dense_5e2jpho9.csv"),
    ("a0p5", "signed_ema_a0p5_1wulaelw.csv"),
    ("exp27", "exp27_damped_ef_qa6sll3h.csv"),
    ("ef_r2", "ef_parent_r2_tilwe80t.csv"),
    ("plain", "plain_u1v94opv.csv"),
]

ALIGN_STEPS = [10, 20, 30, 40, 50, 55, 60, 64, 65, 66, 67]


def load(path):
    """Return dict step->row(dict of float-or-None)."""
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                s = int(float(row["step"]))
            except (ValueError, TypeError):
                continue
            rec = {}
            for k, v in row.items():
                if k == "step":
                    continue
                if v is None or v == "":
                    rec[k] = None
                else:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = None
            # keep last write for a step (val rows may share step with train)
            if s in out:
                for k, v in rec.items():
                    if v is not None:
                        out[s][k] = v
            else:
                out[s] = rec
    return out


def fmt(v, nd=3):
    if v is None:
        return "-"
    if abs(v) >= 1000 or (v != 0 and abs(v) < 0.001):
        return f"{v:.4g}" if abs(v) < 100000 else f"{v:.0f}"
    return f"{v:.{nd}f}"


def trailing_slope(data, metric, end_step, window=10):
    """OLS slope of metric over [end_step-window+1, end_step] (per-step)."""
    xs, ys = [], []
    for s in range(end_step - window + 1, end_step + 1):
        if s in data and data[s].get(metric) is not None:
            xs.append(s)
            ys.append(data[s][metric])
    if len(xs) < 3:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def spike_count(data, metric, end_step, window=10, thresh=16384):
    c = 0
    present = 0
    for s in range(end_step - window + 1, end_step + 1):
        if s in data and data[s].get(metric) is not None:
            present += 1
            if data[s][metric] >= thresh:
                c += 1
    return c, present


def nonzero_clip_count(data, end_step, window=10):
    c = 0
    present = 0
    for s in range(end_step - window + 1, end_step + 1):
        if s in data and data[s].get("len_clip_ratio") is not None:
            present += 1
            if data[s]["len_clip_ratio"] > 0:
                c += 1
    return c, present


def main():
    loaded = {label: load(path) for label, path in RUNS if os.path.exists(path)}

    metrics = [
        ("entropy", "actor/entropy"),
        ("len_mean", "response_length/mean"),
        ("len_max", "response_length/max"),
        ("len_clip_ratio", "response_length/clip_ratio"),
        ("score_mean", "critic/score/mean"),
        ("pg_loss", "actor/pg_loss"),
        ("grad_norm", "actor/grad_norm"),
        ("pg_clipfrac", "actor/pg_clipfrac"),
    ]

    print("# Aligned-step tables\n")
    for short, full in metrics:
        # write aligned CSV
        with open(f"aligned_{short}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + [lab for lab, _ in RUNS if lab in loaded])
            for s in ALIGN_STEPS:
                row = [s]
                for lab, _ in RUNS:
                    if lab not in loaded:
                        continue
                    rec = loaded[lab].get(s, {})
                    row.append(rec.get(short))
                w.writerow(row)
        # markdown
        labs = [lab for lab, _ in RUNS if lab in loaded]
        print(f"## {full}\n")
        print("| step | " + " | ".join(labs) + " |")
        print("|" + "---|" * (len(labs) + 1))
        for s in ALIGN_STEPS:
            cells = []
            for lab in labs:
                rec = loaded[lab].get(s, {})
                cells.append(fmt(rec.get(short)))
            print(f"| {s} | " + " | ".join(cells) + " |")
        print()

    # Scorecard at step-50 state (each run's trailing-10 window ending at 50)
    print("\n# Ignition-precursor scorecard @ step 50\n")
    print("| run | entropy@50 | entropy slope(40-50) | len_mean@50 | len_mean slope(41-50) | "
          "len_max@50 | #len_max>=16384 (41-50) | #clip>0 (41-50) | VERDICT |")
    print("|" + "---|" * 9)
    sc_rows = []
    for lab, _ in RUNS:
        if lab not in loaded:
            continue
        d = loaded[lab]
        end = 50
        ent = d.get(end, {}).get("entropy")
        ent_sl = trailing_slope(d, "entropy", end, 11)  # 40..50 inclusive ~11 pts
        lm = d.get(end, {}).get("len_mean")
        lm_sl = trailing_slope(d, "len_mean", end, 10)
        lx = d.get(end, {}).get("len_max")
        nspk, npr = spike_count(d, "len_max", end, 10)
        nclip, ncpr = nonzero_clip_count(d, end, 10)
        # verdict logic
        flags = []
        if ent is not None and ent < 0.4:
            flags.append("ent<0.4")
        if lm_sl is not None and lm_sl > 5:
            flags.append("len_mean_rising")
        if nspk >= 2:
            flags.append(f"len_max_spikes={nspk}")
        if nclip >= 2:
            flags.append(f"clip_spikes={nclip}")
        verdict = "CLEAN" if not flags else ("WATCH" if len(flags) == 1 else "DANGER")
        verdict += (" (" + ",".join(flags) + ")") if flags else ""
        print(f"| {lab} | {fmt(ent)} | {fmt(ent_sl,4)} | {fmt(lm)} | {fmt(lm_sl,2)} | "
              f"{fmt(lx)} | {nspk}/{npr} | {nclip}/{ncpr} | {verdict} |")
        sc_rows.append([lab, ent, ent_sl, lm, lm_sl, lx, f"{nspk}/{npr}", f"{nclip}/{ncpr}", verdict])

    with open("scorecard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "entropy@50", "entropy_slope_40_50", "len_mean@50",
                    "len_mean_slope_41_50", "len_max@50", "len_max_spikes_41_50",
                    "clip_spikes_41_50", "verdict"])
        for r in sc_rows:
            w.writerow(r)

    # Also dump exp27's full ignition window so the contrast is explicit.
    print("\n# exp27 ignition window (steps 55-67, ground-truth)\n")
    d = loaded["exp27"]
    print("| step | entropy | len_mean | len_max | clip_ratio | score | grad_norm | pg_loss |")
    print("|" + "---|" * 8)
    for s in range(55, 68):
        rec = d.get(s, {})
        print(f"| {s} | {fmt(rec.get('entropy'))} | {fmt(rec.get('len_mean'))} | "
              f"{fmt(rec.get('len_max'))} | {fmt(rec.get('len_clip_ratio'),4)} | "
              f"{fmt(rec.get('score_mean'))} | {fmt(rec.get('grad_norm'))} | {fmt(rec.get('pg_loss'))} |")

    # Endpoint comparison: a0p5 trailing-10 (41-50) full window for visual.
    print("\n# signed_ema a0.5 endpoint window (steps 41-50, ground-truth)\n")
    d = loaded["a0p5"]
    print("| step | entropy | len_mean | len_max | clip_ratio | score | grad_norm | pg_loss |")
    print("|" + "---|" * 8)
    for s in range(41, 51):
        rec = d.get(s, {})
        print(f"| {s} | {fmt(rec.get('entropy'))} | {fmt(rec.get('len_mean'))} | "
              f"{fmt(rec.get('len_max'))} | {fmt(rec.get('len_clip_ratio'),4)} | "
              f"{fmt(rec.get('score_mean'))} | {fmt(rec.get('grad_norm'))} | {fmt(rec.get('pg_loss'))} |")

    # Dense endpoint window (41-50).
    print("\n# dense endpoint window (steps 41-50, ground-truth)\n")
    d = loaded["dense"]
    print("| step | entropy | len_mean | len_max | clip_ratio | score | grad_norm | pg_loss |")
    print("|" + "---|" * 8)
    for s in range(41, 51):
        rec = d.get(s, {})
        print(f"| {s} | {fmt(rec.get('entropy'))} | {fmt(rec.get('len_mean'))} | "
              f"{fmt(rec.get('len_max'))} | {fmt(rec.get('len_clip_ratio'),4)} | "
              f"{fmt(rec.get('score_mean'))} | {fmt(rec.get('grad_norm'))} | {fmt(rec.get('pg_loss'))} |")


if __name__ == "__main__":
    main()
