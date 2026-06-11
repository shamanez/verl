#!/usr/bin/env python3
"""Task #5 addendum: carrier-vs-substrate attribution scorecard.

Adds the columns the main scorecard.csv lacks but that are decisive for the
carrier(merger)-vs-substrate question: max consecutive cap-pin streak, grad_norm
character (mean/max over the endpoint window), and a merger? flag. Writes
scorecard_addendum.csv covering all six runs (plain + ef_r2 are the focus, the
rest are included for contrast). Does NOT touch scorecard.csv.
"""
import csv
import statistics as st

RUNS = [
    ("dense", "dense_5e2jpho9.csv", "no"),
    ("plain", "plain_u1v94opv.csv", "NO (substrate only: powersgd+anchor, spectral.enabled=False)"),
    ("ef_r2", "ef_parent_r2_tilwe80t.csv", "YES (spectral.enabled=True, ef_clip=1 decay=0.9)"),
    ("a0p5", "signed_ema_a0p5_1wulaelw.csv", "YES (signed_ema delta=0.5)"),
    ("exp27", "exp27_damped_ef_qa6sll3h.csv", "YES (ef damped clip=0.5 decay=0.5)"),
    ("ef_r1", "ef_r1_c7fa7kjv.csv", "YES (ef full)"),
]
CAP = 16384


def load(p):
    out = {}
    try:
        rows = list(csv.DictReader(open(p)))
    except FileNotFoundError:
        return out
    for row in rows:
        try:
            s = int(float(row["step"]))
        except (ValueError, TypeError):
            continue
        for k, v in row.items():
            if k == "step" or v in (None, ""):
                continue
            try:
                out.setdefault(s, {})[k] = float(v)
            except ValueError:
                pass
    return out


def col(d, *names):
    """First present of alt column names (csvs differ: len_mean vs ...)."""
    def get(rec):
        for n in names:
            if rec.get(n) is not None:
                return rec[n]
        return None
    return get


def slope(d, getf, a, b):
    xs, ys = [], []
    for s in range(a, b + 1):
        if s in d and getf(d[s]) is not None:
            xs.append(s)
            ys.append(getf(d[s]))
    if len(xs) < 3:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None


def max_consec_pin(d, getmax):
    best = cur = 0
    prev = None
    for s in sorted(d):
        v = getmax(d[s])
        if v is not None and v >= CAP:
            cur = cur + 1 if (prev is not None and s == prev + 1) else 1
            best = max(best, cur)
            prev = s
        else:
            cur = 0
            prev = None
    return best


def main():
    rows = []
    for lab, path, merger in RUNS:
        d = load(path)
        if not d:
            continue
        last = max(d)
        end = 50 if 50 in d and any(d[50].get(k) is not None for k in d[50]) else last
        g_len = col(d, "len_mean")
        g_max = col(d, "len_max")
        g_ent = col(d, "entropy")
        g_clip = col(d, "len_clip_ratio", "clip_ratio")
        g_gn = col(d, "grad_norm")

        ent = g_ent(d.get(end, {}))
        lm = g_len(d.get(end, {}))
        lm_sl = slope(d, g_len, 41, 50)
        lx = g_max(d.get(end, {}))
        # back-half max len_max level (steps 30-50)
        bh = [g_max(d[s]) for s in range(30, 51) if s in d and g_max(d[s]) is not None]
        bh_max = max(bh) if bh else None
        consec = max_consec_pin(d, g_max)
        nspk = sum(1 for s in range(41, 51) if s in d and (g_max(d[s]) or 0) >= CAP)
        nclip = sum(1 for s in range(41, 51) if s in d and (g_clip(d[s]) or 0) > 0)
        gn = [g_gn(d[s]) for s in range(40, 51) if s in d and g_gn(d[s]) is not None]
        gn_mean = st.mean(gn) if gn else None
        gn_max = max(gn) if gn else None

        # verdict: carrier signature = any long-tail emission OR clustering
        if consec >= 2 or (lm_sl is not None and lm_sl > 2):
            verdict = "DANGER (carrier clustering / rising len)"
        elif (bh_max is not None and bh_max >= 4000):
            verdict = "WATCH (carrier present: isolated long-tail spikes)"
        else:
            verdict = "CLEAN (no carrier signature)"

        rows.append([
            lab, merger,
            round(ent, 4) if ent is not None else "",
            round(lm, 1) if lm is not None else "",
            round(lm_sl, 3) if lm_sl is not None else "",
            int(lx) if lx is not None else "",
            int(bh_max) if bh_max is not None else "",
            consec, nspk, nclip,
            round(gn_mean, 2) if gn_mean is not None else "",
            round(gn_max, 2) if gn_max is not None else "",
            verdict,
        ])

    header = ["run", "merger_on", "entropy@end", "len_mean@end",
              "len_mean_slope_41_50", "len_max@end", "len_max_backhalf_max",
              "max_consec_cap_pin", "n_cap_pins_41_50", "n_clip_41_50",
              "grad_norm_mean_40_50", "grad_norm_max_40_50", "verdict"]
    with open("scorecard_addendum.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # print markdown
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main()
