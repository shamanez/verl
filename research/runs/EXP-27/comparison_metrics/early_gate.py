#!/usr/bin/env python3
"""Task #6: retro-test early-warning signals at step <=30 ONLY.

Pretend we cannot see beyond step 30. Score each of the 6 runs on candidate
early discriminators over the window [10, 30], then compare against the known
final outcomes. Writes scorecard_early.csv + prints markdown.
"""
import csv
import statistics as st

# label, file, ground-truth outcome, ground-truth class for confusion matrix
# igniters = exp27(61), ef_r1(30), a0p5(censored-onset 47-48 -> treat IGNITE)
# survivors-within-50 = dense, plain ; ef_r2 = censored-survivor (sensitivity both ways)
RUNS = [
    ("dense", "dense_5e2jpho9.csv", "SURVIVOR"),
    ("plain", "plain_u1v94opv.csv", "SURVIVOR"),
    ("ef_r2", "ef_parent_r2_tilwe80t.csv", "CENSORED?"),   # emitted back-half spikes
    ("a0p5", "signed_ema_a0p5_1wulaelw.csv", "IGNITE"),     # onset 47-48
    ("exp27", "exp27_damped_ef_qa6sll3h.csv", "IGNITE"),    # lock-in 61
    ("ef_r1", "ef_r1_c7fa7kjv.csv", "IGNITE"),              # lock-in 30
]

W_LO, W_HI = 10, 30  # the visible window


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
    def g(rec):
        for n in names:
            if rec.get(n) is not None:
                return rec[n]
        return None
    return g


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


def pctl(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    frac = idx - lo
    if lo + 1 < len(s):
        return s[lo] * (1 - frac) + s[lo + 1] * frac
    return s[lo]


def main():
    rows = []
    for lab, path, outcome in RUNS:
        d = load(path)
        if not d:
            continue
        gmax = col(d, "len_max")
        gmean = col(d, "len_mean")
        gent = col(d, "entropy")
        ggn = col(d, "grad_norm")

        win = [s for s in range(W_LO, W_HI + 1) if s in d]
        maxvals = [gmax(d[s]) for s in win if gmax(d[s]) is not None]
        gnvals = [ggn(d[s]) for s in win if ggn(d[s]) is not None]

        # (a) long-tail spike RATE
        n2000 = sum(1 for v in maxvals if v > 2000)
        n4000 = sum(1 for v in maxvals if v > 4000)
        n8000 = sum(1 for v in maxvals if v > 8000)
        # (b) len_max level stats
        p90_max = pctl(maxvals, 0.90)
        max_max = max(maxvals) if maxvals else None
        # (c) len_mean trailing-10 slope at step 25 and 30
        sl25 = slope(d, gmean, 16, 25)
        sl30 = slope(d, gmean, 21, 30)
        # (d) grad_norm character: spikes > 3x median, first-spike step
        gn_med = st.median(gnvals) if gnvals else None
        gn_spikes = sum(1 for v in gnvals if gn_med and v > 3 * gn_med)
        first_gn_spike = next((s for s in win if ggn(d[s]) is not None and gn_med
                               and ggn(d[s]) > 3 * gn_med), None)
        gn_max = max(gnvals) if gnvals else None
        # (e) entropy decline rate <=30 (conditional, comm-eff only)
        ent_sl = slope(d, gent, W_LO, W_HI)
        ent10 = gent(d.get(10, {}))
        ent30 = gent(d.get(30, {}))

        rows.append({
            "run": lab, "outcome": outcome,
            "n_spike>2000": n2000, "n_spike>4000": n4000, "n_spike>8000": n8000,
            "p90_len_max": round(p90_max) if p90_max is not None else "",
            "max_len_max": round(max_max) if max_max is not None else "",
            "len_mean_slope@25": round(sl25, 3) if sl25 is not None else "",
            "len_mean_slope@30": round(sl30, 3) if sl30 is not None else "",
            "gn_median": round(gn_med, 2) if gn_med is not None else "",
            "gn_spikes>3xmed": gn_spikes,
            "first_gn_spike_step": first_gn_spike if first_gn_spike is not None else "",
            "gn_max": round(gn_max, 2) if gn_max is not None else "",
            "entropy_slope": round(ent_sl, 4) if ent_sl is not None else "",
            "ent@10": round(ent10, 3) if ent10 is not None else "",
            "ent@30": round(ent30, 3) if ent30 is not None else "",
        })

    cols = list(rows[0].keys())
    with open("scorecard_early.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Markdown
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    # --- candidate gate evaluation ---
    print("\n## Candidate gate: ANY len_max>2000 in steps 10-30 => flag susceptible\n")
    def conf(predicate, treat_ef_r2):
        # treat_ef_r2: 'IGNITE' or 'SURVIVOR' for the censored run
        tp = fp = tn = fn = 0
        for r in rows:
            truth = r["outcome"]
            if truth == "CENSORED?":
                truth = treat_ef_r2
            pred = predicate(r)
            if truth == "IGNITE" and pred:
                tp += 1
            elif truth == "IGNITE" and not pred:
                fn += 1
            elif truth == "SURVIVOR" and pred:
                fp += 1
            elif truth == "SURVIVOR" and not pred:
                tn += 1
        return tp, fp, tn, fn

    gate = lambda r: r["n_spike>2000"] >= 1
    for treat in ("IGNITE", "SURVIVOR"):
        tp, fp, tn, fn = conf(gate, treat)
        sens = tp / (tp + fn) if (tp + fn) else None
        spec = tn / (tn + fp) if (tn + fp) else None
        print(f"  ef_r2 treated as {treat}: TP={tp} FP={fp} TN={tn} FN={fn} "
              f"sens={sens} spec={spec}")


if __name__ == "__main__":
    main()
