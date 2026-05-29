#!/usr/bin/env python3
"""Per-step metric extractor for EXP-16 cell train.logs.
Usage: extract_metrics.py <train.log> [<train.log> ...]
Prints step / grad_norm / ppo_kl / pg_clipfrac / clip_lower / rollout_probs_diff / reward.
"""
import re, sys

KEYS = [
    ("grad_norm", "actor/grad_norm"),
    ("ppo_kl", "actor/ppo_kl"),
    ("clipfrac", "actor/pg_clipfrac"),
    ("clip_low", "actor/pg_clipfrac_lower"),
    ("roll_diff", "training/rollout_probs_diff_mean"),
    ("reward", "critic/rewards/mean"),
    ("mask_ratio", "actor/comm_eff/mask_ratio"),
]

def get(line, key):
    m = re.search(re.escape(key) + r":(?:np\.float64\()?(-?[0-9.eE+]+)\)?", line)
    return float(m.group(1)) if m else float("nan")

for path in sys.argv[1:]:
    print("==== " + path + " ====")
    try:
        lines = [l for l in open(path, errors="ignore") if "global_step" in l and "actor/grad_norm" in l]
    except FileNotFoundError:
        print("  (not started)"); continue
    hdr = "%4s " % "step" + " ".join("%11s" % name for name, _ in KEYS)
    print(hdr)
    for l in lines:
        s = re.search(r"step:(\d+)", l)
        st = s.group(1) if s else "?"
        row = "%4s " % st + " ".join("%11.4f" % get(l, k) for _, k in KEYS)
        print(row)
