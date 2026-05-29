#!/usr/bin/env python3
"""Parse EXP-14 test2_cellF metrics jsonl: per-step grad_norm + clean_steps."""
import sys, json

hdr = ["step", "grad_norm", "clean_steps", "mask_ratio", "ppo_kl", "entropy", "pg_loss"]
print("{:>4} | {:>12} | {:>11} | {:>10} | {:>10} | {:>9} | {:>10}".format(*hdr))
print("-" * 90)


def f(x):
    return ("{:.4f}".format(x)) if isinstance(x, (int, float)) else str(x)


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
    except Exception:
        continue
    d = o.get("data", o)
    step = o.get("step", d.get("training/global_step"))
    gn = d.get("actor/grad_norm")
    cs = d.get("actor/comm_eff/clean_steps")
    kl = d.get("actor/ppo_kl")
    ent = d.get("actor/entropy")
    mr = d.get("actor/comm_eff/mask_ratio")
    pg = d.get("actor/pg_loss")
    print("{:>4} | {:>12} | {:>11} | {:>10} | {:>10} | {:>9} | {:>10}".format(
        str(step), f(gn), f(cs), f(mr), f(kl), f(ent), f(pg)))
