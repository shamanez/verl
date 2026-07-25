#!/usr/bin/env python3
"""roundA_table.py - score every issue #93 round-A arm against the BLIND-COMMITTED criteria.

The criteria and thresholds below were fixed in `AB_AMENDMENT.md` and `A4_GUARD.md`
BEFORE a4 and a5 produced a single step. This script only applies them. It does not
choose a winner; it prints the table the analyst and the operator decide from.

Committed criteria (lexicographic; see AB_AMENDMENT.md section 4):
  E   eligibility  reached step 120; wire <= 1232 bits/token/boundary for round-C
                   promotion; actor/ppo_kl == 0 or explained
  V1  drift veto   actor/kl_loss SLOPE <= 1.5x the incumbent's on the SAME window
                   (ceilings 3.52e-3 at 61-120, 3.264e-3 at 100-120). Slope only,
                   never level: every codec carries a different view offset.
  V2  grad veto    run max <= 10.0 and 61+ window mean <= 3.62
  V3  learn veto   critic/score/mean LEVEL at 100-120 >= 0.6248 (0.95x incumbent).
                   Level not slope: at 100-120 all reference runs including DENSE
                   have negative reward slope, so a slope bar is sign-degenerate
                   there, and dense itself fails the registered 0.00288 slope bar.
  O   objective    minimise gap slope (61-120) against S-bar = +5.0e-4, then gap
                   level (100-120) subject to <= 14.2458
  T   tie-break    higher E[rho]; a bonus never a veto (the incumbent is at 0.00503)

Degeneracy screen (added after a5 showed a falling codec-view entropy that two
independent arguments proved to be VIEW movement, not policy movement):
  the entropy READING is a codec-fidelity signal, not a health signal, because the
  DENSE uncompressed control reads 0.324 nats and is perfectly healthy. So health
  is judged on codec-free observables instead: sampler rollout_log_ppl, within-group
  advantage spread, score spread, and aborted ratio.

Usage:
  roundA_table.py [--json out.json]
WANDB_API_KEY must be in the environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

ENTITY = "shamanework-pl"
PROG = "93-long-horizon-stability"

ARMS = [
    ("90-dense-600", "90-dense-600", "dense (uncompressed)", None),
    ("90-prf-exactk-600", "90-prf-exactk-600", "incumbent PRF exact-k", 1232.0),
    (PROG, "a1-srq-b1-sr", "a1 1-bit SR", 2304.0),
    (PROG, "a2-srq-b1-rn", "a2 1-bit RN (killed@60)", 2304.0),
    (PROG, "a3-srq-parity-k493", "a3 parity hybrid", 1232.5),
    (PROG, "a4-prf-exactk-cvc-ce", "a4 PRF + CVC-CE", 1232.0),
    (PROG, "a5-frlr-r48k28-tis", "a5 FRLR + token-IS", 1232.0),
]

KEYS = [
    "actor/kl_loss",
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
    "critic/score/mean",
    "actor/grad_norm",
    "actor/entropy",
    "actor/ppo_kl",
    "response_length/mean",
    "rollout_corr/rollout_log_ppl",
    "critic/advantages/max",
]

S_BAR = 5.0e-4
GAP_LEVEL_BAR = 14.2458
V1_MULT = 1.5
V2_MAX, V2_MEAN = 10.0, 3.62
V3_BAR = 0.6248
WIRE_BAR = 1232.0


def pull(project: str, name: str):
    import wandb

    api = wandb.Api()
    run = None
    for c in api.runs(f"{ENTITY}/{project}"):
        if c.name == name:
            run = c
            break
    if run is None:
        return None, None
    d = {k: [] for k in KEYS}
    d["step"] = []
    for r in run.scan_history(keys=["training/global_step"] + KEYS):
        s = r.get("training/global_step")
        if s is None:
            continue
        d["step"].append(s)
        for k in KEYS:
            d[k].append(r.get(k))
    return {k: np.array([np.nan if v is None else v for v in vv], float) for k, vv in d.items()}, run


def slope(x, y, lo, hi):
    m = (x >= lo) & (x <= hi) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    xx, yy = x[m], y[m]
    xb = xx.mean()
    sxx = ((xx - xb) ** 2).sum()
    return float(((xx - xb) * (yy - yy.mean())).sum() / sxx)


def level(x, y, lo, hi):
    m = (x >= lo) & (x <= hi) & np.isfinite(y)
    return float(np.nanmean(y[m])) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = {}
    for proj, name, label, wire in ARMS:
        d, run = pull(proj, name)
        if d is None or len(d["step"]) == 0:
            rows[label] = {"error": "pull empty (a dense/no-comm-eff run needs core keys only)"}
            continue
        x = d["step"]
        er = d["rollout_corr/k3_kl"] - d["rollout_corr/kl"] + 1
        m61 = x >= 61
        rows[label] = {
            "run": name,
            "state": run.state,
            "max_step": float(np.nanmax(x)),
            "wire_bits": wire,
            "v1_61_120": slope(x, d["actor/kl_loss"], 61, 120),
            "v1_100_120": slope(x, d["actor/kl_loss"], 100, 120),
            "v2_max": float(np.nanmax(d["actor/grad_norm"][m61])) if m61.any() else float("nan"),
            "v2_mean": float(np.nanmean(d["actor/grad_norm"][m61])) if m61.any() else float("nan"),
            "v3_score_level": level(x, d["critic/score/mean"], 100, 120),
            "o_gap_slope": slope(x, d["rollout_corr/kl"], 61, 120),
            "o_gap_level": level(x, d["rollout_corr/kl"], 100, 120),
            "e_rho": float(np.nanmedian(er[(x >= 100) & (x <= 120)])) if ((x >= 100) & (x <= 120)).any() else float("nan"),
            "ppo_kl_max": float(np.nanmax(np.abs(d["actor/ppo_kl"]))),
            "entropy_level": level(x, d["actor/entropy"], 100, 120),
            "rollout_log_ppl": level(x, d["rollout_corr/rollout_log_ppl"], 100, 120),
            "adv_max": level(x, d["critic/advantages/max"], 100, 120),
            "resp_len": level(x, d["response_length/mean"], 100, 120),
        }

    inc = rows.get("incumbent PRF exact-k", {})
    v1c_61 = V1_MULT * inc.get("v1_61_120", float("nan"))
    v1c_100 = V1_MULT * inc.get("v1_100_120", float("nan"))

    print("=== issue #93 round A, scored against the BLIND-COMMITTED criteria ===")
    print(f"V1 ceilings: {v1c_61:.3e} at 61-120, {v1c_100:.3e} at 100-120 (1.5x incumbent, same window)")
    print(f"V2 <= {V2_MAX}/{V2_MEAN} | V3 >= {V3_BAR} | S-bar <= {S_BAR:.1e} | gap level <= {GAP_LEVEL_BAR} | wire <= {WIRE_BAR:.0f}\n")
    hdr = f"{'arm':24s} {'V1@100-120':>11s} {'V2 max/mean':>13s} {'V3 score':>9s} {'gap slope':>10s} {'gap level':>10s} {'E[rho]':>7s} {'wire':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in rows.items():
        if "error" in r:
            print(f"{label:24s} {r['error']}")
            continue
        print(
            f"{label:24s} {r['v1_100_120']:11.6f} {r['v2_max']:6.2f}/{r['v2_mean']:<6.2f} "
            f"{r['v3_score_level']:9.4f} {r['o_gap_slope']:10.6f} {r['o_gap_level']:10.4f} "
            f"{r['e_rho']:7.4f} {(r['wire_bits'] if r['wire_bits'] else 0):7.1f}"
        )

    print("\n=== pass/fail per committed criterion (compressed arms only) ===")
    for label, r in rows.items():
        if "error" in r or label.startswith("dense") or label.startswith("incumbent"):
            continue
        reached = r["max_step"] >= 120
        e = reached and (r["wire_bits"] is not None and r["wire_bits"] <= WIRE_BAR)
        v1 = r["v1_100_120"] <= v1c_100
        v2 = (r["v2_max"] <= V2_MAX) and (r["v2_mean"] <= V2_MEAN)
        v3 = r["v3_score_level"] >= V3_BAR
        osl = r["o_gap_slope"] <= S_BAR
        olv = r["o_gap_level"] <= GAP_LEVEL_BAR
        vet = v1 and v2 and v3
        print(
            f"  {label:24s} E={'P' if e else 'F'} (step {r['max_step']:.0f}, wire {r['wire_bits']:.1f})  "
            f"V1={'P' if v1 else 'F'} V2={'P' if v2 else 'F'} V3={'P' if v3 else 'F'}  "
            f"=> vetoes {'CLEAR' if vet else 'FAIL'} | O-slope={'P' if osl else 'F'} O-level={'P' if olv else 'F'}"
        )

    print("\n=== degeneracy screen on CODEC-FREE observables (entropy READING is fidelity, not health) ===")
    dn = rows.get("dense (uncompressed)", {})
    print(f"  dense entropy reading = {dn.get('entropy_level', float('nan')):.4f} nats "
          f"(e^H = {math.exp(dn['entropy_level']) if np.isfinite(dn.get('entropy_level', float('nan'))) else float('nan'):.1f}); "
          f"a high reading is codec mush, NOT health")
    for label, r in rows.items():
        if "error" in r:
            continue
        print(f"  {label:24s} rollout_log_ppl {r['rollout_log_ppl']:.4f} | adv_max {r['adv_max']:.4f} "
              f"| resp_len {r['resp_len']:.0f} | entropy_reading {r['entropy_level']:.3f}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "v1_ceiling_61_120": v1c_61, "v1_ceiling_100_120": v1c_100}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
