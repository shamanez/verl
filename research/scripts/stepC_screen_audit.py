#!/usr/bin/env python3
"""Q-family geometry screen audit.

Extends research/scripts/geometry_audit.py (Step A) with the SAME projection
convention so UC_act here is comparable to Step A's 0.318. For each family f
(act/grad/adv/tail/hybrid/ticket) the screen dumped a per-boundary orthonormal
basis Q_f (1536x77). The reference grad is G_fresh_anchor@delay_K=0 (per weight
matrix), validated faithful in Step A.

Judge metrics (per family, per target, post-warm = the latest 2 capture ticks):
  UC_f  = ||proj_{Q_f}(G_fresh)||^2 / ||G_fresh||^2          (median over targets)
  OPP_f = ||proj_{Q_f}(G_off)||^2 / ||G_off||^2,
          G_off = G_fresh - proj_{Q_act}(G_fresh)            (act-deflated ref grad)
  AC_f  = ||M Q_f Q_f^T||^2 / ||M||^2   (M = dumped activation A; guardrail only)

Projection convention (IDENTICAL to geometry_audit.audit): for a weight grad g
and a boundary basis q (H x r) with H=q.shape[0]:
  if g.shape[1]==H: proj = (g @ q) @ q.t()      # row-space projection
  elif g.shape[0]==H: proj = q @ (q.t() @ g)    # column-space projection
q_by_H collapses boundaries by H (setdefault) exactly as Step A did.

Winner rule (spec): family f beats control iff UC_f > UC_act AND OPP_f > OPP_act
on the median.

Usage:
    python research/scripts/stepC_screen_audit.py runs/<run>/captures/C1_screen/rank0
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

CAP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/<run>/captures/C1_screen/rank0")
FAMILIES = ["act", "grad", "adv", "tail", "hybrid", "ticket"]


def load(rel: str):
    return torch.load(CAP / rel, map_location="cpu").float()


def median(xs):
    xs = sorted(x for x in xs if x == x)
    n = len(xs)
    if not n:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def load_manifest():
    rows = []
    with open(CAP / "manifest.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_q_by_H(byrole, role, ticks):
    """Step-A convention: latest tick per target (within `ticks`), keyed by H=shape[0]."""
    latest = {}
    for r in byrole[role]:
        if (r["global_step"], r["optimizer_tick"]) not in ticks:
            continue
        t = r["target_name"]
        tick = r["optimizer_tick"]
        if t not in latest or tick >= latest[t][0]:
            latest[t] = (tick, r)
    q_by_H = {}
    for t, (tick, r) in latest.items():
        qt = load(r["path"])
        if qt.dim() == 2:
            q_by_H.setdefault(qt.shape[0], qt)
    return q_by_H


def project(g, q):
    """Step-A projection: returns proj tensor or None if no axis matches H."""
    H = q.shape[0]
    if g.shape[1] == H:
        return (g @ q) @ q.t()
    if g.shape[0] == H:
        return q @ (q.t() @ g)
    return None


def uc_over_targets(byrole, q_by_H, ticks):
    """UC = ||proj_{Q}(G_fresh)||^2/||G_fresh||^2 per G_fresh_anchor weight target.

    Mirrors geometry_audit: for each weight grad, try axis g.shape[1] then
    g.shape[0], break on first projected axis.
    """
    ucs = []
    per = []
    for r in byrole["G_fresh_anchor"]:
        if (r["global_step"], r["optimizer_tick"]) not in ticks:
            continue
        g = load(r["path"])
        if g.dim() != 2:
            continue
        for axis_dim in (g.shape[1], g.shape[0]):
            q = q_by_H.get(axis_dim)
            if q is None:
                continue
            proj = project(g, q)
            if proj is None:
                continue
            ng = float(torch.linalg.norm(g))
            if ng == 0.0:
                break
            uc = float(torch.linalg.norm(proj) ** 2 / (ng * ng))
            ucs.append(uc)
            per.append((r["optimizer_tick"], r["target_name"], uc))
            break
    return ucs, per


def opp_over_targets(byrole, q_by_H, qact_by_H, ticks):
    """OPP = ||proj_{Q_f}(G_off)||^2/||G_off||^2, G_off = G_fresh - proj_{Q_act}(G_fresh).

    Deflate with the SAME-tick act basis (q_act collapsed by H, same convention).
    """
    opps = []
    per = []
    for r in byrole["G_fresh_anchor"]:
        if (r["global_step"], r["optimizer_tick"]) not in ticks:
            continue
        g = load(r["path"])
        if g.dim() != 2:
            continue
        # G_off: deflate g against Q_act on the matching axis
        for axis_dim in (g.shape[1], g.shape[0]):
            qa = qact_by_H.get(axis_dim)
            if qa is None:
                continue
            pa = project(g, qa)
            if pa is None:
                continue
            g_off = g - pa
            qf = q_by_H.get(axis_dim)
            if qf is None:
                break
            pf = project(g_off, qf)
            if pf is None:
                break
            ngoff = float(torch.linalg.norm(g_off))
            if ngoff == 0.0:
                break
            opp = float(torch.linalg.norm(pf) ** 2 / (ngoff * ngoff))
            opps.append(opp)
            per.append((r["optimizer_tick"], r["target_name"], opp))
            break
    return opps, per


def ac_over_boundaries(byrole, fam, ticks):
    """AC_f = ||M Q_f Q_f^T||^2 / ||M||^2 using dumped activation A per boundary."""
    # index A by (gs,tick,target)
    A = {(r["global_step"], r["optimizer_tick"], r["target_name"]): r for r in byrole["A"]}
    acs = []
    for r in byrole[f"Q_{fam}"]:
        key = (r["global_step"], r["optimizer_tick"], r["target_name"])
        if key not in ticks_key(key, ticks):
            pass
        if (r["global_step"], r["optimizer_tick"]) not in ticks:
            continue
        if key not in A:
            continue
        q = load(r["path"])           # (H, r)
        M = load(A[key]["path"])      # (N, H)
        if M.shape[1] != q.shape[0]:
            continue
        proj = (M @ q) @ q.t()        # (N, H)
        nm = float(torch.linalg.norm(M))
        if nm == 0.0:
            continue
        acs.append(float(torch.linalg.norm(proj) ** 2 / (nm * nm)))
    return acs


def ticks_key(key, ticks):  # helper kept for clarity; no-op set membership
    return {key}


def main():
    rows = load_manifest()
    byrole = defaultdict(list)
    for r in rows:
        byrole[r["role"]].append(r)

    # post-warm = latest 2 capture ticks that have family bases
    fam_ticks = sorted({(r["global_step"], r["optimizer_tick"]) for r in byrole["Q_act"]})
    post_warm = set(fam_ticks[-2:])
    out = {"post_warm_ticks": sorted(post_warm), "all_family_ticks": fam_ticks}

    # --- comparability anchor: reproduce Step-A UC via the LIVE Q role ---
    qH_live = build_q_by_H(byrole, "Q", post_warm)
    uc_live, _ = uc_over_targets(byrole, qH_live, post_warm)
    out["UC_via_live_Q_role"] = median(uc_live)  # should land near 0.318 (Step A)
    out["UC_via_live_Q_n"] = len([x for x in uc_live if x == x])

    # --- per-family judge metrics (apples-to-apples: family act is the control) ---
    qact_by_H = build_q_by_H(byrole, "Q_act", post_warm)
    fam_results = {}
    for fam in FAMILIES:
        qH = build_q_by_H(byrole, f"Q_{fam}", post_warm)
        uc, uc_per = uc_over_targets(byrole, qH, post_warm)
        opp, opp_per = opp_over_targets(byrole, qH, qact_by_H, post_warm)
        ac = ac_over_boundaries(byrole, fam, post_warm)
        # per-tick medians
        uc_by_tick = defaultdict(list)
        for tick, _t, v in uc_per:
            uc_by_tick[tick].append(v)
        opp_by_tick = defaultdict(list)
        for tick, _t, v in opp_per:
            opp_by_tick[tick].append(v)
        fam_results[fam] = {
            "UC_median": median(uc),
            "OPP_median": median(opp),
            "AC_median": median(ac),
            "UC_n": len([x for x in uc if x == x]),
            "OPP_n": len([x for x in opp if x == x]),
            "AC_n": len([x for x in ac if x == x]),
            "UC_by_tick": {int(k): median(v) for k, v in sorted(uc_by_tick.items())},
            "OPP_by_tick": {int(k): median(v) for k, v in sorted(opp_by_tick.items())},
        }
    out["families"] = fam_results

    uc_act = fam_results["act"]["UC_median"]
    opp_act = fam_results["act"]["OPP_median"]
    out["UC_act"] = uc_act
    out["OPP_act"] = opp_act

    # winner rule: UC_f > UC_act AND OPP_f > OPP_act
    winners = []
    for fam in FAMILIES:
        if fam == "act":
            continue
        r = fam_results[fam]
        if r["UC_median"] > uc_act and r["OPP_median"] > opp_act:
            winners.append(fam)
    out["winners"] = winners

    print("=" * 78)
    print("Q-family geometry screen")
    print("=" * 78)
    print(f"post-warm ticks (latest 2 family ticks): {sorted(post_warm)}")
    print(f"UC via LIVE Q role (Step-A comparability anchor, ~0.318 expected): "
          f"{out['UC_via_live_Q_role']:.4f}  n={out['UC_via_live_Q_n']}")
    print(f"family-internal control UC_act={uc_act:.4f}  OPP_act={opp_act:.4f}")
    print("-" * 78)
    hdr = f"{'family':8s} {'UC_f':>8s} {'OPP_f':>8s} {'AC_f':>8s}  {'beats?':>8s}  per-tick UC / OPP"
    print(hdr)
    for fam in FAMILIES:
        r = fam_results[fam]
        beats = "control" if fam == "act" else (
            "YES" if (r["UC_median"] > uc_act and r["OPP_median"] > opp_act) else "no")
        uct = " ".join(f"t{k}={v:.3f}" for k, v in r["UC_by_tick"].items())
        oppt = " ".join(f"t{k}={v:.3f}" for k, v in r["OPP_by_tick"].items())
        print(f"{fam:8s} {r['UC_median']:8.4f} {r['OPP_median']:8.4f} {r['AC_median']:8.4f}  "
              f"{beats:>8s}  UC[{uct}] OPP[{oppt}]")
    print("-" * 78)
    print(f"WINNER(S) (UC_f>UC_act AND OPP_f>OPP_act): {winners if winners else 'none_beats_act'}")
    print("-" * 78)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
