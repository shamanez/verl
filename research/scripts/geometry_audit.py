#!/usr/bin/env python3
"""EXP-26 Step A: real-gradient geometry audit over the dumped fp32 tensors.

The Step-A capture (comm_eff.capture.enabled) writes fp32 tensors under
``runs/EXP-26/captures/rank<r>/`` keyed by ``(global_step, optimizer_tick, role,
target_name)`` with a per-row ``manifest.jsonl``. This script reads those dumps
and computes the discriminators the plan's ## Success criteria gate on — the
analyst invokes it and captures stdout into ``runs/EXP-26/analysis.log``:

  cos(G_dense, G_comp)   median over targets, post-warmup tick — H1 (compression
                         is direction-benign iff >= 0.95)
  cos(G_dense, G_corr)   for the active merger — confirms H1 (signed_ema collapses
                         it) / H4 (ef_powersgd improves it over plain PowerSGD)
  Q_act update-capture   ||QQᵀG||²/||G||² per target (the off-principal share) —
                         H2 (does Q_act miss off-principal GRPO UPDATE energy?)
  Q_act activation ratio recomputed ||A-Â||/||A|| vs the logged scalar (the fp32
                         dump-fidelity invariant)
  sign-agreement         magnitude-weighted sign(M)·sign(G_comp) and
                         sign(G_fresh_anchor)·sign(G_comp) at delay_K∈{0,5} — H3
                         (structural ≈coin-flip even at delay_K=0?)

and emits a machine-readable DECISION block:
  DECISION: {go_B_skip_C | go_C_then_B | retire_sign_replacement(confirmed)}

This is a DIAGNOSTIC computation only — it never trains, provisions, or mutates
state. It is intentionally dependency-light (torch.load + stdlib). The DECISION is
a recommendation the analyst reads alongside the training-arm criteria; the
thresholds mirror the plan's ## Step A — geometry audit checklist.

Usage:
    python research/scripts/geometry_audit.py runs/EXP-26 [--rank 0] [--warmup-tick 3]
    python research/scripts/geometry_audit.py runs/EXP-26 --captures-subdir captures
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"geometry_audit: torch is required to load the fp32 dumps ({exc})", file=sys.stderr)
    sys.exit(2)


# Plan ## Step A thresholds (single source: the plan's checklist).
COS_BENIGN = 0.95           # cos(G_dense, G_comp) >= this ⇒ compression direction-benign (H1)
ACT_CAPTURE_MIN = 0.99      # Q_act activation capture ratio floor
UPDATE_CAPTURE_MISS = 0.90  # Q_act UPDATE-energy capture below this ⇒ H2 (misses off-principal)
SIGN_COINFLIP_LO = 0.45     # sign-agreement in [LO, HI] ⇒ ≈coin-flip ⇒ structurally unrecoverable
SIGN_COINFLIP_HI = 0.55


def _load_manifest(cap_root: Path) -> list[dict]:
    mp = cap_root / "manifest.jsonl"
    rows: list[dict] = []
    if not mp.exists():
        return rows
    for line in mp.open():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_tensor(cap_root: Path, rel_path: str):
    p = cap_root / rel_path
    if not p.exists():
        return None
    try:
        return torch.load(p, map_location="cpu").float()
    except Exception:
        return None


def _cos(a, b) -> float:
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        return float("nan")
    if a.shape != b.shape:
        return float("nan")
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan")
    return float((a.flatten() @ b.flatten()) / (na * nb))


def _sign_agreement(m, g) -> float:
    """Magnitude-weighted fraction of entries where sign(m) == sign(g).

    Weighted by |g| so the high-magnitude coordinates (which dominate the update)
    count more. Zero entries (sign 0) contribute 0 agreement. Returns NaN on a
    shape mismatch / empty.
    """
    if m is None or g is None or m.shape != g.shape or g.numel() == 0:
        return float("nan")
    sm = torch.sign(m).flatten()
    sg = torch.sign(g).flatten()
    w = g.abs().flatten()
    wsum = float(w.sum())
    if wsum == 0.0:
        return float("nan")
    agree = (sm == sg).float()
    return float((agree * w).sum() / wsum)


def _index(manifest: list[dict]) -> dict:
    """Index manifest rows by role -> {(gs, tick, target): row}."""
    idx: dict = defaultdict(dict)
    for r in manifest:
        key = (int(r["global_step"]), int(r["optimizer_tick"]), r["target_name"])
        idx[r["role"]][key] = r
    return idx


def _median(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan")
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def audit(run_dir: Path, rank: int, warmup_tick: int, captures_subdir: str) -> dict:
    cap_root = run_dir / captures_subdir / f"rank{rank}"
    manifest = _load_manifest(cap_root)
    out: dict = {"run_dir": str(run_dir), "captures_root": str(cap_root), "n_manifest_rows": len(manifest)}
    if not manifest:
        out["error"] = f"no manifest rows under {cap_root} — was comm_eff.capture.enabled set?"
        return out

    idx = _index(manifest)
    out["roles_present"] = sorted(idx.keys())

    # ---- (a) cos(G_dense, G_comp) and cos(G_dense, G_corr), post-warmup ----
    # G_comp/G_corr are keyed by (gs, tick, target) by the merger (tick=spectral_step);
    # G_dense is keyed at tick=spectral_step+1 by the dense probe so it pairs with
    # the SAME train_batch. Align G_dense[tick] with G_comp/G_corr at the same tick.
    cos_dc_comp, cos_dc_corr = [], []
    gdense = idx.get("G_dense", {})
    gcomp = idx.get("G_comp", {})
    gcorr = idx.get("G_corr", {})
    for (gs, tick, tgt), drow in gdense.items():
        if tick < warmup_tick:
            continue
        dt = _load_tensor(cap_root, drow["path"])
        ck = (gs, tick, tgt)
        if ck in gcomp:
            cos_dc_comp.append(_cos(dt, _load_tensor(cap_root, gcomp[ck]["path"])))
        if ck in gcorr:
            cos_dc_corr.append(_cos(dt, _load_tensor(cap_root, gcorr[ck]["path"])))
    out["cos_Gdense_Gcomp_median"] = _median(cos_dc_comp)
    out["cos_Gdense_Gcorr_median"] = _median(cos_dc_corr)
    out["cos_Gdense_Gcomp_n"] = len([x for x in cos_dc_comp if x == x])
    out["cos_Gdense_Gcorr_n"] = len([x for x in cos_dc_corr if x == x])

    # ---- (b) Q_act capture ratios (activation fidelity + UPDATE energy) ----
    # Activation: recompute ||A-Â||/||A|| from the dumped A/Â and compare to the
    # logged reconstruction_rel_error (the fp32-dump-fidelity invariant).
    A = idx.get("A", {})
    Ah = idx.get("A_hat", {})
    Q = idx.get("Q", {})
    act_fid_drift, act_capture = [], []
    for k, arow in A.items():
        if k not in Ah:
            continue
        a = _load_tensor(cap_root, arow["path"])
        ah = _load_tensor(cap_root, Ah[k]["path"])
        if a is None or ah is None:
            continue
        na = float(torch.linalg.norm(a))
        if na == 0.0:
            continue
        rel = float(torch.linalg.norm(a - ah) / na)
        act_capture.append(1.0 - rel * rel)  # captured energy fraction
        logged = arow.get("reconstruction_rel_error")
        if logged is not None:
            act_fid_drift.append(abs(rel - float(logged)))
    out["Q_act_activation_capture_ratio_median"] = _median(act_capture)
    out["recon_rel_error_dump_fidelity_max_drift"] = max(act_fid_drift) if act_fid_drift else float("nan")

    # Update-energy capture: ||QQᵀG||²/||G||² where G is the dense gradient at the
    # boundary. Q is keyed by boundary (target_name="boundary_<i>"); G is per
    # WEIGHT matrix. We report the AGGREGATE off-principal share of the dense
    # update the activation basis Q fails to span, sampled on the dumped Q whose H
    # matches a G axis. (A representative — the analyst can refine per-target.)
    upd_capture = []
    latest_Q: dict = {}
    for (gs, tick, tgt), qrow in Q.items():
        prev = latest_Q.get(tgt)
        if prev is None or tick >= prev[0]:
            latest_Q[tgt] = (tick, qrow)
    q_by_H: dict = {}
    for tgt, (tick, qrow) in latest_Q.items():
        qt = _load_tensor(cap_root, qrow["path"])
        if qt is not None and qt.dim() == 2:
            q_by_H.setdefault(qt.shape[0], qt)  # (H, r)
    for (gs, tick, tgt), drow in gdense.items():
        if tick < warmup_tick:
            continue
        g = _load_tensor(cap_root, drow["path"])
        if g is None or g.dim() != 2:
            continue
        for _axis_dim, qt in ((g.shape[1], q_by_H.get(g.shape[1])), (g.shape[0], q_by_H.get(g.shape[0]))):
            if qt is None:
                continue
            H = qt.shape[0]
            if g.shape[1] == H:
                proj = (g @ qt) @ qt.t()
            elif g.shape[0] == H:
                proj = qt @ (qt.t() @ g)
            else:
                continue
            ng = float(torch.linalg.norm(g))
            if ng == 0.0:
                continue
            upd_capture.append(float(torch.linalg.norm(proj) ** 2 / (ng * ng)))
            break
    out["Q_act_update_capture_ratio_median"] = _median(upd_capture)
    out["Q_act_update_capture_n"] = len([x for x in upd_capture if x == x])
    ucm = out["Q_act_update_capture_ratio_median"]
    out["Q_act_off_principal_update_share_median"] = (1.0 - ucm) if ucm == ucm else float("nan")

    # ---- (c) sign-agreement at delay_K in {0, 5} ----
    # delay_K=5: sign(M_anchor) vs sign(G_comp)  (M is the K=5 stale EMA)
    # delay_K=0: sign(G_fresh_anchor) vs sign(G_comp)
    M = idx.get("M", {})
    Gfresh = idx.get("G_fresh_anchor", {})
    sign_k5, sign_k0 = [], []
    for (gs, tick, tgt), mrow in M.items():
        ck = (gs, tick, tgt)
        if ck in gcomp:
            sign_k5.append(_sign_agreement(_load_tensor(cap_root, mrow["path"]),
                                           _load_tensor(cap_root, gcomp[ck]["path"])))
    for (gs, tick, tgt), frow in Gfresh.items():
        ck = (gs, tick, tgt)
        if ck in gcomp:
            sign_k0.append(_sign_agreement(_load_tensor(cap_root, frow["path"]),
                                           _load_tensor(cap_root, gcomp[ck]["path"])))
    out["sign_agreement_delayK5_median"] = _median(sign_k5)
    out["sign_agreement_delayK0_median"] = _median(sign_k0)

    out["decision"] = _decide(out)
    return out


def _decide(o: dict) -> str:
    """Map the computed discriminators onto the plan's DECISION enum.

    retire_sign_replacement(confirmed): sign-agreement at delay_K=0 is ≈coin-flip
        (structural — confirms H3; sign-replacement is unrecoverable regardless of
        staleness).
    go_C_then_B: H1 holds (compression benign) AND Q_act MISSES off-principal
        update energy (H2 true) — the basis content is also wrong, so fix Q first.
    go_B_skip_C: H1 holds AND Q_act already captures update energy (H2 false) — the
        sign term was the whole defect, ef_powersgd (Step B) suffices.
    inconclusive: a needed discriminator is NaN (missing dump) — emit STUCK upstream.
    """
    cos_dc = o.get("cos_Gdense_Gcomp_median", float("nan"))
    upd = o.get("Q_act_update_capture_ratio_median", float("nan"))
    s0 = o.get("sign_agreement_delayK0_median", float("nan"))

    if math.isnan(cos_dc):
        return "inconclusive(missing cos(G_dense,G_comp) — check capture_g_dense dumps)"
    if s0 == s0 and SIGN_COINFLIP_LO <= s0 <= SIGN_COINFLIP_HI:
        sign_note = "retire_sign_replacement(confirmed)"
    else:
        sign_note = None
    h1 = cos_dc >= COS_BENIGN
    h2 = (upd == upd) and (upd < UPDATE_CAPTURE_MISS)
    if not h1:
        base = "go_C_then_B(H1 FAILED: compression itself rotates the update — investigate rank/basis)"
    elif h2:
        base = "go_C_then_B"
    elif upd == upd:
        base = "go_B_skip_C"
    else:
        base = "inconclusive(missing Q_act update-capture — check A/Q dumps)"
    return f"{sign_note} + {base}" if sign_note else base


def main() -> int:
    ap = argparse.ArgumentParser(description="EXP-26 Step A geometry audit")
    ap.add_argument("run_dir", help="the EXP-26 run dir (contains captures/)")
    ap.add_argument("--rank", type=int, default=0, help="DP rank subdir to read (default 0)")
    ap.add_argument("--warmup-tick", type=int, default=3, help="ignore ticks below this (post-warmup)")
    ap.add_argument("--captures-subdir", default="captures", help="capture subdir under run_dir")
    ap.add_argument("--emit", default=None, help="also write the audit JSON to run_dir/<name>")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    res = audit(run_dir, args.rank, args.warmup_tick, args.captures_subdir)

    print("=" * 72)
    print("EXP-26 Step A — real-gradient geometry audit")
    print("=" * 72)
    for k, v in res.items():
        if k == "decision":
            continue
        if isinstance(v, float):
            print(f"  {k:42s} = {v:.6f}")
        else:
            print(f"  {k:42s} = {v}")
    print("-" * 72)
    print(f"DECISION: {res.get('decision')}")
    print("-" * 72)
    cos_dc = res.get("cos_Gdense_Gcomp_median", float("nan"))
    cos_corr = res.get("cos_Gdense_Gcorr_median", float("nan"))
    act = res.get("Q_act_activation_capture_ratio_median", float("nan"))
    fid = res.get("recon_rel_error_dump_fidelity_max_drift", float("nan"))
    print(f"  H1 cos(G_dense,G_comp)>={COS_BENIGN}: "
          f"{'PASS' if cos_dc == cos_dc and cos_dc >= COS_BENIGN else 'FAIL/NA'} ({cos_dc:.4f})")
    print(f"  H1 cos(G_dense,G_corr) materially below cos(G_dense,G_comp): "
          f"{'see merger arm' if cos_corr == cos_corr else 'NA'} ({cos_corr:.4f})")
    print(f"  Q_act activation capture>={ACT_CAPTURE_MIN}: "
          f"{'PASS' if act == act and act >= ACT_CAPTURE_MIN else 'FAIL/NA'} ({act:.4f})")
    print(f"  fp32 dump fidelity (recon drift < 1e-3): "
          f"{'PASS' if fid == fid and fid < 1e-3 else 'FAIL/NA'} ({fid:.6f})")

    if args.emit:
        outp = run_dir / args.emit
        outp.write_text(json.dumps(res, indent=2))
        print(f"  wrote audit JSON -> {outp}")

    return 0 if "inconclusive" not in str(res.get("decision", "")) else 3


if __name__ == "__main__":
    raise SystemExit(main())
