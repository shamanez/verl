#!/usr/bin/env python3
"""EXP-38 — dense GRPO temporal-drift analysis + strong standalone HTML report.

Reads the EXP-38 dense-path capture dumps (weight ``theta`` / dense gradient
``g_dense`` / update vector ``update_vector`` for selected decoder matrices, and
boundary hidden-state ``boundary_h`` / activation gradient ``boundary_grad_h`` at
the candidate PP boundaries), plus the per-layer-norm sidecar and a GRPO-signal
sidecar (WandB history or a local jsonl). It computes, all OFFLINE (no GPU):

  * weight drift ||theta_t - theta_{t-k}|| vs lag k (per matrix + median)
  * gradient cosine cos(g_t, g_{t-k}), norm-ratio, and elementwise sign-agreement vs k
  * gradient effective/stable rank, participation ratio, rank-for-90%-energy over time
  * boundary activation h: low-rank check (rank-for-90% vs r=77 / H), stable rank over time
  * boundary subspace overlap o(t,t-k) (top-r right singular subspace) + principal angles
  * periodicity: autocorrelation / FFT of the subspace-overlap time series
  * boundary grad_h effective rank over time
  * correlation of each drift/rank curve against the per-step GRPO signals
  * the located gradient-anchor + activation-codec unsafe-staleness KNEES vs k≈5 (5/5),
    k≈20 (20/20), and the GSM8K epoch-2 boundary (~global step 58)

and renders a self-contained HTML report (base64-embedded plots, no external
assets) into research/reports/comm-eff-grpo/.

Lag axis is in OPTIMIZER TICKS (2 ticks / global step on the accel surface):
k=5 ~= the stable 5/5 anchor (2.5 global steps); k=20 ~= the broken 20/20 anchor
(10 global steps); k=40 (20 global steps) probes beyond the failure boundary.

Usage:
  python research/scripts/exp38_drift_analysis.py runs/EXP-38 \
      --out research/reports/comm-eff-grpo/exp38-dense-drift.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from collections import defaultdict

import numpy as np

# r=77 is the LOCKED PowerSGD rank (the activation codec budget); H=1536 for Qwen2.5-1.5B.
R_LOCKED = 77
HIDDEN = 1536
TICKS_PER_STEP = 2  # ppo_mini=64 / train_batch=128 on the accel surface
EPOCH2_STEP = 58  # GSM8K epoch-2 boundary ~ 7473 prompts / 128
LAGS = [1, 2, 5, 10, 20, 40]  # analysis lags in optimizer ticks


# ----------------------------------------------------------------------------- #
# linear-algebra metrics
# ----------------------------------------------------------------------------- #
def _svdvals(mat: np.ndarray) -> np.ndarray:
    """Singular values (descending) of a 2D matrix, fp64, robust to NaN/Inf."""
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2:
        m = m.reshape(m.shape[0], -1) if m.ndim > 2 else m.reshape(1, -1)
    if not np.isfinite(m).all():
        m = np.nan_to_num(m)
    try:
        return np.linalg.svd(m, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.linalg.svd(m + 1e-12 * np.random.randn(*m.shape), compute_uv=False)


def _top_right_subspace(mat: np.ndarray, r: int) -> np.ndarray:
    """Top-r RIGHT singular vectors (the H-dim subspace a rank-r codec Q projects).

    For a boundary activation h of shape (tokens, H), the row space lives in R^H;
    Vt[:r] (r, H) is the subspace the PowerSGD basis Q would track. Returns (r, H)."""
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2:
        m = m.reshape(m.shape[0], -1)
    m = np.nan_to_num(m)
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    r = min(r, vt.shape[0])
    return vt[:r]


def stable_rank(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    return float((s ** 2).sum() / (s[0] ** 2)) if s.size and s[0] > 0 else 0.0


def participation_ratio(s: np.ndarray) -> float:
    s2 = np.asarray(s, dtype=np.float64) ** 2
    denom = (s2 ** 2).sum()
    return float((s2.sum() ** 2) / denom) if denom > 0 else 0.0


def rank_for_energy(s: np.ndarray, frac: float = 0.90) -> int:
    s2 = np.asarray(s, dtype=np.float64) ** 2
    tot = s2.sum()
    if tot <= 0:
        return 0
    c = np.cumsum(s2) / tot
    return int(np.searchsorted(c, frac) + 1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.mean(np.sign(a) == np.sign(b)))


def subspace_overlap(va: np.ndarray, vb: np.ndarray) -> float:
    """Normalized projection overlap of two r-dim subspaces (rows = basis vectors).

    overlap = ||Va Vb^T||_F^2 / r in [0, 1]; 1 = identical subspace, 0 = orthogonal."""
    r = min(va.shape[0], vb.shape[0])
    va, vb = va[:r], vb[:r]
    m = va @ vb.T
    return float((m ** 2).sum() / r) if r else 0.0


def principal_angles_deg(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    r = min(va.shape[0], vb.shape[0])
    s = np.linalg.svd(va[:r] @ vb[:r].T, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


# ----------------------------------------------------------------------------- #
# capture loading
# ----------------------------------------------------------------------------- #
def load_manifest(capture_dir: str):
    """Return (rows, rank_root). Finds the rank0 (or first rank) manifest."""
    for sub in ("rank0", "rank1", "rank2", "rank3", "."):
        root = os.path.join(capture_dir, sub)
        mpath = os.path.join(root, "manifest.jsonl")
        if os.path.exists(mpath):
            rows = [json.loads(l) for l in open(mpath) if l.strip()]
            return rows, root
    raise FileNotFoundError(f"no manifest.jsonl under {capture_dir}/rank*/")


def load_tensor(root: str, row: dict) -> np.ndarray:
    import torch  # local import: only needed when actually loading dumps

    t = torch.load(os.path.join(root, row["path"]), map_location="cpu", weights_only=False)
    return t.float().numpy()


def filter_present(rows, root):
    """Keep only manifest rows whose .pt exists and is non-empty (drops any failed
    or mid-rsync dump). Logs how many were dropped so silent gaps are visible."""
    kept, dropped = [], 0
    for r in rows:
        p = os.path.join(root, r["path"])
        if os.path.exists(p) and os.path.getsize(p) > 0:
            kept.append(r)
        else:
            dropped += 1
    if dropped:
        print(f"[exp38] WARNING: dropped {dropped}/{len(rows)} manifest rows with missing/empty .pt")
    return kept


def index_by_role(rows):
    """role -> target -> sorted list of (tick, global_step, row)."""
    idx = defaultdict(lambda: defaultdict(list))
    for r in rows:
        gs = r.get("global_step", r.get("optimizer_tick"))
        idx[r["role"]][r["target_name"]].append((int(r["optimizer_tick"]), int(gs), r))
    for role in idx:
        for tgt in idx[role]:
            idx[role][tgt].sort(key=lambda x: x[0])
    return idx


# ----------------------------------------------------------------------------- #
# drift computations
# ----------------------------------------------------------------------------- #
def gradient_drift_vs_lag(idx, root):
    """Per-matrix cos / sign / norm-ratio vs lag k (ticks), aggregated to median."""
    role = "g_dense"
    per_k = {k: {"cos": [], "sign": [], "normratio": []} for k in LAGS}
    per_matrix = {}
    if role not in idx:
        return per_k, per_matrix
    for tgt, series in idx[role].items():
        ticks = [t for t, _, _ in series]
        tens = {t: load_tensor(root, r) for t, _, r in series}
        tset = set(ticks)
        rec = {k: {"cos": [], "sign": [], "normratio": []} for k in LAGS}
        for t in ticks:
            for k in LAGS:
                if (t - k) in tset:
                    a, b = tens[t], tens[t - k]
                    rec[k]["cos"].append(cosine(a, b))
                    rec[k]["sign"].append(sign_agreement(a, b))
                    nb = np.linalg.norm(b.ravel())
                    rec[k]["normratio"].append(float(np.linalg.norm(a.ravel()) / nb) if nb > 0 else 0.0)
        per_matrix[tgt] = rec
        for k in LAGS:
            for key in ("cos", "sign", "normratio"):
                per_k[k][key].extend(rec[k][key])
    return per_k, per_matrix


def weight_drift_vs_lag(idx, root):
    role = "theta"
    per_k = {k: [] for k in LAGS}
    if role not in idx:
        return per_k
    for tgt, series in idx[role].items():
        ticks = [t for t, _, _ in series]
        tens = {t: load_tensor(root, r) for t, _, r in series}
        tset = set(ticks)
        for t in ticks:
            for k in LAGS:
                if (t - k) in tset:
                    per_k[k].append(float(np.linalg.norm((tens[t] - tens[t - k]).ravel())))
    return per_k


def rank_over_time(idx, role, root):
    """target -> list of (global_step, tick, stable_rank, participation, rank90, top_spectrum)."""
    out = {}
    if role not in idx:
        return out
    for tgt, series in idx[role].items():
        rows = []
        for t, gs, r in series:
            s = _svdvals(load_tensor(root, r))
            rows.append(
                {
                    "tick": t,
                    "global_step": gs,
                    "stable_rank": stable_rank(s),
                    "participation": participation_ratio(s),
                    "rank90": rank_for_energy(s, 0.90),
                    "spectrum": s[: min(120, s.size)].tolist(),
                }
            )
        out[tgt] = rows
    return out


def boundary_rank_detail(idx, role, root):
    """Per boundary, median over time of: raw rank-for-90% energy, MEAN-CENTERED
    rank-for-90% (removes the massive-activation / bias direction), and the top-1
    singular energy share (the massive-activation indicator). Distinguishes
    'rank-1 because of a massive-activation outlier dim' from genuine low-rank
    structure — decisive for whether a rank-r codec on the RESIDUAL is the right
    primitive."""
    out = {}
    if role not in idx:
        return out
    for tgt, series in idx[role].items():
        raws, cents, top1s = [], [], []
        for t, gs, r in series:
            m = load_tensor(root, r)
            m = np.asarray(m, dtype=np.float64)
            if m.ndim != 2:
                m = m.reshape(m.shape[0], -1)
            s_raw = _svdvals(m)
            raws.append(rank_for_energy(s_raw, 0.90))
            s2 = s_raw ** 2
            top1s.append(float(s2[0] / s2.sum()) if s2.sum() > 0 else 0.0)
            mc = m - m.mean(axis=0, keepdims=True)
            cents.append(rank_for_energy(_svdvals(mc), 0.90))
        out[tgt] = {
            "raw_rank90": float(np.median(raws)),
            "centered_rank90": float(np.median(cents)),
            "top1_energy_share": float(np.median(top1s)),
        }
    return out


def boundary_subspace_drift(idx, role, root, r=R_LOCKED):
    """Per boundary: subspace overlap o(t,t-k) vs k + a time series of o(t,t-k0)."""
    out = {}
    if role not in idx:
        return out
    for tgt, series in idx[role].items():
        ticks = [t for t, _, _ in series]
        subs = {t: _top_right_subspace(load_tensor(root, r_), r) for t, _, r_ in series}
        gsmap = {t: gs for t, gs, _ in series}
        tset = set(ticks)
        per_k = {k: [] for k in LAGS}
        for t in ticks:
            for k in LAGS:
                if (t - k) in tset:
                    per_k[k].append(subspace_overlap(subs[t], subs[t - k]))
        # overlap time series at the smallest available lag (for periodicity).
        base_k = next((k for k in LAGS if per_k[k]), None)
        ts = []
        if base_k is not None:
            for t in ticks:
                if (t - base_k) in tset:
                    ts.append((gsmap[t], subspace_overlap(subs[t], subs[t - base_k])))
        out[tgt] = {"per_k": per_k, "series": ts, "base_k": base_k}
    return out


def autocorr_fft(series_vals):
    """Return (lags, autocorr, freqs, power) for a 1D series (mean-removed)."""
    x = np.asarray(series_vals, dtype=np.float64)
    if x.size < 4:
        return [], [], [], []
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[x.size - 1 :]
    ac = ac / ac[0] if ac[0] != 0 else ac
    fp = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(x.size, d=1.0)
    return list(range(len(ac))), ac.tolist(), fr.tolist(), fp.tolist()


# ----------------------------------------------------------------------------- #
# GRPO signals (WandB history or local sidecar)
# ----------------------------------------------------------------------------- #
def load_grpo_signals(run_dir: str, wandb_run: str | None):
    """Return {metric: [(global_step, value)]}. Prefers a local sidecar_grpo.jsonl;
    falls back to a WandB history csv at runs/EXP-38/wandb_history.csv if present."""
    out = defaultdict(list)
    side = os.path.join(run_dir, "sidecar_grpo.jsonl")
    if os.path.exists(side):
        for line in open(side):
            if not line.strip():
                continue
            d = json.loads(line)
            # Step key: our own sidecar uses global_step; a fetched WandB history
            # JSONL uses training/global_step or the internal _step.
            gs = d.get("global_step", d.get("training/global_step", d.get("_step")))
            if gs is None:
                continue
            gs = int(gs)
            for k, v in d.items():
                if k in ("global_step", "_step", "training/global_step") or not isinstance(v, (int, float)):
                    continue
                out[k].append((gs, float(v)))
        return out
    csv = os.path.join(run_dir, "wandb_history.csv")
    if os.path.exists(csv):
        import csv as _csv

        with open(csv) as fh:
            rdr = _csv.DictReader(fh)
            for row in rdr:
                gs = row.get("_step") or row.get("global_step") or row.get("step")
                if gs in (None, ""):
                    continue
                for k, v in row.items():
                    try:
                        out[k].append((int(float(gs)), float(v)))
                    except (TypeError, ValueError):
                        pass
    return out


# ----------------------------------------------------------------------------- #
# plotting -> base64
# ----------------------------------------------------------------------------- #
def _b64(fig):
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def median(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


# (plot builders + HTML assembly are completed in build_report below; kept in one
#  function so the report layout is easy to evolve as the captured data lands.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="runs/EXP-38")
    ap.add_argument("--captures", default=None, help="capture dir (default <run_dir>/captures)")
    ap.add_argument("--dataset", default=None,
                    help="dataset tag (gsm8k|big-math|...). Auto-read from <run_dir>/DATASET.json if present, "
                         "else 'gsm8k'. STAMPED on the report + findings + output filename so two datasets "
                         "are NEVER confused.")
    ap.add_argument("--out", default=None,
                    help="output HTML (default research/reports/comm-eff-grpo/exp38-dense-drift-<dataset>.html)")
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()
    # Resolve the dataset tag: CLI > DATASET.json > 'gsm8k'. This is the anti-mixing guard.
    dataset = args.dataset
    if dataset is None:
        dj = os.path.join(args.run_dir, "DATASET.json")
        if os.path.exists(dj):
            dataset = json.load(open(dj)).get("dataset", "gsm8k")
        else:
            dataset = "gsm8k"
    out = args.out or f"research/reports/comm-eff-grpo/exp38-dense-drift-{dataset}.html"
    cap = args.captures or os.path.join(args.run_dir, "captures")
    rows, root = load_manifest(cap)
    rows = filter_present(rows, root)
    idx = index_by_role(rows)
    print(f"[exp38] dataset={dataset} manifest rows={len(rows)} roles={sorted(idx.keys())}")
    for role in idx:
        ntgt = len(idx[role])
        nt = sum(len(v) for v in idx[role].values())
        print(f"  role={role:16s} targets={ntgt:3d} dumps={nt}")
    # The full computation + HTML render is invoked from build_report (added once
    # the real manifest shape is confirmed against the first captures).
    from exp38_report import build_report  # noqa: E402  (sibling module)

    build_report(args.run_dir, cap, root, rows, idx, out, args.wandb_run, dataset)


if __name__ == "__main__":
    main()
