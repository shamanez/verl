"""Whole-tensor forecast-skill ablation for the two-circuit anchor projector.

Given a sparse set of exact RELEX checkpoints (downloaded by download_subset.py),
sweep the ablation grid and, for EVERY floating tensor, measure how well each
projection method forecasts the future exact checkpoint versus the "just reuse the
newest stale checkpoint" baseline. This is the WHOLE-TENSOR generalization of the
live harness's 16-sample causal probe (verl/workers/comm_eff/rank1_probe.py) — the
report explicitly flags that the live probe only scores 4 tensors x 16 coords, so
this closes that gap over all 338 tensors.

Ablation axes (see the plan HTML for the scientific motivation):
  --windows       W = number of source checkpoints (2,3,4,5,6,8,...)
  --horizons      h = gaps ahead to predict (1 = "current fast", 2 = "twice a fast", 3,...)
  --gap           G = step spacing between source checkpoints (default 10 = the cadence)
  --anchors       newest-exact step(s); each defines one (history -> target) instance
  --ranks         SVD rank r for the rank1_relex projector (1 = faithful; >1 = ablation)
  --strengths     alpha horizon strength (1.0 = default; <1 = damped)
  --methods       rank1_relex | relex_from_base | fixed_linear   (stale is always logged)

Emits: <out>/forecast_rows.csv (one row per combo x tensor), <out>/summary.json
(per-combo aggregates). Feed both to make_plots.py.

Time index convention: RELEX step index == the projector "tick". A combo uses
source steps [anchor-(W-1)G, ..., anchor-G, anchor] and target step anchor + hG.
Combos whose required steps are not on disk are skipped (and counted).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_projector as hp  # noqa: E402

# ------------------------------ tensor typing -------------------------------- #
TYPE_PATTERNS = [
    ("embed", r"embed_tokens"),
    ("q_proj", r"self_attn\.q_proj\.weight$"),
    ("k_proj", r"self_attn\.k_proj\.weight$"),
    ("v_proj", r"self_attn\.v_proj\.weight$"),
    ("o_proj", r"self_attn\.o_proj\.weight$"),
    ("gate_proj", r"mlp\.gate_proj\.weight$"),
    ("up_proj", r"mlp\.up_proj\.weight$"),
    ("down_proj", r"mlp\.down_proj\.weight$"),
    ("qkv_bias", r"self_attn\.(q|k|v)_proj\.bias$"),
    ("input_ln", r"input_layernorm\.weight$"),
    ("post_ln", r"post_attention_layernorm\.weight$"),
    ("final_norm", r"^model\.norm\.weight$"),
]


def tensor_type(name: str) -> str:
    for label, pat in TYPE_PATTERNS:
        if re.search(pat, name):
            return label
    return "other"


DECODER_2D = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# ------------------------------ checkpoint IO -------------------------------- #
class CkptStore:
    """Lazy, memory-mapped access to per-step safetensors + the base model."""

    def __init__(self, root: str):
        self.root = root
        self._handles: dict[int, list] = {}
        self._base_handles: list | None = None
        self._names_cache: dict[int, set] = {}

    def _open(self, step: int | None):
        d = os.path.join(self.root, "base_theta0" if step is None else f"global_step_{step}")
        files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors in {d}")
        return [safe_open(os.path.join(d, f), framework="pt", device="cpu") for f in files]

    def handles(self, step: int):
        if step not in self._handles:
            self._handles[step] = self._open(step)
        return self._handles[step]

    def base_handles(self):
        if self._base_handles is None:
            self._base_handles = self._open(None)
        return self._base_handles

    def has_step(self, step: int) -> bool:
        return os.path.isdir(os.path.join(self.root, f"global_step_{step}"))

    def names(self, step: int) -> set:
        if step not in self._names_cache:
            s = set()
            for h in self.handles(step):
                s.update(h.keys())
            self._names_cache[step] = s
        return self._names_cache[step]

    def get(self, step: int, name: str) -> torch.Tensor:
        for h in self.handles(step):
            if name in h.keys():
                return h.get_tensor(name).to(torch.float32)
        raise KeyError(f"{name} not in step {step}")


# ------------------------------ metrics ------------------------------------- #
def whole_tensor_metrics(projected: torch.Tensor, latest: torch.Tensor, actual: torch.Tensor) -> dict:
    """Whole-tensor port of rank1_probe.projection_sample_metrics (fp64 accum)."""
    p = projected.reshape(-1).to(torch.float64)
    s = latest.reshape(-1).to(torch.float64)
    a = actual.reshape(-1).to(torch.float64)
    proj_err = p - a
    stale_err = s - a
    pred_upd = p - s
    act_upd = a - s
    n = a.numel()
    proj_sse = float(torch.sum(proj_err * proj_err).item())
    stale_sse = float(torch.sum(stale_err * stale_err).item())
    pu = float(torch.sum(pred_upd * pred_upd).item())
    au = float(torch.sum(act_upd * act_upd).item())
    if stale_sse == 0.0:
        skill = 0.0 if proj_sse == 0.0 else -1.0
    else:
        skill = 1.0 - proj_sse / stale_sse
    if pu == 0.0 and au == 0.0:
        cos = 1.0
    elif pu == 0.0 or au == 0.0:
        cos = 0.0
    else:
        dot = float(torch.sum(pred_upd * act_upd).item())
        cos = max(-1.0, min(1.0, dot / math.sqrt(pu * au)))
    return dict(
        n=int(n),
        proj_sse=proj_sse,
        stale_sse=stale_sse,
        projected_rmse=math.sqrt(proj_sse / n),
        stale_rmse=math.sqrt(stale_sse / n),
        skill=skill,
        direction_cos=cos,
        pred_update_l2=math.sqrt(pu),
        actual_update_l2=math.sqrt(au),
    )


# ------------------------------ one combo ----------------------------------- #
def run_combo(store, names, *, anchor, W, h, gap, rank, strength, method, skip_embedding, max_numel):
    src_steps = [anchor - (W - 1 - i) * gap for i in range(W)]  # oldest..newest
    target = anchor + h * gap
    ticks = list(src_steps)

    rows = []
    for name in names:
        ttype = tensor_type(name)
        if skip_embedding and ttype == "embed":
            continue
        # snapshots at source steps + the actual target
        snaps = [store.get(s, name) for s in src_steps]
        if max_numel and snaps[0].numel() > max_numel:
            continue
        actual = store.get(target, name)
        latest = snaps[-1]

        if method == "rank1_relex":
            if not torch.is_floating_point(latest):
                continue
            proj, st = hp.project_rank1_tensor(snaps, ticks, target, strength=strength, rank=rank)
            evr, r2, fit_kind = st["evr"], st["r2"], st["fit_kind"]
        elif method == "relex_from_base":
            proj, st = hp.relex_from_base_project(snaps, ticks, target, rank=rank, strength=strength)
            evr, r2, fit_kind = float("nan"), float("nan"), f"from_base_r{rank}"
        elif method == "fixed_linear":
            # decoder 2-D matrices only; others take stale (matches harness scope)
            if ttype in DECODER_2D and latest.dim() == 2 and W >= 2:
                proj = hp.fixed_linear_project(snaps[-1], snaps[-2], h=h * gap, g=gap, strength=strength)
            else:
                proj = hp.stale_baseline(snaps)
            evr, r2, fit_kind = float("nan"), float("nan"), "fixed_linear"
        else:
            raise ValueError(method)

        m = whole_tensor_metrics(proj, latest, actual)
        rows.append(
            dict(
                anchor=anchor,
                W=W,
                horizon=h,
                gap=gap,
                rank=rank,
                strength=strength,
                method=method,
                target=target,
                history=";".join(map(str, src_steps)),
                tensor=name,
                ttype=ttype,
                numel=snaps[0].numel(),
                evr=evr,
                r2=r2,
                fit_kind=fit_kind,
                **m,
            )
        )
    return rows


def aggregate(rows) -> dict:
    """Energy-pooled + macro skill/cos, overall and per tensor-type."""

    def pool(subset):
        if not subset:
            return None
        proj = sum(r["proj_sse"] for r in subset)
        stale = sum(r["stale_sse"] for r in subset)
        pooled_skill = (1.0 - proj / stale) if stale > 0 else 0.0
        macro_skill = sum(r["skill"] for r in subset) / len(subset)
        macro_cos = sum(r["direction_cos"] for r in subset) / len(subset)
        win = sum(1 for r in subset if r["skill"] > 0) / len(subset)
        evrs = [r["evr"] for r in subset if not math.isnan(r["evr"])]
        r2s = [r["r2"] for r in subset if not math.isnan(r["r2"])]
        return dict(
            n_tensors=len(subset),
            pooled_skill=pooled_skill,
            macro_skill=macro_skill,
            macro_direction_cos=macro_cos,
            frac_tensors_win=win,
            evr_mean=(sum(evrs) / len(evrs) if evrs else None),
            r2_mean=(sum(r2s) / len(r2s) if r2s else None),
        )

    key = rows[0]
    out = {k: key[k] for k in ("anchor", "W", "horizon", "gap", "rank", "strength", "method", "target")}
    out["overall"] = pool(rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["ttype"]].append(r)
    out["by_type"] = {t: pool(v) for t, v in sorted(by_type.items())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt_dir", required=True, help="Temp dir from download_subset.py")
    ap.add_argument("--out_dir", required=True, help="Where to write CSV/JSON (temp is fine).")
    ap.add_argument("--windows", default="2,3,4,6,8")
    ap.add_argument("--horizons", default="1,2,3")
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--anchors", default="", help="Comma list of newest-exact steps; default = auto from disk.")
    ap.add_argument("--ranks", default="1")
    ap.add_argument("--strengths", default="1.0")
    ap.add_argument("--methods", default="rank1_relex,relex_from_base,fixed_linear")
    ap.add_argument("--skip_embedding", action="store_true", help="Skip embed_tokens (saves ~8GB RAM at W=8).")
    ap.add_argument("--max_numel", type=int, default=0, help="Skip tensors larger than this (0 = no cap).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    store = CkptStore(args.ckpt_dir)
    windows = [int(x) for x in args.windows.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]
    strengths = [float(x) for x in args.strengths.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    avail = sorted(
        int(m.group(1)) for m in (re.match(r"global_step_(\d+)$", d) for d in os.listdir(args.ckpt_dir)) if m
    )
    print(f"Available steps on disk: {avail}")
    avail_set = set(avail)
    if args.anchors:
        anchors = [int(x) for x in args.anchors.split(",")]
    else:
        anchors = list(avail)  # auto; combos with missing steps are skipped below

    # tensor name set = intersection across all available steps (stable across the grid)
    names = None
    for s in avail:
        names = store.names(s) if names is None else (names & store.names(s))
    names = sorted(names)
    print(f"Tracking {len(names)} tensors common to all steps.")

    all_rows, summaries, skipped = [], [], 0
    for method in methods:
        for W in windows:
            for h in horizons:
                for rank in ranks:
                    r_eff = 1 if method == "fixed_linear" else rank
                    for strength in strengths:
                        for anchor in anchors:
                            src = [anchor - (W - 1 - i) * args.gap for i in range(W)]
                            target = anchor + h * args.gap
                            need = set(src) | {target}
                            if not need <= avail_set:
                                skipped += 1
                                continue
                            rows = run_combo(
                                store,
                                names,
                                anchor=anchor,
                                W=W,
                                h=h,
                                gap=args.gap,
                                rank=r_eff,
                                strength=strength,
                                method=method,
                                skip_embedding=args.skip_embedding,
                                max_numel=args.max_numel,
                            )
                            if not rows:
                                continue
                            all_rows.extend(rows)
                            summ = aggregate(rows)
                            summaries.append(summ)
                            ov = summ["overall"]
                            print(
                                f"{method:16s} W={W} h={h} r={r_eff} a={strength} anchor={anchor} "
                                f"-> pooled_skill={ov['pooled_skill']:+.3f} macro_skill={ov['macro_skill']:+.3f} "
                                f"cos={ov['macro_direction_cos']:+.3f} win={ov['frac_tensors_win']:.2f} "
                                f"evr={ov['evr_mean']}"
                            )
    # write outputs
    csv_path = os.path.join(args.out_dir, "forecast_rows.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "grid": vars(args),
                "available_steps": avail,
                "n_rows": len(all_rows),
                "n_combos": len(summaries),
                "combos_skipped_missing_steps": skipped,
                "summaries": summaries,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {len(all_rows)} rows -> {csv_path}")
    print(f"Wrote {len(summaries)} combo summaries -> {os.path.join(args.out_dir, 'summary.json')}")
    print(f"Skipped {skipped} combos with missing steps.")


if __name__ == "__main__":
    main()
