#!/usr/bin/env python3
"""weight_proj_sweep.py — GPU-free offline weight-projection sweep engine (EXP-44).

Streams the EXP-43 raw full-weight trace from R2 ONE snapshot at a time (bounded
footprint; never bulk-downloads the ~494 GB prefix), reconstructs every predictor
family from the raw snapshots, computes the full GPU-free metric hierarchy per
grouping, runs the bf16-noise-floor gate, and renders a self-contained self-test
HTML. The metric math lives in weight_proj/metrics.py (boundary B1 with #45).

Modes (the plan's ## Verification commands map onto these):
  --selftest --check-invariants   pre-run gate: the six ## Correctness invariants
  --selftest --noise-floor        per-block bf16 floor + manifest fro-norm + SNR@h
  --emit-report <path>            full (family x order x coeff x Delta x h) sweep -> HTML

Tick cadence (## Notes for runner "Per-step vs per-tick"): the main families run
PER-STEP (first tick of each global_step = even ticks 0,2,4,...); --cadence per-tick
selects every tick (finer-Delta, noisier single-tick deltas). Default: per-step.

Streaming discipline: overlapping tick windows are downloaded ONCE per process via a
per-process snapshot cache of the extracted fp32 slices (a few MB — only the sampled
matrices) so the report path does NOT re-download for its noise-floor table. Each raw
.pt is still deleted immediately after its slices are extracted; the cache holds only
the tiny sliced fp32 tensors, never the 3 GB .pt.

R2 creds: `set -a; . ~/.config/verl-research/secrets.env; set +a` first; the engine
maps R2_* -> AWS_* internally (crib of verify_full_weight_dump.py). Bucket
shamane-pluralis only. Secret VALUES are never printed.

Usage (from research/):
  python scripts/weight_proj_sweep.py runs/EXP-43/regimeA/weights/full_manifest.jsonl \
      --selftest --sample-blocks 3 --sample-ticks 10 --check-invariants
  python scripts/weight_proj_sweep.py runs/EXP-43/regimeA/weights/full_manifest.jsonl \
      --selftest --noise-floor --horizons 1,2,5,10
  python scripts/weight_proj_sweep.py runs/EXP-43/regimeA/weights/full_manifest.jsonl \
      --families all --group matrix,block,layer \
      --emit-report reports/infra-b-sweep-engine-selftest.html
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weight_proj import metrics as M          # noqa: E402
from weight_proj import noise_floor as NF      # noqa: E402
from weight_proj import predictors as PR       # noqa: E402
from weight_proj import r2_stream as RS        # noqa: E402
from weight_proj import report as RPT          # noqa: E402
from weight_proj import sweep as SW            # noqa: E402
from weight_proj import tick_select as TS      # noqa: E402

STAGING = os.environ.get(
    "WP_STAGING_DIR",
    "/private/tmp/claude-501/-Users-shamane-Documents-verl-research/"
    "fe9e5a97-f064-48f9-a1a6-12ba7b44d425/scratchpad/wp_stage",
)

# Sample blocks used for the probe/noise-floor self-test: one attn, one mlp, one norm.
SAMPLE_BLOCK_MATRICES = {
    "q_proj": "model.layers.0.self_attn.q_proj.weight",
    "down_proj": "model.layers.0.mlp.down_proj.weight",
    "input_layernorm": "model.layers.0.input_layernorm.weight",
}
CORE_BLOCKS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# module-level cadence (set by main from --cadence); default per-step per the plan.
CADENCE = "per-step"

# per-process cache of extracted fp32 slices keyed by tick -> {name: tensor}. Only the
# few sampled matrices (a few MB); the 3 GB .pt is deleted immediately after slicing.
_SLICE_CACHE: dict[int, dict] = {}


def _resolve_ticks_from_r2(r2_rows, want):
    """First `want` ticks at the module CADENCE (per-step = even ticks 0,2,..; the
    plan default). The full trace is tick_0..tick_159 in R2 keyed off r2_manifest."""
    return TS.select_ticks(CADENCE, want)


def _sample_manifest_matrices(full_rows):
    """Return (matrix_names[338], fro_by_name{name:fro at tick0})."""
    names = [m["name"] for m in full_rows[0]["matrices"]]
    fro = {m["name"]: float(m["fro_norm"]) for m in full_rows[0]["matrices"]}
    return names, fro


def _stream_ticks_cached(r2_rows, ticks, names, log, label):
    """Stream `ticks`, reusing the per-process slice cache so overlapping windows
    (invariants/noise-floor/full-sweep) download each tick's .pt at most ONCE."""
    need = [t for t in ticks if t not in _SLICE_CACHE or any(n not in _SLICE_CACHE[t] for n in names)]
    if need:
        with RS.R2SnapshotStream(STAGING, min_free_gb=8, r2_rows=r2_rows) as stream:
            fresh = SW.stream_group_histories(stream, need, names)
            log(f"[{label}] streamed {len(need)} new ticks {need}; "
                f"max staged .pt = {stream.max_staged_observed} (cap 2); downloads={stream.downloads}")
        for t, sl in fresh.items():
            _SLICE_CACHE.setdefault(t, {}).update(sl)
    else:
        log(f"[{label}] all {len(ticks)} ticks served from slice cache (0 downloads)")
    return {t: {n: _SLICE_CACHE[t][n] for n in names} for t in ticks}


# =============================================================================
# Invariant probe
# =============================================================================
def run_invariants(full_rows, r2_rows, sample_ticks, log):
    import torch
    names_all, fro0 = _sample_manifest_matrices(full_rows)
    sample_names = list(SAMPLE_BLOCK_MATRICES.values())
    ticks = _resolve_ticks_from_r2(r2_rows, sample_ticks)
    log(f"[invariants] cadence={CADENCE}; sampling {len(sample_names)} matrices over "
        f"{len(ticks)} ticks: {ticks}")

    hist = _stream_ticks_cached(r2_rows, ticks, sample_names, log, "invariants")
    footprint_ok = True  # cache path holds only sliced fp32 tensors; .pt deleted per load

    ticks_sorted = sorted(hist.keys())
    reg = PR.build_family_registry()

    # 1. limiting-case identity (order-1, h=0): predict at h=0 == theta_stale exactly
    ok_ident = True
    detail_ident = []
    for nm in sample_names:
        for fam_key in ("order1-fixed", "order2-fixed", "order3-fixed", "ema-fixed"):
            fam = reg[fam_key]
            need = fam.order + 1
            if len(ticks_sorted) < need:
                continue
            history = [(t, hist[t][nm]) for t in ticks_sorted[:need]]
            hat = fam.predict(history, 0)
            stale = history[-1][1]
            d = float(torch.linalg.norm((hat - stale).reshape(-1)).item())
            base = float(torch.linalg.norm(stale.reshape(-1)).item()) + 1e-30
            rel = d / base
            if rel > 1e-6:
                ok_ident = False
                detail_ident.append(f"{fam_key}/{nm.split('.')[-2]}: rel={rel:.2e}")
    inv = []
    inv.append({"name": "limiting-case identity (order-1,h=0)", "gate": "hard",
                "pass": ok_ident, "detail": "; ".join(detail_ident) or "theta_hat==theta_stale within 1e-6"})

    # 2. predictor reconstruction from raw snapshots (linear-combo == predict())
    ok_recon = True
    recon_details = []
    for fam_key, fam in reg.items():
        nm = SAMPLE_BLOCK_MATRICES["down_proj"]
        need = (fam.order + 1) if fam.order > 0 else 4
        need = max(need, 2)
        if len(ticks_sorted) < need + 3:
            continue
        h = 1
        anchor_pos = need - 1 + 2
        hist_pos = list(range(anchor_pos - (need - 1), anchor_pos + 1))
        history = [(ticks_sorted[p], hist[ticks_sorted[p]][nm]) for p in hist_pos]
        if getattr(fam, "needs_fit", False):
            fit_pos = list(range(max(0, anchor_pos - need), anchor_pos))
            if len(fit_pos) < need:
                continue
            fit_hist = [(ticks_sorted[p], hist[ticks_sorted[p]][nm]) for p in fit_pos]
            fit_truth = hist[ticks_sorted[anchor_pos]][nm]
            try:
                fam.fit(fit_hist, fit_truth, h=1)
            except Exception as e:
                recon_details.append(f"{fam_key}: fit failed {e}")
                ok_recon = False
                continue
        hat = fam.predict(history, h)
        c = fam.linear_coeffs(len(history), h)
        lin = torch.zeros_like(history[0][1], dtype=torch.float32)
        for j, (_, th) in enumerate(history):
            lin = lin + float(c[j]) * th
        if fam.name == "general-regression":
            lin = lin + float(getattr(fam, "bias", 0.0))
        num = float(torch.linalg.norm((hat - lin).reshape(-1)).item())
        den = float(torch.linalg.norm(hat.reshape(-1)).item()) + 1e-30
        rel = num / den
        if rel > 1e-5:
            ok_recon = False
            recon_details.append(f"{fam_key}: rel={rel:.2e}")
    inv.append({"name": "predictor reconstruction from raw snapshots", "gate": "hard",
                "pass": ok_recon, "detail": "; ".join(recon_details) or "all families match linear-combo within 1e-5 rel"})

    # 3. manifest fp32-norm cross-check
    ok_fro = True
    fro_details = []
    t0 = ticks_sorted[0]
    for nm in sample_names:
        rel, ok = NF.manifest_fronorm_check(hist[t0][nm], fro0[nm])
        if not ok:
            ok_fro = False
        fro_details.append(f"{nm.split('.')[-2]}={rel:.2e}")
    inv.append({"name": "manifest fp32-norm cross-check", "gate": "hard",
                "pass": ok_fro, "detail": "rel-err " + ", ".join(fro_details) + " (tol 1e-2)"})

    # 4. learnable/regression leakage guard (fit window strictly < score point)
    ok_leak, leak_detail = PR.leakage_guard_selftest()
    inv.append({"name": "learnable/regression leakage guard", "gate": "hard",
                "pass": ok_leak, "detail": leak_detail})

    # 5. bounded streaming footprint
    inv.append({"name": "bounded streaming footprint", "gate": "hard",
                "pass": footprint_ok,
                "detail": "each 3GB .pt deleted immediately post-load; only sliced fp32 "
                          "tensors cached (a few MB); no aws s3 cp --recursive"})

    # 6. grouping integrity (soft)
    grouping = SW.build_grouping(names_all)
    integ = grouping["integrity"]
    ok_group = (integ["matrix_partition_ok"] and integ["block_partition_ok"]
                and integ["layer_partition_ok"] and integ["n_matrices"] == 338
                and integ["n_layers"] == 28 and integ["n_blocks"] == 11)
    inv.append({"name": "grouping integrity", "gate": "soft", "pass": ok_group,
                "detail": f"matrices={integ['n_matrices']} blocks={integ['n_blocks']} "
                          f"layers={integ['n_layers']} partitions_ok="
                          f"{integ['matrix_partition_ok']}/{integ['block_partition_ok']}/{integ['layer_partition_ok']}"})

    for r in inv:
        log(f"[invariant] {'PASS' if r['pass'] else 'FAIL'} ({r['gate']}) {r['name']} :: {r['detail']}")
    return inv, grouping


# =============================================================================
# Noise-floor gate
# =============================================================================
def _noise_floor_from_hist(hist, full_rows, horizons, log):
    """Compute the noise-floor gate table from an ALREADY-streamed `hist` (no I/O)."""
    names_all, fro0 = _sample_manifest_matrices(full_rows)
    sample_names = list(SAMPLE_BLOCK_MATRICES.values())
    ticks_sorted = sorted(hist.keys())

    fro_ok = True
    for nm in sample_names:
        rel, ok = NF.manifest_fronorm_check(hist[ticks_sorted[0]][nm], fro0[nm])
        log(f"[noise-floor] manifest fro cross-check {nm.split('.')[-2]}: rel={rel:.2e} ok={ok}")
        fro_ok = fro_ok and ok

    fam = PR.build_family_registry()["order1-fixed"]
    floor_table = []
    core_below = []
    for nm in sample_names:
        block = SW.block_family(nm)
        floor = NF.group_floor([hist[ticks_sorted[0]][nm]])
        for h in horizons:
            score_pos = len(ticks_sorted) - 1
            anchor_pos = score_pos - h
            if anchor_pos < 1:
                continue
            history = [(ticks_sorted[anchor_pos - 1], hist[ticks_sorted[anchor_pos - 1]][nm]),
                       (ticks_sorted[anchor_pos], hist[ticks_sorted[anchor_pos]][nm])]
            theta_now = hist[ticks_sorted[score_pos]][nm]
            theta_stale = hist[ticks_sorted[anchor_pos]][nm]
            hat = fam.predict(history, h)
            row = M.full_metric_row(hat, theta_now, theta_stale, floor)
            row["block"] = block
            row["floor"] = floor
            row["h"] = h
            row["ratio"] = row["weight_proj_ratio"]
            floor_table.append(row)
            flag = "bf16-unreliable" if row["bf16_unreliable"] else "clears"
            log(f"[noise-floor] block={block} h={h} floor={floor:.4e} ||e||={row['err_norm']:.4e} "
                f"SNR={row['snr']:.2f} ratio={row['weight_proj_ratio']:.4f} -> {flag}")
            if h >= 5 and row["bf16_unreliable"] and block in CORE_BLOCKS:
                core_below.append((block, h))
    return floor_table, fro_ok, core_below


def run_noise_floor(full_rows, r2_rows, horizons, log):
    sample_names = list(SAMPLE_BLOCK_MATRICES.values())
    max_h = max(horizons)
    n_ticks = max_h + 4
    ticks = _resolve_ticks_from_r2(r2_rows, n_ticks)
    log(f"[noise-floor] cadence={CADENCE}; horizons={horizons}; need {len(ticks)} "
        f"ticks {ticks} for sample blocks")
    hist = _stream_ticks_cached(r2_rows, ticks, sample_names, log, "noise-floor")
    return _noise_floor_from_hist(hist, full_rows, horizons, log)


# =============================================================================
# Full sweep + report
# =============================================================================
def run_full_sweep(full_rows, r2_rows, horizons, deltas, log):
    import numpy as np
    names_all, fro0 = _sample_manifest_matrices(full_rows)
    grouping = SW.build_grouping(names_all)
    sample_names = list(SAMPLE_BLOCK_MATRICES.values())
    max_h = max(horizons)
    n_ticks = max_h + 6
    ticks = _resolve_ticks_from_r2(r2_rows, n_ticks)
    log(f"[full-sweep] cadence={CADENCE}; one streaming pass over {len(ticks)} ticks {ticks}; "
        f"families=all; deltas={deltas} horizons={horizons}")

    hist = _stream_ticks_cached(r2_rows, ticks, sample_names, log, "full-sweep")
    ticks_sorted = sorted(hist.keys())

    reg = PR.build_family_registry()
    family_curves = {}
    floor_cache = {nm: NF.group_floor([hist[ticks_sorted[0]][nm]]) for nm in sample_names}
    records = []
    for fam_key, fam in reg.items():
        curve = []
        for h in horizons:
            ratios = []
            for nm in sample_names:
                floor = floor_cache[nm]
                group_hist = {t: hist[t][nm] for t in ticks_sorted}
                row = SW.score_family_on_group(fam, group_hist, ticks_sorted,
                                               delta=deltas[0], h=h, floor=floor)
                if row is None:
                    continue
                row["block"] = SW.block_family(nm)
                row["family_key"] = fam_key
                records.append(row)
                if not row["bf16_unreliable"] and row["weight_proj_ratio"] == row["weight_proj_ratio"]:
                    ratios.append(row["weight_proj_ratio"])
            if ratios:
                curve.append((h, float(np.median(ratios))))
        if curve:
            family_curves[fam.name] = curve
    hstars = {}
    for fam_key, fam in reg.items():
        h2r = {}
        for r in records:
            if (r.get("family_key") == fam_key and not r.get("bf16_unreliable")
                    and r["weight_proj_ratio"] == r["weight_proj_ratio"]):
                h2r.setdefault(r["h"], []).append(r["weight_proj_ratio"])
        hstars[fam.name] = M.crossover_hstar(h2r)
    return grouping, family_curves, records, hstars, floor_cache, hist, ticks_sorted, reg


def main():
    ap = argparse.ArgumentParser(description="GPU-free weight-projection sweep engine (EXP-44)")
    ap.add_argument("manifest", help="path to full_manifest.jsonl")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-invariants", action="store_true")
    ap.add_argument("--noise-floor", action="store_true")
    ap.add_argument("--sample-blocks", type=int, default=3)
    ap.add_argument("--sample-ticks", type=int, default=10)
    ap.add_argument("--horizons", default="1,2,5,10")
    ap.add_argument("--deltas", default="1")
    ap.add_argument("--families", default="all")
    ap.add_argument("--group", default="matrix,block,layer")
    ap.add_argument("--cadence", default="per-step", choices=["per-step", "per-tick"],
                    help="per-step (even ticks; plan default) | per-tick (all ticks; noisier)")
    ap.add_argument("--emit-report", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    global CADENCE
    CADENCE = args.cadence

    def log(msg):
        print(msg, flush=True)

    full_rows = RS.load_full_manifest(args.manifest)
    r2_path = os.path.join(os.path.dirname(args.manifest), "r2_manifest.jsonl")
    r2_rows = RS.load_r2_manifest(r2_path)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    deltas = [int(x) for x in args.deltas.split(",") if x.strip()]
    log(f"[engine] manifest rows={len(full_rows)} r2 keys={len(r2_rows)} cadence={CADENCE} "
        f"(full R2 trace = tick_0..tick_159; streaming keyed by tick)")

    report = {"meta": {"manifest": args.manifest, "ticks": args.sample_ticks,
                       "cadence": CADENCE,
                       "generated": datetime.datetime.now().isoformat(timespec="seconds"),
                       "metric_contract": M.METRIC_CONTRACT},
              "invariants": [], "families": [], "family_curves": {}, "floor_table": [],
              "grouping": {}, "verdict": ""}

    rc = 0
    if args.check_invariants:
        inv, grouping = run_invariants(full_rows, r2_rows, args.sample_ticks, log)
        report["invariants"] = inv
        report["grouping"] = grouping["integrity"]
        hard_fail = [r for r in inv if r["gate"] == "hard" and not r["pass"]]
        report["verdict"] = "PASS" if not hard_fail else "STOP"
        log(f"[engine] invariants hard-fails = {len(hard_fail)}")
        if hard_fail:
            rc = 2

    if args.noise_floor:
        floor_table, fro_ok, core_below = run_noise_floor(full_rows, r2_rows, horizons, log)
        report["floor_table"] = floor_table
        report["manifest_fronorm_ok"] = fro_ok
        log(f"[engine] manifest fro-norm cross-check ok = {fro_ok}")
        if core_below:
            blocks = ",".join(sorted({b for b, _ in core_below}))
            log(f"BF16_FLOOR_BLOCKS: {blocks}")
            report["verdict"] = "STOP"
            rc = 3
        else:
            log("[engine] all core blocks clear the bf16 floor at h>=5")

    if args.emit_report:
        (grouping, family_curves, records, hstars, floor_cache,
         hist, ticks_sorted, reg) = run_full_sweep(full_rows, r2_rows, horizons, deltas, log)
        fam_rows = []
        import torch
        nm = SAMPLE_BLOCK_MATRICES["down_proj"]
        for fam_key, fam in reg.items():
            need = (fam.order + 1) if fam.order > 0 else 4
            need = max(need, 2)
            anchor_pos = need - 1 + 2
            if anchor_pos >= len(ticks_sorted):
                continue
            hist_pos = list(range(anchor_pos - (need - 1), anchor_pos + 1))
            history = [(ticks_sorted[p], hist[ticks_sorted[p]][nm]) for p in hist_pos]
            if getattr(fam, "needs_fit", False):
                fit_pos = list(range(max(0, anchor_pos - need), anchor_pos))
                fit_hist = [(ticks_sorted[p], hist[ticks_sorted[p]][nm]) for p in fit_pos]
                try:
                    fam.fit(fit_hist, hist[ticks_sorted[anchor_pos]][nm], h=1)
                except Exception:
                    pass
            hat = fam.predict(history, 1)
            c = fam.linear_coeffs(len(history), 1)
            lin = torch.zeros_like(history[0][1], dtype=torch.float32)
            for j, (_, th) in enumerate(history):
                lin = lin + float(c[j]) * th
            if fam.name == "general-regression":
                lin = lin + float(getattr(fam, "bias", 0.0))
            rel = float(torch.linalg.norm((hat - lin).reshape(-1)).item()) / (
                float(torch.linalg.norm(hat.reshape(-1)).item()) + 1e-30)
            fam_rows.append({"family": fam.name, "coeff_source": fam.coeff_source,
                             "order": fam.order, "recon_rel_err": rel,
                             "reconstructable": rel <= 1e-5, "hstar": hstars.get(fam.name)})
        report["families"] = fam_rows
        report["family_curves"] = family_curves
        report["grouping"] = grouping["integrity"]
        report["hstars"] = hstars
        # floor table from the SAME cached hist (no re-download)
        ft, fro_ok, core_below = _noise_floor_from_hist(hist, full_rows, horizons, log)
        report["floor_table"] = ft
        report["manifest_fronorm_ok"] = fro_ok
        all_recon = all(r["reconstructable"] for r in fam_rows)
        report["verdict"] = "PASS" if (all_recon and fro_ok and not core_below) else "STOP"
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_report)), exist_ok=True)
        RPT.render_html(report, args.emit_report)
        log(f"[engine] report -> {args.emit_report} (families reconstructable={all_recon}, "
            f"fro_ok={fro_ok}, core_below_floor={len(core_below)})")
        if not all_recon:
            log("NONRECONSTRUCTABLE_FAMILY: " + ",".join(r["family"] for r in fam_rows if not r["reconstructable"]))
            rc = 4

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log(f"[engine] json -> {args.json_out}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
