#!/usr/bin/env python3
"""gate93.py - compute the issue #93 round-gate metrics for one cell.

Pulls a WandB run's history and reports the section-1 gate quantities against
the #90 PRF-exact-k baseline card, so the analyst can write a PASS/REVISE/STOP
verdict. Computes numbers only; it does not decide the verdict.

Plain language (issue #93 vocabulary):
  - reference KL       = actor/kl_loss              (codec-view KL to the ref policy)
  - train-inference gap = rollout_corr/kl           (nats, sampler vs training view)
  - E[rho]             = rollout_corr/k3_kl - rollout_corr/kl + 1  (mean IS weight)
  - within-step identity = actor/ppo_kl             (must stay ~0)

Baseline card (#90 90-prf-exactk-600, from the issue body section 1 / run.json):
  reference KL at 100-120 : 0.156 - 0.203
  train-inference gap     : 14.24 nats  (gate: beat means < 10, target < 3)
  E[rho]                  : 0.0014
  reward slope            : 0.0032 / step  (parity gate: >= 90% = 0.00288)

Usage:
  gate93.py --run <name-or-id> [--project 93-long-horizon-stability]
            [--entity shamanework-pl] [--gate-lo 100] [--gate-hi 120]
            [--json out.json]

WANDB_API_KEY must be in the environment (source the laptop secrets first).
"""
import argparse, json, sys
import numpy as np

# --- #90 PRF-exact-k baseline card (section 1) ---
BASELINE = {
    "run": "90-prf-exactk-600",
    "ref_kl_gate_lo": 0.156,
    "ref_kl_gate_hi": 0.203,
    "gap": 14.24,
    "e_rho": 0.0014,
    "reward_slope": 0.0032,
}
GAP_BEAT = 10.0      # section 1: "train-inference gap < 10 nats"
GAP_TARGET = 3.0     # section 1: "(target < 3)"
SLOPE_PARITY = 0.90  # section 1: "reward slope >= 90 percent of baseline's"

KEYS = [
    "training/global_step",
    "actor/kl_loss", "actor/ppo_kl", "actor/entropy", "actor/grad_norm",
    "rollout_corr/kl", "rollout_corr/k3_kl",
    "critic/rewards/mean", "critic/score/mean",
    "response_length/mean",
    # codec-confinement counters (sanity; presence + non-degenerate)
    "actor/comm_eff/mask_applications",
    "actor/comm_eff/spectral_corrections",
    "actor/comm_eff/anchor_replay_fires",
    "actor/comm_eff/rank1_correction_bypass_ticks",
]


def _fit_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def _win(df, gs, lo, hi):
    """Rows whose global_step is in [lo, hi]; fall back to last ~20% if empty."""
    m = (gs >= lo) & (gs <= hi) & np.isfinite(gs)
    if m.sum() == 0:
        # cell shorter than the nominal window (e.g. a2 early stop) -> tail
        top = np.nanmax(gs)
        m = gs >= max(1.0, top - 20)
    return df[m], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="WandB run name or id")
    ap.add_argument("--project", default="93-long-horizon-stability")
    ap.add_argument("--entity", default="shamanework-pl")
    ap.add_argument("--gate-lo", type=int, default=100)
    ap.add_argument("--gate-hi", type=int, default=120)
    ap.add_argument("--reward-key", default="critic/rewards/mean")
    # Windowed-slope bounds for the registered bars. 0 means "use the gate window".
    ap.add_argument("--slope-lo", type=int, default=0)
    ap.add_argument("--slope-hi", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()

    # resolve run: try id path first, then match by name
    run = None
    try:
        run = api.run(f"{args.entity}/{args.project}/{args.run}")
    except Exception:
        for r in api.runs(f"{args.entity}/{args.project}"):
            if r.name == args.run:
                run = r
                break
    if run is None:
        print(f"ERROR: run '{args.run}' not found in {args.entity}/{args.project}", file=sys.stderr)
        sys.exit(2)

    # WandB's history(keys=...) returns ONLY rows where EVERY requested key is
    # present, so one key a run never logged silently empties the whole frame.
    # That bites on any run with the rollout correction off (the #90 incumbent
    # logs no rollout_corr/rollout_is_* at all). Pull each key on its own and
    # merge on global_step instead.
    # scan_history, NOT history: history() SAMPLES (it returned 13 rows for the
    # incumbent's 21-step gate window), which silently perturbs any slope fitted
    # against a registered threshold. scan_history is unsampled.
    import pandas as pd

    df = None
    for key in KEYS:
        if key == "training/global_step":
            continue
        rows = []
        try:
            for row in run.scan_history(keys=["training/global_step", key]):
                s, v = row.get("training/global_step"), row.get(key)
                if s is None or v is None:
                    continue
                rows.append((float(s), float(v)))
        except Exception:
            continue
        if not rows:
            continue
        part = pd.DataFrame(rows, columns=["training/global_step", key])
        df = part if df is None else df.merge(part, on="training/global_step", how="outer")
    if df is None or len(df) == 0:
        print("ERROR: no history rows", file=sys.stderr)
        sys.exit(2)
    df = df.sort_values("training/global_step").reset_index(drop=True)

    gs = df["training/global_step"].values.astype(float)
    max_step = int(np.nanmax(gs))

    win, _ = _win(df, gs, args.gate_lo, args.gate_hi)

    def gate_mean(key):
        if key not in win:
            return float("nan")
        v = win[key].values.astype(float)
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if len(v) else float("nan")

    def full_slope(key):
        if key not in df:
            return float("nan")
        return _fit_slope(gs, df[key].values.astype(float))

    # Windowed slope. The registered #93 bars are WINDOWED slopes (for example
    # "gap slope 61-120 <= +5.0e-4"), and a full-run fit is not a substitute:
    # these runs open with a large step-1/step-2 codec transient (a5's step-2
    # reference KL was 2.798 nats), so a full-run fit measures that transient
    # decaying and can even flip sign against the matched-window fit. Observed
    # on a5b at 20-45: full-run -0.00649/step versus windowed +0.000512/step.
    def slope_window(key, lo, hi):
        if key not in df:
            return float("nan")
        m = (gs >= lo) & (gs <= hi) & np.isfinite(df[key].values.astype(float))
        if int(m.sum()) < 3:
            return float("nan")
        return _fit_slope(gs[m], df[key].values.astype(float)[m])

    slope_lo = args.slope_lo if args.slope_lo > 0 else args.gate_lo
    slope_hi = args.slope_hi if args.slope_hi > 0 else args.gate_hi

    ref_kl = gate_mean("actor/kl_loss")
    ref_kl_slope = full_slope("actor/kl_loss")
    ref_kl_slope_win = slope_window("actor/kl_loss", slope_lo, slope_hi)
    gap = gate_mean("rollout_corr/kl")
    gap_slope = full_slope("rollout_corr/kl")
    gap_slope_win = slope_window("rollout_corr/kl", slope_lo, slope_hi)
    k3 = gate_mean("rollout_corr/k3_kl")
    e_rho = (k3 - gap + 1.0) if (np.isfinite(k3) and np.isfinite(gap)) else float("nan")

    reward_slope_full = full_slope(args.reward_key)
    reward_slope_gate = _fit_slope(
        win["training/global_step"].values.astype(float),
        win[args.reward_key].values.astype(float),
    ) if args.reward_key in win else float("nan")

    ppo_kl_max = float("nan")
    if "actor/ppo_kl" in df:
        v = np.abs(df["actor/ppo_kl"].values.astype(float))
        v = v[np.isfinite(v)]
        ppo_kl_max = float(np.max(v)) if len(v) else float("nan")

    entropy_last = gate_mean("actor/entropy")
    entropy_slope = full_slope("actor/entropy")
    grad_norm_max = float("nan")
    if "actor/grad_norm" in df:
        v = df["actor/grad_norm"].values.astype(float)
        v = v[np.isfinite(v)]
        grad_norm_max = float(np.max(v)) if len(v) else float("nan")

    # codec confinement counters (last observed)
    def last(key):
        if key not in df:
            return float("nan")
        v = df[key].values.astype(float)
        v = v[np.isfinite(v)]
        return float(v[-1]) if len(v) else float("nan")

    conf = {
        "mask_applications": last("actor/comm_eff/mask_applications"),
        "spectral_corrections": last("actor/comm_eff/spectral_corrections"),
        "anchor_replay_fires": last("actor/comm_eff/anchor_replay_fires"),
        "rank1_bypass_ticks": last("actor/comm_eff/rank1_correction_bypass_ticks"),
    }

    # --- section-1 gate flags (informational; the analyst decides) ---
    flags = {
        "ref_kl_le_baseline": bool(ref_kl <= BASELINE["ref_kl_gate_hi"]) if np.isfinite(ref_kl) else None,
        "gap_lt_10": bool(gap < GAP_BEAT) if np.isfinite(gap) else None,
        "gap_lt_3_target": bool(gap < GAP_TARGET) if np.isfinite(gap) else None,
        "reward_slope_parity": bool(reward_slope_full >= SLOPE_PARITY * BASELINE["reward_slope"]) if np.isfinite(reward_slope_full) else None,
        "ppo_kl_zero": bool(ppo_kl_max < 1e-6) if np.isfinite(ppo_kl_max) else None,
        "e_rho_gt_0p05": bool(e_rho > 0.05) if np.isfinite(e_rho) else None,
    }

    out = {
        "run": run.name,
        "run_id": run.id,
        "state": run.state,
        "max_step": max_step,
        "gate_window": [args.gate_lo, args.gate_hi],
        "gate_window_rows": int(len(win)),
        "metrics": {
            "ref_kl_gate": ref_kl,
            "ref_kl_slope_per_step": ref_kl_slope,
            "ref_kl_slope_window": ref_kl_slope_win,
            "slope_window": [slope_lo, slope_hi],
            "gap_gate": gap,
            "gap_slope_per_step": gap_slope,
            "gap_slope_window": gap_slope_win,
            "e_rho_gate": e_rho,
            "reward_slope_full": reward_slope_full,
            "reward_slope_gate": reward_slope_gate,
            "ppo_kl_max_abs": ppo_kl_max,
            "entropy_gate": entropy_last,
            "entropy_slope_per_step": entropy_slope,
            "grad_norm_max": grad_norm_max,
        },
        "confinement": conf,
        "baseline": BASELINE,
        "gate_flags": flags,
    }

    # --- human-readable ---
    print(f"=== gate93: {run.name} ({run.id}) state={run.state} max_step={max_step} ===")
    print(f"gate window: steps {args.gate_lo}-{args.gate_hi}  ({len(win)} rows"
          + ("" if len(win) else "; FELL BACK to tail") + ")")
    print()
    print(f"  reference KL (actor/kl_loss)   gate={ref_kl:.4f}   slope={ref_kl_slope:+.5f}/step (FULL RUN)"
          f"   [baseline {BASELINE['ref_kl_gate_lo']}-{BASELINE['ref_kl_gate_hi']}]")
    print(f"  train-inference gap (rc/kl)    gate={gap:.3f}    slope={gap_slope:+.5f}/step (FULL RUN)"
          f"   [baseline {BASELINE['gap']}; beat<{GAP_BEAT} target<{GAP_TARGET}]")
    print(f"  >>> WINDOWED slopes over {slope_lo}-{slope_hi} (USE THESE for the registered bars):")
    print(f"        reference KL slope        {ref_kl_slope_win:+.6f}/step")
    print(f"        gap slope                 {gap_slope_win:+.6f}/step")
    print(f"      The FULL RUN slopes above include the step-1/2 codec transient and")
    print(f"      can flip sign against these. Never score a registered bar off them.")
    print(f"  E[rho]=k3-kl+1                 gate={e_rho:.4f}   [baseline {BASELINE['e_rho']}]")
    print(f"  reward slope (full)            {reward_slope_full:+.5f}/step"
          f"   gate-win {reward_slope_gate:+.5f}/step   [baseline {BASELINE['reward_slope']}; 90%={SLOPE_PARITY*BASELINE['reward_slope']:.5f}]")
    print(f"  actor/ppo_kl max|.|            {ppo_kl_max:.2e}   (within-step identity; want ~0)")
    print(f"  entropy                        gate={entropy_last:.3f}   slope={entropy_slope:+.5f}/step")
    print(f"  grad_norm max                  {grad_norm_max:.3f}")
    print(f"  confinement counters (last)    {conf}")
    print()
    print("  section-1 gate flags (informational):")
    for k, v in flags.items():
        print(f"    {k:26s} {v}")
    print()
    print("  NOTE: for sr_quant arms actor/kl_loss carries an SR view-noise floor")
    print("        (~1.9 for a1); judge those on ref_kl_slope, not the absolute gate value.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
