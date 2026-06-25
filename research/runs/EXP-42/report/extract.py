#!/usr/bin/env python3
"""EXP-42 data miner: extract per-cell metric time-series from training logs.
Single source of truth for the report agents. Read-only on logs."""
import re, json, os

RUN = "/Users/shamane/Documents/verl/research/runs"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Metric keys we want from the per-step metric line.
WANT = {
    "response_length_mean": "response_length/mean",
    "score_mean": "critic/score/mean",
    "entropy": "actor/entropy",
    "grad_norm": "actor/grad_norm",
    "clip_ratio": "response_length/clip_ratio",
}
VAL_KEY = "val-core/openai/gsm8k/acc/mean@1"

def strip(s):
    return ANSI.sub("", s)

def parse_metric_line(line):
    """Return (step:int, dict-of-key->float) for a metric line, else None.
    Metric lines look like: ...step:N - key:val - key:val - ..."""
    m = re.search(r"\bstep:(\d+)\b", line)
    if not m:
        return None
    # must be a metric line (has the ' - key:val' structure with our markers)
    if "global_seqlen" not in line and "actor/entropy" not in line:
        return None
    step = int(m.group(1))
    vals = {}
    # split on ' - '
    for tok in line.split(" - "):
        if ":" not in tok:
            continue
        k, _, v = tok.partition(":")
        k = k.strip()
        v = v.strip()
        vals[k] = v
    return step, vals

def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def extract_cell(path):
    """Extract all series for one cell from a log file."""
    series = {k: {} for k in WANT}      # logical_name -> {step: val}
    val = {}                            # step -> acc
    counters = {}                       # step -> {anchor_backwards, lookahead_fires}
    steps_seen = set()

    # lookahead diagnostics, keyed by optimizer tick (the lookahead 'step=')
    # We collect (a) phase per tick from the WARMING/mode line, (b) cos value+kind
    la_phase = {}     # tick -> "raw_stale" | "extrapolated"
    la_mode = {}      # tick -> mode string (fixed_linear / learned_*)
    la_strength = {}  # tick -> strength float
    la_cos = {}       # tick -> (value, cos_kind) where cos_kind in {theta[t-K], theta_hat}
    cfg_mode = None   # mode/strength from the resolved launch command line
    cfg_strength = None
    cfg_disabled = False

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = strip(raw)

            # --- resolved launch command: config fallback for mode/strength ---
            if cfg_mode is None and "lookahead_mode=" in line:
                mcfg = re.search(r"lookahead_mode=(\S+)", line)
                if mcfg:
                    cfg_mode = mcfg.group(1)
                    if cfg_mode == "disabled":
                        cfg_disabled = True
                mscfg = re.search(r"lookahead_strength=([-\d.eE]+)", line)
                if mscfg:
                    cfg_strength = to_float(mscfg.group(1))

            # --- per-step metric line ---
            pm = parse_metric_line(line)
            if pm:
                step, vals = pm
                steps_seen.add(step)
                for logical, key in WANT.items():
                    if key in vals:
                        fv = to_float(vals[key])
                        if fv is not None:
                            series[logical][step] = fv
                if VAL_KEY in vals:
                    fv = to_float(vals[VAL_KEY])
                    if fv is not None:
                        val[step] = fv
                # counters
                ab = vals.get("actor/comm_eff/anchor_backwards")
                lf = vals.get("actor/comm_eff/lookahead_fires")
                counters[step] = {
                    "anchor_backwards": to_float(ab),
                    "lookahead_fires": to_float(lf),
                }
                continue

            # --- lookahead diagnostic: phase/mode line ---
            if "[comm_eff][lookahead]" in line:
                # cos line?
                mc = re.search(r"anchor_align_cos=([-\d.eE]+)\s*\(cos\(([^)]*)\)", line)
                mstep = re.search(r"\bstep=(\d+)\b", line)
                if mc and mstep:
                    tick = int(mstep.group(1))
                    cosval = to_float(mc.group(1))
                    inner = mc.group(2)  # e.g. "g(theta[t-K]), g_live" or "g(theta_hat), g_live"
                    if "theta_hat" in inner:
                        kind = "theta_hat"
                    elif "theta[t-K]" in inner or "theta[t" in inner:
                        kind = "theta[t-K]"
                    else:
                        kind = inner
                    # keep first occurrence per tick (all workers identical)
                    if tick not in la_cos:
                        la_cos[tick] = (cosval, kind)
                    continue
                # phase/mode line
                if mstep:
                    tick = int(mstep.group(1))
                    if "WARMING" in line:
                        if tick not in la_phase:
                            la_phase[tick] = "raw_stale"
                    else:
                        mm = re.search(r"mode=(\S+)", line)
                        ms = re.search(r"strength=([-\d.eE]+)", line)
                        if mm and tick not in la_mode:
                            la_mode[tick] = mm.group(1)
                        if ms and tick not in la_strength:
                            la_strength[tick] = to_float(ms.group(1))
                        if "source_ticks" in line and tick not in la_phase:
                            la_phase[tick] = "extrapolated"
                continue

    # Build anchor_align_cos list: one entry per tick that has a cos value.
    anchor_cos = []
    for tick in sorted(la_cos):
        cosval, kind = la_cos[tick]
        # phase precedence: explicit phase line, else infer from cos kind
        phase = la_phase.get(tick)
        if phase is None:
            phase = "extrapolated" if kind == "theta_hat" else "raw_stale"
        anchor_cos.append({"tick": tick, "value": cosval, "phase": phase, "cos_kind": kind})

    # resolved mode/strength: prefer the diagnostic fire lines, fall back to
    # the resolved launch command (needed when a cell collapsed before the
    # first extrapolated fire, e.g. A75 truncated at step 27).
    mode = None
    strength = None
    if la_mode:
        mode = la_mode[sorted(la_mode)[-1]]
    elif cfg_mode and not cfg_disabled:
        mode = cfg_mode
    if la_strength:
        strength = la_strength[sorted(la_strength)[-1]]
    elif cfg_strength is not None:
        strength = cfg_strength

    last_step = max(steps_seen) if steps_seen else None

    # last counter values
    anchor_backwards = None
    lookahead_fires = None
    if counters:
        last = counters[max(counters)]
        anchor_backwards = last.get("anchor_backwards")
        lookahead_fires = last.get("lookahead_fires")

    # collapse onset: response_length/mean > 2x mean of first-25-global-step values,
    # sustained >=2 consecutive logged steps.
    rl = series["response_length_mean"]
    collapse_step = None
    collapse_threshold = None
    if rl:
        first25 = [v for s, v in sorted(rl.items()) if s <= 25]
        if first25:
            base = sum(first25) / len(first25)
            thr = 2.0 * base
            collapse_threshold = thr
            ordered = sorted(rl.items())
            for i in range(len(ordered) - 1):
                s0, v0 = ordered[i]
                s1, v1 = ordered[i + 1]
                if v0 > thr and v1 > thr:
                    collapse_step = s0
                    break

    def as_pairs(d):
        return [[s, round(d[s], 6)] for s in sorted(d)]

    return {
        "mode": mode,
        "strength": strength,
        "last_step": last_step,
        "collapse_onset_step": collapse_step,
        "collapse_threshold": round(collapse_threshold, 4) if collapse_threshold is not None else None,
        "series": {
            "response_length_mean": as_pairs(series["response_length_mean"]),
            "score_mean": as_pairs(series["score_mean"]),
            "entropy": as_pairs(series["entropy"]),
            "clip_ratio": as_pairs(series["clip_ratio"]),
            "grad_norm": as_pairs(series["grad_norm"]),
        },
        "val": [[s, round(val[s], 6)] for s in sorted(val)],
        "anchor_align_cos": anchor_cos,
        "anchor_backwards": anchor_backwards,
        "lookahead_fires": lookahead_fires,
    }

CELLS = {
    "A25": f"{RUN}/EXP-42/train_A25_internal.log",
    "A50": f"{RUN}/EXP-42/train_A50_internal.log",
    "A75": f"{RUN}/EXP-42/train_A75_internal.log",
    "L":   f"{RUN}/EXP-42/train_L_internal.log",
    "EXP41_ref_5over5": f"{RUN}/EXP-41/verl_train_A.log",
    "EXP41_alpha1p0":   f"{RUN}/EXP-41/verl_train_B.log",
}

out = {"cells": {}, "meta": {
    "surface": "1K GSM8K, Qwen2.5-1.5B-Instruct, 100 steps, test_freq=25, resp 1024",
    "delay_K": 20, "cadence": 20, "signed_ema_alpha": 0.25, "beta_anc": 0.50,
    "powersgd_r": 77, "n_rollouts": 8,
    "note_step_units": "metric 'step:' = global step; lookahead diagnostic 'step='/'tick' = optimizer tick (2 ticks per global step at batch128/mini64)",
    "EXP41_ref_val": {"25": 0.6998, "50": 0.7255, "75": 0.7233, "100": 0.7066},
    "EXP41_B_val100": 0.0478,
}}

for name, path in CELLS.items():
    if not os.path.exists(path):
        out["cells"][name] = {"ERROR": "log not found", "path": path}
        continue
    c = extract_cell(path)
    # EXP-41 B is the documented alpha=1.0 full-catch-up cell; its diagnostic
    # lines predate the strength= field, so set it from context (verdict.md).
    if name == "EXP41_alpha1p0" and c.get("strength") is None:
        c["strength"] = 1.0
        c["strength_source"] = "context (verdict.md: alpha=1.0 full catch-up); not printed in log"
    out["cells"][name] = c

os.makedirs(f"{RUN}/EXP-42/report", exist_ok=True)
with open(f"{RUN}/EXP-42/report/series.json", "w") as f:
    json.dump(out, f, indent=1)

# quick console summary
for name in CELLS:
    c = out["cells"][name]
    if "ERROR" in c:
        print(name, "ERROR", c["path"]); continue
    v25 = dict(c["val"]).get(25)
    print(f"{name}: mode={c['mode']} str={c['strength']} last={c['last_step']} "
          f"collapse={c['collapse_onset_step']} (thr={c['collapse_threshold']}) "
          f"val={c['val']} la_fires={c['lookahead_fires']} anc_bwd={c['anchor_backwards']} "
          f"n_cos={len(c['anchor_align_cos'])}")
