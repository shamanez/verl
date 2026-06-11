#!/usr/bin/env python3
"""Pull full scalar histories from W&B for the EXP-27 3-run comparison.

Entity/project from env. WANDB_API_KEY must be in env (sourced from secrets).
Writes one CSV per run into the cwd, with a fixed column set (NaN where a metric
is absent for that run). Also prints a per-run step-count + final/best-val summary.
"""
import os
import sys
import csv

import wandb

ENTITY = "shamanework-pl"
PROJECT = "verl_compression_research"

RUNS = {
    "dense_5e2jpho9":        "5e2jpho9",
    "signed_ema_a0p5_1wulaelw": "1wulaelw",
    "exp27_damped_ef_qa6sll3h": "qa6sll3h",
    "ef_parent_r2_tilwe80t": "tilwe80t",
    "plain_u1v94opv":        "u1v94opv",
}

# Canonical metric set we want, mapped to a short CSV column name.
METRICS = {
    "actor/entropy": "entropy",
    "response_length/mean": "len_mean",
    "response_length/max": "len_max",
    "response_length/min": "len_min",
    "response_length/clip_ratio": "len_clip_ratio",
    "critic/score/mean": "score_mean",
    "critic/rewards/mean": "rewards_mean",
    "actor/pg_loss": "pg_loss",
    "actor/grad_norm": "grad_norm",
    "actor/pg_clipfrac": "pg_clipfrac",
    "actor/pg_clipfrac_lower": "pg_clipfrac_lower",
    "actor/ppo_kl": "ppo_kl",
    "val-core/openai/gsm8k/acc/mean@1": "val_acc_mean1",
    "actor/comm_eff/spectral/rel_change_mean": "spectral_rel_change_mean",
    "actor/comm_eff/anchor_backwards": "anchor_backwards",
    "actor/comm_eff/spectral_corrections": "spectral_corrections",
    "actor/comm_eff/clean_steps": "clean_steps",
    "actor/comm_eff/mask_applications": "mask_applications",
}

# Some runs may log val under a slightly different key; try alternates.
VAL_ALTS = [
    "val-core/openai/gsm8k/acc/mean@1",
    "val/test_score/openai/gsm8k",
    "val-core/openai/gsm8k/acc/mean",
    "val/openai/gsm8k/acc/mean@1",
]


def pull_run(api, run_id):
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    # Discover the actual keys present in this run's history summary.
    rows = []
    # scan_history with explicit keys + step gives every logged value.
    want_keys = list(METRICS.keys()) + VAL_ALTS + ["_step", "global_step"]
    want_keys = sorted(set(want_keys))
    for h in run.scan_history(keys=None, page_size=2000):
        rows.append(h)
    return run, rows


def coalesce_val(row):
    for k in VAL_ALTS:
        v = row.get(k)
        if v is not None:
            return v
    return None


def main():
    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    api = wandb.Api(timeout=60)

    summary_lines = []
    for label, run_id in RUNS.items():
        try:
            run, rows = pull_run(api, run_id)
        except Exception as e:
            print(f"FAILED {label} ({run_id}): {e}", file=sys.stderr)
            summary_lines.append(f"{label}\t{run_id}\tFETCH_FAILED\t{e}")
            continue

        # Determine step key: prefer global_step then _step.
        out_path = f"{label}.csv"
        cols = ["step"] + list(METRICS.values())
        max_step = -1
        n_train_rows = 0
        val_points = []  # (step, val)
        score_at = {}    # step -> score
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                step = r.get("global_step")
                if step is None:
                    step = r.get("_step")
                rowout = [step]
                has_train = False
                for wkey, _short in METRICS.items():
                    v = r.get(wkey)
                    if wkey == "val-core/openai/gsm8k/acc/mean@1" and v is None:
                        v = coalesce_val(r)
                    rowout.append(v)
                    if wkey == "actor/entropy" and v is not None:
                        has_train = True
                w.writerow(rowout)
                if step is not None and isinstance(step, (int, float)):
                    if step > max_step:
                        max_step = step
                if has_train:
                    n_train_rows += 1
                valv = coalesce_val(r)
                if valv is not None and step is not None:
                    val_points.append((step, valv))
                sc = r.get("critic/score/mean")
                if sc is not None and step is not None:
                    score_at[step] = sc

        # Build a summary line.
        best_val = max([v for _, v in val_points], default=None)
        final_val = val_points[-1][1] if val_points else None
        val_str = "; ".join(f"{int(s)}:{v:.4f}" for s, v in val_points) if val_points else "none"
        summary_lines.append(
            f"{label}\t{run_id}\tmax_step={max_step}\ttrain_rows={n_train_rows}"
            f"\tbest_val={best_val}\tfinal_val={final_val}\tvals=[{val_str}]"
        )
        print(f"wrote {out_path}  max_step={max_step} train_rows={n_train_rows} "
              f"best_val={best_val} final_val={final_val}")

    print("\n=== SUMMARY ===")
    for ln in summary_lines:
        print(ln)


if __name__ == "__main__":
    main()
