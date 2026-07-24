# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Issue #93 section-6 gate table for one finished cell.

Pulls the WandB history of a run in the 93-long-horizon-stability project and
prints/writes the pre-registered gate reads at the step-100..120 window:

  * reference KL         actor/kl_loss at matched step vs the #90 baseline
  * train-inference gap  rollout_corr/kl (nats): pass < 10, target < 3
  * E[rho]               rollout_corr/k3_kl - rollout_corr/kl + 1
  * reward slope         OLS slope of critic/score/mean over the window,
                         parity = slope >= 0.9 x baseline slope
  * ppo_kl identity      actor/ppo_kl == 0 through the window
  * confinement          comm_eff counters finite, no NaN rows, entropy > 0

Baseline card (#90 90-prf-exactk-600 at steps 100-120): kl_loss 0.156-0.203,
gap 14.24 nats, E[rho] 0.0014, reward slope 0.0032/step.

Usage:
  python gate93.py --run <wandb_run_name> [--entity shamanework-pl]
      [--project 93-long-horizon-stability] [--window 100 120] [--out gate.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BASELINE = {
    "kl_loss_lo": 0.156,
    "kl_loss_hi": 0.203,
    "gap_nats": 14.24,
    "e_rho": 0.0014,
    "reward_slope": 0.0032,
}

KEYS = [
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/entropy",
    "actor/grad_norm",
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
    "critic/score/mean",
    "training/global_step",
]


def ols_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def main() -> None:
    import wandb

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run display name, e.g. a1-srq-b1-sr")
    ap.add_argument("--entity", default="shamanework-pl")
    ap.add_argument("--project", default="93-long-horizon-stability")
    ap.add_argument("--window", type=int, nargs=2, default=[100, 120])
    ap.add_argument("--slope-window", type=int, nargs=2, default=[20, 120],
                    help="steps used for the reward-slope OLS fit")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    api = wandb.Api()
    runs = [r for r in api.runs(f"{args.entity}/{args.project}")
            if r.name == args.run]
    if not runs:
        raise SystemExit(f"no run named {args.run} in {args.entity}/{args.project}")
    run = sorted(runs, key=lambda r: r.created_at)[-1]

    hist = list(run.scan_history(keys=KEYS))
    rows = [h for h in hist if h.get("training/global_step") is not None]
    if not rows:
        raise SystemExit("history empty (run still initializing?)")
    step_of = lambda h: h["training/global_step"]  # noqa: E731
    rows.sort(key=step_of)
    last = step_of(rows[-1])

    lo, hi = args.window
    win = [h for h in rows if lo <= step_of(h) <= hi]
    if not win:
        win = rows[-10:]
        lo, hi = step_of(win[0]), step_of(win[-1])

    def series(key, rs):
        return [(step_of(h), h[key]) for h in rs
                if h.get(key) is not None and not (isinstance(h[key], float) and math.isnan(h[key]))]

    kl = series("actor/kl_loss", win)
    gap = series("rollout_corr/kl", win)
    k3 = {s: v for s, v in series("rollout_corr/k3_kl", win)}
    ppo = series("actor/ppo_kl", win)
    ent = series("actor/entropy", win)
    slo, shi = args.slope_window
    reward_rows = [(s, v) for s, v in series("critic/score/mean", rows) if slo <= s <= shi]

    kl_med = sorted(v for _, v in kl)[len(kl) // 2] if kl else float("nan")
    gap_med = sorted(v for _, v in gap)[len(gap) // 2] if gap else float("nan")
    rho = [k3[s] - v + 1 for s, v in gap if s in k3]
    rho_med = sorted(rho)[len(rho) // 2] if rho else float("nan")
    slope = ols_slope([s for s, _ in reward_rows], [v for _, v in reward_rows])
    ppo_max = max((abs(v) for _, v in ppo), default=float("nan"))
    ent_min = min((v for _, v in ent), default=float("nan"))

    gates = {
        "ref_kl_le_baseline": kl_med <= BASELINE["kl_loss_hi"],
        "gap_lt_10": gap_med < 10,
        "gap_lt_3_target": gap_med < 3,
        "reward_slope_parity": slope >= 0.9 * BASELINE["reward_slope"],
        "ppo_kl_zero": ppo_max == 0.0,
        "entropy_positive": ent_min > 0,
    }
    hard = ["ref_kl_le_baseline", "gap_lt_10", "reward_slope_parity", "ppo_kl_zero", "entropy_positive"]
    result = {
        "run": args.run, "wandb_id": run.id, "last_step": last,
        "window": [lo, hi],
        "ref_kl_median": kl_med, "gap_nats_median": gap_med,
        "e_rho_median": rho_med, "reward_slope_per_step": slope,
        "reward_points_in_fit": len(reward_rows),
        "ppo_kl_absmax": ppo_max, "entropy_min": ent_min,
        "baseline": BASELINE, "gates": gates,
        "hard_gates_pass": all(gates[g] for g in hard),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
