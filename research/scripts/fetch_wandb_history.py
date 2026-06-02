"""Fetch a WandB run's FULL per-step history into a local metrics JSONL.

verl logs only to console+wandb (`trainer.logger=["console","wandb"]`), so the
per-step training-signal curves the analyst needs (`critic/score/mean`,
`actor/pg_loss`, `actor/grad_norm`, `actor/comm_eff/*`) live in WandB, not on the
box. This pulls a run's complete `scan_history()` and writes one JSON object per
logged step to `runs/EXP-<ID>/metrics/<name>.jsonl`, so the curve-match survives
box teardown and analyze.py / diff_against_baseline.py can read it.

Usage:
    # by display (experiment) name — picks the most-recent matching run
    python research/scripts/fetch_wandb_history.py \
        --project comm_eff_curve_match_m4 \
        --run-name curvematch_dense_ref_50step \
        --out runs/EXP-18/metrics/curvematch_dense_ref_50step.jsonl

    # by wandb run id (the 8-char hash) — unambiguous
    python research/scripts/fetch_wandb_history.py \
        --project comm_eff_curve_match_m4 --run-id abcd1234 \
        --out runs/EXP-18/metrics/foo.jsonl

Writes the JSONL (chronological, one row per `_step`) plus a sidecar
`<out>.config.json` holding the run's config + summary + state for provenance.
Each row carries `_step`, `step` (= training/global_step when present), and every
scalar metric logged at that step. Requires WANDB_API_KEY in the environment
(source ~/.config/verl-research/secrets.env first).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_run(api, entity: str, project: str, run_name: str | None, run_id: str | None):
    """Return a wandb Run, resolving by id (exact) or display name (latest match)."""
    if run_id:
        return api.run(f"{entity}/{project}/{run_id}")
    if not run_name:
        raise SystemExit("fetch_wandb_history: pass --run-name or --run-id")
    # Filter the project for runs whose display name matches; pick the most recent.
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": run_name}))
    if not runs:
        # Fall back to a manual scan (older wandb servers ignore display_name filter).
        runs = [r for r in api.runs(f"{entity}/{project}") if r.name == run_name]
    if not runs:
        raise SystemExit(
            f"fetch_wandb_history: no run named {run_name!r} in {entity}/{project}. "
            "Check the project/entity, or pass --run-id."
        )
    # created_at is an ISO string; lexical sort is chronological → last is newest.
    runs.sort(key=lambda r: getattr(r, "created_at", "") or "")
    return runs[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--entity", default="shamanework-pl")
    ap.add_argument("--run-name", default=None, help="WandB display/experiment name")
    ap.add_argument("--run-id", default=None, help="WandB run id (8-char hash); unambiguous")
    ap.add_argument("--out", required=True, type=Path, help="output JSONL path")
    ap.add_argument("--page-size", type=int, default=2000)
    args = ap.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        print("fetch_wandb_history: WANDB_API_KEY not set — "
              "`source ~/.config/verl-research/secrets.env` first", file=sys.stderr)
        return 2

    import wandb  # imported here so --help works without the package

    api = wandb.Api(timeout=60)
    run = _resolve_run(api, args.entity, args.project, args.run_name, args.run_id)
    print(f"fetch_wandb_history: run={run.name} id={run.id} state={run.state} "
          f"({args.entity}/{args.project})")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with args.out.open("w") as fh:
        # No `keys=` → full history (every scalar logged at each step). For a
        # 50-step run this is small; page_size keeps memory bounded for long runs.
        for row in run.scan_history(page_size=args.page_size):
            # Normalise a `step` field from the trainer's global_step when present
            # so analyze.py / the curve-match can index by training step directly.
            gs = row.get("training/global_step", row.get("_step"))
            if gs is not None and "step" not in row:
                try:
                    row["step"] = int(gs)
                except (TypeError, ValueError):
                    pass
            fh.write(json.dumps(row) + "\n")
            n += 1

    # Provenance sidecar: config + summary + state.
    sidecar = args.out.with_suffix(args.out.suffix + ".config.json")
    try:
        summary = {k: v for k, v in dict(run.summary).items()
                   if isinstance(v, (int, float, str, bool)) or v is None}
    except Exception:
        summary = {}
    sidecar.write_text(json.dumps({
        "run_name": run.name, "run_id": run.id, "state": run.state,
        "entity": args.entity, "project": args.project,
        "config": dict(run.config), "summary": summary,
    }, indent=2, default=str))

    print(f"fetch_wandb_history: wrote {n} rows → {args.out}")
    print(f"fetch_wandb_history: provenance → {sidecar}")
    if n == 0:
        print("fetch_wandb_history: WARNING — 0 rows (run may not have logged history yet)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
