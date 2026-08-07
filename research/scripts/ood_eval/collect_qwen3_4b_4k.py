#!/usr/bin/env python3
"""Collect the Qwen3-4B-Base 4k pair's eval matrix into one results.json.

Reads the per-tag/per-benchmark ``train.log`` files ``ood_eval.sh`` leaves under
``$OOD_EVAL_ROOT/<tag>/<bench>/`` and, when they are reachable, the two training
logs, then writes:

  $OOD_EVAL_ROOT/results.json   machine-readable, consumed by report_qwen3_4b_4k.py
  stdout                        the same numbers as a plain table

Nothing here talks to the network or to WandB: the on-box logs are authoritative
(WandB drops the final step of a run on the atexit teardown race).
"""

from __future__ import annotations

import json
import os
import re
import sys

# Benchmark order and grouping. IN-DOMAIN is the MATH family the models were
# trained on; everything else is out of domain, with mmlu_stem last because it is
# the non-math capability-preservation anchor rather than another math set.
IN_DOMAIN = ["math500"]
OOD = ["gsm8k", "minerva", "olympiad", "amc23", "aime24", "aime25", "aime26", "hmmt25", "mmlu_stem"]
BENCHES = IN_DOMAIN + OOD

PRETTY = {
    "math500": "MATH-500",
    "gsm8k": "GSM8K",
    "minerva": "Minerva",
    "olympiad": "OlympiadBench",
    "amc23": "AMC23",
    "aime24": "AIME24",
    "aime25": "AIME25",
    "aime26": "AIME26",
    "hmmt25": "HMMT25",
    "mmlu_stem": "MMLU-STEM",
}
# Sampling protocol per benchmark, mirrored from eval_qwen3_4b_4k.sh's BENCHES.
PROTOCOL = {
    b: ("avg@8, t=0.7" if b in ("amc23", "aime24", "aime25", "aime26", "hmmt25") else "greedy mean@1") for b in BENCHES
}

ACC_RE = re.compile(r"acc/mean@[0-9]+['\"]?[: ]+([0-9.]+)")


def read_acc(root: str, tag: str, bench: str):
    """Last acc/mean@N in a bench log, or None when the bench never produced one."""
    path = os.path.join(root, tag, bench, "train.log")
    if not os.path.exists(path):
        return None
    try:
        with open(path, errors="replace") as fh:
            hits = ACC_RE.findall(fh.read())
    except OSError:
        return None
    return float(hits[-1]) if hits else None


# Training-log scraping. The console line carries `global_step:N` alongside the
# metric dict; a bare `step:` prefix is NOT the step number (it also matches
# timing_s/step), so anchor on global_step only.
STEP_RE = re.compile(r"global_step[\"']?[: ]+([0-9]+)")
VAL_RE = re.compile(r"val-core/[^\s:'\"]*acc/mean@[0-9]+['\"]?[: ]+([0-9.]+)")
METRIC_TMPL = r"{key}['\"]?[: ]+(-?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)"
TRAIN_KEYS = {
    "score": r"critic/score/mean",
    "response_length": r"response_length/mean",
    "response_clip_ratio": r"response_length/clip_ratio",
    "grad_norm": r"actor/grad_norm",
    "kl_loss": r"actor/kl_loss",
    "ppo_kl": r"actor/ppo_kl",
    "entropy": r"actor/entropy",
    "step_seconds": r"timing_s/step",
}


def read_training(path: str) -> dict:
    """Per-step training series plus the in-domain validation curve."""
    out = {"val": {}, "series": {k: {} for k in TRAIN_KEYS}, "log": path, "last_step": None}
    if not path or not os.path.exists(path):
        return out
    compiled = {name: re.compile(METRIC_TMPL.format(key=key)) for name, key in TRAIN_KEYS.items()}
    try:
        fh = open(path, errors="replace")
    except OSError:
        return out
    with fh:
        for line in fh:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            out["last_step"] = step if out["last_step"] is None else max(out["last_step"], step)
            v = VAL_RE.search(line)
            if v:
                out["val"][step] = float(v.group(1))
            for name, rx in compiled.items():
                hit = rx.search(line)
                if hit:
                    try:
                        out["series"][name][step] = float(hit.group(1))
                    except ValueError:
                        pass
    return out


def main() -> int:
    root = os.environ.get("OOD_EVAL_ROOT", "/workspace/runs/ood-eval-4b")
    steps = [int(s) for s in os.environ.get("STEPS", "500 400 300 200 100").split()]
    steps = sorted(steps)
    arms = {
        "commeff": os.environ.get("COMMEFF_LOG", "/workspace/runs/qwen3-4b-4k-commeff-500/train.log"),
        "dense": os.environ.get("DENSE_LOG", "/workspace/runs/qwen3-4b-4k-dense-500/train.log"),
    }

    tags = ["base"] + [f"{arm}{s}" for s in steps for arm in ("commeff", "dense")]
    table = {tag: {b: read_acc(root, tag, b) for b in BENCHES} for tag in tags}

    results = {
        "run_id": os.environ.get("RUN_ID", "qwen3-4b-4k-500"),
        "model": os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Base"),
        "steps": steps,
        "benches": BENCHES,
        "in_domain": IN_DOMAIN,
        "ood": OOD,
        "pretty": PRETTY,
        "protocol": PROTOCOL,
        "eval": table,
        "training": {arm: read_training(path) for arm, path in arms.items()},
    }
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    def fmt(v):
        return f"{v:10.4f}" if v is not None else f"{'.':>10s}"

    final = steps[-1]
    hdr = f"{'bench':12s} " + " ".join(f"{t:>10s}" for t in tags) + f"  ce{final}-d{final}"
    print(f"# {results['run_id']}  {results['model']}  (results.json in {root})")
    print(hdr)
    print("-" * len(hdr))
    for b in BENCHES:
        ce, de = table.get(f"commeff{final}", {}).get(b), table.get(f"dense{final}", {}).get(b)
        d = (ce - de) if (ce is not None and de is not None) else None
        marker = "  <- in-domain" if b in IN_DOMAIN else ""
        print(
            f"{b:12s} "
            + " ".join(fmt(table[t][b]) for t in tags)
            + f"  {f'{d:+.4f}' if d is not None else 'n/a':>9s}{marker}"
        )

    missing = [(t, b) for t in tags for b in BENCHES if table[t][b] is None]
    if missing:
        print(f"\n# {len(missing)} of {len(tags) * len(BENCHES)} cells missing (rerun eval_qwen3_4b_4k.sh to fill):")
        for t, b in missing[:20]:
            print(f"#   {t}/{b}")
        if len(missing) > 20:
            print(f"#   ... and {len(missing) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
