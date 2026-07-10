"""Convert AIME-2024 (math-ai/aime24) to a verl VAL-ONLY parquet, boxed reward route (issue #63).

Operator directive 2026-07-08: validation uses EXACTLY math-ai/aime24 (HF), split
`test`, 30 problems, and the WandB metric keys must honestly read as AIME. The rows
therefore carry data_source="math-ai/aime24", which routes to math_reward (last
\\boxed{} span + is_equiv) via the entry added to verl/utils/reward_score/__init__.py
on the harness branch — NOT the `startswith("aime")` math_dapo branch (that extractor
ignores \\boxed{} and scores boxed responses 0/-1; verified in the #63 CPU gate).

Schema of math-ai/aime24 split=test (verified via datasets-server 2026-07-08):
  id: str · problem: str · solution: "\\boxed{<answer>}" · url: str
ground_truth = the bare answer inside the boxed span (e.g. "204").

Row schema mirrors research/scripts/prepare_rlvr_math.py build_row(). VAL ONLY.

Usage (on the box, from the run payload):
  python3 prepare_aime_boxed.py --local_save_dir /workspace/data/aime2024_boxed
"""
from __future__ import annotations

import argparse
import os
import re

import datasets

# Same instruction prepare_rlvr_math.py appends: tells the model to emit \boxed{}.
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
# HONEST name (operator directive): routes to math_reward via the harness-branch
# router entry; WandB keys become val-core/math-ai/aime24/*.
DATA_SOURCE = "math-ai/aime24"
HF_ID = "math-ai/aime24"

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _bare_answer(solution: str) -> str | None:
    """math-ai/aime24 `solution` is the boxed answer string, e.g. '\\boxed{204}'."""
    s = str(solution).strip()
    m = _BOXED_RE.search(s)
    if m:
        return m.group(1).strip()
    return s if s else None


def build_row(problem: str, answer: str, idx: int) -> dict:
    content = problem.strip() + " " + INSTRUCTION
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"split": "test", "index": idx, "answer": answer, "hf_id": HF_ID},
    }


def load_rows(limit: int | None = None) -> list[dict]:
    ds = datasets.load_dataset(HF_ID, split="test")
    out, seen = [], set()
    for ex in ds:
        problem = ex.get("problem")
        gt = _bare_answer(ex.get("solution") or "")
        if not problem or not str(problem).strip() or not gt:
            continue
        key = str(problem).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(build_row(str(problem), gt, len(out)))
        if limit and len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local_save_dir", required=True)
    ap.add_argument("--out-name", default="val.parquet", help="output parquet filename (val-only)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all; aime24 test has 30)")
    args = ap.parse_args()

    out = os.path.expanduser(args.local_save_dir)
    os.makedirs(out, exist_ok=True)
    rows = load_rows(args.limit if args.limit > 0 else None)
    if len(rows) != 30:
        print(f"prepare_aime_boxed: WARNING expected 30 problems, got {len(rows)}")
    if not rows:
        print("prepare_aime_boxed: FATAL no rows converted")
        return 1

    ds = datasets.Dataset.from_list(rows)
    val_path = os.path.join(out, args.out_name)
    ds.to_parquet(val_path)
    print(f"prepare_aime_boxed: val={len(ds)} -> {val_path}")
    print(f"prepare_aime_boxed: data_source={DATA_SOURCE}")
    ex = ds[0]
    print("SAMPLE prompt:", ex["prompt"][0]["content"][:180])
    print("SAMPLE ground_truth:", ex["reward_model"]["ground_truth"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
