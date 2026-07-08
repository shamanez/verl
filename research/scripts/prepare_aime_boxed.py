"""Convert AIME-2024 to a verl VAL-ONLY parquet with the boxed reward route (issue #63).

AIME-2024 is a 30-problem validation surface for distilled-reasoning RL runs. The
native `BytedTsinghua-SIA/AIME-2024` ships `data_source="math_dapo"` + an
"Answer: $Answer" prompt — that routes to the math_dapo "Answer:"-regex extractor
(verl/utils/reward_score/__init__.py), which IGNORES \\boxed{} and would score
every boxed R1-Distill response 0. So this converter REWRITES the row:

  * data_source -> "HuggingFaceH4/MATH-500"  (routes to math_reward: last \\boxed{}
    span + is_equiv, plain 0.0/1.0) — a math_reward alias DISTINCT from deepscaler's
    "DigitalLearningGmbH/MATH-lighteval" so the two-file val list emits TWO distinct
    val-core/<data_source>/reward/mean keys (no metric-key collision).
  * prompt      -> extra_info.raw_problem (the clean problem, no Answer boilerplate)
    + the \\boxed{} INSTRUCTION (same instruction prepare_rlvr_math.py uses).
  * ground_truth-> reward_model.ground_truth (plain string int, e.g. "540").

Row schema mirrors research/scripts/prepare_rlvr_math.py build_row() exactly.
Emits VAL ONLY (default <save_dir>/val.parquet) — AIME is a 30-row directional
eval surface, never a train set.

Sources (first that loads wins):
  BytedTsinghua-SIA/AIME-2024   split=train  (primary; extra_info.raw_problem + reward_model.ground_truth)
  Maxwell-Jia/AIME_2024         split=train  (fallback; Problem + Answer)

Usage (on the box, PREPARE-authored, runs from the run payload):
  python3 prepare_aime_boxed.py --local_save_dir /workspace/data/aime2024_boxed
"""
from __future__ import annotations

import argparse
import os

import datasets

# Same instruction prepare_rlvr_math.py appends: tells the model to emit \boxed{}.
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
# math_reward alias, DISTINCT from deepscaler's DigitalLearningGmbH/MATH-lighteval.
DATA_SOURCE = "HuggingFaceH4/MATH-500"


def build_row(problem: str, answer: str, idx: int, hf_id: str) -> dict:
    content = problem.strip() + " " + INSTRUCTION
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"split": "test", "index": idx, "answer": answer, "hf_id": hf_id},
    }


def _rows_bytedtsinghua(limit: int | None):
    hf_id = "BytedTsinghua-SIA/AIME-2024"
    ds = datasets.load_dataset(hf_id, split="train")
    out = []
    for idx, ex in enumerate(ds):
        extra = ex.get("extra_info") or {}
        problem = extra.get("raw_problem")
        rm = ex.get("reward_model") or {}
        gt = rm.get("ground_truth")
        if problem is None or gt is None or str(problem).strip() == "" or str(gt).strip() == "":
            continue
        out.append(build_row(str(problem), str(gt).strip(), idx, hf_id))
        if limit and len(out) >= limit:
            break
    return out


def _rows_maxwell(limit: int | None):
    hf_id = "Maxwell-Jia/AIME_2024"
    ds = datasets.load_dataset(hf_id, split="train")
    out = []
    for idx, ex in enumerate(ds):
        problem = ex.get("Problem")
        gt = ex.get("Answer")
        if problem is None or gt is None or str(problem).strip() == "":
            continue
        out.append(build_row(str(problem), str(gt).strip(), idx, hf_id))
        if limit and len(out) >= limit:
            break
    return out


def load_rows(limit: int | None = None) -> list[dict]:
    """Return converted AIME rows; primary source first, fallback on any failure."""
    try:
        rows = _rows_bytedtsinghua(limit)
        if rows:
            return rows
    except Exception as e:  # noqa: BLE001
        print(f"prepare_aime_boxed: primary source failed ({type(e).__name__}: {str(e)[:120]}); trying fallback")
    return _rows_maxwell(limit)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local_save_dir", required=True)
    ap.add_argument("--out-name", default="val.parquet", help="output parquet filename (val-only)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all; AIME-2024 has 30)")
    args = ap.parse_args()

    out = os.path.expanduser(args.local_save_dir)
    os.makedirs(out, exist_ok=True)
    rows = load_rows(args.limit if args.limit > 0 else None)
    if not rows:
        print("prepare_aime_boxed: FATAL no rows converted from any source")
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
