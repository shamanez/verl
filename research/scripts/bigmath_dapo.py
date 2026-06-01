"""Preprocess gshasiri/Big-Math-RL-Verified-filtered to verl parquet format.

Mirrors examples/data_preprocess/gsm8k.py's row schema, but routes the reward to
verl's DAPO math verifier (data_source="math_dapo" -> math_dapo.compute_score),
which extracts the last \\boxed{...} from the model output and does a normalized
LaTeX/numeric comparison against the verified ground-truth answer. This is the
robust, established math-RL reward (GSM8K's "#### <num>" regex is numeric-only and
would mis-grade Big-Math's diverse answers).

Source columns: problem (str), answer (str, verified final answer), source,
domain (list), llama8b_solve_rate (float, difficulty proxy; lower = harder).
Splits: train (123,602), validation (6,506).

Output: <save_dir>/train.parquet + <save_dir>/test.parquet, consumed by the
vast_comm_eff launcher via DATA_DIR (it skips its gsm8k prep when both exist).

Usage (on the box):
  python3 research/scripts/bigmath_dapo.py --local_save_dir /root/data/bigmath \
    --train-cap 20000 --val-size 500 --seed 42
"""
from __future__ import annotations

import argparse
import os

import datasets

# DAPO/MATH-style instruction — the reward extracts the final \boxed{} answer.
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
DATA_SOURCE = "math_dapo"  # routes to verl.utils.reward_score.math_dapo.compute_score


def make_map_fn(split: str):
    def process_fn(example, idx):
        problem = example["problem"]
        answer = str(example["answer"]).strip()
        content = problem + " " + INSTRUCTION
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "problem": problem,
                "source": example.get("source"),
                "llama8b_solve_rate": example.get("llama8b_solve_rate"),
            },
        }

    return process_fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-dataset", default="gshasiri/Big-Math-RL-Verified-filtered")
    ap.add_argument("--local_save_dir", default="/root/data/bigmath")
    ap.add_argument("--train-cap", type=int, default=20000,
                    help="shuffle then keep this many train rows (>> steps*batch needed); 0 = keep all")
    ap.add_argument("--val-size", type=int, default=500,
                    help="shuffle then keep this many validation rows for periodic eval speed; 0 = keep all")
    ap.add_argument("--max-solve-rate", type=float, default=None,
                    help="if set, keep only train rows with llama8b_solve_rate <= this (harder subset)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = os.path.expanduser(args.local_save_dir)
    os.makedirs(out, exist_ok=True)

    ds = datasets.load_dataset(args.hf_dataset)
    train = ds["train"]
    # The dataset names its held-out split "validation".
    test = ds["validation"] if "validation" in ds else ds["test"]

    # Drop rows with empty/unusable answers.
    train = train.filter(lambda e: e["answer"] is not None and str(e["answer"]).strip() != "")
    test = test.filter(lambda e: e["answer"] is not None and str(e["answer"]).strip() != "")

    if args.max_solve_rate is not None:
        train = train.filter(
            lambda e: e.get("llama8b_solve_rate") is not None
            and float(e["llama8b_solve_rate"]) <= args.max_solve_rate
        )

    train = train.shuffle(seed=args.seed)
    test = test.shuffle(seed=args.seed)
    if args.train_cap and args.train_cap > 0:
        train = train.select(range(min(args.train_cap, len(train))))
    if args.val_size and args.val_size > 0:
        test = test.select(range(min(args.val_size, len(test))))

    train = train.map(make_map_fn("train"), with_indices=True, remove_columns=train.column_names)
    test = test.map(make_map_fn("test"), with_indices=True, remove_columns=test.column_names)

    train_path = os.path.join(out, "train.parquet")
    test_path = os.path.join(out, "test.parquet")
    train.to_parquet(train_path)
    test.to_parquet(test_path)
    print(f"bigmath_dapo: wrote train={len(train)} -> {train_path}")
    print(f"bigmath_dapo: wrote test={len(test)} -> {test_path}")
    print(f"bigmath_dapo: data_source={DATA_SOURCE}  instruction={INSTRUCTION!r}")
    # Sanity sample
    ex = train[0]
    print("SAMPLE prompt:", ex["prompt"][0]["content"][:160])
    print("SAMPLE ground_truth:", ex["reward_model"]["ground_truth"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
