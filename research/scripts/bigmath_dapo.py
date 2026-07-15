"""Preprocess gshasiri/Big-Math-RL-Verified-filtered to verl parquet format.

This is a non-default dataset utility. It emits the same parquet row schema as
the current MATH preparation and uses
``DigitalLearningGmbH/MATH-lighteval`` reward routing: extract the last
``\\boxed{}`` answer and compare it with ``is_equiv``.

Source columns: problem (str), answer (str, verified final answer), source,
domain (list), llama8b_solve_rate (float, difficulty proxy; lower = harder).
Splits: train (123,602), validation (6,506).

Output: <save_dir>/train.parquet + <save_dir>/test.parquet, consumed by the
comm-eff launcher when DATA_DIR points at this directory. The prepared files are
used instead of the launcher's default MATH fallback prep.

Usage (on the box):
  python3 research/scripts/bigmath_dapo.py --local_save_dir /root/data/bigmath \
    --train-cap 20000 --val-size 500 --seed 42
"""
from __future__ import annotations

import argparse
import os

import datasets

# Prompt instructs \boxed{} output; reward verifier must extract \boxed{}.
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
# data_source is a ROUTING KEY (not the literal HF id). "DigitalLearningGmbH/MATH-lighteval"
# routes to math_reward.compute_score (last \boxed{} over the full solution + is_equiv
# normalised comparison), returning a plain float 0.0/1.0. This mirrors the proven
# min_rl_add recipe (examples/data_preprocess/math_dataset.py + examples/min_rl_trainer/
# run_llama3.2_1b_minrl.sh, exp "big-math-minirl-...-mathstyle").
# Do NOT use "math_bigmath": that custom entry returned {"pred": None}, and verl's
# process_validation_metrics does np.mean over every reward-extra key -> np.mean([None,...])
# -> "NoneType / int" TypeError that crashes val_before_train. A float return carries no
# "pred" key, so validation aggregation is safe (string preds are skipped; None is not).
DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"


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
