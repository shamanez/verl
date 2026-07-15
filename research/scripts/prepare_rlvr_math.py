"""Preprocess the 5 RLVR-paper math datasets (issue #62) to verl parquet format.

Additive companion to research/scripts/bigmath_dapo.py: same row schema, same
routing key. Every dataset emits data_source="DigitalLearningGmbH/MATH-lighteval"
so the reward routes to math_reward.compute_score (last \\boxed{} over the full
response + is_equiv normalized comparison, plain float 0.0/1.0). This is the ONLY
route proven safe here — do NOT use "math_bigmath" (returns {"pred": None} ->
np.mean crash in process_validation_metrics) or "math_dapo" ("Answer:" regex
ignores \\boxed{}, scored all -1 on 2026-06-01). See reward_score/__init__.py.

The prompt always ends with the boxed instruction so the model is told to emit
\\boxed{}, which is what the reward verifier extracts. For datasets whose native
prompt carries a DIFFERENT answer-format instruction (DAPO-Math-17k: "...Answer:
$Answer..."), that boilerplate is stripped first so the two instructions do not
conflict.

Ground-truth per dataset (probed 2026-07-07):
  math        EleutherAI/hendrycks_math      answer = \\boxed{} span of `solution`  (7 subject configs, concat)
  numina-cot  AI-MO/NuminaMath-CoT           answer = \\boxed{} span of `solution`
  deepscaler  qingy2024/DeepScaleR-40k       answer = `solution`
                                               (bare short answer; carve test from train)
  skywork-or1 Skywork/Skywork-OR1-RL-Data    answer = json.loads(reward_model.ground_truth)[0]
                                               (split "math"; carve test from train)
  dapo-math   BytedTsinghua-SIA/DAPO-Math-17k answer = reward_model.ground_truth (plain str; train-only -> carve test)

Output: <save_dir>/train.parquet + <save_dir>/test.parquet, consumed by the
canonical comm-eff launcher via DATA_DIR. MATH is the default surface; when both
files exist the launcher consumes them instead of invoking its MATH fallback
prep. On-box DATA_DIR dirs: math, numina_cot, deepscaler, skywork_or1, dapo_math
(slug with '-' -> '_').

Usage (on the box, one dir per dataset):
  python3 research/scripts/prepare_rlvr_math.py --dataset math \
    --local_save_dir /root/data/math --train-cap 20000 --val-size 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import re

import datasets

from verl.utils.reward_score import math_reward

# Prompt instructs \boxed{} output; the reward verifier extracts the last \boxed{}.
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
# data_source is a ROUTING KEY, not a literal HF id (see module docstring).
DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"

# DAPO-Math-17k wraps its problem in an "Answer:"-format instruction that conflicts
# with our \boxed{} instruction. Strip the known prefix/suffix; leave the problem.
_DAPO_PREFIX_RE = re.compile(
    r"^\s*Solve the following math problem step by step\..*?answer to the problem\.\s*",
    re.DOTALL,
)
_DAPO_SUFFIX_RE = re.compile(
    r"\s*Remember to put your answer on its own line after\s*[\"']?Answer:?[\"']?\.?\s*$",
    re.DOTALL,
)


def _extract_boxed(solution) -> str | None:
    """Final answer = the last \\boxed{} span of a worked solution (nested-safe)."""
    if solution is None:
        return None
    boxed = math_reward.last_boxed_only_string(str(solution))
    if boxed is None:
        return None
    try:
        return math_reward.remove_boxed(boxed).strip()
    except Exception:
        return None


def _first_of_jsonlist(gt) -> str | None:
    """Skywork ground_truth is a JSON list string ('["15625"]'); DAPO is a plain
    string ('34'). Return the first element for a list, else the stripped string."""
    if gt is None:
        return None
    if isinstance(gt, list | tuple):
        return str(gt[0]).strip() if gt else None
    s = str(gt).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).strip()
        except Exception:
            pass
    return s


def _strip_dapo(content: str) -> str:
    content = _DAPO_PREFIX_RE.sub("", content)
    content = _DAPO_SUFFIX_RE.sub("", content)
    return content.strip()


# Per-dataset extraction spec. configs=[None] => single default config.
# test_split=None => no native held-out split; carve one from train.
SPECS = {
    "math": {
        "hf_id": "EleutherAI/hendrycks_math",
        "configs": [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ],
        "train_split": "train",
        "test_split": "test",
        "problem": lambda r: r["problem"],
        "answer": lambda r: _extract_boxed(r.get("solution")),
    },
    "numina-cot": {
        "hf_id": "AI-MO/NuminaMath-CoT",
        "configs": [None],
        "train_split": "train",
        "test_split": "test",
        "problem": lambda r: r["problem"],
        "answer": lambda r: _extract_boxed(r.get("solution")),
    },
    "deepscaler": {
        "hf_id": "qingy2024/DeepScaleR-40k",
        "configs": [None],
        "train_split": "train",
        "test_split": None,
        "problem": lambda r: r["question"],
        "answer": lambda r: str(r["solution"]).strip() if r.get("solution") is not None else None,
    },
    "skywork-or1": {
        "hf_id": "Skywork/Skywork-OR1-RL-Data",
        "configs": [None],
        "train_split": "math",  # non-standard splits: math | code
        "test_split": None,
        "problem": lambda r: r["prompt"][0]["content"],
        "answer": lambda r: _first_of_jsonlist(r.get("reward_model", {}).get("ground_truth")),
    },
    "dapo-math": {
        "hf_id": "BytedTsinghua-SIA/DAPO-Math-17k",
        "configs": [None],
        "train_split": "train",
        "test_split": None,
        "problem": lambda r: _strip_dapo(r["prompt"][0]["content"]),
        "answer": lambda r: _first_of_jsonlist(r.get("reward_model", {}).get("ground_truth")),
    },
}

SLUGS = list(SPECS.keys())


def build_row(spec, split: str, problem: str, answer: str, idx: int) -> dict:
    content = problem.strip() + " " + INSTRUCTION
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"split": split, "index": idx, "answer": answer, "hf_id": spec["hf_id"]},
    }


def map_example(spec, split: str):
    """Return a (example, idx) -> row|None mapper; None rows are dropped."""
    get_p = spec["problem"]
    get_a = spec["answer"]

    def fn(example, idx):
        try:
            problem = get_p(example)
            answer = get_a(example)
        except Exception:
            return None
        if problem is None or str(problem).strip() == "":
            return None
        if answer is None or str(answer).strip() == "":
            return None
        return build_row(spec, split, str(problem), str(answer), idx)

    return fn


def _load_concat(spec, split: str):
    """Load one split, concatenating multiple subject configs if present."""
    parts = []
    for cfg in spec["configs"]:
        if cfg is None:
            parts.append(datasets.load_dataset(spec["hf_id"], split=split))
        else:
            parts.append(datasets.load_dataset(spec["hf_id"], cfg, split=split))
    return datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def iter_mapped(slug: str, n: int, split: str = "train"):
    """Stream up to n mapped rows (no full download) — used by the CPU preflight.
    `split` is the logical split ("train"/"test"); train-only datasets ignore it
    and stream their sole split."""
    spec = SPECS[slug]
    native = spec["train_split"] if (split == "train" or spec["test_split"] is None) else spec["test_split"]
    out = []
    fn = map_example(spec, split)
    idx = 0
    for cfg in spec["configs"]:
        if len(out) >= n:
            break
        args = (spec["hf_id"],) if cfg is None else (spec["hf_id"], cfg)
        stream = datasets.load_dataset(*args, split=native, streaming=True)
        for ex in stream:
            row = fn(ex, idx)
            idx += 1
            if row is not None:
                out.append(row)
            if len(out) >= n:
                break
    return out


def _prep_split(spec, split_label, native_split, cap, seed):
    ds = _load_concat(spec, native_split)
    ds = ds.shuffle(seed=seed)
    if cap and cap > 0:
        ds = ds.select(range(min(cap, len(ds))))
    mapped = ds.map(map_example(spec, split_label), with_indices=True, remove_columns=ds.column_names)
    # map cannot drop rows by returning None -> None rows survive as all-None dicts;
    # filter them out on the mapped dataset.
    mapped = mapped.filter(lambda r: r.get("data_source") is not None)
    return mapped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=SLUGS)
    ap.add_argument("--local_save_dir", required=True)
    ap.add_argument("--train-cap", type=int, default=20000, help="shuffle then keep this many train rows; 0 = keep all")
    ap.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="held-out eval rows; for train-only datasets these are carved from train",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spec = SPECS[args.dataset]
    out = os.path.expanduser(args.local_save_dir)
    os.makedirs(out, exist_ok=True)

    if spec["test_split"] is not None:
        train = _prep_split(spec, "train", spec["train_split"], args.train_cap, args.seed)
        test = _prep_split(spec, "test", spec["test_split"], args.val_size, args.seed)
    else:
        # Train-only dataset: carve a held-out test split from the front, train from the rest.
        full = _load_concat(spec, spec["train_split"]).shuffle(seed=args.seed)
        val_n = min(args.val_size, len(full) // 2) if args.val_size else 0
        test_raw = full.select(range(val_n))
        train_raw = full.select(range(val_n, len(full)))
        if args.train_cap and args.train_cap > 0:
            train_raw = train_raw.select(range(min(args.train_cap, len(train_raw))))
        test = test_raw.map(map_example(spec, "test"), with_indices=True, remove_columns=test_raw.column_names).filter(
            lambda r: r.get("data_source") is not None
        )
        train = train_raw.map(
            map_example(spec, "train"), with_indices=True, remove_columns=train_raw.column_names
        ).filter(lambda r: r.get("data_source") is not None)

    train_path = os.path.join(out, "train.parquet")
    test_path = os.path.join(out, "test.parquet")
    train.to_parquet(train_path)
    test.to_parquet(test_path)
    print(f"prepare_rlvr_math[{args.dataset}]: train={len(train)} -> {train_path}")
    print(f"prepare_rlvr_math[{args.dataset}]: test={len(test)} -> {test_path}")
    print(f"prepare_rlvr_math[{args.dataset}]: data_source={DATA_SOURCE}")
    if len(train):
        ex = train[0]
        print("SAMPLE prompt:", ex["prompt"][0]["content"][:160])
        print("SAMPLE ground_truth:", ex["reward_model"]["ground_truth"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
