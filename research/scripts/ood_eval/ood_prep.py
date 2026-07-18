#!/usr/bin/env python3
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

"""Build in-domain + OOD benchmark eval parquets in the MATH test.parquet schema.

Ten benchmarks (in-domain held-out, distribution-shift, and competition math, plus
MMLU-STEM knowledge): gsm8k, math500, minerva, olympiad, amc23, aime24/25/26,
hmmt25, mmlu_stem.

Each benchmark becomes /root/data/ood/<bench>/{test.parquet, train.parquet->MATH}.
Reusing data_source="DigitalLearningGmbH/MATH-lighteval" routes scoring to
verl's math_reward boxed verifier -- byte-identical to training-time eval.
train.parquet is a symlink to the real MATH train only to satisfy the launcher's
existence guard; it is never read under trainer.val_only=True.
"""
import argparse
import os

import datasets

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"  # routing key -> math_reward.compute_score
OOD_ROOT = os.environ.get("OOD_ROOT", "/root/data/ood")
MATH_TRAIN = os.environ.get("MATH_TRAIN", "/root/data/math/train.parquet")


def row(problem, gold, bench, idx):
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": problem.strip() + " " + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(gold).strip()},
        "extra_info": {"split": "test", "index": idx, "bench": bench, "answer": str(gold).strip()},
    }


def mc_row(problem, gold_letter, bench, idx):
    # Self-contained multiple-choice instruction. The RELEX jinja suffix-replace
    # only fires on the MATH suffix, so this passes through unchanged.
    instr = "Put the letter of the correct answer within \\boxed{}."
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": problem.strip() + " " + instr}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": gold_letter},
        "extra_info": {"split": "test", "index": idx, "bench": bench, "answer": gold_letter},
    }


def gsm8k():
    d = datasets.load_dataset("openai/gsm8k", "main", split="test")
    return [row(r["question"], r["answer"].split("####")[-1].strip().replace(",", ""), "gsm8k", i)
            for i, r in enumerate(d)]


def math500():
    d = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [row(r["problem"], r["answer"], "math500", i) for i, r in enumerate(d)]


def minerva():
    d = datasets.load_dataset("math-ai/minervamath", split="test")
    return [row(r["question"], r["answer"], "minerva", i) for i, r in enumerate(d)]


def olympiad():
    d = datasets.load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    out = []
    for i, r in enumerate(d):
        fa = r["final_answer"]
        gold = fa[0] if isinstance(fa, (list, tuple)) and fa else str(fa)
        out.append(row(r["question"], gold, "olympiad", i))
    return out


def aime24():
    d = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return [row(r["Problem"], r["Answer"], "aime24", i) for i, r in enumerate(d)]


def amc23():
    d = datasets.load_dataset("math-ai/amc23", split="test")
    return [row(r["question"], r["answer"], "amc23", i) for i, r in enumerate(d)]


# --- paper-matched competition sets (RELEX Table 1 sources) ---
def aime25():
    d = datasets.load_dataset("TianHongZXY/AIME2025", split="test")
    return [row(r["problem"], r["answer"], "aime25", i) for i, r in enumerate(d)]


def aime26():
    d = datasets.load_dataset("MathArena/aime_2026", split="train")
    return [row(r["problem"], r["answer"], "aime26", i) for i, r in enumerate(d)]


def hmmt25():
    d = datasets.load_dataset("MathArena/hmmt_feb_2025", split="train")
    return [row(r["problem"], r["answer"], "hmmt25", i) for i, r in enumerate(d)]


STEM = ["abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning"]


def mmlu_stem(cap=40):
    out, idx = [], 0
    letters = "ABCD"
    for subj in STEM:
        d = datasets.load_dataset("cais/mmlu", subj, split="test")
        for j in range(min(cap, len(d))):
            r = d[j]
            q = r["question"] + "\n" + "\n".join(f"{letters[k]}) {c}" for k, c in enumerate(r["choices"]))
            out.append(mc_row(q, letters[r["answer"]], "mmlu_stem", idx))
            idx += 1
    return out


BENCHES = {"gsm8k": gsm8k, "math500": math500, "minerva": minerva, "olympiad": olympiad,
           "aime24": aime24, "amc23": amc23, "mmlu_stem": mmlu_stem,
           "aime25": aime25, "aime26": aime26, "hmmt25": hmmt25}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of benchmark names")
    args = ap.parse_args()
    names = args.only if args.only else list(BENCHES)
    for name in names:
        try:
            rows = BENCHES[name]()
            d = datasets.Dataset.from_list(rows)
            bdir = os.path.join(OOD_ROOT, name)
            os.makedirs(bdir, exist_ok=True)
            d.to_parquet(os.path.join(bdir, "test.parquet"))
            link = os.path.join(bdir, "train.parquet")
            if not os.path.exists(link):
                os.symlink(MATH_TRAIN, link)
            print(f"OK   {name:10s} {len(d):5d} rows -> {bdir}/test.parquet")
        except Exception as e:
            print(f"FAIL {name:10s} {repr(e)[:140]}")
