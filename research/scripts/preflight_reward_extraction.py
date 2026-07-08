"""CPU correctness gate for issue #62 — the MONEY GATE (no GPU until this is green).

Proves the full prompt -> generation -> extraction -> ground_truth reward loop is
correct for every new model x dataset BEFORE any GPU is provisioned. Uses NO GPU
and NO model generation: the "response" side of each check is synthesized so the
reward route (default_compute_score on the locked "DigitalLearningGmbH/MATH-lighteval"
key -> math_reward.compute_score) is exercised deterministically on CPU.

Emits three headline counts the plan's verification command asserts:
  tokenizers_ok : #models whose AutoTokenizer + apply_chat_template load and keep the
                  \\boxed{} instruction (target 8 = 5 test + 3 integrate-only).
  schemas_ok    : #datasets whose prep mapping yields the 4 Verl fields, carries the
                  boxed instruction in prompt[-1].content, and round-trips through a
                  train+test parquet write/read (target 5).
  pairs_pass    : #(test-model x dataset) pairs passing checks 1-5 (target 25 = 5x5).

Checks per pair (model m, dataset d), on a REAL mapped row of d:
  1  apply_chat_template(m, row.prompt) preserves the \\boxed{} instruction (model axis).
  2  row.ground_truth is non-empty (dataset answer extraction succeeded).
  3  nested balanced-brace: last_boxed_only_string / remove_boxed pull the correct span
     from a nested \\boxed{\\frac{a}{b}}.
  4  a synthesized response containing \\boxed{<gt>} extracts back to gt (is_equiv).
  5  default_compute_score(route, correct_resp, gt) == 1.0 AND (route, wrong_resp, gt) == 0.0.

Usage (laptop, CWD=research/):
  python scripts/preflight_reward_extraction.py --emit runs/62-rlvr-models-datasets/preflight.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import datasets
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from verl.utils.reward_score import default_compute_score, math_reward

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_rlvr_math as prep  # noqa: E402

ROUTE = prep.DATA_SOURCE
INSTR_MARK = "\\boxed"

# 5 test models (GPU-smoked) + 3 integrate-only (CPU load only) = 8 tokenizers.
TEST_MODELS = {
    "qwen25-math-1p5b": "Qwen/Qwen2.5-Math-1.5B",
    "qwen3-1p7b-base": "Qwen/Qwen3-1.7B-Base",
    "r1-distill-qwen-1p5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "open-nemotron-1p5b": "nvidia/OpenReasoning-Nemotron-1.5B",
    "qwen3-4b-base": "Qwen/Qwen3-4B-Base",
}
INTEGRATE_ONLY = {
    "r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "qwen3-8b-base": "Qwen/Qwen3-8B-Base",
}
ALL_MODELS = {**TEST_MODELS, **INTEGRATE_ONLY}
DATASETS = prep.SLUGS  # math, numina-cot, deepscaler, skywork-or1, dapo-math

SAMPLE_N = 5           # real rows streamed per dataset
NESTED = r"reasoning \boxed{\frac{a}{b}} tail"   # nested-brace probe
WRONG_ANSWER = "-98765.4321"   # sentinel guaranteed non-equivalent to any real gt


def load_tokenizers():
    out = {}
    for slug, hid in ALL_MODELS.items():
        rec = {"hf_id": hid, "loads": False, "apply_ok": False, "boxed_survives": False,
               "error": None}
        try:
            tok = AutoTokenizer.from_pretrained(hid, trust_remote_code=True)
            rec["loads"] = True
            txt = tok.apply_chat_template(
                [{"role": "user", "content": "What is 2+2? " + prep.INSTRUCTION}],
                tokenize=False, add_generation_prompt=True)
            rec["apply_ok"] = bool(txt and txt.strip())
            rec["boxed_survives"] = INSTR_MARK in txt
            rec["_tok"] = tok
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        out[slug] = rec
    return out


def validate_schema(slug):
    """Stream a small sample, map via prep, check 4 Verl fields + boxed instruction,
    then round-trip a tiny train+test parquet. Returns (ok, detail, sample_rows)."""
    detail = {"slug": slug, "train_rows": 0, "test_rows": 0, "fields_ok": False,
              "boxed_in_prompt": False, "parquet_roundtrip": False, "error": None}
    try:
        train = prep.iter_mapped(slug, SAMPLE_N, split="train")
        test = prep.iter_mapped(slug, SAMPLE_N, split="test")
        detail["train_rows"] = len(train)
        detail["test_rows"] = len(test)
        if not train or not test:
            detail["error"] = "empty sample"
            return False, detail, train
        req = ("data_source", "prompt", "ability", "reward_model", "extra_info")

        def row_ok(r):
            if any(r.get(k) in (None, "", []) for k in req):
                return False
            gt = r["reward_model"].get("ground_truth")
            if gt in (None, ""):
                return False
            return isinstance(r["prompt"], list) and r["prompt"][-1]["content"].strip() != ""

        detail["fields_ok"] = all(row_ok(r) for r in train + test)
        detail["boxed_in_prompt"] = all(INSTR_MARK in r["prompt"][-1]["content"] for r in train + test)

        with tempfile.TemporaryDirectory() as td:
            for label, rows in (("train", train), ("test", test)):
                p = os.path.join(td, f"{label}.parquet")
                datasets.Dataset.from_list(rows).to_parquet(p)
                back = pq.read_table(p).to_pylist()
                assert len(back) == len(rows) and back[0]["reward_model"]["ground_truth"]
            detail["parquet_roundtrip"] = True

        ok = detail["fields_ok"] and detail["boxed_in_prompt"] and detail["parquet_roundtrip"]
        return ok, detail, train
    except Exception as e:
        detail["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return False, detail, []


def check_pair(tok, row):
    """Run checks 1-5 on one (model tokenizer, mapped dataset row). Returns dict."""
    gt = str(row["reward_model"]["ground_truth"]).strip()
    res = {}

    # 1 — model chat template keeps the boxed instruction
    try:
        templ = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        res["c1_boxed_in_template"] = INSTR_MARK in templ
    except Exception as e:
        res["c1_boxed_in_template"] = False
        res["c1_error"] = f"{type(e).__name__}: {str(e)[:80]}"

    # 2 — dataset answer extraction produced a non-empty ground truth
    res["c2_gt_nonempty"] = gt != ""

    # 3 — nested balanced-brace extraction pulls the correct span
    nb = math_reward.last_boxed_only_string(NESTED)
    res["c3_nested_brace"] = (nb == r"\boxed{\frac{a}{b}}"
                              and math_reward.remove_boxed(nb) == r"\frac{a}{b}")

    # 4 — synthesized \boxed{gt} extracts back to gt (is_equiv)
    synth = f"Working... the final answer is \\boxed{{{gt}}}."
    try:
        span = math_reward.last_boxed_only_string(synth)
        extracted = math_reward.remove_boxed(span) if span else None
        res["c4_boxed_extract"] = extracted is not None and math_reward.is_equiv(extracted, gt)
    except Exception as e:
        res["c4_boxed_extract"] = False
        res["c4_error"] = f"{type(e).__name__}: {str(e)[:80]}"

    # 5 — route scores correct -> 1.0 and wrong -> 0.0
    correct_resp = f"Some reasoning. Answer: \\boxed{{{gt}}}"
    wrong_resp = f"Some reasoning. Answer: \\boxed{{{WRONG_ANSWER}}}"
    try:
        sc_c = default_compute_score(ROUTE, correct_resp, gt)
        sc_w = default_compute_score(ROUTE, wrong_resp, gt)
        res["c5_correct_1"] = float(sc_c) == 1.0
        res["c5_wrong_0"] = float(sc_w) == 0.0
    except Exception as e:
        res["c5_correct_1"] = False
        res["c5_wrong_0"] = False
        res["c5_error"] = f"{type(e).__name__}: {str(e)[:80]}"

    res["pass"] = all(res.get(k) for k in
                      ("c1_boxed_in_template", "c2_gt_nonempty", "c3_nested_brace",
                       "c4_boxed_extract", "c5_correct_1", "c5_wrong_0"))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", required=True, help="path to write preflight.json")
    args = ap.parse_args()

    report = {"route": ROUTE, "models": {}, "datasets": {}, "pairs": {}}

    # --- model axis (8 tokenizers) ---
    toks = load_tokenizers()
    tokenizers_ok = 0
    for slug, rec in toks.items():
        ok = rec["loads"] and rec["apply_ok"] and rec["boxed_survives"]
        tokenizers_ok += int(ok)
        report["models"][slug] = {k: v for k, v in rec.items() if k != "_tok"}
        report["models"][slug]["ok"] = ok

    # --- dataset axis (5 schemas) + cache a real sample row per dataset ---
    schemas_ok = 0
    ds_sample = {}
    for slug in DATASETS:
        ok, detail, sample = validate_schema(slug)
        schemas_ok += int(ok)
        report["datasets"][slug] = detail
        ds_sample[slug] = sample[0] if sample else None

    # --- 25 pairs (5 test models x 5 datasets) ---
    pairs_pass = 0
    for mslug in TEST_MODELS:
        tok = toks[mslug].get("_tok")
        for dslug in DATASETS:
            key = f"{mslug}|{dslug}"
            row = ds_sample.get(dslug)
            if tok is None or row is None:
                report["pairs"][key] = {"pass": False,
                                        "skip_reason": "tokenizer or dataset sample unavailable"}
                continue
            r = check_pair(tok, row)
            pairs_pass += int(r["pass"])
            report["pairs"][key] = r

    report["tokenizers_ok"] = tokenizers_ok
    report["schemas_ok"] = schemas_ok
    report["pairs_pass"] = pairs_pass
    report["gate_green"] = (tokenizers_ok == 8 and schemas_ok == 5 and pairs_pass == 25)

    os.makedirs(os.path.dirname(os.path.abspath(args.emit)), exist_ok=True)
    with open(args.emit, "w") as f:
        json.dump(report, f, indent=2)

    print(f"tokenizers_ok={tokenizers_ok}/8  schemas_ok={schemas_ok}/5  pairs_pass={pairs_pass}/25  "
          f"GATE {'GREEN' if report['gate_green'] else 'RED'}")
    if not report["gate_green"]:
        for slug, m in report["models"].items():
            if not m["ok"]:
                print(f"  MODEL FAIL {slug}: {m}")
        for slug, d in report["datasets"].items():
            if not (d["fields_ok"] and d["boxed_in_prompt"] and d["parquet_roundtrip"]):
                print(f"  DATASET FAIL {slug}: {d}")
        for k, r in report["pairs"].items():
            if not r.get("pass"):
                print(f"  PAIR FAIL {k}: {r}")
    return 0 if report["gate_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
