"""Measure AQ-SGD's buffer reuse distance from the real data and tokenizer.

AQ-SGD's activation buffer only scores a hit if it spans the REUSE DISTANCE,
which in RLVR is one full epoch of prompts. That need is

    n_examples * mean_prompt_tokens * n_boundaries * H * 2 bytes

and nothing in the comm_eff config subtree knows any of those four numbers, so
it is measured here rather than assumed. Run on the box, from the tree root,
before launching a codec arm.

Measured for Pair 1 (Qwen2.5-Math-1.5B on MATH, RELEX chat template,
7 compressed boundaries over 28 decoder layers, H=1536), 2026-09-10:

    rows 7498, prompt tokens mean 111.4, median 82, p95 280, max 1326
    0.1502 GiB per buffered token position
    scope=prompt one-epoch need  16.7 GiB  -> a 16 GiB cap covers 96%
    scope=all    one-epoch need 102.9 GiB  -> a 16 GiB cap covers 15.5%
    prompt share of boundary traffic 16.3%   <- the ceiling on AQ-SGD's mechanism
    600 steps x 128 prompts over 7498 problems = 10.2 epochs

The 16.3% is the number the ablation turns on: response tokens are resampled
every step, so only the prompt prefix recurs with its activation intact, and
that prefix is a sixth of what crosses a boundary.
"""

import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

df = pd.read_parquet(os.path.expanduser("~/data/math/train.parquet"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True)
tmpl = open("examples/grpo_trainer/relex_qwen_chat_template.jinja").read()
msgs = list(df["prompt"].iloc[0])

d = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("=== default template ===")
print(repr(d)[:260])
print("  tokens:", len(tok(d).input_ids))

r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=tmpl)
print("=== RELEX template ===")
print(repr(r)[:260])
print("  tokens:", len(tok(r).input_ids))

use_relex = len(tok(r).input_ids) > 10
kw = {"chat_template": tmpl} if use_relex else {}
L = []
for p in df["prompt"].head(2000):
    t = tok.apply_chat_template(list(p), tokenize=False, add_generation_prompt=True, **kw)
    L.append(len(tok(t).input_ids))
L = np.array(L)
n, H, B, ITEM = len(df), 1536, 7, 2
per_pos = n * B * H * ITEM
need = per_pos * L.mean()
cap = 16 * 1024**3
print()
print("template used:", "RELEX" if use_relex else "default")
print(f"rows {n} | prompt tokens mean {L.mean():.1f} median {np.median(L):.0f} p95 {np.percentile(L,95):.0f} max {L.max()}")
print(f"buffer per buffered token position: {per_pos/1024**3:.4f} GiB")
print(f"scope=prompt one-epoch need {need/1024**3:.1f} GiB -> 16 GiB cap covers {min(100, 100*cap/need):.0f}%")
need_all = per_pos * (L.mean() + 574)
print(f"scope=all (574-tok mean response) need {need_all/1024**3:.1f} GiB -> 16 GiB covers {100*cap/need_all:.1f}%")
print(f"prompt share of boundary traffic: {100*L.mean()/(L.mean()+574):.1f}%")
