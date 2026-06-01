"""Base-model capability: GSM8K vs Big-Math, identical \boxed format + math_reward.

Measures how well the UNTRAINED Qwen2.5-1.5B-Instruct (no RL) solves each dataset,
to test 'is GSM8K easy for this model' — the thesis explaining why masked+clean@20
reaches dense parity on GSM8K but stalls on Big-Math. Same prompt format + verifier
for both removes the format confound.
"""
import random
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.reward_score import math_reward

INSTR = "Let's think step by step and output the final answer within \\boxed{}."
N = 200
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

# GSM8K test (numeric answer after ####)
gsm = load_dataset("openai/gsm8k", "main", split="test")
random.seed(0)
gidx = random.sample(range(len(gsm)), N)
gsm_pg = [(gsm[i]["question"], gsm[i]["answer"].split("####")[-1].strip().replace(",", "")) for i in gidx]

# Big-Math test (already \boxed-style; bare answer)
bm = pq.read_table("/root/data/bigmath/test.parquet").to_pylist()
random.seed(0)
bm_s = random.sample(bm, min(N, len(bm)))
bm_pg = [(r["extra_info"]["problem"], r["reward_model"]["ground_truth"]) for r in bm_s]

llm = LLM(model=MODEL, gpu_memory_utilization=0.6, max_model_len=4096, enforce_eager=True, dtype="bfloat16")
sp = SamplingParams(temperature=0.0, max_tokens=1024)

print("=" * 70)
for name, pg in [("GSM8K", gsm_pg), ("Big-Math", bm_pg)]:
    chats = [tok.apply_chat_template([{"role": "user", "content": q + " " + INSTR}],
                                     tokenize=False, add_generation_prompt=True) for q, _ in pg]
    outs = llm.generate(chats, sp)
    correct = boxed = 0
    for (q, gt), o in zip(pg, outs):
        gen = o.outputs[0].text
        boxed += int(math_reward.last_boxed_only_string(gen) is not None)
        if math_reward.compute_score(gen, gt) == 1.0:
            correct += 1
    print(f"{name:>9}: base Qwen2.5-1.5B-Instruct \\boxed acc = {correct}/{len(pg)} = {correct/len(pg):.3f}  (emitted boxed {boxed}/{len(pg)})")
print("=" * 70)
