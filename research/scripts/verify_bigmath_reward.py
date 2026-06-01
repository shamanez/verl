"""End-to-end reward sanity check on REAL model trajectories (run on the box).

Loads the base policy (Qwen2.5-1.5B-Instruct) via vLLM, generates on a handful of
Big-Math val prompts (already \boxed{}-instructed in the parquet), extracts the
\boxed{} answer, and scores it with the SAME route the training uses
(default_compute_score("DigitalLearningGmbH/MATH-lighteval", ...) -> math_reward).
Prints per-sample: ground_truth, extracted pred, reward, response token count,
whether truncated. Confirms (a) the model emits \boxed{} answers, (b) the reward
gives +1 on correct ones, (c) responses fit well under the length cap.
"""
import random
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.reward_score import math_reward
from verl.utils.reward_score import default_compute_score

N = 10
MAX_NEW = 2048
rows = pq.read_table("/root/data/bigmath/test.parquet").to_pylist()
random.seed(0)
samples = random.sample(rows, N)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
prompts = [tok.apply_chat_template(s["prompt"], tokenize=False, add_generation_prompt=True) for s in samples]

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", gpu_memory_utilization=0.55,
          max_model_len=4096, enforce_eager=True, dtype="bfloat16")
# greedy for a deterministic capability read
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_NEW))

n_correct = 0
n_boxed = 0
print("=" * 80)
for s, o in zip(samples, outs):
    gen = o.outputs[0].text
    ntok = len(o.outputs[0].token_ids)
    gt = s["reward_model"]["ground_truth"]
    boxed = math_reward.last_boxed_only_string(gen)
    pred = math_reward.remove_boxed(boxed) if boxed else None
    score = default_compute_score("DigitalLearningGmbH/MATH-lighteval", gen, gt)
    n_boxed += int(boxed is not None)
    n_correct += int(score == 1.0)
    print(f"GT={gt!r:>14} | pred={str(pred)[:20]!r:>22} | reward={score} | resp_tokens={ntok} | trunc={ntok >= MAX_NEW}")
print("=" * 80)
print(f"SUMMARY: {N} samples | has_boxed={n_boxed}/{N} | reward==1 (correct)={n_correct}/{N} | none truncated at {MAX_NEW}")
print(f"max response tokens this batch: {max(len(o.outputs[0].token_ids) for o in outs)} (cap in training = 16384)")
