# Qwen3-4B-Base at 4k: compressed against dense, 500 steps each

Two GRPO runs on MATH, identical on every axis except how much of the boundary
activation crosses the wire. One 4x H200 box, sequential arms, then a shared
in-domain plus out-of-domain evaluation, then one HTML comparison.

## The two runs

| | **Run A: compressed** | **Run B: dense control** |
|---|---|---|
| order | first | second |
| launcher | `run_qwen3_4b_4k_500_fsdp.sh commeff` | `run_qwen3_4b_4k_500_fsdp.sh dense` |
| experiment name | `qwen3-4b-4k-commeff-500` | `qwen3-4b-4k-dense-500` |
| model | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B-Base` |
| data | MATH (`prepare_rlvr_math.py --dataset math`) | same |
| context | 1024 prompt + 3072 response = 4096 | same |
| prompts per step / mini-batch | 128 / 128 (one optimizer tick per step) | same |
| rollouts per prompt | 8 | same |
| optimizer | AdamW 1e-6, weight decay 0.01, clip 1.0 | same |
| reference KL | `low_var_kl`, coefficient 0.001 | same |
| steps | 500 | 500 |
| validation | step 0, 100, 200, 300, 400, 500 | same |
| checkpoints | 100, 200, 300, 400, 500 | same |
| **boundary transport** | **PRF exact-k, p=0.95, constant rescale, masking the train forward, the old-logprob recompute and the reference forward, 8 pipeline shards** | **`COMM_EFF_ENABLED=false`** |
| bits per token per boundary | 2048 (128 of 2560 coordinates) | 40960 (5.0 percent of it) |
| anchor circuit | paired dense replay, cadence/delay 20/20 ticks, `rollout_batch` scope, CPU state, `owns_q=false` | inert |
| weight projection | rank-1 RELEX, W2 secant, strength 1, `auto`, `stale_correct` | inert |
| gradient merger | signed EMA over all floating parameters, `beta_anc=0.25`, `alpha=0.25` | inert |
| WandB | project `qwen3-4b-4k-500` | same project, second run |
| R2 | `.../qwen3-4b-4k-500/qwen3-4b-4k-commeff-500/checkpoints/` | `.../qwen3-4b-4k-dense-500/checkpoints/` |

`COMM_EFF_ENABLED=false` is the only science delta. Everything else, including
the chat template, the data order seed, the sampling shape and every optimizer
constant, is shared.

## Run D: the same compressed run with the gradient merger switched off

Run A collapsed. It reached step 200 and stopped with the score falling away,
and the mechanism identified afterwards was the gradient merger: between anchor
fires the compressed gradient keeps its own magnitude but borrows its signs from
a stale dense average, which pushes every coordinate at once in a fixed
direction. Run D is the direct ablation of that one mechanism.

| | **Run D: no sign correction** |
|---|---|
| launcher | `run_qwen3_4b_4k_500_fsdp.sh nosign` |
| experiment name | `qwen3-4b-4k-nosign-500` |
| WandB | project `qwen3-4b-4k-500`, alongside runs A and B |
| delta against run A | `spectral.signed_ema_alpha` 0.25 to 1.0, one number |
| what that does | the merger computes `alpha*G + (1-alpha)*abs(G)*sign(M)`, so at 1.0 it returns `G` bit for bit and the anchor's signs never reach the optimizer |
| anchor circuit | UNCHANGED: still fires every 20 ticks, still replays the paired dense batch, still maintains `M` |
| weight projection | UNCHANGED: rank-1 RELEX, W2 secant, strength 1 |
| optimizer state | UNTOUCHED, no swap, no reset (that is the separate `optreset` arm) |
| everything else | identical to run A, including the codec, the batch shape, the schedule and the checkpoint cadence |

`M` becomes a quantity the run maintains and never reads. That is deliberate:
the alternative way to switch the merger off, `spectral.enabled=false`, is
rejected outright by the config validator whenever the weight projection is on,
and it would additionally make the engine's anchor hook return before it
snapshots anything, deleting the anchor and the projection along with the
merger. That is three changes. `alpha=1.0` is one, and it leaves step timing,
host memory and the whole slow circuit comparable to run A.

`alpha=1.0` is also the endpoint of an axis this project has already swept:
0.25 was chosen as the best value and 0.5 was measurably worse.

A pre-flight gate proves the claim on CPU before any GPU time is spent. It warms
`M` so that every one of its signs opposes the gradient, the worst case the
correction can present, then asserts that the merger returns the gradient
unchanged bit for bit, in both float32 and bfloat16, and that the same tensors
at `alpha=0.25` move by 1.5 times their own norm. A knob that were dead would
fail the second half of that gate.

Two guards refuse the run rather than let it become a different experiment
quietly: `COMM_EFF_SPECTRAL_ENABLED` must stay `true`, and
`COMM_EFF_OPT_RESET_ENABLED` must stay `false`.

Confirm it took, once the log has a few steps:

```bash
grep -m1 "\[comm_eff\]\[signed_ema\] enabled" /workspace/runs/qwen3-4b-4k-nosign-500/train.log
```

That line reports `alpha=1.0 ... identity=true`.

## Why 128 prompts per step and not the 512 in CLAUDE.md

128/128 is the surface every piece of long-horizon evidence in this project sits
on: the 600-step compressed and dense pair, and all twelve stability arms. It
gives exactly one optimizer tick per global step, so the anchor's cadence of 20
ticks reads directly as "every 20 global steps" instead of every 10. It is also
what makes two 500-step arms fit one box. Set `TRAIN_BATCH_SIZE=512
PPO_MINI_BATCH_SIZE=256` to train the CLAUDE.md surface instead, at roughly four
times the cost per step and a doubled anchor rate.

## Commands

```bash
tmux new -s q4b
bash examples/grpo_trainer/run_qwen3_4b_4k_500_all_fsdp.sh
```

That is the whole study: smoke gate, compressed arm, dense arm, evaluation
matrix, report. Every stage is resumable and skips finished work, so the same
command restarts a killed session.

Individual stages, if they are wanted separately:

```bash
SMOKE=1 bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh commeff
bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh commeff
bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh dense
bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh nosign
bash research/scripts/ood_eval/eval_qwen3_4b_4k.sh
python3 research/scripts/ood_eval/report_qwen3_4b_4k.py \
  --results /workspace/runs/ood-eval-4b/results.json \
  --out     /workspace/runs/ood-eval-4b/qwen3-4b-4k-comparison.html
```

## Gates that run before a GPU is touched

| gate | claim |
|---|---|
| GPUs | exactly 4 detected |
| host RAM | at least 320 GiB, the anchor replicates its clone, its snapshots and the fp32 gradient EMA per rank |
| disk | at least 900 GiB, ten kept 4B checkpoints plus the merged evaluation models |
| aws | present, installed from the official v2 zip if not, with a 256 MB multipart part size |
| R2 | `R2_BUCKET` repaired to `shamane-pluralis` in the on-box secrets file, then a real bucket listing |
| exact-k | keeps exactly 128 of 2560 coordinates per token at p=0.95 |
| model shape | the checkpoint really is hidden 2560 with 36 layers |
| boundaries | 36 layers over 8 shards gives `[4, 9, 14, 19, 23, 27, 31]` |
| anchor | its replay clone inherits gradient checkpointing |
| eval context | `max_model_len` pinned to 4096, or vLLM falls back to this model's 32768 and refuses to boot |

All eight pass on this checkout today, verified on the laptop.

## Evaluation

Ten benchmarks per checkpoint, dense in every case (the evaluation never runs
the compressed path, it measures what the weights can do).

| group | benchmarks | protocol |
|---|---|---|
| in domain | MATH-500 | greedy mean@1 |
| out of domain, math | GSM8K, Minerva, OlympiadBench, AMC23, AIME24, AIME25, AIME26, HMMT25 | greedy mean@1, avg@8 at temperature 0.7 for the five competition sets |
| out of domain, capability retention | MMLU-STEM | greedy mean@1, boxed letter |

Roster: the untrained base, then both arms at 500, then both arms at 400, 300,
200 and 100. The headline three come first so a truncated evaluation still
answers the question. Benchmarks are fanned over two GPU pairs and a benchmark
that already produced a result is never re-run.

## The smoke gate

The first stage runs 25 steps of the compressed arm with no validation, no
checkpoints and nothing written to R2, then reads four numbers off it and
refuses to start the 500-step arms if two of them are out of range.

25 steps, not 5, because at 128/128 there is one optimizer tick per global step,
so the anchor first fires at step 20 and its unsharded replay clone only enters
the memory peak there. The 1.5B reference run measured 34.2 GB allocated at step
1 and 109.0 GB at step 25, against a lifetime maximum of 110.1. A short probe
under-reports the peak by a factor of three.

| reading | reference value on the 1.5B run | gate |
|---|---|---|
| median s/step, step 1 dropped | 91 | none, it prices the box |
| peak GB allocated | 109.0 of 141 | fails above 125 |
| response length mean | 774 rising to 792, of a 2048 cap | none, it feeds the judgement below |
| response clip ratio | 0.12 to 0.14 | fails above 0.35 |

If the memory gate trips, the ladder is `ROLLOUT_GPU_MEM_UTIL` 0.60 to 0.50,
then `LOG_PROB_MAX_TOKEN_LEN_PER_GPU` 24576 to 18432, then
`PPO_MAX_TOKEN_LEN_PER_GPU`.

## Cost and wall clock

The only hard per-step measurement in this repository is the 600-step 1.5B pair
at this same 128/128 shape on one H200: 72.7 s/step compressed, 68.4 s/step
dense, with a 574-token mean response and 2 percent of responses clipped.
Scaling that by 2.7x for parameters and 1.3x for the longer response window, and
dividing by roughly 3.2x for four GPUs, gives about 80 to 90 s/step. Box quality
alone has moved this number by 1.57x between two nominally identical H200 hosts,
so the honest band is:

| | low | high |
|---|---|---|
| per arm | 11 h | 18 h |
| both arms plus validation | 24 h | 38 h |
| evaluation matrix | 8 h | 14 h |
| **total box time** | **32 h** | **52 h** |
| **cost at about $10.5/hr** | **$340** | **$550** |

The smoke stage replaces this estimate with a measured median in well under
an hour, before either arm starts.

## Risks

| risk | why it applies here | mitigation |
|---|---|---|
| responses press the 3072 cap | Qwen3 base models are wordier than the Qwen2.5-Math model this recipe was tuned on, and truncation feedback at the cap is what ended the previous long-context attempt | smoke stage hard-fails above a 0.35 clip ratio, the report plots response length, and `MAX_RESPONSE_LENGTH` plus `MAX_MODEL_LEN` raise it together |
| anchor peak memory at the first fire | its replay clone does not shard, so per-rank peak is set by the token budget rather than the GPU count, and the peak appears once, at the first fire | gradient checkpointing on the clone is a launch gate, vLLM held at 0.60, actor budget 18432 and log-prob budgets cut to 24576, watch step 20 |
| the R2 mirror kills the run at the first save | a missing `aws` binary raises out of the save path rather than skipping, and it has cost 49 H200 steps before | installed and credential-checked in preflight, with the 256 MB part size that avoids the multipart failure on large shards |
| clean run exits non-zero | the engine writes `done.flag` into a directory it never creates when the log lives outside the repo | the directory is created up front, and completion requires both a zero exit and the flag |
| reference KL looks alarming on the compressed arm | it is measured through the compressed view, so its scale is not comparable to the dense arm's | read each arm's divergence only against itself, judge on the response-length slope and validation |
| box quality | measured 1.57x spread on identical hardware | the smoke stage prices the actual box before committing |
| the evaluation never boots | this model reports a 32768 position limit while the engine hardcodes chunked prefill off, so vLLM rejects the default 8192 token batch | `max_model_len` and `max_num_batched_tokens` are passed through a new, empty-by-default hook in the shared eval script |

## Deliverables

1. Two WandB runs in project `qwen3-4b-4k-500`.
2. Five checkpoints per arm in R2 under
   `s3://shamane-pluralis/autonomous-harness-rlvr-compression/qwen3-4b-4k-500/<arm>/checkpoints/global_step_<N>/`.
3. `results.json` and `RESULTS.txt`, the full 11 by 10 accuracy matrix.
4. `qwen3-4b-4k-comparison.html`, the comparison page: headline numbers, a
   grouped bar chart per benchmark with in-domain separated from out of domain,
   the complete numeric table with the compressed-minus-dense column, and six
   training-dynamics panels.

## Files

| path | role |
|---|---|
| `examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh` | one arm, end to end, from a bare box |
| `examples/grpo_trainer/run_qwen3_4b_4k_500_all_fsdp.sh` | smoke, both arms, evaluation, report |
| `research/scripts/ood_eval/eval_qwen3_4b_4k.sh` | merge, evaluate, tabulate |
| `research/scripts/ood_eval/collect_qwen3_4b_4k.py` | logs to `results.json` |
| `research/scripts/ood_eval/report_qwen3_4b_4k.py` | `results.json` to the HTML page |
