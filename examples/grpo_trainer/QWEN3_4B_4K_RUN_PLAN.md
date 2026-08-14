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

## Run D: the sign correction only ever applied to a fresh M

Run A collapsed. It reached step 200 and stopped with the score falling away,
and the mechanism identified afterwards was the gradient merger: the compressed
gradient keeps its own magnitude but borrows its signs from a dense average,
which pushes every coordinate at once in a fixed direction. What run D tests is
narrower and sharper than "turn the merger off". It tests whether the damage
comes from **reusing a frozen average between fires**.

### The two cadences, and why they are not the same knob

Run A's own counters make the situation concrete. Over its 200 steps it logged
`anchor_replay_fires = 10` against `spectral_corrections = 72038`, and that
second number factors exactly as 398 floating parameters times 181 ticks, where
181 is 200 steps minus the 19 warmup ticks that ran before `M` was ready.

| | knob | run A | effect |
|---|---|---|---|
| refresh of `M` | `anchor.cadence` | 20 | the anchor replays dense and folds `G_anchor` into `M`, on ticks 20, 40, 60 and so on |
| application of `M` | `spectral.cadence` | **1** | the signs are pushed into the gradient on **every** tick, whether or not `M` moved |

So on tick 20 the correction reads a freshly refreshed `M`, and then ticks 21
through 39 each apply that same frozen `M` again. Over 500 steps that is 481
applications of which only 25 are fresh: stale beats fresh nineteen to one.

`delay_K = 20` is a third quantity and not a cadence at all. It sets how far back
the anchor reaches for the paired weights and rollout batch it replays, which is
why the projection horizon is 20 ticks wide.

### The run

| | **Run D: fresh M only** |
|---|---|
| launcher | `run_qwen3_4b_4k_500_fsdp.sh freshm` |
| experiment name | `qwen3-4b-4k-freshm-500` |
| WandB | project `qwen3-4b-4k-500`, alongside runs A and B |
| delta against run A | `spectral.cadence` 1 to 20, locked equal to `anchor.cadence`, one number |
| what that does | the correction is applied on the fire ticks only, where `M` was refreshed earlier in the same tick, and skipped on the nineteen ticks between |
| corrections over 500 steps | 25, every one of them on a fresh `M`, against run A's 481 of which 456 were stale |
| merger strength | UNCHANGED, `alpha=0.25`, `beta_anc=0.25` |
| anchor circuit | UNCHANGED: fires every 20 ticks, replays the paired dense batch, maintains `M` |
| weight projection | UNCHANGED: rank-1 RELEX, W2 secant, strength 1 |
| optimizer state | UNTOUCHED, no swap, no reset (that is the separate `optreset` arm) |
| everything else | identical to run A, including the codec, the batch shape, the schedule and the checkpoint cadence |

Ordering is what makes a fire-tick correction read a fresh `M`, and it is a fact
of the engine rather than an assumption. `BaseEngine.train_batch` calls the
anchor refresh at the top, then the compressed forward and backward, then the
gradient correction, then the optimizer step. Both hooks advance their own
counter on every call, so with equal cadences they land on exactly the same
ticks and the correction always reads an `M` written moments earlier in that
tick. Simulating 500 ticks against the real predicates gives 25 corrections, 25
fresh, 0 stale, against run A's 481 corrections with 456 stale.

The launcher derives the merger cadence from the anchor cadence rather than
hardcoding 20, and then refuses the run if the two are unequal. That guard is
load-bearing: the config validator only requires that the anchor cadence be
divisible by the merger cadence, so a value like 10 passes validation and
quietly puts half the corrections back on stale ticks, which is run A's
behaviour under run D's name.

Two further guards refuse the run rather than let it drift into another
experiment: `COMM_EFF_SPECTRAL_ENABLED` must stay `true`, and
`COMM_EFF_OPT_RESET_ENABLED` must stay `false`. A fourth refuses
`signed_ema_alpha=1.0`, which would silently make this run E.

### Run E, the zero point of the same axis

`run_qwen3_4b_4k_500_fsdp.sh nosign` (`qwen3-4b-4k-nosign-500`) sets
`signed_ema_alpha` to 1.0 instead. The merger computes
`alpha*G + (1-alpha)*abs(G)*sign(M)`, so at 1.0 it returns `G` bit for bit and
`M` never reaches the optimizer at all. That places the three arms on one dose
axis, which is worth having because run D alone changes both the staleness and
the number of applications and cannot separate them:

| | corrections over 500 steps | of which stale |
|---|---|---|
| run A | 481 | 456 |
| run D | 25 | 0 |
| run E | 0 | 0 |

`alpha=1.0` rather than `spectral.enabled=false` because the latter is rejected
outright by the config validator whenever the weight projection is on, and it
would additionally make the engine's anchor hook return before it snapshots
anything, deleting the anchor and the projection along with the merger.

Confirm either arm took, once the log has a few steps:

```bash
grep -m1 "\[comm_eff\]\[signed_ema\] enabled" /workspace/runs/qwen3-4b-4k-freshm-500/train.log
```

It reports the resolved `alpha` and `cadence`.

Confirm it took, once the log has a few steps:

```bash
grep -m1 "\[comm_eff\]\[signed_ema\] enabled" /workspace/runs/qwen3-4b-4k-nosign-500/train.log
```

That line reports `alpha=1.0 ... identity=true`.

## Runs G and H: the delayed_ef merger, on the same two-cadence axis

`run_qwen3_4b_4k_500_fsdp.sh delayedef` (`qwen3-4b-4k-delayedef-500`) is run A
with ONLY the merger swapped, signed_ema to delayed_ef:

    delta      = M - G_comp          refreshed once per anchor fire
    G_corr(t)  = G_comp(t) + lambda * delta

with `lambda=1.0` and `beta_anc=0.0` (part of the swap: M must be the latest
fire's raw dense anchor gradient, not an EMA of older fires). The codec, the
anchor cadence/delay 20/20, the rank-1 RELEX projection at strength 1, and
`spectral.cadence=1` are all run A's values, so the held `delta` is re-applied
STALE on every one of the 19 ticks between fires, the exact analog of run A's
stale-M reuse. On the fire tick itself the algebra collapses:
`G_corr = G_comp + (M - G_comp) = G_anchor`, the anchor's dense gradient
computed at the RELEX-projected weights on the same batch.

`run_qwen3_4b_4k_500_fsdp.sh delayedef-fresh` (`qwen3-4b-4k-delayedef-fresh-500`)
is run G with the freshm one-number change, `spectral.cadence` 1 to 20 locked to
the anchor cadence. The residual is then never re-applied stale: every 20th tick
the gradient IS the anchor gradient, every other tick is the untouched
compressed gradient.

The pair asks of the ADDITIVE merger family exactly what runs A and D asked of
the sign-replacement family: does the stale reuse between fires cause the
collapse, or does the merger formula itself? Two cautions carried over from the
pre-fork record (the mergers were pruned from this lineage on 2026-07-15 and
recovered by archaeology). First, delayed_ef historically ran against a
RING-paired compressed gradient at a truly visited stale point, scoring 96
percent of dense at 1.5B, while this port pairs against the CURRENT tick's
compressed gradient at the RELEX-projected point (same batch, no ring), which is
the only pairing the projected anchor admits. Second, no historical run ever
combined delayed_ef with RELEX, so these two arms are also the first clean test
of that combination.

Confirm either arm took:

```bash
grep -m1 "\[comm_eff\]\[delayed_ef\] enabled" /workspace/runs/qwen3-4b-4k-delayedef-500/train.log
```

That line reports the resolved `lambda`, `beta_anc`, `cadence` and
`identity=false`. In WandB, `comm_eff/delayed_ef_refreshed` must factor as
398 x anchor fires for both arms; `comm_eff/delayed_ef_held` grows by
398 x 19 per fire interval on run G and stays 0 on run H.

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
bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh freshm
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
