# Run 99: PRF exact-k on R1-Distill-Qwen-1.5B / DeepScaleR at 16384 context

Replication of the run-90 surface (`90-prf-exactk-600`) and the run-96 launcher
pattern (`96-qwen3-8b-prf-exactk-16k-1000`) on a different model and dataset, with
the step-600 dense-against-compressed capability audit that the system-status
report draws its "Capability at step 600" figure from.

Base branch `autonomous-harness-v1`, whose `verl/` tree is byte-identical to the
one run 96 trained on. **No selective compression.** The `p_by_boundary`
dense-middle lever is run 97's and the launcher refuses to start if it is set.

## Surface

| | value | vs run 90 |
|---|---|---|
| model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | changed |
| data | DeepScaleR (`qingy2024/DeepScaleR-40k`), cap 20000, val 500, seed 42 | changed |
| prompt | the model's own template, `<｜User｜>` / `<｜Assistant｜><think>` | changed |
| context | 1024 prompt + 15360 response = 16384 | 3072 -> 16384 |
| batch / mini | 128 / 128, one on-policy tick per generation | unchanged |
| rollout | n=8, TP=1, gpu_mem 0.72 | unchanged |
| optimizer | AdamW 1e-6, low_var_kl 0.001, entropy 0 | unchanged |
| codec | `prf_mask` p=0.95, exact-k, constant rescale, pp_size 8 | unchanged |
| wire budget | 77 of 1536 coords per token per boundary, 1232 bits | unchanged |
| boundaries | 28 layers over 8 stages, cuts at [3, 7, 11, 15, 18, 21, 24] | model-derived |
| anchor | paired dense replay 20/20 ticks, `rollout_batch`, CPU state, `owns_q=false` | unchanged |
| weights | rank-1 RELEX, W2, strength 1, `auto`, `stale_correct` | unchanged |
| signed EMA | `beta_anc=0.25`, `alpha=0.25`, all floating params, CPU | unchanged |
| steps | 600, val every 100, save every 100 | val 150 -> 100 |

Two arms, one variable apart:

- `dense` runs the same surface with `COMM_EFF_ENABLED=false`.
- `prf` runs the project-default codec.

Checkpoints mirror to R2 under
`autonomous-harness-rlvr-compression/99-r1distill-deepscaler-600/<arm>/checkpoints/`.

## Why the prompt change is load-bearing

The base launcher pins RELEX's Qwen ChatML template, which is correct for
Qwen2.5-Math-1.5B and wrong for R1-Distill. R1-Distill's own template wraps turns
in `<｜User｜>` / `<｜Assistant｜>` and opens the assistant turn inside `<think>`,
which is what the model was distilled to continue from. Feeding it ChatML would
tokenize `<|im_start|>` as ordinary text, drop the `<think>` opener, and quietly
evaluate a differently-prompted model with no error anywhere. Both the training
launcher and the eval driver rewrite the override to `null` in a generated copy
and gate that the rewrite took.

## Why 16384 and not the 4096-token protocol

This is the one place the run departs from the project protocol, and it departs
deliberately. R1-Distill is a long-CoT model, its own template opens the
assistant turn inside `<think>`, and issue #63 ran it at 16384 response tokens.
At a 3072-token cap a completion that never closes `</think>` emits no
`\boxed{}` and scores 0, so a large share of the reward signal would be "ran out
of tokens" rather than "got it wrong". That is the truncation-feedback mechanism
that sparked the run-96 collapse. 16384 total gives the model room to finish, at
the cost of a longer step.

The measurement stays either way: verl logs `response_length/clip_ratio` every
step, which is exactly the truncation rate. At 15360 it should be small. Run 96
collapsed at this same cap, so a RISING clip ratio remains the earliest warning
this surface gives. The dense arm runs first and prices it, and the sweep driver
prints the per-arm summary between arms.

## Order of operations

```bash
# On the box, inside tmux. Both arms, sequential, dense first.
bash examples/grpo_trainer/run_99_both_arms.sh

# Single arm, if the box is only free for one.
ARM=prf bash examples/grpo_trainer/run_r1distill_deepscaler_600.sh

# After both arms finish: the step-600 capability audit, every gate first.
DRY_RUN=1 bash research/scripts/ood_eval/ckpt_eval.sh
bash research/scripts/ood_eval/ckpt_eval.sh

# The figure.
python3 research/scripts/ood_eval/plot_dense_vs_compressed.py \
  --results /workspace/runs/99-r1distill-deepscaler-600-eval/RESULTS_99-r1distill-deepscaler-600.tsv \
  --title "Capability at step 600: dense against 95 percent compressed"
```

The audit's in-domain benchmark is DeepScaleR's held-out split, scored with
verl's own validation sampling, so it cross-checks the in-training val at the
same step. The out-of-domain suite is the standard ten: math500, gsm8k, minerva,
olympiad, amc23, mmlu_stem, aime24, aime25, aime26, hmmt25.

`EVAL_STEPS` defaults to `600` because that is the figure. Set
`EVAL_STEPS="200 600"` to add the step-200 column, and the driver resumes per
cell rather than redoing the work already on disk.

## CPU gates run before any GPU was requested

- exact-k keeps exactly 77 of 1536 coordinates per token, 1232 bits per token per
  boundary, verified against the real codec.
- 28 layers over 8 stages give boundaries [3, 7, 11, 15, 18, 21, 24], verified.
- `build_anchor_module` enables gradient checkpointing on the anchor clone.
- The real R1-Distill config matches (hidden 1536, 28 layers) and its tokenizer
  renders `<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜><think>` with no
  ChatML markers.
- The launcher's patch pipeline was exercised against the real base launcher: the
  response length and two batch scalars rewrite, the prompt length is inherited,
  and the generated launcher execs with `custom_chat_template=null` and no
  remaining reference to the RELEX template.
- `R2_BUCKET` is pinned to `shamane-pluralis`. The shipped secrets file sets it to
  the key PREFIX, and `r2_sink.py` hard-refuses any other bucket, so an unpinned
  run would fail every upload at the first save with the compute already spent.
- `bash -n` on all three shell drivers, `py_compile` on all three Python files,
  and `tests/special_sanity/check_example_naming.py` green.
- The tabulator and plotter were exercised end to end on synthetic per-cell logs.
