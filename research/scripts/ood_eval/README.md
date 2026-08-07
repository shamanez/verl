# In-domain + OOD capability eval harness

A small, reproducible harness for evaluating GRPO checkpoints (e.g.
`Qwen/Qwen2.5-Math-1.5B`) on in-domain **and** out-of-domain benchmarks with the
*exact* prompt template and boxed scorer used at training-time validation. It was
built for the KL dose-response study (does two-circuit compression drift damage
OOD / base capability while in-domain val still looks healthy?), but the scripts
are generic.

## Why it reuses the training launcher

Every benchmark is scored by running the normal GRPO launcher in
`trainer.val_only=True` mode with `COMM_EFF_ENABLED=false`. That guarantees the
prompt template, chat formatting, generation config, and the `\boxed{}` answer
extraction / `is_equiv` scorer are byte-identical to training-time validation, so
eval numbers sit directly beside training val curves. All ten benchmarks are
tagged with a single `data_source = "DigitalLearningGmbH/MATH-lighteval"`, which
routes scoring to `verl.utils.reward_score.math_reward` uniformly (no per-bench
scorer, no `aime*`/`math_dapo` "Answer:" regex path).

## Files

| File | Role |
|---|---|
| `ood_prep.py` | Build `test.parquet` for the 10 benchmarks in the MATH schema. Parameterized by `OOD_ROOT` (default `/root/data/ood`) and `MATH_TRAIN`. No credentials. |
| `ood_eval.sh` | Evaluate ONE model on ONE benchmark via val-only. Parameterized by `VERL_DIR`, `OOD_EVAL_ROOT`, `OOD_DATA_ROOT`, `LAUNCHER`, `SHIM_DIR`. |
| `ood_run_all.sh` | Matrix orchestrator: merge each FSDP checkpoint (local or R2), eval each model x 10 benchmarks fanned over a GPU-pair pool, tabulate. Roster is an editable example. **Its `BENCHES` table is the single definition of the sampling protocol** and is parsed, not copied, by `ckpt_eval.sh`. |
| `ckpt_eval.sh` | Multi-arm capability audit, driven entirely by env. Discovers each arm's R2 layout, verifies shard completeness before downloading, pulls / merges / size-verifies / deletes one checkpoint at a time, evals in-domain + all 10 OOD benchmarks per model, tabulates. Resumes per cell. |
| `tabulate_arms.py` | Turn the per-cell `train.log` files into the printed table plus a `RESULTS_<run>.tsv`, with `arm - base` and `compressed - dense` delta columns. |
| `plot_dense_vs_compressed.py` | The capability figure from that TSV: paired bars per benchmark, dense in blue and compressed in green, in-domain separated from OOD by a dashed divider. |

## Multi-arm audit (`ckpt_eval.sh`)

Built for a run that writes one R2 regime per arm, so a single listing under
`autonomous-harness-rlvr-compression/<RUN_ID>/` finds both:

```bash
# Defaults audit dense + prf at step 600 for run 99.
bash research/scripts/ood_eval/ckpt_eval.sh

# Every preflight gate, no downloads. Run this first, always.
DRY_RUN=1 bash research/scripts/ood_eval/ckpt_eval.sh

# A different run, different arms, more steps.
RUN_ID=... ARMS="dense prf" EVAL_STEPS="200 600" \
  bash research/scripts/ood_eval/ckpt_eval.sh

# Then the figure.
python3 research/scripts/ood_eval/plot_dense_vs_compressed.py \
  --results /workspace/runs/<RUN_ID>-eval/RESULTS_<RUN_ID>.tsv \
  --title "Capability at step 600: dense against 95 percent compressed"
```

Two properties worth knowing before trusting a number it prints:

- **The eval prompt is rewritten to match training.** The base launcher pins
  RELEX's Qwen ChatML template. For any model that is not a Qwen chat model that
  is the wrong prompt, so the driver rewrites the override to `null` in its
  generated launcher copy and gates that the rewrite took. The merged-model check
  also refuses a checkpoint whose tokenizer renders ChatML.
- **Nothing is trusted that was not verified.** A step is only downloaded once
  every rank shard is listed, every pulled object is size-checked against R2, and
  a merge only counts as done once its weights are on disk and large enough, at
  which point `.merge_ok` is written. `config.json` alone is never a success
  marker, because `save_pretrained` writes it before the weights.

## Benchmarks and sampling protocol

| Benchmark | Type | Sampling |
|---|---|---|
| MATH500, GSM8K, Minerva, OlympiadBench, MMLU-STEM | in-domain / distribution-shift / knowledge | greedy, mean@1 (temp 0) |
| AMC23, AIME24, AIME25, AIME26, HMMT25 | competition (high variance) | avg@8, temp 0.7, top_p 0.8 |

MMLU-STEM is rendered as A-D multiple choice with a "put the letter in `\boxed{}`"
instruction (19 STEM subjects, capped per subject). GSM8K gold is comma-stripped.

## Usage

```bash
# 1. Build the benchmark parquets (once).
python research/scripts/ood_eval/ood_prep.py                 # all 10
python research/scripts/ood_eval/ood_prep.py --only math500 gsm8k   # a subset

# 2a. One model on one benchmark:
research/scripts/ood_eval/ood_eval.sh /path/to/merged_model math500 mytag 0,1

# 2b. The full matrix over a roster of checkpoints (edit ROSTER in the script):
research/scripts/ood_eval/ood_run_all.sh
# -> writes $OOD_EVAL_ROOT/RESULTS.txt (tag x benchmark table with delta columns)
```

## Checkpoint sources (R2) and credentials

`ood_run_all.sh` merges FSDP checkpoints to clean HF models. If a checkpoint is not
present locally it is pulled from an S3-compatible store (Cloudflare R2). **No
credential is stored in these scripts.** They `source ~/.config/verl-research/secrets.env`
(off-repo, `chmod 600`), which must define:

```
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
R2_CKPT_BUCKET=<bucket-holding-your-checkpoints>
```

The key prefix under the bucket is `R2_PREFIX` (default matches the reference
study). If your checkpoints are already merged locally, no R2 access is needed.

## Known eval caveats (uniform across all columns, so deltas are unaffected)

- OlympiadBench gold retains `$...$`; the string-match grader is slightly stricter
  than a sympy verifier, deflating absolute OlympiadBench scores by a few points.
- MMLU-STEM is 0-shot boxed-letter on a math model (the model is not a natural MC
  answerer); treat it as a relative knowledge-retention probe.
- Max generation length is set by the launcher; competition problems occasionally
  truncate. Applied identically to every checkpoint.
