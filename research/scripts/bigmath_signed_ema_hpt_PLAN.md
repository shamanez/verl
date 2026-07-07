# Big-Math signed_ema (α, β) hyperparameter-tuning ablation — RUNBOOK

**Paste this whole file into a fresh session.** It is self-contained: instance
access, environment, dataset, the ablation grid, run/monitor/judge, and WandB
naming. Assumes the Vast.ai box below is **already ON**.

**Goal.** Re-tune the two comm-eff merger knobs — **α** (`signed_ema_alpha`, the
sign-correction weight) and **β** (`beta_anc`, the M EMA decay) — for the HARD
dataset (Big-Math) at HIGH anchor latency (cadence/delay_K = 20/20), **without
weight projection**. The incumbent (α=0.25, β=0.50) was tuned on GSM8K at LOW
latency; it has never had a proper Big-Math ablation. Training-curve check, no validation.

---

## 0. Instance & environment (do this first)

**Box (already provisioned, 1×H200, CUDA-13 driver):**
```bash
ssh -i ~/.ssh/vast_ai -p 40276 root@84.8.116.228 -L 8080:localhost:8080
```
(If the box IP/port changed, update this line. Repo lives at `/workspace/verl`.)

**This instance does NOT have our verl docker image.** We use a venv named
**`run-verl`** that reproduces the image exactly (torch **2.11.0+cu130**, vllm
**0.20.2**, transformers **5.3.0**, flash-attn **2.8.3** built from source), using
the instance's own GPU driver + nvcc (no CUDA toolkit install).

```bash
# activate it (this is all you need if it is already built):
source /workspace/venvs/run-verl/bin/activate

# if it is NOT built yet (fresh box), build it ONCE (idempotent; ~15-20 min, the
# flash-attn compile dominates; instant no-op once built):
bash /workspace/verl/research/scripts/setup_run_verl_env.sh
```
The ablation driver auto-activates `run-verl`, so for the run itself you don't
even need to activate manually.

**Dataset** — Big-Math already prepped at `/root/data/bigmath`
(`train.parquet` 123,600 rows / `test.parquet` 500). If missing:
```bash
source /workspace/venvs/run-verl/bin/activate
python /workspace/verl/research/scripts/bigmath_dapo.py \
  --local_save_dir /root/data/bigmath --train-cap 0 --val-size 500 --seed 42
```
**Secrets** — `~/.config/verl-research/secrets.env` (HF_TOKEN + WANDB_API_KEY) is
already present. **1×H200 ⇒ the 7 cells run SEQUENTIALLY** (one GRPO run
saturates the GPU; true parallelism needs more boxes).

---

## 1. The two circuits and what α, β do

**Fast circuit** — normal actor train pass; pipeline-boundary activations
PowerSGD-compressed (rank r=77) → biased+noisy gradient `G_noisy`.
**Anchor circuit** — uncompressed, no-optimizer clone from a `delay_K`-stale
snapshot, fired every `cadence` ticks → raw `G_anchor`, EMA'd into `M`; the anchor
also **owns Q** (the PowerSGD basis).

**Merger (`signed_ema`)** rewrites the fast gradient before `optimizer.step()`
(`verl/workers/comm_eff/spectral_filter.py:440`):
```
G_corr = α·G_noisy + (1 − α)·|G_noisy|·sign(M)
```
Magnitude from the fast compressed grad; **sign from the stale-but-uncompressed
anchor** — compression corrupts small-coord signs, the anchor's are cleaner.
- **α** ∈ [0,1]: `1.0` → no correction; `0.0` → sign fully from anchor; `0.25` → incumbent.

**M is built by the β EMA at each anchor fire** (`spectral_filter.py:324`):
```
M ← β·M + (1 − β)·G_anchor
```
- **β** ∈ [0,1]: `0.0` → M = newest fire (freshest, no memory); `0.9` → long smooth
  average (stalest direction, lowest variance); `0.5` → incumbent.

**Cadence arithmetic:** cadence/delay_K are in **optimizer ticks**; 128/64 batching
= 2 ticks/step ⇒ 75 steps = 150 ticks. At cadence=20 the anchor fires **7×**, so M
is refreshed 7× and β blends across those 7 fires; delay_K=20 ticks ≈ **10 steps**
stale. M is cold (merger inert) until the first fire (~step 10), so cells only
diverge after that. This is the **k-collapse regime** (collapse historically
visible ~step 61); 75 steps captures its onset.

---

## 2. The grid — cross/plus design centered on the incumbent (7 runs)

| # | α | β | Run name | What this cell checks |
|---|-----|-----|-------------------|-----------------------|
| 1 | 0.25 | 0.50 | `hpt_bm_a0.25_b0.50` | **Incumbent** on Big-Math @ 20/20 — reference + pipeline/mem canary |
| 2 | 1.00 | 0.50 | `hpt_bm_a1.00_b0.50` | **Merger OFF** — no sign correction → the k-collapse floor to beat |
| 3 | 0.00 | 0.50 | `hpt_bm_a0.00_b0.50` | **Max sign correction** — sign fully from the 10-step-stale anchor |
| 4 | 0.50 | 0.50 | `hpt_bm_a0.50_b0.50` | Alpha interior |
| 5 | 0.75 | 0.50 | `hpt_bm_a0.75_b0.50` | Alpha — light touch |
| 6 | 0.25 | 0.00 | `hpt_bm_a0.25_b0.00` | **Freshest M** (no memory) — hypothesis: wins at high latency |
| 7 | 0.25 | 0.90 | `hpt_bm_a0.25_b0.90` | **Smoothest/stalest M** |

Alpha swept at β=0.50 (cells 2,5,4,1,3); β swept at α=0.25 (cells 6,1,7); center
(0.25,0.50) shared. Ordered so the biggest-signal comparisons land first — early-killable.
Follow-up if wanted: a small 2×2 grid around the joint winner (edit the `GRID` array).

---

## 3. Fixed surface (every cell identical except α, β)

Mirrors the proven EXP-58 Big-Math 1×H200 surface; only the merger varies.

| Axis | Value |
|---|---|
| Env | `run-verl` venv (torch 2.11.0 / vllm 0.20.2 / transformers 5.3.0 / flash-attn 2.8.3) |
| Model / loss | Qwen2.5-1.5B-Instruct, vanilla GRPO, no-KL no-entropy |
| Response / prompt | 4096 / 1024 |
| Rollout | vLLM, TP=1, n=8, gpu_mem_util=0.45 |
| Batch | train 128 / mini 64, **dynamic bsz ON**, ppo_max_token 18432 |
| Substrate | comm-eff ON, PowerSGD r=77, anchor owns Q, **cadence/delay_K = 20/20**, paired replay, snapshot+M on CPU |
| Weight projection | **OFF** (lookahead defaults) |
| Merger | `signed_ema`, diagnostics=false |
| Schedule | **75 steps, NO validation** (val_before_train=False, test_freq=100000), **NO checkpoint** (save_freq=100000) |

---

## 4. Run · monitor · judge

**Run** (in tmux on the box; the driver auto-activates `run-verl`, preps data if
missing, and runs the 7 cells sequentially):
```bash
cd /workspace/verl
tmux new -s hpt
DRY_RUN=1 bash research/scripts/bigmath_signed_ema_hpt.sh   # sanity-print the grid first
bash research/scripts/bigmath_signed_ema_hpt.sh             # run the full grid
```
Idempotent/resumable: a cell with `runs/<name>/done.flag` is skipped;
`START_CELL=k` resumes; `ONLY_CELL=k` runs one; `FORCE=1` reruns.
OOM fallback (shrink response before escalating GPUs):
`PPO_MAX_TOKEN_LEN_PER_GPU=9216 …`, then `MAX_RESPONSE_LENGTH=2048 …`.

**WandB (distinct from any `rlvr-compression` run):**
- project **`verl_compression_research`**, entity **`shamanework-pl`**
- group **`hyperparam_tuning_bigmath_signed_ema_c20d20`**  ← all 7 cells land here
- run names **`hpt_bm_a{α}_b{β}`**

**Monitor** — background ~30-min poll: tmux liveness, per-GPU util (idle GPU while
tmux alive ⇒ stall), and `Traceback|OOM|NaN` in `runs/<name>/train.log`.

**Judge** (no val — training curve only), primary → secondary:
1. `critic/rewards/mean` (score) trajectory over steps ~10–75 — higher/more-monotone wins.
2. `actor/grad_norm` — stability (no explosion = the k-collapse signature).
3. `actor/pg_loss` shape; `response_length` (no collapse/runaway).
4. comm-eff counters: `comm_eff/bytes_ratio` (≈equal across cells), `merger_coldM_fallbacks` (→0 after ~step 10), `anchor_backwards` (=7).

**Close-out** — backfill the last 1–2 steps to WandB from `train.log`; tear the box
down once idle + synced.

---

## 5. What this does NOT do
- No code changes (config-only sweep of a working method).
- No validation / eval accuracy — training-curve dynamics only.
- Does not touch any concurrent weight-projection run (separate box, separate group).
- Only α and β vary; substrate, surface, dataset, latency held fixed.
