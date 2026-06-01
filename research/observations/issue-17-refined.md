# M3 — Long-horizon lossy-compression GRPO: periodic full-gradient refresh (clean_cadence K=20) over 2 epochs — the CORE single-run diagnostic

## Project context (self-contained — harness agents do not read `major-goal/`)

Communication-efficient, pipeline-parallel GRPO. The training path runs under per-(token, channel) activation masking at the pipeline-stage boundaries (`h̃ = h·mask`, inverted-dropout rescale `1/(1−p)`); rollouts come from ordinary unmasked vLLM. Model/algorithm/data fixed: `Qwen/Qwen2.5-1.5B-Instruct`, vanilla GRPO **no-KL no-entropy**, GSM8K. "Done" (`research/.claude/GOAL.md`) = stable + parity-with-dense + measured savings + one launcher.

This issue is **entirely about understanding** what masked GRPO *does* over a long horizon when we inject the **true dense gradient only periodically (every 20 steps)** and apply **no correction** to the train-inference mismatch in between: does it keep learning, diverge, entropy-collapse, saturate, or **slowly drift into a weight state the periodic clean step can no longer repair** — and does the masked actor (a deliberately different function from the unmasked vLLM sampler) degrade the **true** policy toward random, or does on-policy GRPO's structure protect the saved weights. It is the **core characterization run** that the next PP-comm-efficient algorithm will be designed around.

## Relation to prior issues — supersedes the anchor+spectral plan in #11

EXP-16 (#16, PASS, PR #10) established, on 4×B200 Qwen2.5-1.5B/GSM8K (B200 was a forced substitution — no H200 rentable 2026-05-30 — **not** a memory requirement; see "Multi-GPU mandate" below), p=0.9 per-(token,channel) mask:

- **Rescale is necessary for *unbiasedness*** (not as a grad-norm tamer): no-rescale → `grad_norm ~2698`, val 0.082; rescale → `grad_norm ~5–8` (masked) / `~0.38` (dense), val 0.729. The residual ~13–20× masked grad-norm is the **variance** penalty `≈ p/(1−p)`, bounded by Adam scale-invariance + grad-clip=1.0.
- **Pure masked p=0.9 does NOT learn**: reward flat `0.126 → 0.147` over 10 steps.
- **Anchor + spectral as configured does NOT learn and does NOT close the train-inference gap**: `anchor@2+spectral@2` → reward `0.140 → 0.131`, **GSM8K val 0.080 ≈ random**, masked pearson(actor,rollout) `~0.0045`, spectral `rel_change ~0.24`. Root cause: spectral is a *linear reweighting of the masked gradient* in the anchor's SVD basis; masking makes `G_mask` nearly **orthogonal** to the dense direction (cos≈0), so no linear projection recovers it — and the clean anchor gradient is **never applied** (it only feeds the EMA `M`, by design / verified in `anchor.py`).
- **The periodic CLEAN (unmasked) optimizer step DOES learn**: `clean@5` over 50 steps → reward `0.13 → 0.778`, **GSM8K val 0.729 vs dense 0.741 (within 1.2 pts)**, train-inference gap **stationary** (pearson slope +4e-5/step, R²=0.14). `clean@4`/20 → val 0.696.

**Lesson:** the clean signal must be **applied** (clean_cadence), not used as projection geometry (anchor+spectral). This issue takes that one proven-working lever to a **sparse** cadence (K=20, 4× sparser than the validated K=5) over a **long** horizon (2 epochs) to find where it breaks. **#11 should be closed as superseded** (its M95+AP anchor+spectral K=20 hypothesis is falsified by EXP-16).

## Harness fields (parsed by research-planner)

```yaml
kind: experiment
code_change: false            # every knob already exists (clean_cadence, test_freq, epochs) — pure config
milestone: M3
baseline_run: EXP-16 dense_ref (no-KL, 25-step, val 0.741)   # nearest no-KL control; EXP-3 (with-KL, 0.789) for reference only
depends_on: [16]
seed_replicates: 1
budget_gpu_hr: 96
budget_dph_max: 24.0
max_parallel: 1
target_modules: []            # no source edits; config-only run from vast-ai-workload
escalate_to_codex_if:
  - "actor/grad_norm non-finite"
  - "NaN detected"
  - "nan|NaN|inf|Inf in (loss|grad_norm|reward|log_prob)"
  - "mask applied on (rollout|log_prob|ref|val|infer|checkpoint)"
  - "clean step did not freeze mask counters"
  - "RuntimeError: CUDA out of memory"
  - "FSDP .*(shard|reduce|reduction).* error"
  - "cgroup pids.max .* (<=|too tight)"
  - "VAST_API_KEY found in container env"
```

## Hypothesis

Comm-eff masked GRPO (`p=0.9`, rescale ON, `mask_recompute=true`, per-(token,channel) mask at the 7 pipeline boundaries `[3,7,11,15,18,21,24]` of Qwen2.5-1.5B) with the optimizer stepping on the **true unmasked dense gradient every 20 steps** (`clean_cadence=20`, anchor+spectral **OFF**), run for **2 epochs (~116 steps)** on GSM8K, **keeps learning and the periodic clean step keeps fully repairing the weights** (no irreparable drift):

- GSM8K `val/test_score` rises (no divergence, no entropy collapse, no plateau-then-fall) to `≥ step0 + 0.05`;
- `actor/grad_norm` stays a **bounded sawtooth** — masked-step peaks ~5–8, resetting to the dense `~0.4` on each clean step (20/40/60/80/100); no NaN/Inf;
- **each clean step fully re-anchors**: clean-step `grad_norm` stays ~0.4 across all of 20/40/60/80/100 (does **not** trend upward), and the post-clean repair of val/gap does **not** weaken over the run — i.e. the 19 masked steps between clean steps do not accumulate damage the clean step cannot undo;
- the **true policy does not drift toward random**: clean-step (true-policy) `actor/entropy` stays sharp (~0.4, sharpening like dense's 0.38→0.24), it does **not** climb toward the masked-forward value (~5.9), and clean-step `val` stays ≫ 0;
- the **train-inference mismatch stays STATIONARY** (does not grow) even at this 4×-sparser clean cadence — `training/rollout_probs_diff_mean`, `rollout_corr/kl`, and masked-step pearson(actor,rollout) have ≈ 0 slope over the run, including when binned by steps-since-last-clean;
- masked-step `pg_clipfrac` stays bounded (well below the no-rescale ~0.15), not saturating — the ratio clip is containing each masked update;
- boundary-activation communication savings ≈ **85.5%** of dense (19/20 steps masked at p=0.9).

**Falsified if:** `val/test_score` diverges, collapses, or saturates-then-drops; `grad_norm` non-finite; clean-step (true-policy) entropy collapses toward 0 **or** climbs toward the masked value while reward falls; clean-step `grad_norm` **trends upward** across 20/40/60/80/100 (repair losing ground to drift); or the train-inference gap (pearson / `rollout_corr/kl` / clipfrac) **grows monotonically** (a ratchet, not a clean-resettable sawtooth) over the run.

## Run config (single run; pure-config, launch from `vast-ai-workload`) — UNCHANGED from the validated EXP-16 shape

Launcher: `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`

```bash
PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=grpo_mask_channel_p0p9_rescale_clean_every20_2epoch \
COMM_EFF_ENABLED=true \
COMM_EFF_MASK_ENABLED=true \
COMM_EFF_MASK_P=0.9 \
COMM_EFF_MASK_RESCALE=true \
COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=20 \
COMM_EFF_ANCHOR_ENABLED=false \
COMM_EFF_SPECTRAL_ENABLED=false \
TOTAL_EPOCHS=2 \
TOTAL_TRAINING_STEPS=116 \
TEST_FREQ=10 \
VAL_BEFORE_TRAIN=True \
USE_DYNAMIC_BSZ=True \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

Everything else stays at launcher defaults to match EXP-16/EXP-3 shape: `ACTOR_LR=1e-6`, `ROLLOUT_N=8`, `TRAIN_BATCH_SIZE=128`, `PPO_MINI_BATCH_SIZE=64` (⇒ 2 optimizer sub-steps/global-step), `MAX_PROMPT=1024 / MAX_RESPONSE=16384`, no-KL no-entropy, rollout correction **STRICTLY OFF** (`rollout_is=rollout_rs=null, bypass_mode=false`; `calculate_log_probs=True` is a read-only train-inference *diagnostic*, not a correction). `USE_DYNAMIC_BSZ=True` is the EXP-16-proven perf knob (MFU 0.75%→13.86%, step 129s→37s, peak 62 GB) and is correct here because the per-(token,channel) mask is packing-invariant.

Schedule: GSM8K train ≈ 7473 ex / 128 ≈ 58 steps/epoch → 2 epochs ≈ **116 steps**. Clean (dense) steps fire at **20, 40, 60, 80, 100**; validations at **0, 10, 20, …, 110**.

## What to measure — grouped by question (one run answers all the main claims)

> All metrics below are **existing WandB scalars** or analyst-side post-processing of them (binning, trajectory overlay, slope fits). No new logging ⇒ `code_change:false` holds. "Masked step" = any step not a multiple of 20; "clean step" = 20/40/60/80/100.

### Group A — Does it learn, and how fast vs the true dense gradient?
1. **LEARNING** — `critic/score/mean` (reward) + `val/test_score` every 10 steps. Monotone climb toward the dense reference (0.741 no-KL / 0.789 with-KL)?
2. **LEARNING SPEED vs dense (sample efficiency of the compressed path)** — overlay #17's reward/val trajectory against three EXP-16 references on the same step axis: dense `cell6` (val 0.741), `clean@5`/50 (0.729), `clean@4`/20 (0.696). Report (i) steps-to-reach reward ≥ 0.5, (ii) final-val gap to dense, (iii) reward slope. This quantifies the **cost of sparse clean cadence**: with only ~5 true-gradient steps over 116, how much slower / lower-ceiling is K=20 than the dense path? (Diagnostic number, not pass/fail — parity is a bonus here.)

### Group B — Is the *true* policy healthy, or drifting toward random?
3. **TRUE-POLICY ENTROPY (clean steps only)** — `actor/entropy` **on the clean steps** is the *true policy* entropy (masked-step ~5.9 is a mask artifact — the flattened masked *forward*, not the weights — ignore it for policy health). Dense ref went 0.38 → 0.24 (healthy sharpening). Track the clean-step entropy trajectory across 20/40/60/80/100: monotone-decreasing (sharpening, healthy) vs **rising** (drifting toward a flat/random policy).
4. **RANDOM-POLICY CHECK (the masking is on activations, not weights)** — the mask zeroes 90% of channels in the *training forward*, so the masked policy *looks* near-uniform (entropy ~5.9). But the **weights are not masked** — the question "is the model becoming a totally random policy because we mask in the trainer?" is about the **true** policy, which is what vLLM rollouts and the clean step see. Evidence the weights are NOT random: (a) clean-step `val/test_score` stays ≫ 0 (a random GSM8K policy scores ≈ 0); (b) clean-step entropy stays sharp (~0.4), nowhere near `log|V|`. **Track the masked-entropy ↔ clean-entropy gap each cadence**: if the *clean*-step entropy starts climbing toward the masked value, the weights themselves are degrading toward the flat regime — the real failure this run must rule out.

### Group C — Train-inference mismatch with NO correction (masked actor ≠ vLLM sampler)
5. **TRAIN-INFERENCE GAP** — the masked actor is *deliberately* a different function from the unmasked vLLM sampler; the GRPO importance ratio absorbs it (and `old_logprob` is masked too ⇒ pre-update ratio ≈ 1, `ppo_kl ≈ 5e-4`). Metrics: `training/rollout_probs_diff_mean`, `rollout_corr/kl`, masked-step pearson(actor,rollout). At `clean@5` these were **flat** (slope ≈ 0, R²=0.14). **New at K=20:** (a) global slope over the run — stationary or growing? (b) **bin each masked step's gap by steps-since-last-clean (0..19)** — *flat vs position* ⇒ the masked policy is self-stationary (gap is mask-caused, not drift); *rising with position* ⇒ within-window drift that the clean step resets (a sawtooth in the gap itself). K=20 gives 4× longer windows than the validated K=5 to expose any such drift.

### Group D — Clean-step repair vs irreparable drift (the core long-horizon question)
6. **CLEAN-STEP REPAIR DYNAMICS** — at each clean step the true dense gradient is *applied*. Two competing outcomes this run adjudicates:
   - *(H-repair)* on-policy GRPO + bounded masked updates keep the weights in a good basin; each clean step fully re-anchors and learning compounds ⇒ **clean-step `grad_norm` stays ~0.4** across 20/40/60/80/100, and the post-clean improvement in val/gap does not weaken.
   - *(H-drift)* the 19 masked steps between clean steps slowly corrupt the weights into a region the single clean step cannot climb out of ⇒ **clean-step `grad_norm` RISES** over the run (0.4 → …), the per-clean-step "repair delta" (metric just-after minus just-before the clean step, for val/gap/entropy) **shrinks**, and val measured *at* clean steps saturates-then-declines.
   Measurables: (i) clean-step `grad_norm` trajectory across the 5 clean steps — flat ⇒ repair, monotone-rising ⇒ accumulating drift; (ii) per-clean-step repair delta on val/gap/entropy — constant/growing ⇒ healthy, shrinking ⇒ damage outpacing repair; (iii) val *at* clean steps climbs (20<40<60<80<100) vs peaks-then-falls.
7. **CLIPPING as the protective valve (does on-policy GRPO protect the weights?)** — masked-step `actor/pg_clipfrac` / `pg_clipfrac_lower`. The ratio clip bounds the magnitude of each masked update *regardless* of how flat the masked forward is, and advantages come from the **correct unmasked vLLM rollouts** — both are candidate mechanisms for why the saved weights don't corrupt even though the masked actor differs from the sampler. Stable ~0.04 ⇒ the clip is containing it; **climbing toward saturation between clean steps** ⇒ the masked policy is outrunning the K=20 re-anchor (protection overwhelmed — points to needing a *continuous* on-policy-masked correction, not a sparser clean step). (no-rescale was ~0.15.)

### Group E — Stability, saturation, savings
8. **DIVERGENCE** — `actor/grad_norm` finite and a bounded sawtooth (resets to ~0.4 at clean steps); zero NaN/Inf.
9. **SATURATION** — does reward/val plateau or peak-then-decline before step 116? (distinct from Group D: this is the overall curve shape; D is the clean-step repair trend.)
10. **SAVINGS** — 19/20 steps masked at p=0.9 ⇒ ≈ **85.5%** of boundary-activation traffic saved vs dense; log the concrete number.

## Mandatory consistency checks (cheap, must hold)

- Clean steps fire at exactly 20/40/60/80/100: `comm_eff/mask_applications/{train,old_logprob}` **freeze** (no increment), `grad_norm → ~0.4`, `ppo_kl → ~1e-5`, `pg_clipfrac → ~4e-4`, `pg_clipfrac_lower → 0`.
- Masking confined to actor-train: `comm_eff/mask_applications/{rollout,ref_logprob,val,infer,ckpt} == 0` every step.
- Anchor/spectral OFF ⇒ `comm_eff/{anchor_backwards,spectral_corrections} == 0` all steps.

## Success criteria (machine-checkable)

- [ ] Reaches `global_step ≥ ~116` (2 epochs); no NaN/Inf in any loss/grad/reward/log-prob field.
- [ ] `actor/grad_norm` finite throughout; masked-step median `< 10`, clean-step `~0.4`.
- [ ] `val/test_score_final ≥ val/test_score_step0 + 0.05` (still learns under sparse-clean compression).
- [ ] **Clean-step `grad_norm` does NOT trend upward** across 20/40/60/80/100 (each clean step still fully re-anchors — repair not losing ground to drift).
- [ ] **True policy not drifting to random**: clean-step (true-policy) `entropy` neither collapses toward 0 nor climbs toward the masked ~5.9 while reward declines; clean-step `val` stays ≫ 0.
- [ ] **Train-inference gap stationary**: |slope| of pearson, `rollout_corr/kl`, `rollout_probs_diff_mean` over masked steps within noise (R² < ~0.3 or slope ≈ 0); the steps-since-clean binning shows no monotone within-window growth, or any growth fully resets at each clean step (clean-resettable sawtooth, not a ratchet).
- [ ] Masked-step `pg_clipfrac` not saturating (stays well below ~0.15; ideally bounded ≲ 0.08).
- [ ] Clean steps fire at 20/40/60/80/100 with counters frozen + ratio≡1; masking confined to actor-train.
- [ ] Learning-speed gap vs dense / clean@5 / clean@4 reported (steps-to-threshold + final-val gap + slope).
- [ ] Boundary-activation comm-savings number logged (~85.5%).
- [ ] (Optional) `COMM_EFF_ENABLED=false` short re-check reproduces dense GRPO (already proven EXP-16).

## Analyst predicate

- **PASS** — learns (`val ≥ step0 + 0.05`), non-divergent / non-collapsing through 2 epochs, train-inference gap stationary (or a clean-resettable sawtooth), clean-step `grad_norm` not trending upward (clean step keeps fully repairing), true policy not drifting to random. Parity with dense is a **bonus, not required** — this is a characterization run, not a parity certification.
- **REVISE** (≤ 2 iterations) with concrete `next_actions:` — if it learns but with a worrying trend (gap or clipfrac climbing, **clean-step grad_norm trending up / repair delta shrinking**, or val plateaus early), tighten the cadence (K 20→10, fresher dense injection so the clean step re-anchors before drift accumulates) and re-characterize; if it learns cleanly, the natural follow-up is the comm-savings/quality trade sweep `K ∈ {10, 20, 40, never}`.
- **STOP** — divergence / entropy collapse / val crash / **irreparable drift no K recovers** ⇒ the periodic clean step does not scale to sparse cadence. A real, loggable **negative result** that reshapes the algorithm design: it would push away from "apply the true dense gradient more often" toward a **cheaper continuous on-policy-masked correction** that holds the masked actor close to the sampler every step instead of re-anchoring periodically.

## Open questions this run feeds into the new PP-comm-efficient algorithm

1. **Learning speed / sample efficiency** — is masked + clean@20 materially **slower to learn** than the true dense gradient, and by how much? With only ~5 clean steps over 116, what fraction of dense's final score does the compressed path reach, and is the masked gradient between clean steps *contributing* learning or merely *not corrupting*? (Clean ablation: clean-only vs masked+clean at matched cadence — a fast follow-up; within this run, attribute the rise to masked-windows vs clean steps by reading reward per step.)
2. **Irreparability** — does the between-clean masked drift **accumulate** into damage the periodic clean step can no longer repair over a long horizon (rising clean-step `grad_norm`, shrinking repair delta, saturate-then-fall), or does each clean step fully re-anchor indefinitely? This is the question that decides whether "periodic clean refresh" is a viable *long-run* strategy or only a short-run crutch.
3. **On-policy GRPO as a weight-protection mechanism** — does GRPO's on-policy structure (importance-ratio clipping bounding each masked update + advantages computed from the *correct* unmasked vLLM rollouts + `old_logprob` recomputed under the same mask so the pre-update ratio ≈ 1) inherently **protect the saved weights from corruption** even though the masked actor is a different function from the vLLM sampler — i.e. is the policy *allowed* to drift "a bit" from vLLM each step without the weights degrading? If yes, the next algorithm can lean on that tolerance and mask more aggressively between cheap corrections.
4. **Where it breaks** — at what clean cadence does learning break (this run = K=20; the trend it shows sets the next sweep `K ∈ {10, 20, 40, never}`), and does the train-inference gap's stationarity survive the sparser re-anchoring?

## Multi-GPU mandate + locked template

Multi-GPU only, 4 ≤ num_gpus ≤ 8, **H100/H200** (EXP-16's B200 was a one-off substitution for H200 unavailability on 2026-05-30 — not required), locked template `verl-research-vllm020` (hash `3b0f8b726ac3036d6c007bfa13b6d75f`, id 447527), skills only. Anchor **and** spectral are **OFF**, so neither the ~3 GB anchor clone nor the spectral GPU-EMA/SVD is allocated; the EXP-16 dense/clean path on this exact shape peaked at **~62 GB/GPU** (`USE_DYNAMIC_BSZ=True`, `max_token_len=98304`) — so **4×H200 ≥140 GB fits the full 16K-response shape with > 2× headroom, unchanged**. No reduction of batch / response length / rollout-n is needed. Fallback **8×H100 ≥80 GB**.

```yaml
gpu_count: 1
gpu_filter_chain:
  - "num_gpus=4 gpu_name=H200 gpu_ram>=140 cuda_max_good>=13.0 reliability>=0.97 rentable=true verified=true"
  - "num_gpus=8 gpu_name=H100 gpu_ram>=80 cuda_max_good>=13.0 reliability>=0.97 rentable=true verified=true"
  - "num_gpus=8 gpu_name=H200 gpu_ram>=140 cuda_max_good>=13.0 reliability>=0.97 rentable=true verified=true"
max_dph: 24.0
max_gpu_hr: 96
max_parallel: 1
wall_clock_hr: 12
iterations: 2
```
