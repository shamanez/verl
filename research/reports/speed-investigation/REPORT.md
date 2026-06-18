# VERL Speed Investigation — report + orchestrator verification

> **Status:** GPU-free analysis, delivered 2026-06-18 (Workflow wf_f6d3cb93-5bc, 26 agents, 16/20 knobs admissible after adversarial neutrality critic, 4 rejected).
> **ALL GPU-consuming steps remain gated on the EXP-35 issue being CLOSED** (operator directive).

## ⚠️ ORCHESTRATOR VERIFICATION (read before trusting §2 knob #1 and the FAST-CONFIG)

The report's single highest-leverage knob (#1: "raise `ppo_max_token_len_per_gpu` 3000→8192") and its KL caveat rest on values the profiler read from the *launch command*, which passes **duplicated Hydra overrides** (launcher defaults, then the authoritative comm_eff block; Hydra is last-wins). I confirmed the **resolved** config from verl's own OmegaConf dump in `runs/EXP-35/exp-35-c3-a050/train.log`:

- `'ppo_max_token_len_per_gpu': 24576` (actor) and `32768` (log_prob/ref) — **NOT 3000**.
- `'use_kl_loss': False`, and **0** `actor/kl_loss:` metrics logged — KL is OFF (matches the fixed control surface).

**Consequences:**
1. **Knob #1 is MOOT** — the actor token budget is already 24576, above the 8192 the report proposes. The run is **not** over-partitioned. By the report's own §3 verdict, this means **config-only Tier A will NOT clear <25 min**.
2. The KL caveat is void (KL confirmed off — no math discrepancy).
3. **The real path to <25 min** is therefore: **(Tier C) a code patch (exp/* branch) gating the comm_eff diagnostic overhead** — the ~200 debug prints/step + the ~196 per-matrix synchronous `relative_change().item()` GPU→CPU syncs/step + the relevance probe (knobs #10–12) — which attack the measured dominant sink (the 9.4 min/cell anchor circuit) and align with the existing "diagnostics-off on production arms" policy; **plus (Tier B) memory-gated device placement** (#2 `spectral.ema_device=gpu`, #3 `anchor.snapshot_device=gpu`), which attack the anchor CPU↔GPU shuffling but add 6–18 GB to the ~124/143 GB anchor-step peak and must pass a memory canary. Minor neutral I/O (#6 HF-offline cells 2–5, #8 `save_freq=0`, #7 `forward_prefetch`, #4 chunked-prefill) rounds it out.

**Bottom line:** still achievable, but it requires a code change + a memory canary, not a one-line config flip. The biggest reliable win is stripping the per-step diagnostic syncs/prints from the anchor/spectral path.

## Workflow profiler logs (verbatim)
- profile: top sinks = #1 update_actor = 1004.6s total / 16.7 min = 67.8% of the 24.7 min step-compute (c3-a050; c1-a000 identical within noise). This is the actor backward+optimizer phase. | #1a Within update_actor, the ANCHOR is the single biggest sink: normal update_actor avg 8.8s vs anchor avg 37.1s => the anchor second-backward + signed_ema spectral-correction merger adds ~28.3s on each of the 20 anchor-firing steps = 566s / 9.4 min of pure anchor overhead (38% of the WHOLE cell's step-time spent on the anchor circuit alone). Anchor fires on 20/49 post-warmup steps = 40.8% (cadence/delay_K=5 in optimizer-ticks = every ~2.5 global steps, confirmed via comm_eff/anchor_backwards counter 0->20). | #1b The non-anchor part of update_actor (~263s / 4.4 min over 30 steps) is the irreducible GRPO actor forward+backward+step; that is hard-locked compute.
- investigate: 20 candidates -> 20 unique
- critique: 16 admissible / 4 rejected

---

I'll synthesize the report directly from the profiler data and admissible-knob verdicts provided. No file investigation is needed — all the load-bearing evidence is in the brief. Let me produce the decision-ready markdown.

# EXP-35 Speed Investigation — Get one cell under 25 min without changing the optimization math

**Goal:** cut one EXP-35 training cell from the observed ~28 min to **under 25 min**, touching **only** pure engineering/scheduling axes that leave the loss, gradients, RNG stream, and rollout distribution bit-for-bit or statistically identical. Hard-locked controls (batch 128, mini 64, lr 1e-6, n=8, model, GRPO loss, total steps 50, max_response 2048, the PowerSGD r=77 + mandatory anchor + signed_ema substrate) are **out of scope**.

> **Live-config caveat up front (changes the math of every estimate below).** The profiler reconciled wall-clock against the *actual launch command in the log*, which uses `actor.ppo_max_token_len_per_gpu=3000`, `log_prob_max_token_len_per_gpu=4096/8192`, `update_weights_bucket_megabytes` at the **config default 2048** (not 4096 — verified the two live launchers set no override), and `use_kl_loss=True kl_loss_coef=0.001 low_var_kl`. These differ from the values in the orchestrator brief (24576 / 32768 / no-KL). The KL discrepancy does **not** affect the speed knobs (it changes the optimization math and is hard-locked either way), but the **`ppo_max_token_len_per_gpu=3000`** value is load-bearing: it means the run is heavily over-partitioned and the single highest-leverage knob has large headroom. Confirm the resolved Hydra config before trusting the FAST-CONFIG below.

---

## 1. Where the 28 minutes goes

Authoritative total: the 50 `timing_s/step` values sum to **1481.7 s = 24.7 min** of step-compute (c3-a050; c1-a000 identical within noise). Add in-log init (~73–83 s) and the pre-Ray shell/docker/HF-import phase, and the ~28 min/cell observation reconciles.

| Phase | Total / cell | Share of step-compute | Notes |
|---|---|---|---|
| **update_actor** | 1004.6 s / 16.7 min | **67.8%** | The dominant sink. Splits into anchor vs non-anchor below. |
| — anchor overhead | 566 s / 9.4 min | 38% of whole cell | ~28.3 s **extra** on each of 20 anchor-firing steps (37.1 s anchor vs 8.8 s normal update_actor). Driver: second backward on the replayed paired batch (snapshot_device=cpu ⇒ CPU↔GPU shuffling) + signed_ema spectral merger over all 196 matrices (cadence=1). First anchor (step 3) costs +20 s one-time for M-matrix/buffer alloc. |
| — non-anchor actor compute | ~263 s / 4.4 min | over 30 normal steps | Irreducible GRPO forward+backward+step. **Hard-locked.** |
| **gen (vLLM rollout)** | 233.6 s / 3.9 min | 15.8% | ~4.4 s/step steady; step 1 alone is 16.7 s (CUDA-graph capture warmup). |
| **reward** | 132.4 s / 8.9% | — | **Overlaps gen/old_log_prob** — sum-of-parts exceeds step total by 2–4.5 s/step. True marginal wall-clock ≈ 0. Do **not** count as additive. |
| **old_log_prob** | 112.6 s / 1.9 min | 7.6% | Policy log-prob recompute for the importance ratio. |
| **update_weights** | 111.5 s / 1.9 min | 7.5% | FSDP→vLLM weight resync, ~2.2 s flat every step. |
| adv | 5.1 s | 0.3% | Negligible. |
| **testing (val)** | ~15.4 s total | ~1% | Fires only at steps 25 & 50 (test_freq=25), greedy single-pass. Both "useful" (no wasted val@55 since total=50). |
| **save_checkpoint** | ~4.4 s total | <1% | Fires only at step 50 (save_freq=50). |

**Per-step profile:** NORMAL step ≈ 17.6–17.9 s; ANCHOR step ≈ 46.4–46.8 s. Anchor fires on **20/49 post-warmup steps = 40.8%** (cadence/delay_K=5 in optimizer ticks ⇒ every ~2.5 global steps; confirmed via `comm_eff/anchor_backwards` 0→20). `ref` has **no** separate timed phase under the KL path. **`gen / old_log_prob / update_weights / adv` are byte-identical between anchor and normal steps** — the entire anchor delta lives inside `update_actor`.

**Init:** ~73–83 s in-log (Ray + FSDP load + vLLM build + CUDA-graph capture). Operator's observed "~4–5 min" wall-clock additionally includes the pre-Ray shell/docker/HF-resolve/verl-import phase that precedes the first logged timestamp, so in-log ~1.3 min is a **lower bound**. `val_before_train` was False in both cells ⇒ no val@0 inflating init.

**Headline:** the time is in two places — the **anchor circuit (9.4 min, 38%)**, which is substrate and mostly hard-locked *as math* but has CPU↔GPU device-placement and micro-batch-count engineering levers, and the **non-anchor `update_actor` fixed-per-micro-batch overhead**, which the over-partitioned `ppo_max_token_len_per_gpu=3000` inflates. Both of those, plus a cluster of small neutral I/O knobs, are the admissible surface.

---

## 2. Ranked admissible knobs

Ranked by (expected speedup × confidence ÷ risk). All passed the neutrality critic. "Neutrality class" — **NEUTRAL** = bit-for-bit identical trajectory; **NUMERIC_ONLY** = FP-reduction-order or sampling-realization differences only, statistically identical, no bias.

| # | Knob | Expected speedup | How to enable (exact) | Neutrality class & verification |
|---|---|---|---|---|
| **1** | **Raise `ppo_max_token_len_per_gpu`** (3000 → 8192–12288) | **Largest.** Directly attacks #1 sink. Halving micro-batch count cuts the fixed per-micro-batch overhead in both the 263 s non-anchor and 566 s anchor update_actor (anchor replays the *same* paired batch ⇒ inherits the lower count). Plausibly **several s/step on anchor steps**. | `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192` (sweep upward, watch peak mem). | **NUMERIC_ONLY.** Token-mean divides by the **global** all-reduced token count (core_algos.py:1173), so the partition only reorders the FP sum ⇒ bit-identical for the dense control, FP-order-only for the PowerSGD sketch `V=Σ Mᵀ(M@Q)` (count never enters `Q=orth(V)`). Verify: A/B 3000 vs 8192 with codec ON — grad_norm/loss/codec rel_change/val@25 must track within step-noise with no divergent drift over 50 steps; dense-OFF control must be bit-identical to ~1e-5. **Assert no OOM on anchor steps first.** |
| **2** | **`spectral.ema_device` cpu → gpu** | Removes ~196 small H2D copies/step (M_anchor per matrix) from the every-step signed_ema merger + the per-fire D2H. Plausibly **low-seconds/step**. | `+actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu` (hydra, no code change). | **NEUTRAL.** Device placement only; the EMA math runs on GPU regardless, fp32 `.to()` round-trip is byte-preserving. Verify: 5–8 step canary vs cpu, same seed — loss/grad_norm/rel_change match (ideally bit-identical), **watch `max_memory_allocated`** (adds ~6 GB fp32 M-state to HBM; `ema_device=cpu` is a STANDING OOM guard per FIXED_CONTROL_SURFACE.md:116). |
| **3** | **`anchor.snapshot_device` cpu → gpu** | Removes a full-model (~3 GB bf16) H2D copy per anchor fire (20 fires) from the 37 s anchor update_actor. **Potentially the largest device-placement lever** if it fits. | `+actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=gpu` (hydra, no code change). | **NEUTRAL.** Snapshot placement only; `copy_(s.to(p.device,p.dtype))` preserves bf16 bytes ⇒ G_anchor + M-EMA bit-identical. Verify: per-fire clone-state hash identical cpu vs gpu; trajectory bit-identical. **Highest OOM risk** — delay_K=5 ⇒ up to ~6 full-model bf16 snapshots (~18 GB) resident in HBM. Flip #2 first; only flip #3 if the canary shows headroom. |
| **4** | **`enable_chunked_prefill` → True** | Smooths the gen wave (interleaves prefill with decode). On short-prompt GSM8K, ~5–15% of gen ⇒ **~0.2–0.6 min/cell**. | `actor_rollout_ref.rollout.enable_chunked_prefill=True` (currently pinned False). `max_num_batched_tokens=8192 > max_model_len` so no extra knob. | **NUMERIC_ONLY (rng_stream realization).** Scheduler-only; does not bias the rollout *distribution*, and old_log_prob is train-recomputed so the update is unaffected. BUT chunked attention tiling → bit-different prompt logits → under temp>0 sampling a fixed seed draws **different tokens** ⇒ different realized trajectory. **Must be enabled on ALL cells incl. the baseline — not a free swap against an already-collected baseline.** Verify: 10-step A/B, grad_norm/pg_loss within seed-noise band, rollout_probs_diff not widened. |
| **5** | **`update_weights_bucket_megabytes` 2048 → 4096** | Consolidates the ~3 GB bf16 resync into one bucket. Sub-fraction of the 2.2 s/step phase ⇒ **~0.1–0.3 min/cell**. | `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096`. (Profiler's "already 4096" note is wrong — live launchers set no override ⇒ default 2048; lever is **not** exhausted.) | **NEUTRAL.** Pure transport chunking; bytes/dtype identical (`view(uint8)` copy, no cast). Verify: hash vLLM-loaded params bitwise-equal 2048 vs 4096; watch transient staging buffer (larger is the safe direction). Low value. |
| **6** | **`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`** (cells 2–5) | Skips per-file HF-hub HEAD/metadata round-trips at init on already-cached cells. **A few s to tens of s/cell** init, network-latency-bound. Nothing on cell 1. | In `launch.sh` `run_cell()`, add both env vars to every cell **after** the first (or pre-`hf download` once, export for all). | **NEUTRAL.** Same cached bytes, zero network. **Precondition:** cell 1 must run ONLINE to populate cache (errors loudly otherwise). Verify: offline cell still logs "Loading weights" cache-hit <1s; val@0/grad_norm match. |
| **7** | **`forward_prefetch=True`** (FSDP1) | Overlaps next-layer all-gather with current-layer forward in both normal & anchor update_actor. NVLink already cheap ⇒ ~3–8% of forward, **likely <1 min/cell**. | `actor_rollout_ref.actor.fsdp_config.forward_prefetch=true` (+ ref worker optionally). | **NEUTRAL.** All-gather is deterministic data movement, no reduction reorder ⇒ stricter than NUMERIC_ONLY. Verify: scalars **bit-identical** (not just within-noise); any digit drift ⇒ investigate. Static-graph caveat satisfied (196 targets registered once). |
| **8** | **`save_freq=0`** (disable step-50 checkpoint) | ~4.4 s/cell. Verdict is scalar-based (val@25/@50 from log/W&B); checkpoint never re-read. | `SAVE_FREQ=0` in run_cell env, or `trainer.save_freq=0` override. **Keep `test_freq=25`.** | **NEUTRAL.** Save gate short-circuits; happens after final optimizer step, draws no RNG. Verify: 50-step scalar streams bit-identical; no `global_step_50/` dir written. |
| **9** | **`forward_prefetch` for ref** — folded into #7. | — | — | — |
| **10** | **Gate/throttle spectral debug prints + flush** (`spectral_filter.py:1199-1204`, also 354/386/1222) | Removes ~196–392 synchronous `flush=True` writes/step (~9.6k–19.6k lines/cell, 2.3 MB log). **Sub-second to low-single-digit s/cell** + cheaper rsync/heartbeat. | **Code patch (exp/* branch):** wrap print in `if os.environ.get('COMM_EFF_SPECTRAL_VERBOSE','0')=='1':` (default off) or drop `flush=True`. **Keep `state.spectral_rel_change[name]=rel` unconditional** so W&B is unchanged. | **NEUTRAL.** Diagnostic-only, runs after writeback. Verify: spectral_corrections count, grad_norm, rel_change_mean bit-identical. Gate-don't-delete (analyst greps `rel_change=`). |
| **11** | **Defer/batch per-matrix `relative_change().item()`** (`spectral_filter.py:1197`) | Removes ~196 synchronous GPU→CPU syncs/step from the non-anchor update_actor critical path. **Sub-second to low-seconds/step** plausibly. | **Code patch (exp/* branch):** stash norm tensors, single batched `.item()`/`.tolist()` post-loop; or gate behind the same flag as #10. Keep writing `spectral_rel_change`. | **NEUTRAL (measurement_only).** rel feeds only W&B, never the optimizer; writeback(grad, g_proj) untouched. Verify: post-writeback grad tensors bit-identical; nsys shows ~196 fewer syncs/step. Batched-item may shift logged scalar at last ULPs (expected, measurement-only). |
| **12** | **Gate relevance-probe + anchor-canary prints** (`transformer_impl.py:1234-1267/1726-1729/1809-1819` probe; `1661-1671` canary print) | Probe: removes per-micro-batch padded-tensor materialization + reduce from the 37 s anchor path (20 fires). Canary print: ~0. **Low-seconds aggregate at most.** | **Code patch (exp/* branch):** gate probe install + prints behind a debug flag. **Keep the canary `assert _can_ok`** (1672-1678) — bitwise staleness guard for snapshot_device. | **NEUTRAL (measurement_only).** Probe is detached/fp32/scalar-only, swallows its own exceptions, never touches loss/G_anchor/EMA. Verify: grad/loss/M bit-identical incl. the 20 anchor steps; only the `logp_mad=` / `anchor-canary` lines disappear. |
| **13** | **Warm Ray head across cells** | Amortizes ~2.2 s Ray bootstrap × 4 reused cells = **~8–12 s over the sweep**. Does **not** skip per-cell vLLM build / CUDA-graph / FSDP load. | `ray start --head --num-gpus=4 --port=6379 && export RAY_ADDRESS=127.0.0.1:6379` before the loop; `ray stop` after. | **NEUTRAL.** Control-plane only; seeding is config-derived, not Ray-derived. **Fragility risk (gates adoption, not neutrality):** an ignited cell can leave un-reaped actors/GPU mem and OOM a later cell; current cold-boot is fully isolated. Verify each cell logs "Connecting to existing Ray cluster" + nvidia-smi reaped between cells. Poor cost/benefit. |
| **14** | **Keep `tensor_model_parallel_size=1`** (do NOT raise) | Confirmation, not a change. TP=1/DP=4 is throughput-optimal for a 1.5B model; raising to TP=2 adds per-decode all-reduce and would **slow** gen. | No change. Ensure launcher default `ROLLOUT_TP=2` does not leak in; force `=1`. | **NEUTRAL as kept.** (Raising TP *would* be NON_NEUTRAL — reshards reductions → perturbs sampled tokens.) Verify resolved config shows `=1`. |

---

## 3. Candidate FAST-CONFIG

Apply together. Tiered so you can stop at the config-only tier if it already clears the bar.

**Tier A — config-only, no code edit, lowest risk (apply first):**
```bash
# hydra overrides on the existing launcher / run_cell
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192          # #1  biggest lever
+actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu        # #2  ~196 H2D/step gone
actor_rollout_ref.rollout.enable_chunked_prefill=True            # #4  smooth gen (ALL cells)
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096  # #5
actor_rollout_ref.actor.fsdp_config.forward_prefetch=true        # #7
trainer.save_freq=0                                              # #8  (keep test_freq=25)
# env, cells 2-5 only (cell 1 ONLINE):
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                           # #6
```

**Tier B — add if Tier A is short of <25 min AND the memory canary shows headroom:**
```bash
+actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=gpu     # #3  ~3GB H2D/fire gone — HIGH OOM risk
```

**Tier C — code patch on an exp/* branch (only if still short; modest, neutral):**
- Gate spectral debug prints + drop `flush=True` (#10)
- Defer/batch the per-matrix `relative_change().item()` (#11)
- Gate the relevance probe + canary print, **keep the canary assert** (#12)

**Skip:** Warm Ray (#13) — ~8–12 s for real cross-cell-contamination fragility; not worth it on a 5-cell sweep.

### Projected wall time

This is a **rough projection**, not a measurement — the brief explicitly did not separately time several of these levers.

| Source | Conservative saving | Optimistic saving |
|---|---|---|
| #1 ppo_max_token_len (halve micro-batch overhead across 263 s non-anchor + 566 s anchor) | ~1.0 min | ~3.0 min |
| #2 ema_device gpu (~196 H2D/step removed) | ~0.5 min | ~1.5 min |
| #4 chunked prefill | ~0.2 min | ~0.6 min |
| #5 weight bucket | ~0.1 min | ~0.3 min |
| #6 HF offline (cells 2–5; per-cell here ≈ ¼ of sweep saving) | ~0.1 min | ~0.4 min |
| #7 forward_prefetch | ~0.1 min | ~0.5 min |
| #8 save_freq=0 | ~0.07 min | ~0.07 min |
| #10–12 print/sync gating (Tier C) | ~0.1 min | ~1.0 min |
| **Tier A+C total** | **~2.2 min** | **~7.4 min** |
| **+ Tier B #3 snapshot_device gpu** | ~+0.5 min | ~+2.0 min |

**Verdict on clearing <25 min:** Starting from ~28 min, **Tier A alone plausibly lands in the 25.8–24.5 min band** — i.e., it *might* clear 25 min on the optimistic end but is **not guaranteed**. The single load-bearing bet is **#1 (`ppo_max_token_len_per_gpu`)**: if the live value really is 3000, the run is so over-partitioned that raising it should comfortably exceed the conservative estimate and pull the cell under 25 min on its own. **Adding Tier C (and Tier B if memory allows) makes <25 min likely.** Confidence is gated entirely on confirming the live `ppo_max_token_len_per_gpu` value first.

### Memory-headroom conflicts (the real constraint)

Anchor steps already peak **~124/143 GB**. Three of the knobs push HBM **up**:
- **#1 ppo_max_token_len ↑** raises per-micro-batch activation memory — the most likely to OOM on anchor steps. **Sweep upward (8192 first), not straight to 12288.**
- **#2 ema_device=gpu** adds ~6 GB fp32 M-state to HBM (it is a standing OOM guard).
- **#3 snapshot_device=gpu** adds up to ~18 GB (≈6 full-model bf16 snapshots) — the riskiest.

These **stack on the same ~124 GB anchor peak**. Do **not** flip #1 + #2 + #3 simultaneously without a memory canary. Recommended order: flip #2 (small) and #1 (to 8192) together, measure `max_memory_allocated` on an anchor step; only then consider #3. If OOM, drop #3, then back #1 to 6144, then #2 last. All three fail *loudly* (crash) rather than silently corrupting the trajectory, so neutrality is never at risk from the memory dimension — only feasibility.

---

## 4. Empirical validation spec (gates promotion)

Run **once EXP-35 closes** (do not perturb the in-flight scientific run). Use the SAME provisioned box/topology and a FIXED global seed. One **fast-config cell vs the locked-config cell**, plus a short canary tier.

**(a) Prove <25 min:**
1. Run one full 50-step cell with the FAST-CONFIG. Record total wall (init + step-compute), and the per-step `timing_s/step` mean for NORMAL (~18 s target ↓) and ANCHOR (~45 s target ↓) steps separately.
2. **Pass = total cell wall < 25 min.** If between 25 and 28, identify which Tier-A knob underdelivered (most likely #1 didn't merge enough micro-batches — check `num_micro_batches/tick` dropped) and add Tier B/C.
3. Confirm **no OOM on anchor steps**: `torch.cuda.max_memory_allocated` stays under 143 GB with margin.

**(b) Prove neutrality — split by claimed class:**

*Bit-identical knobs (#2, #3, #5, #6, #7, #8, #10–13):* run a **5–10 step canary, same seed**, fast-config vs locked. Assert **bit-for-bit** equality (not within-noise) of: per-step `actor/grad_norm`, `actor/pg_loss`, `pg_clipfrac`, the post-writeback gradient / optimizer state_dict checksum after step 1, the `comm_eff/spectral/rel_change_mean` series, the PowerSGD/anchor canary lines, and val@25/val@50. **Any last-bit drift falsifies the NEUTRAL claim** for that knob → isolate it (toggle one at a time) and downgrade.

*NUMERIC_ONLY knobs (#1, #4):* these are **statistically identical, not bitwise** (FP-reduction-order for #1; sampling-realization for #4). Run the fast cell and compare grad_norm/pg_loss/val@25/val@50 against a **seed-jitter band** of the locked config (i.e., the locked config rerun with 2–3 different seeds to establish the run-to-run noise envelope). **Pass = fast-config curves fall inside that band with no monotone divergent drift over 50 steps.** For #1 specifically, also run a **dense-OFF control** (comm_eff disabled) at 3000 vs 8192 — there grad_norm/loss must match to ~1e-5 fp reduction noise, empirically validating the partition-invariance proof. For #4, confirm `rollout_probs_diff` (train-recompute vs vLLM logprob) is **not widened** vs locked — chunked prefill must not grow the train/inference mismatch.

**(c) Confirm the science verdict is unmoved:** val@25 and val@50 from the fast-config cell must land within the locked-config seed-noise band. If val moves outside the band, **stop** — a knob is not as neutral as claimed.

**Promotion rule:** FAST-CONFIG is promoted to the EXP-35 launcher **only if both (a) wall < 25 min AND (b) every knob passes its class-appropriate neutrality check.** If a single knob fails (b), drop it and re-project; if that drops below 25 min, the remaining set is still promotable.

---

## 5. Gaps & residual risks

**Not separately profiled (estimates are inferred, not measured).** The brief explicitly did NOT time: the spectral debug prints/flushes (#10), the per-matrix `.item()` syncs (#11), the relevance probe (#12), the chunked-prefill gen win (#4), the weight-bucket consolidation (#5), the device-placement H2D/D2H transfers (#2, #3), and forward_prefetch overlap (#7). Their savings are mechanism-based plausibility estimates. The **only** firmly-sized lever is #1 (attacks the measured 67.8% update_actor sink), and even its magnitude depends on how many micro-batches actually merge — which depends on the live token budget.

**Config discrepancy unresolved.** The brief states 24576/32768/no-KL; the *actual log* shows 3000/4096/8192 and `use_kl_loss=True`. **Before applying anything, dump the resolved Hydra config / launch.sh and confirm the real `ppo_max_token_len_per_gpu`.** If it's already ~24576 (not 3000), the #1 lever is largely exhausted and the FAST-CONFIG **will not** clear <25 min on config-only Tier A — you'd be reliant on Tier B+C, which is far less certain. This single fact is the largest uncertainty in the whole report.

**Memory is the binding constraint, not neutrality.** #1, #2, #3 all raise the ~124/143 GB anchor peak and stack. The brief's headroom signals are mixed (EXP-29 `max_mem 30.77` suggests room, but that may predate the full 196-matrix signed_ema M-state). Treat every memory estimate as a hypothesis to be measured on a canary, not trusted.

**NUMERIC_ONLY realization drift compounds through RL.** #4 (and #1 under the codec) perturb the *realized* trajectory at the fp/sampling-noise level, then that noise compounds through the multi-step RL feedback loop. The rollout *distribution* is unbiased, but the specific run diverges — which is why both **must be set on all cells including any re-collected baseline** and validated against a seed-noise band, not against a single locked run. If the seed-noise band itself is wide (RL is noisy), distinguishing "neutral knob" from "real regression" at val@25 may be hard with a single cell — consider 2–3 seeds for the band.

**Version-availability unknowns.** `forward_prefetch` (#7) is a documented FSDP1 arg and the run uses `strategy=fsdp` (FSDP1 path fires), but its presence/behavior on the exact PyTorch build in `verl-research-vllm020` was asserted, not checked on the box — verify before relying on it. `enable_chunked_prefill` (#4) requires `max_num_batched_tokens ≥ max_model_len` (8192 > ~2560, satisfied) but the vLLM 0.20 scheduler behavior under this codec was not exercised live.

**Code-patch knobs need an exp/* branch.** #10–12 (and any verl/ edit) cannot be hydra flips — they require an `exp/<N>-<slug>` branch (the protect-hook also allows `vast-ai-workload`). Gate-don't-delete: the analyst greps `rel_change=` and `merger_coldM_fallbacks=`, and the snapshot canary `assert` is a real bitwise staleness guard for `snapshot_device` — keep it.

**Rejected knobs (appendix) — do NOT revisit as neutral:**
- **`gpu_memory_utilization` 0.55→0.75:** REJECT. KV-pool size changes resident batch size → with CUDA graphs on, different captured-graph/GEMM tilings → perturbed logits → under temp>0 sampling, **different rollout tokens → changed rollout distribution**. Crosses the explicit hard bar.
- **`free_cache_engine` / KV-retention across steps:** REJECT. The level-1 KV-only path is a **no-op stub** in this vLLM async server (`vllm_async_server.py:630-640`); the real `_sleep_hybrid` uses level=2. Unimplemented + any real implementation risks serving off **stale prefixes** computed under prior-step weights → rollout-dist change.
- **`cudagraph_capture_sizes` restriction:** REJECT. Changes which padding bucket / eager-vs-graph path the hot 1024-seq rollout wave takes → reorders attention reductions → perturbed logits → different sampled tokens.
- **Single-process multi-cell sweep:** REJECT as proposed. Reusing the vLLM engine carries the sampler RNG forward across cells (seeded at engine build, `vllm_async_server.py:268`) → alters rollout distribution; optimizer/anchor-ring/M-EMA/Q/global_step/W&B all inherit state. Would only be neutral with a perfect determinism reset + a passing bitwise canary against a cold-start cell — a build-and-verify item, not an admissible flip today.

**Bottom line:** The plan is plausible-but-not-guaranteed to clear <25 min, and the entire confidence rests on confirming `ppo_max_token_len_per_gpu` is really 3000 (not 24576) on the live box. Confirm that one value, apply Tier A with a memory canary on anchor steps, measure, and only escalate to Tier B/C if short. Neutrality is well-grounded for every kept knob; the real risk is OOM (feasibility) and the magnitude of the inferred savings.
