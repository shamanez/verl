# EXP-26: Diagnose the SFT→GRPO merger mismatch with a real-gradient geometry audit, then test direction-preserving, RLVR-native compression — #25 follow-up

Move the comm-efficient GRPO frontier from "guess a merger and run a full training arm" to **"first measure the real update geometry, then change the merger / `Q` only where the geometry says it is broken."** #25 falsified the `signed_ema` merger and proved the substrate. This issue diagnoses *why* with real gradients and real matrices, then tests two direction-preserving candidates against parity — **no anchor-sign replacement, no objective change, no dense-surpass claim before parity is recovered.**

> **Gating.** Step A (the real-gradient geometry audit) is a **diagnostic gate**. The training arms (Steps B/C/E) do **not** launch until Step A returns a decision. #24 (closed, EF-PowerSGD on the PowerSGD residual) is **carried forward here as Step B**, now gated behind the audit instead of launched blind.

> **Non-negotiable substrate invariants — realism constraints (do NOT relax, even slightly, in any arm).** The north-star is a *realistic* communication-efficient, decentralized pipeline-parallel setting, so three properties of the **training path** are fixed for every arm here and may not be tuned, ablated, or "temporarily" relaxed:
> 1. **`Q` is updated ONLY by the anchor network** (`anchor.owns_q=true`). The fast/compressed circuit is a strict read-only consumer of `Q` (fast `maybe_update_basis` is fail-closed). No arm may let the fast path write `Q`.
> 2. **A full, uncompressed, full-coverage gradient pass happens ONLY inside the anchor network.** The fast/training path is *always* compressed — there is no fresh full-rank "clean step" on it. That fresh dense step is exactly the unrealistic crutch (`clean_cadence`) the anchor replaced: full-H transfer, impossible on a real decentralized link.
> 3. **Anchor staleness is mandatory** (`delay_K ≥ 5`, optimizer ticks). The anchor reference is `delay_K`-stale by construction and is never made fresh — on a real decentralized-PP link the full-gradient reference is itself stale on a slow link. `delay_K=0` is **forbidden as a training configuration.**
>
> **These bind the METHOD/training path only.** Step A's audit additionally captures two *measurement-only* probes — a parallel uncompressed `G_dense` backward and a `delay_K=0` fresh anchor gradient — that exist solely to compute the geometry diagnostics (update cosine; structural-vs-staleness sign decomposition). Like a validation pass, these probes **never feed the optimizer, are not part of the method, and are removed after the audit.** They do not relax invariants 2–3.

**Prior-experiment history (W&B):** https://wandb.ai/shamanework-pl/verl_compression_research?nw=nwusershamanework
Reference runs (read, never re-run): dense `5e2jpho9` (val@50 **0.7536**) · A0 PowerSGD r77+fresh-clean@5 `oquyeic3` (**0.7415**) · no-refresh PowerSGD r77 floor = EXP-23 A1 (**0.6914**) · EXP-25 `signed_ema` α=0.5 `1wulaelw` (**0.7066**) / α=0.3 `r8kc702g` (0.6164) / α=0.0 `uyrpaftw` (0.3541) · α=0 + KL@0.001 `5hormzfk` (0.6793, below floor).

**Labels:** `research:claim` `kind:experiment` `milestone:M6` (code_change — Step A adds capture instrumentation; Steps B/C add a new merger + `Q` families; validated by training curves + the geometry audit).

---

## Planner fields (parse straight into `.claude/plans/26.md`)

```yaml
kind:            experiment        # Step A is diagnostic, but B/C/E are code-change training runs that can only be validated on Vast → experiment + code_change:true (not implementation/investigation)
code_change:     true
milestone:       M6
baseline_run:    EXP-25            # references (dense/A0/floor/EXP-25 arms) read from W&B; the EXP-20/23 run dirs were cleared in the #25 clean-slate
depends_on:      [EXP-25]          # terminal STOP → references + the falsified mechanism this issue starts from
seed_replicates: 1                 # directional curve-match, single seed, matching EXP-20/23/25
promote_launcher_as: none          # promote a canonical launcher only when a method recovers PARITY with a understood mechanism (human decides)
non_negotiables:                   # REALISM constraints — bind the TRAINING path of EVERY arm; never tune/ablate/relax (see the invariants callout)
  - anchor_owns_Q                  # Q updated ONLY by the anchor; fast path read-only on Q (maybe_update_basis fail-closed)
  - full_pass_only_in_anchor       # the only uncompressed full-coverage gradient pass is the anchor's; fast/training path is ALWAYS compressed; NO fresh clean step
  - mandatory_staleness            # anchor is delay_K>=5 stale by construction; delay_K=0 forbidden as a training config (a real decentralized link is stale)
  # NB: Step A's parallel-uncompressed G_dense + delay_K=0 fresh-anchor grad are MEASUREMENT-ONLY probes (never fed to the optimizer), NOT a relaxation of the above.
hypothesis: >
  On the fixed GSM8K surface (Qwen2.5-1.5B-Instruct, vanilla GRPO no-KL/no-entropy, lr 1e-6,
  train_batch 128, ppo_mini 64, n=8, max_response 16384, anchor on + owns Q, PowerSGD r=77,
  cadence/delay_K 5, no clean step), the #25 lag is caused by the MERGER corrupting the live
  GRPO update DIRECTION, not by rank-77 PowerSGD compression (which ties dense at 0.7415).
  A real-gradient geometry audit (Step A) will show (a) plain-PowerSGD update cosine to dense
  is high (>=0.95 post-warmup) while signed_ema update cosine collapses, and (b) whether the
  activation-energy basis Q_act also misses off-principal GRPO update energy. A direction-
  preserving error-feedback PowerSGD merger (Step B) then recovers val@50 to the
  PowerSGD/fresh-clean band (>= 0.7414 = floor+0.05, within ~1 pt of A0 0.7415) WITHOUT any
  length/clip collapse. FALSIFIED for a given merger if its best arm val@50 <= floor+0.02 = 0.7114
  OR its dense-vs-compressed update cosine does not improve over plain PowerSGD.
target_modules:
  - verl/workers/comm_eff/powersgd_activation.py        # Step A: dump A, Â=(A@Q)Qᵀ, Q at the projection hook (:381-382, :413-424); Step C: Q_grad/Q_adv/Q_tail/Q_hybrid/Q_ticket families at fixed rank 77
  - verl/workers/comm_eff/spectral_filter.py            # Step A: dump G_comp (merger input :307), M/G_anchor (:181), rel_change (:310); Step B: new correction_mode=ef_powersgd (direction-preserving residual EF), NO sign term
  - verl/workers/comm_eff/anchor.py                      # Step A: dump K-stale G_anchor + the delay_K=0 fresh anchor grad for the sign-agreement decomposition
  - verl/workers/engine/fsdp/transformer_impl.py        # Step A: parallel UNCOMPRESSED fast backward to capture G_dense alongside G_comp at the same step; fp32 dump of post-merger pre-Adam p.grad
  - verl/workers/config/comm_eff.py                      # config: correction_mode=ef_powersgd + residual clip/decay; q_basis={act,grad,adv,tail,hybrid,ticket}; diagnostic-capture flags
  - examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh  # wire the new env vars; keep the substrate defaults
compute:
  max_dph:      24.0
  max_gpu_hr:   60                 # Step A short diagnostic (~3-5 GPU-hr) + Step B EF arms + Step C Q sweep; staged, gated by Step A
```

---

## Background

### What #25 proved

1. **The substrate is correct and realistic.** The anchor circuit — a full-coverage (all 196 weight-matrix gradients), DP-mean-reduced, `delay_K`-stale gradient EMA `M`, refreshed from a no-hook isolated clone, that is the **only** thing that updates the PowerSGD basis `Q` — passed every on-box probe gate (R1 coverage/DP-reduce/cold-M guard; R2 anchor-owns-`Q`, fast net never writes `Q`; off-path PowerSGD parity `reconstruction_rel_error=0.024`). This is the *realistic* setting (continuously-maintained stale anchor, no impractical periodic dense "clean step"), and it is **not** relitigated here.
2. **Compression is benign.** Plain PowerSGD r=77 + fresh-clean@5 ties dense within ~1 pt (0.7415 vs 0.7536). `reconstruction_rel_error≈0.024` is small and **stationary** through the whole run — it does not spike at collapse. The dropped off-subspace activation energy is `≈0.024² ≈ 0.06%`. PowerSGD is **not** the immediate problem.
3. **The `signed_ema` merger is falsified.** `G_corr = α·G_comp + (1−α)·|G_comp|·sign(M)`. The α-sweep is **monotonic and net-harmful**: val@50 = 0.3541 (α=0) < 0.6164 (α=0.3) < 0.7066 (α=0.5), with the best arm at the least-correction edge — *every unit of signed correction makes RL worse*. Best-α 0.7066 is below the falsification line (floor+0.02 = 0.7114) and far below A0 (0.7415) and dense (0.7536). **STOP.**

So the lag (0.047 below dense; 0.035 below plain PowerSGD) is **caused by the merger added on top of the substrate, not by rank-77 compression.**

### Why `signed_ema` failed (the mechanism, from real EXP-25 data)

On a coordinate where `sign(M)` disagrees with `sign(G_comp)`, the merger reduces to `(2α−1)·|G_comp|·sign(G_comp)` — a **dose-graded direction corruption**: α=0 fully reverses the live step at full magnitude, α=0.3 is 40% reversed, α=0.5 zeroes the coordinate (the knee), α→1 is plain PowerSGD. The merger's *entire* effect is on the disagreeing fraction, and:

- The stale anchor sign **disagrees with the live compressed gradient on ~50% of magnitude-weighted coordinates, every step** (warm `rel_change` median = **1.416 ≈ √2** ⇒ disagree-energy/total = ½). This is **structural, not staleness/EMA/compression**: it is already 50.4% at the first warm comparison (delay≈4, no EMA history), flat across all 50 steps, and **uniform across all 7 matrix types and all 28 layers** (it does not track the 7 compressed boundaries). It is the coin-flip signature of two *different* estimators (`M`: stale/clean/uncompressed/β-smoothed; `G_comp`: fresh/compressed) of a **near-zero-mean per-coordinate GRPO gradient** whose true sign is ill-defined where the GRPO group-relative advantages cancel.
- Replacing the live sign with the stale sign **destroys the per-coordinate sign-cancellation that is GRPO's implicit step-size regularizer** — grad-norm inflates from dense's 0.387 to α=0's 3.3–11. Under the no-KL/no-entropy surface there is no brake, so the persistent wrong-direction force ignites a **response-length reward-hack** (α=0 → 5863 tok, α=0.3 → 16K cap; `clip_ratio` 0.29/0.91). **Length explosion, not low entropy, is the val-killer** (dense trains at *lower* entropy 0.122 with bounded length and the best val; α=0.5 keeps low entropy *and* bounded length and survives). A KL brake (α=0 + `kl_loss_coef=0.001`, `5hormzfk`) closes the length channel but lands **0.6793, below the floor** — confirming the brake is a guardrail, not a cure: the direction bias still drags val below codec parity.

PowerSGD-only (no merger, same codec, same low entropy, same recon error) hits 0.741 with no collapse. **The merger is the entire pathology.**

### Why the failure is an SFT→GRPO / RLVR mismatch

`signed_ema` is an **SFT-era correction idea**: use a clean, stable, low-variance reference statistic to repair a noisy path by trusting the clean *direction*. That is sound when the target distribution is fixed (SFT). It is wrong for GRPO/RLVR:

- **SFT can use a clean teacher direction; GRPO needs the live on-policy direction.** The GRPO update depends on the current batch's sampled responses, group-normalized advantages, importance ratios, and clipping. The current sign pattern *is* part of the signal. A clean-but-stale sign is a **worse** estimator of the live update direction than the noisy-but-live compressed sign (which is direction-faithful to dense). #25 used the anchor as a sign oracle; for RLVR that is false.
- **RLVR is sensitive to small biased drift.** RLVR steps are small and KL-proximal even without explicit KL; a "small" correction that is nearly an independent sign field (`rel_change≈√2`) is not a perturbation, it changes the optimization objective. SFT tolerates larger deterministic bias; GRPO does not.

This is why the next phase must **stop treating the anchor as a sign oracle** and keep it only as: owner of `Q`, geometry meter, basis refresher, and support for **direction-preserving** residual/preconditioner state. The fast path must preserve the live GRPO update direction.

### What paper findings motivate the next plan

- **`arXiv:2511.08567` — "The Path Not Taken: RLVR Provably Learns Off the Principals."** Three-Gate theory: on-policy RL imposes a one-step KL leash (Gate I), pretrained geometry steers those small steps into **low-curvature, off-principal, spectrum-preserving** directions (Gate II), and bf16 hides the small off-preferred updates as apparent parameter sparsity (Gate III). RLVR preserves top singular spectra and rotates principal subspaces *less* than SFT; principal-only / principal-targeted (PiSSA-style) updates underperform and destabilize, while non-principal "safe" masks track dense. **Implication:** activation **reconstruction** error is not enough — `Q_act` may reconstruct activations well yet miss the off-principal directions where GRPO actually learns. Diagnose with **real per-matrix update tensors in fp32** (bf16 sparsity is a Gate-III readout, not the object); measure principal/off-principal update preservation, subspace rotation, and update cosine.
- **`arXiv:2505.11711` — "RL Finetunes Small Subnetworks."** RL updates a small (~5–30%) but **nearly full-rank** subnetwork spread across layers; SFT updates are denser; KL/clipping barely change the sparsity (on-policy data drives it). **Implication:** a rank-77 activation codec can reconstruct well yet restrict the sparse/full-rank update GRPO wants. **Error-feedback PowerSGD** is attractive precisely because it keeps low-rank *communication* while accumulating the omitted residual over time — recovering update energy without overriding direction.
- **`arXiv:2602.01599` — "The Multiple Ticket Hypothesis."** On Qwen2.5-1.5B / GSM8K, training a *random* 1% of parameters matches/exceeds full RLVR; 20 disjoint 1% masks all succeed (~0.005 Jaccard); structured (first/last-layer) masks lose to random at fixed budget; effective Fisher rank is tiny (~44). **Implication:** a good RLVR compressor need only preserve the **policy-relevant low-dimensional update effect**, not the exact dense update or the top activation-energy basis. Use random / off-principal / principal **parameter-mask controls as a geometry probe** for what `Q` should preserve (Step D) — not as a new objective.
- **`arXiv:2509.04259` — "RL's Razor."** On-policy RL is biased toward **KL-minimal** solutions; KL from the base policy predicts forgetting better than weight movement. **Implication:** a compressor must preserve the KL-minimal on-policy path; a merger that looks small in activation/weight space but increases KL drift / worsens update cosine is not RLVR-native. Make forward KL, update cosine, and sign-agreement first-class.
- **Pass@k / coverage (`arXiv:2507.14843`, `arXiv:2504.13837`).** RLVR often sharpens pass@1 while narrowing high-k coverage; token-entropy ≠ answer diversity; raising temperature post-training need not restore coverage. **Implication:** do **not** make a dense-surpass claim from greedy val alone — and only after parity is recovered.

### Why the geometry audit must use real gradients and real matrices

The failure is a **direction/geometry** claim, and every cheap proxy already in the logs is provably blind to it: `reconstruction_rel_error` was small and *stationary* while the run collapsed (compression benign), entropy declined in healthy arms too, and bf16 apparent sparsity is a Gate-III precision artifact, not the true update structure. The discriminating signal lives only in the **per-target update direction in fp32** — `cos(G_dense, G_comp)`, the off-principal share of the dense update, `Q`'s capture ratio *for update energy* (not activation energy), and sign-agreement at `delay_K=0`. None of these can be read off a scalar log or inferred from reconstruction; they require the real activation, reconstructed activation, live/compressed/dense gradient, post-merger update matrix, `Q`, and anchor `M` captured **at the same step on the same prompts**. Hence Step A is a tensor-capture audit, run before any training arm spends compute.

### When existing artifacts are insufficient — the short few-step diagnostic

The EXP-25/dense/PowerSGD logs and W&B scalars do **not** contain per-target gradient/update tensors or `G_dense` alongside `G_comp`, so the audit cannot be done from existing artifacts. Run a **short, few-step diagnostic job** (≈5–10 optimizer ticks, the fixed 4×H200/8×H100 surface, ~3–5 GPU-hr) with capture instrumentation enabled:

- three arms on the **same prompts and rollout settings**: dense, plain PowerSGD r=77, and the EXP-25 anchor+`signed_ema` arm;
- a **parallel uncompressed fast backward** so `G_dense` is captured at the same step as `G_comp` (the never-logged EXP-20 success criterion; the `COLLAPSE_GRADIENT_FLOW_ANALYSIS §1.2 / §8.1` OPEN);
- dump (see **Required tensor captures**) real activations, reconstructed activations, live/compressed/dense gradients, post-merger pre-Adam update matrices, `Q`, and anchor `M`/`G_anchor` (both `delay_K=5` and a `delay_K=0` fresh anchor grad), all in **fp32**, keyed by `(global_step, optimizer_tick, target_name, shape, dtype, norm, rank)`.

---

## Hypotheses

- **H1 (primary, decided by Step A).** The #25 lag is merger **direction corruption**, not rank-77 compression. Prediction: `cos(G_dense, G_comp) ≥ 0.95` post-warmup for plain PowerSGD, while `cos(G_dense, G_corr)` collapses for `signed_ema`.
- **H2 (`Q` geometry, decided by Step A).** `Q_act` (activation-energy basis) captures activation energy well (≈0.9994) but **under-captures GRPO update energy**, with the deficit concentrated in **off-principal** directions (per `2511.08567`). If true → an RLVR-native `Q` (Step C) is needed; if `Q_act` already captures update energy → skip Step C, go straight to EF (Step B).
- **H3 (sign disagreement is structural).** Sign-agreement(`M`, `G_comp`) ≈ 50% **even at `delay_K=0`** (fresh anchor — a *measurement-only* probe; training staleness stays `delay_K≥5` per the invariants). If true → sign-replacement is permanently unrecoverable and must never return; if it rises sharply at `delay_K=0` → the disagreement was staleness, a different (still non-sign-replacement) fix applies.
- **H4 (direction-preserving fix recovers parity).** Error-feedback on the PowerSGD residual (Step B) re-injects the dropped `(I−QQᵀ)` energy **without overriding direction**, recovering val@50 into the PowerSGD/fresh-clean band with improved update cosine and no length/clip collapse.
- **H5 (parity ceiling — frames the non-goal).** Because only ~0.06% of activation energy is dropped, "anchor corrects compression bias" has a **realistic ceiling of parity with dense**, not surpass. A surpass claim would require an extra-dense information channel (the stale clean anchor is not one — dense already sees full uncompressed activations every step). **Parity first; surpass is out of scope here.**

---

## Experiment plan

Ordered so no full training arm runs with an unclear failure mode. **Step A gates everything.**

### Step A — Real-gradient geometry audit *(diagnostic gate; code_change = capture instrumentation; ~3–5 GPU-hr)*
Run the short few-step diagnostic above (dense / plain-PowerSGD / EXP-25-signed_ema arms, parallel uncompressed backward, fp32 dumps). Compute, per target per step: principal/off-principal decomposition of the **dense** update; the same projection of `G_comp`, `G_corr`, and `M`; `Q_act` capture ratio for **update** energy; `cos(G_dense, G_comp)` and `cos(G_dense, G_corr)`; sign-agreement(`M`,`G_comp`) and sign-agreement(`G_anchor_fresh`,`G_comp`) at `delay_K∈{0,5}`; per-layer spectral drift + principal-subspace rotation; bf16-zero vs fp32-nonzero update fraction.
**Decision rule:**
- If `Q_act` captures update energy well **and** the only defect is the sign term → go to **Step B** (EF), skip C.
- If `Q_act` misses off-principal update energy → run **Step C** (`Q` sweep) before/with B.
- If sign-agreement stays ≈ coin-flip at `delay_K=0` → **permanently retire all sign-replacement mergers** (already the plan; this hard-confirms it).

### Step B — Direction-preserving EF-PowerSGD *(the #24 primitive, carried forward)*
Per compressed activation tensor: `u_t = h_t + e_t`; `y_t = u_t Q_t`; `ĥ_t = y_t Q_tᵀ`; `e_{t+1} = decay·clip(u_t − ĥ_t)`. Constraints: anchor owns `Q`; fast path reads `Q`; residual per-target, shape-aware, **reset on shape mismatch**, **norm-clipped relative to activation norm**, detached; **no sign term anywhere.** Arms: EF-PowerSGD vs plain PowerSGD vs dense, 50→100 steps, val@25.

### Step C — RLVR-native `Q` sweep *(only if Step A says `Q_act` misses update energy)*
At **fixed total rank 77**: `Q_act` (control) · `Q_grad` (anchor-owned basis from live GRPO gradient right-singular stats) · `Q_adv` (advantage-weighted activation stats) · `Q_tail` (after removing top activation principals) · `Q_hybrid` (split-rank: activation-energy + off-principal grad directions) · `Q_ticket` (informed by Step D). Judge by update cosine + off-principal preservation + greedy val + length/clip, **not** by activation reconstruction. Do not change rank allocation and merger simultaneously unless the earlier gate already passed.

### Step D — Sparse-ticket diagnostic *(geometry probe only, not an objective)*
Small controlled parameter-update masks: all-dense · random 1% · random 5% · off-principal/low-magnitude 1% · principal/high-magnitude 1%. Interpretation: off-principal/random beating principal supports the off-principal reading and motivates `Q_tail`/`Q_ticket`; principal winning means the task does not follow the paper's geometry strongly here; all sparse failing → keep as a paper-level note, do not let it drive the pipeline-compression design. (Parameter sparsity does not itself reduce inter-stage activation comms — this is diagnostic for what `Q` should preserve.)

### Step E — Dense-matched communication run *(only after B or C identifies a stable method)*
Minimum comparison: dense control · plain PowerSGD reference · best EF-PowerSGD arm · best RLVR-native `Q` arm (if different). Report measured inter-stage communication volume vs dense.

---

## Required tensor captures (Step A)

All dumped in **fp32**, at the same `(global_step, optimizer_tick)` for the dense / PowerSGD / signed_ema arms on identical prompts, keyed with `{target_name, shape, dtype, ‖·‖, rank/projection-stats}`. Capture sites are real code locations.

| Tensor | Symbol | Where (file:line) | Feeds |
|---|---|---|---|
| Real boundary activation | `A` | `powersgd_activation.py:381-382` (hook input, pre-projection) | activation energy spectrum; principal basis of activations |
| Reconstructed activation | `Â=(A Q)Qᵀ` | `powersgd_activation.py:381-382` (post-projection) | reconstruction error (sanity vs 0.024); dropped `(I−QQᵀ)A` |
| Projection basis | `Q` (per boundary, rank 77) | `powersgd_activation.py:413-424,581-583` (anchor-owned `Q←orth(V)`) | `Q` capture ratio for **update** energy; subspace overlaps |
| Live compressed gradient | `G_comp` (per target, post FSDP all-reduce, pre-merger) | `spectral_filter.py:307` (merger input `gm`) | `cos(G_dense,G_comp)`; off-principal share; sign-agreement |
| Dense gradient (parallel uncompressed backward) | `G_dense` | `transformer_impl.py` (added uncompressed fast backward) | the reference for **every** cosine / principal split |
| Post-merger update matrix (pre-Adam) | `G_corr` | `spectral_filter.py:307` output → `transformer_impl.py` write-back, **before** AdamW | `cos(G_dense,G_corr)`; what the optimizer actually consumed |
| Anchor raw gradient | `G_anchor` (`delay_K=5`) **and** `delay_K=0` fresh | `anchor.py:115` (clean PG bwd) → `spectral_filter.py:181` | sign-agreement decomposition (structural vs staleness, H3) |
| Anchor gradient EMA | `M` (β=0.95) | `spectral_filter.py:181` | sign(`M`) vs sign(`G_comp`); `M` energy principal/off-principal |
| Per-target update `rel_change` | `‖G_corr−G_comp‖/‖G_comp‖` | `spectral_filter.py:310` | cross-check the √2 disagreement signature |

The `G_dense` parallel uncompressed backward and the `delay_K=0` fresh anchor gradient are **measurement-only probes** (see the non-negotiable invariants callout): they are captured to compute the diagnostics, **never feed the optimizer**, and are removed after the audit — they do not relax the "full pass only in the anchor" / mandatory-staleness invariants.

Derived audit outputs (per target, per step): SVD of `G_dense` → top-k principal subspace (k at 90% energy) vs tail; fraction of `G_dense`/`G_comp`/`G_corr`/`M` energy in each; `‖QQᵀ G‖²/‖G‖²` for the **gradient** vs the activation; all four cosines; sign-agreement at `delay_K∈{0,5}`; per-layer principal-subspace rotation; bf16-zero-but-fp32-nonzero update fraction.

---

## Metrics

**Primary geometry metrics (Step A, from real fp32 tensors):**
- **Update cosine** `cos(G_dense, G_comp)` and `cos(G_dense, G_corr)` per target/step (EXP-20's never-logged success criterion; benign compression predicted ≳ 0.95–0.98 post-warmup).
- **Principal/off-principal update preservation** — share of dense update energy in top-k vs tail, and how much each codec/merger preserves in each.
- **`Q` capture ratio for update energy** `‖QQᵀ G‖²/‖G‖²` vs the activation capture ratio (≈ 0.9994).
- **Sign agreement** sign(`M`)·sign(`G_comp`) and sign(`G_anchor_fresh`)·sign(`G_comp`), magnitude-weighted, at `delay_K∈{0,5}`.
- **Spectral drift / subspace rotation** per layer, dense vs compressed.
- **bf16 apparent sparsity vs fp32 update magnitude.**

**Training / guardrail metrics (every training arm — the EXP-25 discriminators):**
- **greedy val@25 / val@50** (primary outcome).
- **train reward** `critic/score/mean` per step (watch **peak-then-crash**, the EXP-25 fingerprint).
- **`response_length/mean`** — ignition alarm at **> 2× the step-10 baseline**.
- **`clip_ratio` / response-length clip** — danger band **0.3–0.9**.
- **`grad_norm`** — dense reference ≈ 0.387; merger-inflation alarm.
- **residual norm** (EF arms) `‖e_t‖/‖h_t‖` — clip alarm.
- **entropy slope** (context only — NOT a discriminator; dense trains at the lowest entropy).
- **`rollout_ppl`** (exploration proxy, comparable across the anchor boundary; `rollout_probs_diff_mean` is **not** comparable across anchor vs non-anchor arms).
- **pass@k / answer-level entropy** — measured **only after parity**, to avoid a false surpass claim.

---

## Stop conditions

Stop a merger / arm immediately if any holds:
- it uses an anchor sign to **replace** the live sign (sign-replacement is retired);
- magnitude-weighted sign-disagreement reappears near the EXP-25 √2 pattern under a merger meant to preserve direction;
- `cos(G_dense, G_corr)` collapses (does not improve over plain PowerSGD);
- **response length explodes** (> 2× step-10 baseline) while val/train look superficially healthy;
- `clip_ratio` enters the **0.3–0.9** danger band;
- `val@50` falls **below the no-refresh floor (0.6914)** without a clear diagnostic reason;
- **any realism invariant is broken on the training path** — the fast path updates `Q`; any full/uncompressed/full-coverage gradient pass runs on the fast/training path (the full pass must live only in the anchor); or anchor staleness is relaxed below `delay_K=5` / set fresh.

---

## Acceptance & promotion criteria

**A method is promoted** (to Step E / a canonical launcher) only if **all** hold:
- it is **direction-preserving** (no sign replacement) and keeps **anchor ownership of `Q`**;
- it **improves update cosine** or off-principal update preservation over plain PowerSGD (Step A / per-arm capture);
- `val@50 ≥ 0.7414` (= floor+0.05, within ~1 pt of A0 fresh-clean 0.7415) — i.e. **parity with the PowerSGD/fresh-clean band**, the mechanism-understood reference;
- it triggers **no** length/clip collapse alarm;
- it **materially reduces** measured inter-stage communication vs dense (reported as a number, Step E);
- its result is **explained by the geometry audit**, not an uninspected training artifact.

**Issue-level acceptance** (this issue is well-formed): it states the EXP-25 failure mechanism before proposing experiments; requires principal/off-principal diagnostics + update cosine computed from **real** gradients/matrices; requires the length/clip collapse alarms; explains why each arm tests a specific paper-derived hypothesis; and keeps the GRPO verifier/objective fixed.

**Non-goals (hard):**
- **No anchor-sign replacement** — `signed_ema` and any sign-oracle merger are retired.
- **No training-objective / verifier changes** — vanilla GRPO, no-KL/no-entropy stays the fixed control surface. KL / entropy / length caps may appear **only** as explicitly-labeled *guardrail diagnostics* on a separate lineage, never as a fix for a merger and never folded into the control surface.
- **No broad rank sweep before the geometry is diagnosed** (rank 77 is held; `Q` *content* is the lever, not its size, until Step A says otherwise).
- **No dense-surpass claim before parity is recovered** with a mechanism understood from the audit (H5: the anchor-corrects-bias ceiling is parity, ~0.06% recoverable energy).
- **No relaxation of the realism invariants** (the substrate-invariants callout): `Q` updated **only** by the anchor; the **full gradient pass only inside the anchor** (training path always compressed, no fresh clean step); **mandatory `delay_K≥5` staleness**. These define the realistic decentralized-PP scenario — they are not hyperparameters and may not be tuned or ablated. (Step A's measurement-only `G_dense` / `delay_K=0` probes are not an exception — they never touch the optimizer.)

---

## References

**Internal:** `CLAUDE.md` · `CODE_WALKTHROUGH.md` · `research/.claude/GOAL.md` · `research/runs/SUMMARY.md` · `research/runs/FIXED_CONTROL_SURFACE.md` · `research/runs/next_phase_after_#25.md` · `research/runs/UNWANTED_HOOKS_AND_SILENT_FAILURES.md` · `research/runs/EXP-25/verdict.md` · `research/runs/EXP-25/DEEP_FINDINGS.md` · `research/runs/EXP-25/COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` · `research/runs/EXP-25/ENTROPY_COLLAPSE_FINDINGS.md` · `research/diagnostics/ENTROPY_COLLAPSE_WATCH.md`

**External:**
- "The Path Not Taken: RLVR Provably Learns Off the Principals" — arXiv:2511.08567 — https://arxiv.org/abs/2511.08567
- "Reinforcement Learning Finetunes Small Subnetworks in Large Language Models" — arXiv:2505.11711 — https://arxiv.org/abs/2505.11711
- "The Multiple Ticket Hypothesis: Random Sparse Subnetworks Suffice for RLVR" — arXiv:2602.01599 — https://arxiv.org/abs/2602.01599
- "RL's Razor: Why Online Reinforcement Learning Forgets Less" — arXiv:2509.04259 — https://arxiv.org/abs/2509.04259
- "The Invisible Leash: Why RLVR May Not Escape Its Origin" — arXiv:2507.14843 — https://arxiv.org/abs/2507.14843
- "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?" — arXiv:2504.13837 — https://arxiv.org/abs/2504.13837
- Vogels et al., "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization", NeurIPS 2019 — https://arxiv.org/abs/1905.13727
