# Next Phase After #25: Diagnose The SFT-to-GRPO Mismatch

Status: updated June 9, 2026.

Scope: this document focuses only on why the current communication-efficient GRPO method lags dense, why an SFT-motivated method does not transfer cleanly to RLVR/GRPO, and what should be tested next. The GRPO verifier/objective is fixed and is not an experiment axis.

The fixed control surface remains the one in `research/runs/FIXED_CONTROL_SURFACE.md`: Qwen2.5-1.5B-Instruct, GSM8K, vanilla GRPO, no KL, no entropy bonus, PowerSGD rank 77, anchor circuit on, `owns_Q=True`, `delay_K=5`, full matrix coverage, and no clean cadence. The intended variable is the compression/merger primitive.

## Bottom Line

#25 did not prove that communication-efficient GRPO is weak. It proved that the current merger is the wrong object for GRPO.

Plain PowerSGD is close to dense. The anchor substrate is now reliable. The failure appears when the method borrows an SFT-style correction idea: use a clean/stable anchor statistic to repair a noisy compressed path. In GRPO, that correction overwrites live on-policy direction information that the optimizer depends on.

The next phase should therefore stop treating the anchor as a sign oracle. The anchor should own `Q`, measure geometry, refresh compression bases, and support direction-preserving residuals. The fast path must preserve the live GRPO update direction.

## What #25 Established

Strong facts from the local runs:

- Dense control reaches `val@50 = 0.7536`.
- Plain PowerSGD `r=77` with fresh clean refresh every 5 reaches `val@50 = 0.7415`.
- No-refresh PowerSGD reaches `val@50 = 0.6914`.
- EXP-25 `signed_ema` best arm is `alpha=0.5`, `val@50 = 0.7066`, below the STOP line and far below dense.
- The `signed_ema` alpha sweep is monotonic harmful: `alpha=0.5 > alpha=0.3 > alpha=0.0`.
- Compression reconstruction error is small and stable; PowerSGD itself is not the immediate problem.
- Anchor-circuit implementation checks are green: full target coverage, DP-reduced `M`, cold-M guard, anchor-owned `Q`, and read-only fast-path `Q`.

The important numeric interpretation:

- Plain PowerSGD/fresh-clean is only `0.0121` below dense at step 50.
- EXP-25 `signed_ema` is `0.0469` below dense.
- EXP-25 is also `0.0349` below the plain PowerSGD/fresh-clean reference.

So the lag is not primarily caused by rank-77 activation compression. It is caused by the merger added on top of that substrate.

## Why The Current Method Lags

The proximal cause is sign corruption.

When `M` becomes warm, the anchor sign disagrees with the live compressed GRPO gradient on roughly half the magnitude-weighted coordinates. At coordinates where signs disagree, `signed_ema` applies this effective factor:

```text
effective factor = 2 * alpha - 1
```

That means:

- `alpha=0.0`: full direction reversal on disagreeing coordinates.
- `alpha=0.3`: partial direction reversal.
- `alpha=0.5`: zeroes disagreeing coordinates.
- `alpha -> 1.0`: returns toward plain PowerSGD.

This matches the observed alpha sweep. The more the stale anchor sign controls the update, the worse training gets.

The deeper cause is object mismatch:

- The anchor statistic is clean but stale.
- The fast path is noisy but live.
- GRPO's useful update is not just a denoised supervised gradient; it is an on-policy, group-relative, clipped policy-gradient estimate.
- Replacing the live sign with an anchor sign changes the optimization objective rather than merely compressing it.

This is why the collapse is not explained by low entropy alone. Low entropy appears in non-collapsing arms too. The bad arms show a length-exploit feedback loop because the update direction is distorted.

## Why SFT Intuition Mismatches GRPO

The base paper, "The Path Not Taken: RLVR Provably Learns Off the Principals" (`arXiv:2511.08567`), is directly relevant. Its main implication for this project is that RLVR does not use the same adaptation geometry as SFT.

### 1. SFT likes principal structure; RLVR may learn off-principal

SFT gradients are tied to fixed supervised targets. SFT-style compression and PEFT methods often work well when they preserve high-energy/principal directions, because those directions explain much of the supervised loss geometry.

The RLVR paper argues that RLVR gains are associated with small, spectrum-preserving updates in off-principal or low-curvature regions. If we compress only the directions that SFT would preserve, we may preserve activation reconstruction while dropping the directions where GRPO actually learns.

Implication: activation reconstruction error is not enough. We need to measure update preservation in principal and off-principal subspaces.

### 2. SFT can use clean teacher direction; GRPO needs live on-policy direction

In SFT, a clean reference direction can be a useful stabilizer because the target distribution is fixed. In GRPO, the update depends on current sampled responses, group-normalized advantages, importance ratios, and clipping. The current batch's sign pattern is part of the signal.

EXP-25 used the anchor as if clean sign meant better sign. For GRPO, that is false. Clean-but-stale sign can be worse than noisy-but-live sign.

Implication: future mergers must be direction-preserving. They may reduce variance or carry residuals, but they must not replace live signs with stale signs.

### 3. SFT tolerates larger deterministic bias; GRPO is sensitive to small biased drift

RLVR updates are small and KL-proximal even without explicit KL in many settings. A small biased correction can dominate the actual learning signal. The EXP-25 `rel_change` near `sqrt(2)` is a warning that the anchor correction is not a small perturbation; it is nearly an independent sign field.

Implication: any correction must be audited by update cosine, sign agreement, KL movement, length dynamics, clip ratio, and validation.

### 4. SFT compression can optimize reconstruction; GRPO compression must optimize credit assignment

A low-rank activation codec can reconstruct activations well but still damage the policy update if it removes directions tied to advantage-weighted credit assignment. GRPO is not trying to imitate a fixed answer distribution; it is relocating probability mass among sampled responses.

Implication: `Q` should be selected by policy-update preservation, not only activation-energy preservation.

### 5. SFT sparsity/low-rank results do not imply the same sparse or low-rank object for RLVR

"Reinforcement Learning Finetunes Small Subnetworks in Large Language Models" (`arXiv:2505.11711`) and "The Multiple Ticket Hypothesis" (`arXiv:2602.01599`) both indicate that RLVR can operate through small or highly redundant parameter subsets. But that does not automatically mean activation communication should use the same principal low-rank basis as SFT.

The useful RLVR object may be:

- sparse in parameters,
- off-principal in update geometry,
- low-dimensional in policy/Fisher space,
- but not simply top-rank in activation energy.

Implication: we need geometry-aware compression experiments, not only stronger versions of the current SFT-like compression.

## Paper Findings To Carry Into The Next Issue

These are point-form findings for the next agent. Read them together with the EXP-25 facts above. The goal is not to summarize the papers broadly; the goal is to extract what matters for diagnosing why the current method lags and designing the next experiment plan.

### `arXiv:2511.08567`: RLVR Learns Off The Principals

Core findings:

- The paper proposes a Three-Gate Theory for RLVR updates.
- Gate I: on-policy RL imposes a one-step KL leash. Even without explicit KL, clipping/on-policy updates keep each step small and policy-proximal.
- Gate II: pretrained model geometry steers those small steps away from principal/high-curvature directions and into lower-curvature, spectrum-preserving directions.
- Gate III: bf16 precision makes the bias appear as parameter sparsity by hiding small updates in non-preferred regions.
- RLVR preserves top singular spectra and rotates principal subspaces less than SFT.
- SFT tends to target principal weights/high-energy directions and causes more spectral drift.
- RLVR update masks have low overlap with principal-weight masks.
- The paper warns that sparsity alone is a surface readout; the deeper object is optimization geometry.
- Principal-only sparse RL masks perform poorly relative to dense.
- Non-principal/low-magnitude "safe" masks track dense more closely.
- Principal-targeted PEFT/LoRA variants such as PiSSA do not provide the expected RLVR gain and can become unstable at high learning rates.
- Low-rank LoRA can still work for RLVR when it does not force principal-direction updates, because the update can be small and off-principal.

What this means for #25:

- `signed_ema` behaved like an SFT-era correction: it trusted a stable clean direction over the live RLVR direction.
- The failure is consistent with forcing the optimizer toward a stale/principal/stabilized direction instead of preserving the live off-principal GRPO step.
- `Q_act` may be reconstructing activations well while still missing part of the RLVR-relevant update geometry.
- The next issue should require principal/off-principal update decomposition before launching another full training run.
- The next issue should treat bf16 apparent sparsity carefully; use fp32 optimizer/update statistics when diagnosing true update structure.

Required measurements from this paper:

- real gradients and real per-matrix update tensors, not only scalar logs or proxy summaries,
- per-layer spectral drift,
- top-k principal subspace rotation,
- overlap between update mask and principal mask,
- overlap between update mask and low-magnitude/non-principal mask,
- `Q` capture ratio for update energy in principal vs off-principal components,
- forward KL movement from base/current policy,
- update cosine dense vs compressed.

### `arXiv:2602.01599`: Multiple Tickets For RLVR

Core findings:

- Training only a random 1% of parameters can match or exceed full-parameter RLVR in the tested Qwen2.5 0.5B/1.5B settings.
- The paper tests Qwen2.5-1.5B on GSM8K and MATH-500, making it close enough to our model/task family to matter.
- Twenty independent 1% masks succeed with very small pairwise overlap, around `0.005` Jaccard.
- This argues against a single privileged lottery ticket.
- Performance remains strong around 99% to 99.95% sparsity in their sweep.
- Performance degrades sharply at more extreme sparsity, around 99.99% and beyond.
- Their explanation is KL-constrained policy optimization: only a low-dimensional policy-relevant update subspace matters.
- Their Fisher argument assumes low effective rank, delocalized eigenvectors, and small per-step updates.
- They report an effective Fisher rank around 44 in one Qwen2.5-0.5B Alphabet Sort analysis.
- Structured masks such as first-layer-only or last-layer-only are worse than random masks at fixed budget.
- They also run without explicit KL in the relevant setup, so the result is not only an explicit-KL artifact.

What this means for #25:

- The anchor does not need to identify exact privileged coordinates.
- A good RLVR compression method may only need to preserve the policy-relevant low-dimensional update effect, not the exact dense update.
- Random/off-principal ticket diagnostics are useful for learning what geometry GRPO tolerates.
- Parameter-ticket results do not directly solve inter-stage activation communication. They are a diagnostic and design clue for `Q`, not the final communication method.
- If a random/off-principal ticket works but a principal ticket fails in our setup, that strongly supports a tail-aware or ticket-aware `Q`.

Required measurements from this paper:

- real per-parameter or per-matrix gradient/update masks from an actual GRPO run,
- random 1% and 5% parameter-mask controls,
- principal/high-magnitude mask,
- off-principal/low-magnitude mask,
- mask overlap/Jaccard across seeds if more than one seed is run,
- gradient Fisher/effective-rank estimate if feasible,
- comparison of sparse-mask update cosine and greedy validation against dense.

### `arXiv:2505.11711`: RL Finetunes Small Subnetworks

Core findings:

- RL fine-tuning updates only a small subnetwork, roughly 5% to 30% of parameters, across multiple models and algorithms.
- The algorithms studied include PPO, GRPO, DPO, and related RL/post-training objectives.
- SFT updates are much denser than RL updates.
- The sparse RL update is not simply low-rank. Most updated parameter matrices are nearly full-rank.
- The sparsity is spread across layers and matrices, not isolated to one convenient module.
- Finetuning only the identified subnetwork can recover, and sometimes exceed, full fine-tuning performance.
- Subnetworks overlap across seeds, datasets, and algorithms more than random guessing.
- One reported seed-overlap example is around 60%, showing partial but not complete reuse.
- KL regularization and gradient clipping have limited impact on observed update sparsity.
- In-distribution/on-policy data appears more important for sparsity than explicit regularization.
- Their GRPO KL/no-KL comparison reports similar sparsity levels, roughly 69.8% vs 68.8%.

What this means for #25:

- Do not assume the RLVR update is naturally represented by a single low-rank object just because PowerSGD is low-rank.
- Activation communication can be low-rank while the downstream parameter update remains sparse and nearly full-rank.
- This creates a possible mismatch: a rank-77 activation basis may preserve reconstruction but restrict the sparse/full-rank update pattern GRPO wants.
- EF-PowerSGD is attractive because it can keep low-rank communication while accumulating omitted residual information over time.
- Sparse-ticket diagnostics should be interpreted alongside `Q` diagnostics: one probes parameter-update tolerance, the other probes activation-communication loss.

Required measurements from this paper:

- real update matrices from dense and compressed GRPO steps,
- per-matrix update sparsity,
- per-matrix update rank,
- sparse/full-rank characterization of dense vs compressed updates,
- layerwise update distribution,
- whether omitted PowerSGD residual energy maps to sparse/full-rank parameter updates.

### `arXiv:2509.04259`: RL's Razor

Core findings:

- On-policy RL is biased toward KL-minimal solutions among the many solutions that solve the new task.
- SFT can reach similar task performance while moving farther from the base policy.
- KL divergence from the base/current policy predicts forgetting better than raw weight movement or sparsity in their experiments.
- GRPO and 1-0 Reinforce reach task performance with smaller KL shifts than offline objectives such as SFT/SimPO.
- The critical factor is on-policy data, not merely the presence of negative examples.
- The paper reinforces that RL and SFT can reach similar scores through different distributional paths.

What this means for #25:

- A compression method should preserve the KL-minimal/on-policy path, not merely recover a plausible supervised direction.
- A biased merger can look small in weight or activation space but still push the policy off the RLVR path.
- Forward KL, token-level KL, and update cosine should be first-class diagnostics for the next issue.
- If a method improves reconstruction but increases KL drift or worsens update cosine, it is probably not RLVR-native.

Required measurements from this paper:

- real policy-gradient/update measurements from the fixed GRPO path,
- per-step forward KL from current/old policy,
- cumulative KL to base/reference policy when available,
- KL delta dense vs compressed,
- update cosine vs dense,
- old-log-prob/current-log-prob drift under rollout-correction settings.

### Pass@k And Support-Coverage Papers

These are secondary for the immediate #25 follow-up, but useful for avoiding a false "surpass dense" claim.

Relevant findings:

- Current RLVR often improves pass@1 while narrowing high-k coverage.
- Some papers argue RLVR mostly sharpens probability mass over solutions already accessible to the base model.
- Base models can sometimes outperform RLVR-trained models at large `k` because they retain broader answer support.
- Token-level entropy is not the same as answer-level diversity.
- Increasing temperature after training does not necessarily restore the base model's coverage.
- SFT/distillation can expand support in ways standard RLVR often does not.

What this means for #25:

- The next issue should not make dense-surpass claims from greedy validation alone unless parity is first established.
- If a compressed method seems better than dense, measure whether it improves pass@k/answer coverage or only changes greedy mode selection.
- Answer-level entropy and pass@k are diagnostics after the core lag is fixed. They should not distract from the immediate merger/Q diagnosis.

Required measurements from these papers:

- greedy validation,
- sampled mean@k,
- pass@k,
- answer-level entropy,
- token-level entropy,
- overlap of solved problem sets between dense and compressed,
- response length and answer/error categories.

## Updated Role Of The Anchor Circuit

The anchor circuit remains mandatory, but its role must change.

Keep:

- anchor owns `Q`;
- fast circuit reads `Q`;
- anchor refreshes clean geometry;
- anchor provides diagnostics;
- anchor supports bounded residual or preconditioner state.

Remove:

- anchor as sign oracle;
- anchor as direct replacement for live GRPO direction;
- merger logic that assumes clean stale sign is safer than live compressed sign.

The anchor should answer these questions:

- Which directions does dense GRPO actually update?
- Does current `Q` preserve those directions?
- Is compression bias principal, off-principal, or random-looking?
- Does EF/residual logic reduce bias without changing update direction?
- Is the fast path still KL-proximal and on-policy-like?

## Q Is The Main Compression Object

`Q` is still the right lever because it controls what information crosses the pipeline boundary. But the current `Q` should be treated as an SFT-style baseline, not as the final RLVR basis.

Current likely issue:

- `Q_act` preserves top activation-energy directions.
- Those directions may be excellent for reconstruction.
- They may not be the directions GRPO uses for off-principal, small, spectrum-preserving adaptation.

Candidate `Q` families at fixed total rank 77:

- `Q_act`: current activation-energy basis. Control.
- `Q_grad`: anchor-owned basis from live GRPO gradient/right-singular statistics.
- `Q_adv`: basis from advantage-weighted activation statistics.
- `Q_tail`: basis after removing top activation/principal components.
- `Q_hybrid`: split-rank basis, e.g. activation-energy plus off-principal GRPO-gradient directions.
- `Q_ticket`: basis or mask family informed by sparse-ticket diagnostics from `arXiv:2602.01599`.

Do not judge these only by activation reconstruction error. Judge them by:

- dense/compressed update cosine,
- principal/off-principal update preservation,
- greedy validation,
- pass@k/coverage when relevant,
- response length,
- clip ratio,
- validation trajectory,
- entropy slope,
- residual norm.

## Moving Forward: First Diagnose, Then Change The Merger

The next phase should be ordered to avoid another expensive run with an unclear failure mode.

### Step A: Real-Gradient Geometry Audit

Goal: identify whether the current method lags because `Q` preserves the wrong subspace, because the merger corrupts signs, or both.

This must use real gradients and real matrices from the fixed GRPO path. Existing dense, PowerSGD, no-refresh, and EXP-25 logs/checkpoints can be used, but only if they contain the required tensors. If not, run a short diagnostic job for a few optimizer steps and capture the tensors directly.

Minimum short-run requirement if existing artifacts are insufficient:

- run dense and current compressed/anchor paths for the same prompts and rollout settings,
- capture real activation matrices at compression targets,
- capture real compressed/reconstructed activations,
- capture real per-target gradients or activation-gradient products,
- capture real parameter update matrices before optimizer state hides the raw direction,
- capture `Q` and any residual/anchor statistics at the same steps,
- log global step, minibatch/optimizer tick, target name, tensor shape, dtype, norm, and rank/projection stats.

Required outputs:

- Principal/off-principal decomposition of dense GRPO updates.
- Principal/off-principal decomposition of PowerSGD and EXP-25 updates.
- `Q_act` capture ratio for gradient/update energy, not only activation energy.
- Dense vs compressed update cosine per target using real gradients/updates.
- Sign agreement between live compressed gradient, anchor statistic, and dense reference at `delay_K=0` and current `delay_K`.
- Spectral drift and principal subspace rotation across training.
- bf16-aware apparent sparsity vs fp32 update magnitude.

Decision after Step A:

- If `Q_act` captures update energy well, prioritize EF/residual merger.
- If `Q_act` misses off-principal update energy, prioritize RLVR-native `Q`.
- If anchor sign disagreement remains near coin-flip even at `delay_K=0`, permanently remove sign-merger ideas.

### Step B: Direction-Preserving EF-PowerSGD

Goal: recover the plain PowerSGD/fresh-clean band without sign replacement.

For each compressed activation tensor `h_t`:

```text
u_t       = h_t + e_t
y_t       = u_t Q_t
h_hat_t   = y_t Q_t^T
e_{t + 1} = decay_clip(u_t - h_hat_t)
```

Design constraints:

- Anchor owns `Q`.
- Fast path reads `Q`.
- Residuals are per target and shape-aware.
- Reset residual on shape mismatch.
- Clip residual norm relative to activation norm.
- Detach residual state unless intentionally testing a differentiable residual.
- No sign replacement.

Promotion gate:

- `val@50` returns to the plain PowerSGD/fresh-clean band or better.
- Update cosine improves over plain PowerSGD.
- Length, clip ratio, and validation trajectory stay outside EXP-25 collapse alarms.

### Step C: RLVR-Native Q Sweep

Goal: identify whether the compression basis should preserve activation principals, GRPO-gradient directions, off-principal directions, or a hybrid.

Arms:

- `Q_act`,
- `Q_grad`,
- `Q_adv`,
- `Q_tail`,
- `Q_hybrid`,
- `Q_ticket` if Step A/D supports it.

Hold total rank at 77. Do not change rank allocation and merger at the same time unless the earlier gate is already passed.

Primary readouts:

- greedy val,
- update cosine,
- off-principal update preservation,
- response length,
- clip ratio,
- validation trajectory,
- entropy slope.

### Step D: Sparse-Ticket Diagnostic

Goal: use `arXiv:2602.01599` as a geometry probe, not as a separate objective.

Run small controlled masks:

- all-parameter dense update,
- 1% random parameter update mask,
- 5% random parameter update mask,
- off-principal/low-magnitude 1% mask,
- principal/high-magnitude 1% mask.

Interpretation:

- If off-principal or random sparse masks match dense better than principal masks, that supports the RLVR-off-principal reading.
- If principal masks win, the current task/model may not follow the paper's geometry strongly enough for tail-heavy `Q`.
- If all sparse masks fail, keep sparse tickets as a paper-level observation but do not let them drive the main pipeline-compression plan.

This step is diagnostic because parameter sparsity does not directly reduce inter-stage activation communication. Its value is identifying the update geometry that `Q` should preserve.

### Step E: Dense-Matched Communication Run

Goal: test communication-efficient GRPO after the diagnosis is resolved.

Only run after Step B or C identifies a stable method.

Minimum comparison:

- dense control,
- plain PowerSGD reference,
- best EF-PowerSGD arm,
- best RLVR-native `Q` arm if different.

Success criteria:

- matches or beats plain PowerSGD/fresh-clean,
- materially reduces inter-stage communication,
- preserves live update direction,
- does not reintroduce EXP-25 length/clip collapse,
- has a clear explanation from the geometry audit.

Dense surpass should be treated as a later claim. First prove parity with a method whose mechanism is understood.

## Handoff For A Future Issue

The future issue should be framed as a #25 follow-up, not as a broad new research direction.

Suggested issue title:

```text
Diagnose SFT-to-GRPO mismatch in communication-efficient GRPO and test direction-preserving RLVR-native compression
```

Problem statement:

- #25 proved the anchor/PowerSGD substrate is usable.
- #25 falsified `signed_ema`.
- The gap is caused by the merger corrupting live GRPO direction, not by rank-77 PowerSGD alone.
- Recent RLVR papers suggest the current method is too SFT-like: it preserves or injects stable/principal directions, while RLVR learns through small, KL-proximal, often off-principal updates.

Non-goals:

- no anchor-sign replacement,
- no training-objective changes,
- no broad rank sweep before geometry is diagnosed,
- no dense-surpass claim before parity is recovered,
- no fast-path ownership of `Q`.

Minimum experiment plan for the issue:

1. Run the real-gradient geometry audit on existing dense, PowerSGD, no-refresh, and EXP-25 artifacts if they contain the needed tensors; otherwise run a short few-step diagnostic to collect real gradients, activations, update matrices, `Q`, and anchor statistics.
2. Decide whether the immediate blocker is merger direction corruption, `Q` subspace mismatch, or both.
3. Implement EF-PowerSGD as the first direction-preserving merger candidate.
4. If `Q_act` misses update energy, run a fixed-rank RLVR-native `Q` sweep.
5. Use sparse-ticket masks only as diagnostics for update geometry.
6. Compare against dense, plain PowerSGD, and no-refresh references under the fixed control surface.

Issue acceptance criteria:

- The issue contains exact runs/arms, metrics, stop conditions, and promotion criteria.
- The issue lists the EXP-25 failure mechanism before proposing new experiments.
- The issue requires principal/off-principal diagnostics and update cosine computed from real gradients/matrices.
- The issue requires length/clip collapse alarms.
- The issue explains why each arm tests a specific paper-derived hypothesis.
- The issue keeps the GRPO verifier/objective fixed.

## Stop Conditions

Stop a merger immediately if:

- it uses anchor sign to replace live sign;
- magnitude-weighted sign disagreement reappears near the EXP-25 pattern;
- update cosine vs dense collapses;
- response length explodes while validation/training signals look superficially healthy;
- clip ratio enters the EXP-25 danger band;
- `val@50` falls below the no-refresh floor without a clear diagnostic reason;
- fast path updates `Q`.

## Promote Conditions

Promote a method if:

- it is direction-preserving;
- it keeps anchor ownership of `Q`;
- it improves update cosine or off-principal preservation;
- it matches or beats plain PowerSGD/fresh-clean under the fixed surface;
- it reduces inter-stage communication;
- its win is explained by the geometry audit rather than by an uninspected training artifact.

## Practical Notes

- Keep `resolved_params` proof mandatory: codec, rank, cadence, `delay_K`, target count, and `owns_Q`.
- Log cadence in both optimizer/minibatch ticks and global steps; EXP-25 showed that cadence accounting can mislead.
- Preserve cold-M and anchor guards even if `M` becomes diagnostic only.
- Avoid global hooks and module swaps that can silently desynchronize anchor/fast targets.
- If KL or length caps are used, label them as guardrail diagnostics, not as part of the fixed no-KL/no-entropy control surface.

## References

Internal:

- `CLAUDE.md`
- `CODE_WALKTHROUGH.md`
- `research/.claude/GOAL.md`
- `research/PROGRESS.md`
- `research/LOG.md`
- `research/runs/SUMMARY.md`
- `research/runs/FIXED_CONTROL_SURFACE.md`
- `research/runs/UNWANTED_HOOKS_AND_SILENT_FAILURES.md`
- `research/runs/EXP-25/verdict.md`
- `research/runs/EXP-25/DEEP_FINDINGS.md`
- `research/runs/EXP-25/ENTROPY_COLLAPSE_FINDINGS.md`
- `research/runs/EXP-25/COLLAPSE_GRADIENT_FLOW_ANALYSIS.md`
- `research/runs/EXP-25/PATH_TO_SURPASS_DENSE.md`
- `research/runs/EXP-25/SURPASS_DENSE_STRATEGY.md`
- `research/diagnostics/ENTROPY_COLLAPSE_WATCH.md`

External:

- "The Path Not Taken: RLVR Provably Learns Off the Principals", arXiv:2511.08567, https://arxiv.org/abs/2511.08567
- "The Multiple Ticket Hypothesis: Random Sparse Subnetworks Suffice for RLVR", arXiv:2602.01599, https://arxiv.org/abs/2602.01599
- "Reinforcement Learning Finetunes Small Subnetworks in Large Language Models", arXiv:2505.11711, https://arxiv.org/abs/2505.11711
- "RL's Razor: Why Online Reinforcement Learning Forgets Less", arXiv:2509.04259, https://arxiv.org/abs/2509.04259
- "The Invisible Leash: Why RLVR May Not Escape Its Origin", arXiv:2507.14843, https://arxiv.org/abs/2507.14843
- "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?", arXiv:2504.13837, https://arxiv.org/abs/2504.13837
- Vogels et al., "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization", NeurIPS 2019, https://arxiv.org/abs/1905.13727
