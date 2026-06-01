# Team brief — explain masked+clean GRPO vs dense (comm-eff-theory team)

You are a teammate on the `comm-eff-theory` agent team. You do NOT inherit the lead's
conversation history — this file + your spawn prompt are your context. Read it fully.

## The system under study
Private verl fork building a **communication-efficient, pipeline-parallel GRPO trainer**.
Model: **Qwen2.5-1.5B-Instruct**. RL: **vanilla GRPO, no-KL, no-entropy** (pure RLVR; reward is
verifiable correctness of a `\boxed{}` / `####` answer). See `CODE_WALKTHROUGH.md` and
`research/.claude/GOAL.md` for the method.

The "comm-eff method" (the variable under study here):
- **Per-(token, channel) activation masking** at the 7 pipeline-stage boundaries
  `[3,7,11,15,18,21,24]` of the 28-layer model, with **drop probability p=0.9** (≈85.5% of
  boundary activation traffic dropped), **rescale ON** (surviving activations scaled by 1/(1-p)
  to keep expectation), `mask_recompute=true`. Anchor + spectral correction are **OFF**.
- **`clean_cadence=20`**: every 20th optimizer step uses the **true, unmasked dense gradient**
  (a full clean forward/backward); the other 19/20 steps use the masked forward.
- Crucial implementation fact: the PPO ratio is computed between the **masked old-logprob and
  masked new-logprob** (both masked → self-consistent → ratio ≈ 1, `pg_clipfrac` ≈ 0.03). The
  mask corrupts the **gradient-estimation forward**, NOT the deployed policy. **Validation is
  always run with the UNMASKED (true) forward** — so val accuracy measures the real policy (the
  weights), not the masked forward.
- `calculate_log_probs=True` captures vLLM's own generation probs to compute the
  `rollout_corr/*` train-inference DIAGNOSTICS (these are NOT used in the loss).

## The empirical findings to explain (from our runs)
1. **GSM8K (EXP-17, masked clean@20):** val **0.085 → 0.735**, reward 0.108 → 0.749, over 116
   steps. Dense GSM8K reference (EXP-16 cell6) ≈ **0.741**. So masked-clean@20 ≈ **dense parity**
   on GSM8K despite a ~95% communication cut. (Step-0 0.085 was a `####`-format artifact; RL
   quickly elicited the latent capability → 0.49 by step 30.)
2. **Big-Math-RL-Verified-filtered (harder, competition math):**
   - **EXP-19 masked clean@20:** val **flat ~0.55** (0.56→0.55), reward flat ~0.40. NO learning.
   - **EXP-20 dense (comm-eff OFF):** val **climbing 0.558 → 0.608**, reward **0.41 → 0.56**.
     Real (if modest) learning. → On hard data, DENSE learns but masked-clean@20 STALLS.
3. **rollout_corr/* (train-inference consistency), EXP-19, WandB run `zejoupvf`:**
   - **Masked steps:** `kl ≈ 16.8`, `pearson(actor,rollout) ≈ 0.004` (≈ decorrelated),
     `ppl_ratio ≈ 2×10⁷`, `training_log_ppl ≈ 17` vs `rollout_log_ppl ≈ 0.36`,
     `rollout_probs_diff_mean ≈ 0.85`, `chi2_seq = -1` (seq-level IS weight underflows to ~0).
   - **Clean steps (20/40/60/80):** `kl ≈ 0.0003`, `pearson ≈ 0.9996`, `ppl_ratio ≈ 1.0`,
     `training_log_ppl ≈ rollout_log_ppl ≈ 0.3`. → a **clean-resettable sawtooth**, not a ratchet.
   - `grad_norm`: masked steps ~4–5; clean steps ~0.2–0.4. `pg_clipfrac` ~0.03–0.04 throughout.
   (EXP-16/EXP-17 established the same sawtooth on GSM8K; gap is mask-caused + flat/stationary.)

## The questions this team must answer
A. **What kind of policy is the masked actor following, vs dense?** Distinguish the masked
   *training-time forward* (≈ decorrelated from the sampler, near-random per `pearson≈0.004`) from
   the *actual policy* = the weights (evaluated unmasked). What objective/gradient does the masked
   update actually optimize? Is the masked forward a biased/stochastic estimator of the true
   policy gradient (relate to dropout, DropConnect, random projection / sketched gradients,
   structured-noise SGD, gradient compression with error-feedback)?
B. **Why does a ~20-million× train-inference perplexity mismatch still LEARN (on GSM8K)?** What is
   the minimal condition for SGD/policy-gradient ascent under such a corrupted forward (positive
   expected inner product with the true gradient? unbiasedness from rescale? clip protection?),
   and what role does the **clean step every 20** play (bias removal / re-anchoring / variance
   reduction)?
C. **Why does it work on GSM8K but stall on Big-Math?** Tie to: GSM8K is *easy* for this model
   (elicitation of a latent ~73% capability, weak/coarse gradient suffices) vs Big-Math is *hard*
   (requires capability the model lacks; dense extracts modest signal, the lossy masked gradient
   cannot). I.e. the masked-gradient's information loss is tolerable when the task only needs
   elicitation, fatal when it needs genuine learning.
D. **Literature:** recent RLVR results where **noisy / random / spurious / 1-shot rewards still
   improve** Qwen-class models (and the "RLVR elicits vs teaches" debate), plus any
   gradient-compression / activation-masking / stale-gradient training that is analogous. Map each
   to our setting.

## Deliverables (write under research/findings/theory/)
- `theory.md` — answers A, B, C with explicit math/mechanism (theorist).
- `literature.md` — annotated bibliography + relevance mapping (lit-scout).
- `empirical_check.md` — does the theory match EXP-17/19/20 + rollout_corr numbers? contradictions (empiricist).
- `REPORT.md` — the synthesized explanation (synthesizer), citing the above.

## Rules
- Research/analysis only. Do NOT launch training, touch Vast.ai, or edit `verl/` source. Write
  only under `research/findings/theory/`. The box is busy running EXP-20 (do not interfere).
- Be rigorous and skeptical: challenge each other's claims; mark speculation vs established fact.
- Data sources you may read: `research/runs/EXP-17/`, `research/runs/EXP-19/`, `research/runs/EXP-20/`
  (train.log + metrics), `CODE_WALKTHROUGH.md`. WandB run ids: EXP-17=`t03dn4nh`, EXP-19=`zejoupvf`.
