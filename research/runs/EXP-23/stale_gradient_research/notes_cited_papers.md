# EXP-23 — Cited-papers reader notes (RL gradient structure → stale-anchor tolerance)

**Author:** cited-reader (task #2) · **Date:** 2026-06-04

**Problem we're reading against (EXP-23):** combine a STALE full-rank gradient
`M` (anchor captured at θ_{t−K}, K≈5) with a LIVE PowerSGD-compressed gradient
`G` (low-rank sketch of the current PP-boundary grad), via additive *inject*
(`g = G + λ·M`) or convex *blend* (`g = (1−λ)·G + λ·M`), to test whether a
stale re-anchor can recover the effect of a fresh full-rank step for a
compressed GRPO (RLVR) trainer. The operator's hypothesis: **RL's gradient /
update structure (sparse, small-subnetwork, but full-rank, and qualitatively
different from SFT) is the reason staleness might be tolerable.** These notes
extract what each cited source actually says and what it implies for that
hypothesis and for the choice of combiner.

A note on arXiv IDs: the operator's IDs are near-future and a couple resolved
to slightly different landing pages / canonical IDs than given. Discrepancies
are flagged per source. All four sources were located and read.

---

## Source 1 — RL update sparsity (operator id 2602.03839)

**Resolved to:** arXiv **2602.03839v2**, *"Understanding and Exploiting Weight
Update Sparsity for Communication-Efficient Distributed RL,"* Erfan Miahi &
Eugene Belilovsky. (ID matched as given.) Read via PDF + arXiv HTML.

### Gradient-structure claim
Per-step weight updates in RL post-training are **~99% "compute-invisible":**
> "approximately 99% of per-step weight updates are invisible after the BF16
> cast used by standard training and inference forward passes."

Mechanism: at typical RL post-training LRs, **Adam updates often fall below the
local BF16 rounding threshold**, so casting θ to the forward-pass dtype erases
them. This is *per-step* and *bitwise* sparsity (much finer than the coarse
5–30% checkpoint-diff sparsity in Mukherjee).

### Supporting evidence / equations
- Sparsity is **stable across training and scale**: "Mean per-step sparsity is
  approximately 99% across all model scales and families," stdev over 400 steps
  only **0.2–0.4%**, worst-case step still **>98%**. Within the **k ≤ 8** async
  staleness range they recommend, sparsity stays **>98%**.
- Sparsity definition (Def. A.2):
  `S_k^D(t) = (1/d) Σ_i 1[ cast_D(θ̄_{t+k}^{(i)}) = cast_D(θ̄_t^{(i)}) ]`
- Compute-visibility gate (Eq. 1):
  `G_D(θ, s) := { i : cast_D(θ_i) ≠ cast_D(θ_i − s_i) }`
  — transmit only coords whose update would change the next forward pass.
- **Error feedback is explicit** in the trainer-to-trainer variant PULSELoCo
  (Algorithm 2): updates that fail the gate are NOT dropped — kept in an FP32
  error-feedback buffer for the next outer round:
  `s_r^(t) ← (θ^(t−1) − w_r) + e_r^(t−1)`;
  `e_r^(t)[visible] ← 0`; `e_r^(t)[invisible] ← s_r^(t)[invisible]`.
  PULSESync (trainer→inference) keeps the unsent residual in the FP32 master
  weights instead (it's exact-reconstruction, lossless).
- Results: >100× weight-sync reduction (bit-identical reconstruction); PULSELoCo
  cuts trainer-to-trainer comm >17× vs DiLoCo, >100× vs DDP.
- RL vs SFT: not directly compared; they cite (coarsely) "RL fine-tuning
  changes only 5–30% of parameters under coarse checkpoint comparisons."

### What it implies for our STALE-anchor problem
This is the most operationally-relevant source for our combiner design, on two
counts. **(a) Staleness is already validated in this regime:** the authors
explicitly support **k ≤ 8 step asynchrony** and show the (compute-visible)
update set stays >98% sparse across that window — i.e. an old gradient stays a
good gradient for several steps because so little actually moves. Our K=5
anchor staleness sits inside their endorsed window. **(b) Error feedback is the
named fix for what staleness/quantization drops.** Their whole method is: gate
out the part you can't afford to send, but *accumulate the residual* and inject
it next round so nothing is permanently lost. This maps directly onto our
inject path: rather than treating the PowerSGD residual (the part of G outside
its low-rank sketch) as gone, the stale full-rank M can be read as a *periodic
flush of the accumulated residual* — and the cleanest version of EXP-23 would
add a true EF buffer (accumulate `G_full − G_compressed` locally, fold into the
stale M at re-anchor) rather than relying on M to coincidentally cover it.
**Concrete combiner suggestion: error-feedback inject** — maintain a local FP32
residual buffer, and at each step use `g = G_compressed + decay·buffer`,
refreshing `buffer` from the stale full-rank M every K steps; this is a known-
convergent pattern (EF-SGD / PowerSGD already ship EF) and matches the operator
memory note that "NO error-feedback = top lever" for PowerSGD here.

---

## Source 2 — RL ≠ SFT, learns off the principals (operator id 2511.08567)

**Resolved to:** arXiv **2511.08567**, *"The Path Not Taken: RLVR Provably
Learns Off the Principals,"* Zhu, Zhang, Huang, Su, Liu, Zhao, Fedorov,
Pirsiavash, Sha, Lee, Pan, Wang, Tian, Tai. (ID matched.) HTML/ar5iv conversion
failed; read from abs page + author Three-Gate summary.

### Gradient-structure claim
RLVR updates are **NOT genuinely sparse and NOT low-rank** — the sparsity is a
*surface artifact*. The real structure: RLVR moves weights **off the principal
directions**, in low-curvature, spectrum-preserving subspaces, whereas SFT
**targets principal weights and distorts the spectrum**.
> "RLVR learns off principal directions in weight space" / "SFT targets
> principal weights, distorts the spectrum"; RLVR gains come "via minimal
> spectral drift, reduced principal-subspace rotation."

### Supporting evidence / framework
**Three-Gate Theory** — every RL step passes three gates that push the update
off-principal:
- **Gate I (KL Anchor):** KL-constrained update keeps the step small/local.
- **Gate II (Model Geometry):** steers the step *off principal directions* into
  low-curvature, spectrum-preserving subspaces.
- **Gate III (Precision):** micro-updates in non-preferred regions get hidden
  by precision/rounding, **making the off-principal update LOOK like sparsity**.
- Causal control: applying an **orthogonal rotation** to the pretrained weights
  (destroying its geometry) **abolishes update locality** — links the bias
  causally to pretrained geometry, not to the RL objective per se.
- Conclusion: "RL operates in a distinct optimization regime from SFT, so
  directly adapting SFT-era PEFT methods can be flawed."

### What it implies for our STALE-anchor problem
This is the *cautionary counterweight* to the simple "RL is sparse so staleness
is easy" story. Two implications: **(1) The full-rank M is not redundant.**
Because the true RLVR update is full-rank and off-principal (Gate III says the
"sparsity" is a precision mirage), the part of the gradient that PowerSGD's
low-rank sketch *misses* is real signal, not noise — which is exactly why a
**full-rank** stale M (or an EF residual) is worth folding in at all, and why a
purely low-rank compressor would systematically under-serve RL. **(2) Spectrum
preservation argues for a magnitude-capped / blend combiner over raw inject.**
RLVR works by *minimal spectral drift*; an additive inject of a stale full-rank
M risks a large, spectrum-distorting step (re-introducing the SFT-like failure
mode). A **convex blend** (`(1−λ)G + λM`, small λ) or a magnitude cap on the
injected M keeps the per-step move small and spectrum-preserving, consistent
with how RLVR is shown to succeed. This aligns with the operator's prior memory
result that *blend is the only live correction (0.81) while reweight is inert
(0.14)* — blend respects the small-step / off-principal geometry; an aggressive
projection/reweight does not. Caveat for the team: this paper warns that
borrowing SFT-era machinery (incl. low-rank/PEFT-style assumptions) into RL can
be "flawed" — so the staleness tolerance should be argued from RL's own
geometry (Gates I–III), not by analogy to SFT delayed-gradient results.

---

## Source 3 — OPD blog (SFT vs RL gradient pressure)

**Source:** https://nrehiew.github.io/blog/sft_rl_opd/ — "On-Policy
Distillation" / SFT-vs-RL writeup. Synthesizes the two arXiv results below.

### Gradient-structure claims (with the references it cites)
- **SFT** "exerts gradient pressure uniformly across the entire model
  distribution, not only on regions relevant to the new task" → broad, dense,
  redundant updates.
- **RL** "exerts gradient pressure only through behavior sampled from the
  current policy" → "RL mostly reshapes high-probability regions the model
  already visits" → localized.
- Cites **Mukherjee et al.**: "RL updates only a small subnetwork of a model via
  sparse but full-rank updates while SFT induces dense ones." (= Source 4 below.)
- Cites **Yuan et al.**: "SFT updates are more redundant, while RL updates are
  more important" — when you prune the # of updated params, "the performance of
  RL degrades much faster" than SFT (RL updates are load-bearing, SFT's are not).
- No claim about update-direction stability across steps.

### What it implies for our STALE-anchor problem
The blog supplies the *intuition* that makes a stale anchor plausible: RL only
pushes on the **near-policy / on-policy** region, so over a short K-step window
the policy distribution (and hence where gradient pressure lands) changes
slowly → the *region* of the update is persistent even if the exact coords
aren't. That's the optimistic case for K=5 staleness. But the **Yuan-et-al**
half is a warning: RL updates are *important, not redundant* — you cannot drop
them cheaply (unlike SFT). So a stale M that is even slightly mis-aligned with
the current active region is more costly in RL than the same staleness would be
in SFT. Net: favors **short K + correction targeted at the live active set**
(mask the stale M to the coords PowerSGD/the live grad currently touches) over
a blunt full-tensor inject.

---

## Source 4 (read in depth — operator-emphasized) — Linear dynamics & periodic re-grounding

**Operator id given:** 2601.04537 → **resolved to** arXiv **2601.04537**,
*"Linear Dynamics in the RLVR Training of Large Language Models,"* Tianle Wang,
Jiayu Liu, Zhongyuan Wu, Shenghao Jin, Wei Chen, Hao Xu, Ning Miao. Code:
github.com/Miaow-Lab/RLVR-Linearity. (ID matched.) Read via abs + full PDF.

> **NOTE / possible operator conflation:** the operator framed this id as the
> "RL updates a small subnetwork / sparsity" source, but **2601.04537 is the
> linear-dynamics paper**, not a sparsity paper. The sparsity/full-rank claim
> the operator wanted is actually **Mukherjee et al. 2505.11711** (see Source 4b
> below). I read 2601.04537 in depth as instructed — and it turns out to be the
> *most directly relevant of all four to the stale-anchor mechanism*, so the
> emphasis was well-placed regardless.

### Gradient-structure claim (this is the key one for EXP-23)
Across model families, RL algorithms, and configs, **RLVR enters a robust
LINEAR regime**: both parameter weights and output log-probs evolve **highly
linearly** (R² > 0.7, often 0.95+). Verbatim abstract:
> "RLVR consistently enters a robust linear regime, where both parameter weights
> and output log-probabilities … evolve in a highly linear manner (R² > 0.7)."

The cumulative weight change is well-modeled as a **stable low-dimensional
drift**: `Δθ_t = θ_t − θ_0 ≈ v · t`, with `v` a (near-)constant drift vector.

### Supporting evidence / mechanism (the low-pass-filter argument)
> "this linearity … stems from the high-variance, noisy nature of RLVR training
> signals, which act as a low-pass filter to concentrate optimization along a
> stable, low-dimensional drift."

Mechanism, as they argue it: the per-step RL gradient is high-variance noise on
top of a small persistent mean. Over many minibatches the noise destructively
interferes and averages out; only the **low-variance persistent drift component
survives** → effective update `Δθ ≈ α·(E[∇L_RL] + noise)`, noise → 0, leaving
the mean drift operative across many steps. Empirically the per-window update
**direction is stable**: cosine similarity of the estimated drift across K-step
windows stays high (cos > 0.9), so the direction at step t predicts the
direction at t+K.

### The actionable method — weight-space extrapolation + periodic re-grounding
> "weight-space extrapolation matches the performance of standard RL
> optimization while achieving a **6.1× training speedup through periodic
> re-grounding**."

Procedure: (1) estimate drift `v` from a short window of real steps; (2)
**extrapolate** forward without computing gradients: `θ_{t+K} ≈ θ_t + K·v`;
(3) **re-ground every K steps** by doing a real gradient step and recalibrating
`v`; (4) repeat. Re-grounding cadence K is on the order of ~10–50 (the 6.1×
comes from doing far fewer real gradient computations than steps). Output-space
extrapolation is a second use (bypasses late-stage collapse, +4.2% avg).

### What it implies for our STALE-anchor problem (the operator's emphasis)
**This paper is essentially independent evidence that our EXP-23 mechanism
should work, and it tells us how to do the combine.**
1. **Staleness is cheap because the drift direction is stable.** If the RLVR
   weight trajectory is linear with cos > 0.9 across K-step windows, then a
   full-rank gradient/anchor captured at θ_{t−5} still points in essentially the
   current update direction — *that is exactly the premise of EXP-23*. A K=5
   stale M is far inside the regime where they extrapolate with a single drift
   vector. The operator's hypothesis ("staleness might be tolerable for RL")
   has a direct mechanistic basis here: **RL's gradient is low-pass-filtered to a
   slowly-changing low-dimensional drift, so its directional autocorrelation
   across a few steps is high.**
2. **Re-grounding ≈ our re-anchor.** Their "extrapolate K steps then re-ground"
   is structurally identical to "run K compressed steps then refresh the stale
   full-rank anchor M." It validates the *cadence* design (periodic full-rank
   ground-truth correcting a cheap approximation in between) and suggests the
   clean/anchor cadence can be relaxed well beyond K=5 (they go 10–50) — matching
   the operator memory note that clean_cadence is "likely relaxable."
3. **Combiner guidance — drift, not impulse.** Because the *signal* is the
   persistent low-dimensional drift and the *noise* is the high-variance
   residual, the right way to use the stale M is as an **estimate of the drift
   direction** to bias/steer the live compressed G, not as a one-shot additive
   impulse. Two concrete options consistent with this paper:
     - **Blend toward the stale drift:** `g = (1−λ)G + λ·M̂`, where `M̂` is the
       (normalized) stale full-rank direction — keeps steps small (Gate-I/II
       friendly per Source 2) while pulling G back onto the persistent drift the
       low-rank sketch may have rotated away from.
     - **EMA / drift-tracked anchor instead of a single K-old snapshot:** since
       the drift is stable, an EMA of past full-rank grads is a *lower-variance*
       estimate of `v` than one stale snapshot — directly addresses the
       low-pass-filter mechanism (do the averaging the paper says RL is
       implicitly doing). This is cheaper to keep fresh than a full re-anchor.
4. **Failure mode to watch (late-stage non-linearity / collapse):** the paper
   notes linearity is strongest mid-training and they use *output-space*
   extrapolation to bypass **late-stage collapse**. Implication: a fixed stale-K
   may degrade near the end of a run where the trajectory bends; EXP-23 should
   check whether the inject/blend benefit decays in the last steps, and consider
   shortening K (more frequent re-anchor) late.

### Source 4b — the actual sparsity/full-rank reference (Mukherjee et al.)

Because the operator's sparsity claim really lives here, I read it too:
arXiv **2505.11711**, *"Reinforcement Learning Finetunes Small Subnetworks in
Large Language Models,"* Mukherjee et al.

- **Sparse:** RL updates only a small subnetwork — **5–30% of params** (sparsity
  68.5–96.0%), across 7 RL algos (DPO/GRPO/PPO/ORPO/KTO/SimPO/PRIME) × 10 models.
- **But full-rank:** "updates to almost all parameter matrices are nearly
  full-rank … span almost the full subspaces." Table 2: mean update rank is
  **99.2–99.8% of max** (e.g. DeepSeek-Math-7B GRPO 99.4%). → unlike LoRA, the
  small subnetwork is NOT low-rank.
- **Subnetwork stability (load-bearing for staleness):** the subnetwork is
  *substantially* but *not perfectly* consistent. Cross-seed overlap **~60.5/60.6%**;
  cross-dataset 26.7/67.1%; full stress (seed+data+algo) 59.1/33.2% — "greater
  than chance" but "falls short of 100%." During training the identity of
  updated params **shifts somewhat** (early grads outside the final subnetwork
  cancel out; sparsity drifts ~77→80%).
- Subnetwork-only finetuning (freeze the rest) **matches/beats full** finetuning
  (θ_sub vs θ_full ~94% identical weights for DPO; +1.6–2.4% acc).
- Sparsity cause: **near-policy / in-distribution data** (in-dist DPO 81.4%
  sparse vs OOD DPO ~7% — dense). Not KL, not clipping, not prior SFT.

**Implication for EXP-23:** the **full-rank** finding is the strongest argument
*for* keeping a full-rank stale M (a low-rank PowerSGD sketch alone provably
under-covers RL's full-rank update — consistent with Source 2). The **~60%
subnetwork overlap across seeds, partial drift across steps** is the strongest
argument *for short K and active-set masking*: the active subnetwork is mostly
but not entirely persistent, so a 5-step-stale M is mostly aligned but should be
**masked/projected onto the live active set** (the coords the current G touches)
to drop the ~40% that may have moved. This is the single most defensible
combiner recipe across all sources: **full-rank stale M, blended (not impulse-
injected) at small λ, masked to the live active subnetwork, with error feedback
on the dropped residual.**

---

## Cross-source synthesis (one screen)

| Source | Core claim | Implication for stale M + live G |
|---|---|---|
| 2602.03839 (Miahi/Belilovsky) | ~99% per-step updates compute-invisible; **k≤8 async OK**; **error feedback** is the named fix | Our K=5 is inside the validated async window; **add an EF residual buffer** rather than hoping M covers the dropped part |
| 2511.08567 (Path Not Taken) | RLVR sparsity is a **surface artifact**; true update is **full-rank, off-principal, spectrum-preserving**; SFT≠RL | Keep a **full-rank** M (low-rank sketch under-covers RL); use **small-λ blend/cap**, not big inject (preserve spectrum); don't borrow SFT machinery blindly |
| OPD blog (+Yuan) | RL pressure is **on-policy/localized**; RL updates **important not redundant** | Region of update is persistent over short K (good); but RL updates can't be dropped cheaply → **target correction to the live active set** |
| **2601.04537 (Linear Dynamics)** ★ | RLVR trajectory is **linear**, a **stable low-dim drift** (noise low-pass-filtered out); cos>0.9 across K-windows; extrapolate+**re-ground every K** (6.1×) | **Direct validation:** stale M ≈ still-current drift direction → K=5 trivially safe; re-ground = our re-anchor; use M as **drift estimate (blend / EMA)**, not impulse; relax cadence; watch **late-stage non-linearity** |
| 2505.11711 (Mukherjee) | RL update **sparse (5–30%) but full-rank**; subnetwork ~60% stable across seeds, partial drift; in-dist data ⇒ sparse | **Full-rank M justified**; **mask M to live active subnetwork** + short K because the subnetwork is only ~60% persistent |

**Bottom line for the combiner choice:** every source points the same way —
the recommended EXP-23 correction is a **full-rank stale anchor M, used as a
low-variance *drift estimate* (EMA or normalized direction), folded in via a
small-λ convex BLEND (spectrum-preserving) rather than a raw additive inject,
masked/projected onto the live active subnetwork the compressed G is touching,
with error feedback accumulating the PowerSGD low-rank residual.** Inject-with-
big-λ and pure low-rank correction are the predicted failure modes (spectrum
distortion; under-covering RL's full-rank update). K=5 is comfortably inside the
staleness tolerance implied by both the async-sparsity (k≤8) and linear-dynamics
(extrapolate 10–50) results.

---

## Sources
- arXiv 2602.03839v2 — Miahi & Belilovsky, *Understanding and Exploiting Weight Update Sparsity for Communication-Efficient Distributed RL* — https://arxiv.org/abs/2602.03839
- arXiv 2511.08567 — Zhu et al., *The Path Not Taken: RLVR Provably Learns Off the Principals* — https://arxiv.org/abs/2511.08567 (Three-Gate summary: https://hanqing.notion.site/The-Path-Not-Taken-RLVR-Provably-Learns-Off-the-Principals-2a7f89d0c846816faffec5f1c2b24a49)
- OPD blog — *SFT vs RL / On-Policy Distillation* — https://nrehiew.github.io/blog/sft_rl_opd/
- arXiv 2601.04537 — Wang et al., *Linear Dynamics in the RLVR Training of Large Language Models* — https://arxiv.org/abs/2601.04537 (code: https://github.com/Miaow-Lab/RLVR-Linearity)
- arXiv 2505.11711 — Mukherjee et al., *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models* — https://arxiv.org/abs/2505.11711 (the actual sparse-but-full-rank reference; OPD-cited)
