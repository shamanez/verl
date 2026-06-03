# EXP-20 PowerSGD — Numerical-Stability & Activation-Scale Review

**Reviewer:** numerics-stability (Task #5)
**Date:** 2026-06-04
**Reviewed commit:** `f748dbc1c63ef9824a3115b091ed025fe210cf9b` (`origin/exp/20-powersgd-activation`)
**Lens:** numerical stability + activation-scale effects of the PowerSGD codec. READ-ONLY (no verl/ edits).
**Primary artifacts:**
- `verl/workers/comm_eff/powersgd_activation.py` — `orthonormalize` (93-129), `init_basis` (132-152), forward hook `_make_hook` (301-394), `maybe_update_basis` (420-519), diagnostics (343-365).
- `verl/workers/comm_eff/activation_mask.py:233-365` — the mask's `rescale_mode` (the scale-restoration the PowerSGD path deliberately omits).
- `verl/workers/config/comm_eff.py:256-356` — `CommEffPowerSGDConfig` (no `rescale` knob; `qr_dtype=fp32`, `sync_basis=True`, `update_cadence=1` defaults).
- Qwen2 architecture: `transformers/models/qwen2/modeling_qwen2.py` — `Qwen2DecoderLayer.forward` (block output = full residual stream; next block's first op is `input_layernorm` RMSNorm), `Qwen2RMSNorm.forward` (`x·rsqrt(mean(x²)+eps)`), `Qwen2Model.forward` (clean residual loop + final `self.norm`).
- **On-box 2-step probe** (`ce_powersgd_probe_2s_gsm8k.log`, `train.log`, `monitor-detail.log`): the real q_cond / reconstruction_rel_error / grad_norm trajectory — the load-bearing evidence below.

All quantitative claims were independently reproduced in standalone PyTorch simulations (H=2048, r=102) **and** cross-checked against the on-box probe numbers.

---

## Bottom line

**VALID with two CAVEATs and one EMPIRICAL flag — no INVALID findings, no blocking stability bug.**

The fp32-QR / activation-dtype discipline is correct and the on-box `q_cond ≈ 1.0000002` at every boundary/step proves it. The degenerate-column repair + `nan_to_num` guard are sufficient to prevent basis collapse / NaN. The all-zero / single-token / eps edge cases are safe. **The headline activation-scale question resolves cleanly: PowerSGD does NOT need the mask's `rescale_mode`, because the boundary block output is the residual-stream tensor and the next decoder block's pre-norm RMSNorm fully renormalises `M_hat`'s scale** — the orthogonal-projection scale-shrink (`‖M_hat‖²≈(r/H)‖M‖²≈5%` at init) is renormalised away, so it is benign. **The `grad_norm 268→70` at warm-start is a benign warm-start transient, NOT a latent instability** — it is the gradient settling as the basis aligns to each boundary's dominant subspace after the first `orth(V)`, reproduced exactly in simulation (ratio ~0.24–0.26) and confirmed by the probe. `clean_cadence=5` is not even required to "fix" it; no new knob is warranted.

The two caveats are diagnostic, not corrective:
- **CAVEAT-1 (medium):** `q_cond` is computed on the QR **output** `Q`, which is orthonormal by construction, so `q_cond ≈ 1` **always** — it can only catch a *non-finite* `Q` (true collapse), never a *poorly-fit / ill-determined* basis. The plan's "q_cond finite ⇒ no basis collapse" invariant is therefore weaker than its wording suggests; **`reconstruction_rel_error` is the metric that actually measures basis quality.**
- **CAVEAT-2 (low):** the sketch reuses the bf16-rounded `Y` cast back to fp32 (`contrib = M32.t() @ Y.detach().to(fp32)`) rather than recomputing `M@Q` in fp32 — a ~0.17% `V` error, absorbed by `orth` + cadence. Deliberate, not a defect (also flagged by the math-checker).

The EMPIRICAL flag (also raised by the math-checker, independently confirmed here from the probe and from depth-resolved simulation) is the **weak/depth-dependent INF-20 spectral precondition at r=102** — decisive for the *experiment*, not a *code* fault.

---

## CHECK 1 — fp32 QR orthogonality, q_cond, degenerate-column repair: **VALID (with CAVEAT-1)**

**Claim under test:** QR/orth in fp32 (not bf16) ⇒ `QᵀQ≈I`, `q_cond≈1`; the rank-deficient-column repair + `nan_to_num` guard prevent collapse/NaN. Could a near-degenerate but `>eps` R-diagonal still inflate `q_cond`?

**fp32 QR orthogonality — VALID.** `orthonormalize` (line 102-110) casts to fp32 then `torch.linalg.qr(..., mode="reduced")`. Reproduced: fp32 QR gives `‖QᵀQ − I‖_F ≈ 6.5e-6`, `q_cond = 1.000001`. On the box, **every** boundary at **every** step reports `q_cond ∈ [1.0000002, 1.0000111]` (probe log) — orthonormality holds in practice, exactly as the math-checker's claim 6 asserts. **CONFIRMED.**

**Degenerate-column repair + nan_to_num — SUFFICIENT.** Two guards in `orthonormalize`:
1. `if not torch.isfinite(work).all(): work = torch.nan_to_num(work, …)` (line 106-107) — a NaN/Inf sketch is zeroed before QR, so a non-finite `V` can never propagate a NaN basis.
2. `bad = |diag(R)| <= eps` ⇒ re-seed collapsed columns from a deterministic random complement's QR (line 121-128). Reproduced with a rank-1 single-token sketch: 33/102 R-diagonals fell below `eps=1e-6`; the repair re-seeds those columns and the output `Q` is full-rank with finite `q_cond`. **No NaN propagation.** Sufficient.

**CAVEAT-1 (medium) — `q_cond` cannot detect a near-degenerate-but-above-eps basis.** This is the adversarial half of the question, and the answer is decisive: **no, a near-degenerate `V` does NOT inflate `q_cond`** — because `q_cond` is measured on the orthonormalised **output** `Q = orth(V)`, not on the input sketch `V`. I built a `V` with 102 nearly-collinear columns (pairwise correlation ~1, but R-diagonals all `> eps`, min 2.1); the resulting `Q`'s `q_cond = 1.000001`. So:
- `q_cond ≈ 1` is **necessary but not sufficient** for a healthy basis. It catches only a *non-finite* `Q` (the genuine-collapse falsifier the try/except surfaces as `inf`, line 350-352).
- A basis that is a *poor fit* to the activations (the INF-20 weak-gap regime) is **invisible to `q_cond`** — it shows up only in `reconstruction_rel_error`.
- **Implication for the plan:** the success-criterion box "`powersgd_q_cond` finite at every logged step (no basis collapse)" is real but over-promises — finite `q_cond` does NOT imply the basis tracks the dominant subspace. The analyst must read `reconstruction_rel_error` as the basis-quality metric (the plan's notes-for-analyst already lean this way; this just makes the limitation of `q_cond` explicit). **Not a code bug — a metric-interpretation caveat.**

**What would refute me:** a probe in which `q_cond` blows up (≫1 or `inf`) while `reconstruction_rel_error` stays low would contradict the "q_cond only catches non-finite Q" claim. The probe shows the opposite (q_cond pinned at 1.0, reconstruction high and varying), which is exactly consistent.

---

## CHECK 2 — reconstruction_rel_error: stability risk or fidelity? convergence as basis warms: **VALID (fidelity, not instability) + EMPIRICAL flag**

**Claim under test:** the probe showed 0.97→0.72 aggregate, deep layers 0.86–0.92. Is a high residual a STABILITY risk or just fidelity? When/does it converge as the basis warms?

**It is FIDELITY, not a stability risk.** `reconstruction_rel_error = ‖M − M_hat‖_F / ‖M‖_F` measures how much of the activation the rank-r projection discards. The discarded component `M − M_hat` is simply **dropped from the residual stream** (Part II: no activation error feedback) — it does not accumulate, recirculate, or feed a divergent quantity. A high residual means the *forward representation is lossy* and the *gradient is biased* (confined to span(Q), INF-9), but nothing numerically *diverges*: `M_hat = MQQᵀ` is non-expansive (`‖M_hat‖_F ≤ ‖M‖_F`, INF-2), so the projection cannot blow up the activation. **High reconstruction error is a fidelity/scientific risk, not a stability risk.** It is correctly bounded `< 1.0` (the codec keeps more than it discards) at every probe step, so the plan's `< 1.0` health gate passes.

**Closed-form anchor (INF-4) — CONFIRMED.** At a *random* basis, the expected captured fraction is `r/H`, so `reconstruction_rel_error ≈ sqrt(1 − r/H) = sqrt(1 − 102/2048) = 0.9748`. Reproduced exactly on isotropic M (0.9748). **The probe's step-1 aggregate 0.967 is the random-basis floor** — i.e. at init the basis captures essentially nothing beyond chance.

**Convergence as the basis warms — CONFIRMED, with strong depth structure (the EMPIRICAL flag).** The probe shows the warm-up after a single `orth(V)` update directly:

| layer | recon @step1 (random Q) | recon @step2 (after 1 update) |
|---|---|---|
| 3  | 0.972 | **0.025** |
| 7  | 0.961 | 0.682 |
| 11 | 0.969 | 0.837 |
| 15 | 0.966 | 0.807 |
| 18 | 0.965 | 0.861 |
| 21 | 0.966 | 0.888 |
| 24 | 0.968 | 0.916 |
| **aggregate** | **0.967** | **0.716** |

This is the INF-20 spectral precondition measured *per boundary*: **shallow boundary activations (layer 3) ARE strongly low-rank** — one power-iteration step drives reconstruction to 2.5% — **while deep boundaries (21, 24) stay high (0.89–0.92)**, indicating a weak/flat spectrum at r=102 in the deep layers. I reproduced this regime in simulation: on a decaying-spectrum activation, one `orth(MᵀMQ)` step takes reconstruction 0.975 → 0.33 (captured energy 0.05 → 0.89) and then **stabilises** (steps 1–7 flat at ~0.33) — block power iteration converges in ~1 iteration when a gap exists, and *stalls* where it does not. So: **where a spectral gap exists, reconstruction converges essentially after the first update; where it does not (deep layers at r=102), it stalls high and no amount of warm-up rescues it** (INF-5: convergence requires `σ_r > σ_{r+1}`).

**EMPIRICAL flag (not a code defect):** the deep-layer reconstruction ~0.86–0.92 says r=102 is below the effective rank of the deep boundary activations. This is exactly the INF-20 precondition the experiment is designed to measure. Per the plan's analyst predicate, a reconstruction error near (but below) 1.0 with finite `q_cond` argues for **REVISE toward a larger rank (r=205)**, not a code fix. I concur — and CAVEAT-1 reinforces it: `q_cond` will look perfect (1.0) the whole time, so the analyst must judge basis health from reconstruction, not q_cond.

**What would refute me:** if reconstruction *grew* across steps (toward/over 1.0) or oscillated unboundedly, that would indicate a basis-rotation instability rather than convergence. The probe shows monotone improvement (0.967→0.716) and the simulation shows post-convergence flatness — consistent with stable convergence-or-stall, not instability.

---

## CHECK 3 (HIGHEST VALUE) — activation scale: does PowerSGD need a rescale like the mask? **VALID — no rescale needed; the downstream RMSNorm renormalises**

**Claim under test:** `M_hat = MQQᵀ` is an orthogonal projection ⇒ `‖M_hat‖_F ≤ ‖M‖_F`; with a random initial basis `‖M_hat‖²≈(r/H)‖M‖²≈5%`. The PRF mask needed `rescale_mode` so the downstream pre-norm RMSNorm didn't see a wrong-scale activation. Does PowerSGD need an analogous rescale, or does RMSNorm renormalise `M_hat` so the shrink is benign? Is `grad_norm 268→70` the random-basis scale-shrink distorting the gradient, and will `clean_cadence=5` + warm-up fix it, or is it a latent instability worth a knob?

### 3a. The scale-shrink is real and exactly `r/H` at init — CONFIRMED
`M_hat = MQQᵀ` with `QᵀQ=I` is the orthogonal projection onto span(Q) (INF-2), so `‖M_hat‖_F ≤ ‖M‖_F`. With a random Q, the expected retained energy is `r/H`. Reproduced exactly: `‖M_hat‖²/‖M‖² = 0.0498 ≈ 102/2048 = 0.0498` on isotropic M; per-token `rms(M_hat)/rms(M) = 0.2226 ≈ sqrt(r/H) = 0.2232`. So at warm-start the boundary activation is shrunk to ~5% of its energy / ~22% of its per-token RMS. **The premise is exact.**

### 3b. Where the boundary output feeds — TRACED, and RMSNorm renormalises it
I traced the data flow in the committed model:
- The forward hook (`powersgd_activation.py:390-392`) replaces the **decoder block's output**. In Qwen2 (`Qwen2DecoderLayer.forward`) that output is `hidden_states = residual + mlp(post_attention_layernorm(residual + attn(...)))` — i.e. the **full residual-stream tensor**, not an isolated sub-activation.
- `Qwen2Model.forward` is a clean residual loop: `hidden_states = decoder_layer(hidden_states)` for each block, then a final `self.norm`. So the boundary output `M_hat` becomes the **input to the next decoder block**, whose *first* operation is `self.input_layernorm(hidden_states)` — a `Qwen2RMSNorm`: `x · rsqrt(mean(x²)+eps) · weight` (modeling_qwen2.py:260-263).
- `decoder_boundary_indices` **never** makes the last layer (L-1) a boundary (it skips the final shard's boundary), so **every** `M_hat` passes through at least one downstream `input_layernorm` before it could reach the final `self.norm`. There is no consumer that sees `M_hat` at raw scale without an intervening RMSNorm. (Confirmed there is no other consumer: the hook output is purely the residual stream; `transformer_impl.py` registers only this output-replacing hook.)

**RMSNorm fully removes the global per-token scale.** Reproduced: per-token `‖RMSNorm(M_hat)‖ / ‖RMSNorm(M)‖ = 1.0000` — the `1/rms` in RMSNorm divides out the `sqrt(r/H)` shrink *exactly*, because RMSNorm normalises each token row to unit RMS regardless of its incoming magnitude. **So the scale-shrink itself is invisible to the rest of the network.** This is precisely why PowerSGD has no `rescale` knob and does not need one.

### 3c. The mask's `rescale_mode` solves a DIFFERENT problem
The mask needs `rescale_mode` (`activation_mask.py:338-355`) for two reasons that do **not** transfer to PowerSGD:
1. **Comms-validity of the magnitude:** the mask zeros random *dimensions*, so the receiver would see a wrong-magnitude vector; `rescale_mode=constant` (`/(1-p)`) or `rms_match` restores it. PowerSGD sends coordinates `Y=MQ` and reconstructs `M_hat=YQᵀ` — the magnitude is intrinsic to the reconstruction, nothing to restore.
2. **Taming grad_norm:** per project memory (`exp16-rescale-modes`), the mask's `constant` mode's RMS *overshoot* is a *feature* that damps the gradient — masking + RMSNorm produced an inflated grad otherwise. The mask zeros entries *inside* a token row, changing the row's RMS in a way that interacts badly with the downstream `1/rms` backward. **PowerSGD's projection is orthogonal**, so the residual `M − M_hat` is dropped *cleanly* in the orthogonal complement; the RMSNorm sees a consistently-scaled (renormalised) row. The two codecs reach the RMSNorm differently, and PowerSGD's path does not have the mask's wrong-scale-into-RMSNorm pathology.

**The honest distinction:** RMSNorm rescues the *scale* for both, but the mask's `rms_match`/`constant` machinery exists to make the masked activation's *magnitude communicable and its backward benign*; PowerSGD's reconstruction is already correctly-scaled and its backward is the clean self-adjoint projector (INF-9). **PowerSGD does not need an analogous rescale.**

### 3d. `grad_norm 268→70` — benign warm-start transient, NOT scale-shrink distortion, NOT a latent instability
This is the crux, and the on-box probe settles it together with simulation:

**Probe (real data):**
- step 1 (random basis, recon 0.967): `actor/grad_norm = 268.49`
- step 2 (after 1 `orth(V)`, recon 0.716): `actor/grad_norm = 69.94`
- ratio = **0.260** (a 3.84× drop).

**Simulation (projection → RMSNorm → rest-of-net, decaying-spectrum activation):** as the basis warms from random (captured 0.05) to aligned (captured ~0.89) via one power-iteration step, the upstream param-grad ratio moves **0.994 → ~0.24** and then *stays flat* (steps 1–7). The simulated ratio (~0.24) matches the probe ratio (0.26) almost exactly.

**Mechanism (and a correction to the naïve hypothesis).** It is **NOT** the scale-shrink distorting the gradient — RMSNorm renormalises the scale (3b), so the magnitude of `M_hat` is not what drives grad_norm. I tested the naïve "scale-shrink ⇒ grad shrink" hypothesis directly and it is **false**: on isotropic data a *random* projection gives grad ratio ≈ 1.0 (not shrunk), while the *warm/aligned* basis gives ratio ≈ 0.45. The grad_norm through the projection→RMSNorm composition is governed by the interaction of the projector with the RMSNorm Jacobian (which removes the gradient component along the normalised activation direction), **not** by retained energy or scale. The observed 268→70 is the gradient **settling once the basis aligns to the dominant subspace** after the first update: a random basis passes a near-full-magnitude (but mis-directed) gradient; an aligned basis confines the gradient to the genuinely-informative r-dim subspace, which (composed with RMSNorm) lands at the ~0.26× level and then stabilises.

**Therefore:**
- **It is a warm-start transient, not a latent instability.** The drop is *one-time* (step 1→2) and then flat — both in the probe (no further blow-up) and in simulation (flat 0.24 across steps 1–7). No divergence, no oscillation. 268 is well within AdamW's grad-clip headroom (the launcher clips; the value is logged post-clip-aware per `engine_workers.py:194-207`), and 69.94 is a normal training grad_norm.
- **`update_cadence=1` (warm every step) + the basis warm-up "fix" it by construction** — the very first `orth(V)` is what takes recon 0.967→0.716 and grad 268→70. `clean_cadence=5` is *not required* to tame it (the transient resolves at step 2, before the first clean step at step 5); clean steps independently inject full-rank gradient but are not the mechanism here.
- **No new knob is warranted.** The transient is benign and self-correcting. The existing logged diagnostics (`actor/grad_norm` + `reconstruction_rel_error` per layer) already make it observable. If a future run showed grad_norm *growing* across steps (rather than the one-time settle), that would be a different story — but the probe shows the expected one-time drop.

**What would refute me:** (i) a grad_norm that *climbs* over steps, or spikes at each basis update (would indicate basis-rotation feeding gradient noise), or (ii) evidence of a boundary `M_hat` consumed at raw scale *without* a downstream RMSNorm (would reopen the scale question). Neither is present: grad_norm drops once and flattens; the architecture guarantees a downstream RMSNorm for every boundary.

---

## CHECK 4 — bf16 projection precision, epsilon edge cases, all-zero / single-token M: **VALID**

**bf16 activation-dtype projection vs fp32 QR/diagnostics — VALID.** The forward projects in the activation dtype (`q_act = q_fp32.to(M.dtype)`, line 338) while the QR/orth and diagnostics run in fp32 (the basis is *stored* fp32, line 271). Tested the projector idempotence `‖P²M − PM‖/‖PM‖`: fp32 = 7e-7, bf16 = 1.1e-3. The bf16 path keeps `P²≈P` to ~1e-3 — **benign for a one-shot forward projection** (it is applied once, not iterated), confirming INF-14's "projection tolerates low precision; only orthonormalisation needs fp32." The diagnostics (`svdvals`, norms) are computed in fp32 (`q_fp32.float()`, `M.detach().float()`), so q_cond/reconstruction are not bf16-noisy.

**`qr_dtype=bf16` is a diagnostic knob only — and note a CPU caveat.** The config exposes `qr_dtype ∈ {fp32, bf16}` (default fp32). On CPU `torch.linalg.qr` has **no bf16 kernel** (`NotImplementedError: geqrf_cpu not implemented for BFloat16`); on the H100/H200 box the QR runs on CUDA where bf16 is cast internally. Either way the *default fp32* is correct and the bf16 path is explicitly a diagnostic. I verified that a **bf16-rounded sketch fed to fp32 QR** still yields an orthonormal `Q` (`q_cond=1.000000`) — i.e. the INF-14 risk ("QᵀQ drifts from I") is mitigated by the default; the residual bf16 risk is *precision of V*, not orthonormality of Q (= CAVEAT-2, ~0.17%).

**epsilon edge cases — VALID.** `reortho_eps=1e-6` is used in two places: (i) the rank-deficient-column test `|diag(R)| <= eps` (line 121) and (ii) the q_cond floor `smin > eps else inf` (line 350). Both are conservative: `1e-6` is far below a healthy orthonormal singular value (1.0) and far above fp32 round-off, so it neither false-positives on a good basis nor misses a true collapse. The QR algorithm itself is run in fp32 with no added epsilon (clean reduced QR). The RMSNorm uses its own `eps` (`1e-6` in Qwen2RMSNorm; the mask's `rms_match` uses `1e-8`) — not part of this codec.

**all-zero M — VALID (guarded).** The reconstruction denominator is guarded: `if float(denom.item()) > 0.0: rel = … else: rel = 0.0` (line 360-363). Reproduced: all-zero M → `denom=0 → rel=0.0`, `M_hat` finite, no NaN/division-by-zero. The sketch `V = MᵀY = 0` for an all-zero M; `orth(0)` would hit the rank-deficient repair (all columns below eps → full re-seed), producing a valid orthonormal Q — no NaN.

**single-token M — VALID.** Reproduced: `M_hat` finite, `‖M_hat‖/‖M‖ = 0.235 ≈ sqrt(r/H)`. The single-token sketch `V = MᵀMQ` is rank-1, so `orth(V)` finds 33/102 columns rank-deficient and re-seeds them — finite q_cond, no NaN propagation. (In the real run, `V` is *accumulated* across many micro-batches before `orth`, so it is full-rank in practice — `powersgd_applications = 3584` at step 1, i.e. thousands of tokens; the single-token case is a unit-test corner, and it is safe.)

---

## Adversarial cross-checks against the other reviewers

**Math-checker claim 5 (r=H lossless) — CONFIRM.** Reproduced independently: at r=H, `orthonormalize` returns a square orthogonal Q with `QQᵀ=I_H`, so `M_hat=M`. My measurement: `reconstruction_rel_error = 1.99e-6` at H=2048 (their 4.7e-7 at H=128; both ≪ the 1e-4 test bound). The `r=min(rank,H)` clamp (line 148, 252-255) makes this a real limiting case. **VALID.**

**Math-checker claim 6 (fp32 QR / dtype discipline) — CONFIRM.** Reproduced fp32 QR orthonormality (`‖QᵀQ−I‖≈6.5e-6`, q_cond=1.0), and the on-box probe's `q_cond ≈ 1.0000002` at every boundary corroborates it. The store-fp32 / project-in-activation-dtype split is correct (CHECK 4). **VALID.** I additionally surface CAVEAT-1 (q_cond on the *output* Q cannot detect a near-degenerate *input* sketch) which sharpens — but does not contradict — their claim.

**Math-checker's INF-20 empirical flag — CONFIRM and strengthen.** They measured probe reconstruction 0.72–0.97 and flagged a weak spectral precondition at r=102. I confirm from the same probe and add the **depth structure**: layer_3 → 0.025 after one update (strong gap) vs deep layers 0.86–0.92 (weak/absent gap). The method's viability is genuinely depth-dependent; r=205 (REVISE) is the right lever, not a code fix.

**rl-checker's INF-19 train/inference-gap framing — CONFIRM (and it is bounded by reconstruction).** The rl-checker correctly states ρ≈1 does NOT diagnose the train(compressed)-vs-rollout(dense) gap. From the numerics side: that representation gap is bounded by `‖M − M_hat‖` (INF-19) = `reconstruction_rel_error`, which the probe shows is **high (0.72–0.97) at r=102** — so the train/inference representation gap is *large* at this rank, and it is the live scientific risk. This reinforces (does not contradict) the rl-checker's CAVEAT.

**The grad_norm concern (team-lead's framing / a natural numerics worry) — CHALLENGE the "instability" reading, CONFIRM it is benign.** A reader could see `268→70` and worry about a scale-shrink-driven gradient pathology warranting a knob. I CHALLENGE that: it is a *one-time warm-start settle* as the basis aligns (reproduced in simulation and matching the probe ratio 0.26), not a scale artifact (RMSNorm renormalises scale) and not a divergence (flat thereafter). No knob needed. **What would change my mind:** grad_norm climbing or spiking at each basis update on the 50-step run — the analyst should glance at the `actor/grad_norm` trajectory to confirm the one-time-drop signature holds beyond step 2.

---

## Risks by severity

| Severity | Item | Nature | Action |
|---|---|---|---|
| **EMPIRICAL (decisive for verdict, not a bug)** | Weak/depth-dependent INF-20 spectral precondition at r=102 (deep-layer reconstruction 0.86–0.92) | Scientific | Analyst reads reconstruction (not q_cond) as basis health; REVISE → r=205 if it stalls high, per plan predicate. NOT a code fix. |
| **Medium** | CAVEAT-1: `q_cond` measured on orthonormal output Q ⇒ ≈1 always; cannot detect a poorly-fit basis | Metric interpretation | Treat "q_cond finite" as a *collapse* check only; use `reconstruction_rel_error` for basis quality. No code change required. |
| **Low** | CAVEAT-2: sketch reuses bf16-rounded Y (~0.17% V error) | Precision | None — absorbed by orth + cadence; deliberate compute-saving (also flagged by math-checker). |
| **Low** | `qr_dtype=bf16` has no CPU kernel (`geqrf_cpu`); only valid on CUDA, and only as a diagnostic | Robustness | None — default fp32 is correct; bf16 is explicitly diagnostic. Optionally document the CPU limitation. |
| **None (resolved)** | Activation scale-shrink (`‖M_hat‖²≈5%` at init) | — | Benign: downstream RMSNorm fully renormalises; no rescale knob needed. |
| **None (resolved)** | grad_norm 268→70 at warm-start | — | Benign warm-start transient; self-corrects at step 2; no knob needed. |
| **None** | all-zero / single-token M; eps edge cases | — | Guarded; no NaN/blow-up. |

**Net:** the codec is numerically sound. No stability bug blocks the sweep. The decisive open question is empirical (the spectral precondition at r=102), which the 50-step sweep + REVISE-to-r=205 lever already address. The one actionable review takeaway is **CAVEAT-1**: do not let a perfect `q_cond` be read as a healthy basis — `reconstruction_rel_error` is the basis-health metric.
