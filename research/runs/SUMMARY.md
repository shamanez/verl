# Research Runs Summary

Durable record (full run dirs de-bloated; provenance = this file + W&B + git
history + merged code). North-star + "done": `../.claude/GOAL.md`. The two
active fronts: `../reports/priority-1-anchor-staleness-k-collapse.html` and
`../reports/priority-2-compression-train-inference-mismatch.html`.

## Current baseline (the problem state)

The baseline every future test compares against is
`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`:

- **merger** — `signed_ema` (α=0.25, β_anc=0.50)
- **substrate** — locked PowerSGD r=77 anchor circuit (anchor owns `Q`, clean=0,
  paired replay, `disable_custom_all_reduce`)
- **surface** — resp 1024, dynamic-bsz, rollout TP=1, gpu_mem 0.55, 50 steps, val@25/50, diagnostics off
- **anchor latency** — `cadence`/`delay_K` = **20/20** — the **k-collapse regime**

The baseline deliberately runs at high latency, where the method **fails** (see
below). That failure is the problem Priority 1 targets. Exact values:
`FIXED_CONTROL_SURFACE.md` (not duplicated here).

## Settled background (locked — do not relitigate)

- **Substrate locked** — PowerSGD r=77 on the mandatory anchor circuit reaches
  dense parity at ~5% gradient comm. The anchor is mandatory and is the only
  thing that updates `Q`; the two-circuit structure is mandatory.
- **Stable/parity holds only at LOW anchor latency.** At `cadence`/`delay_K` = 5/5
  the comm-eff `signed_ema` run reached **val@50 ≈ 0.736** vs a dense control
  **≈ 0.766** on the older 2K surface (n=1 each; rollout nondeterminism ≈ ±0.024/draw)
  — i.e. parity-band at ~5% gradient comm. **Goals 1–3 (stable / parity / savings)
  are met at low latency; Goal 4 (one canonical surpass launcher) is open.**
- **Reference floors** — no-merger PowerSGD ≈ 0.63; dense full-gradient band ≈ 0.75–0.78.

## The k-collapse finding (why the baseline sits at 20/20)

Anchor latency is the failure knob. At **20/20** the method breaks; at 5/5 it is
stable/near-dense. The stale anchor gradient rotates ~orthogonal to the live
gradient by k≈10–20 (cos 0.51→0.18@k5→~0@k10→−0.01@k20, norm preserved ⇒ pure
rotation). Two collapse symptoms, one cause (off-policy staleness `K>τ`):

| anchor latency | merger | outcome |
|---|---|---|
| 5/5 | signed_ema | STABLE, near-dense |
| 20/20 | signed_ema | terminal collapse ~step 61 (entropy collapse + length explosion) |
| 20/20 | (additive mergers) | stalls (grad_norm grows, sub-baseline plateau) |
| dense control | — | sails through to ≈0.78 ⇒ compression-specific, not an epoch effect |

Full argument: `../reports/priority-1-anchor-staleness-k-collapse.html`.

## EXP-42 weight-projection accuracy (M4 measurement, 2026-06-29)

Measures the primitive the look-ahead anchor rests on: does the projected weight
`theta_hat = theta_stale + alpha (theta_stale - theta_old)` land closer to the
current weight than the raw-stale weight, in weight space, versus horizon. One
operator 1xH200, single-GPU (operator-authorised), two regimes, per-tick
count-sketch of the 196 decoder matrices replayed offline on the MacBook.

- Regime A (plain GRPO, val@80=0.7695): at the operating horizon h = K = 10
  (alpha = 1) median `weight_proj_ratio = 0.972 < 1` (projection helps),
  `dir_cos = 0.549`. Crossover `h* = 10`.
- Regime B (PowerSGD r=77 codec only, val@80=0.0788, collapsed = allowed data):
  at h = 10 ratio `= 1.083 > 1` (no help). Crossover `h* = 5`. Codec-active
  confirmed on 1 GPU (powersgd_applications 19838 vs 0; recon rel-error ~0.97;
  the in-graph projection fires without PP or DP).
- `dir_cos` stays positive at every horizon in both regimes, so the overshoot
  past `h*` is a MAGNITUDE effect (alpha steps past `theta_now` along an aligned
  direction), not a weight-space sign flip. This refines the prior-collapse
  reading: the sign flip seen in the prior extrapolated-anchor-cosine runs is not
  a weight-space direction reversal at h up to 20.
- `learned_linear` (scalar-mean residual) is inert versus `fixed_linear` at the
  operating point.
- Consequence for the look-ahead method: weight projection beats doing nothing
  only at LOW staleness in the clean regime (h up to K=10), and compression
  halves that window (h up to 5). A gradient-accuracy follow-up is gated to the
  clean regime at fixed_linear, h up to 10.
- Deliverables (run dir de-bloated 2026-06-30): HTML reports at
  `reports/exp42-weight-projection-accuracy.html` + `reports/exp42-dense-weight-behavior.html` +
  `reports/exp42-dense-deep-analysis.html` + `reports/exp42-prior-gradient-probe.html`; verdict at
  `.claude/plans/42-verdict.md`; analysis tooling at `research/scripts/{weight_proj_sweep,build_report,build_dense_report,build_dense_report_v2}.py`.
  Code: `exp/42-weight-accuracy` @ 531dd5e9. The widened completeness extension (all matrices incl.
  embeddings, RMSNorm gains, biases) with ADAPTIVE Q (owns_q=false) is built + pushed and DEFERRED to
  a fresh session via `.claude/plans/42-corrected-rerun-prompt.md` (scaffold
  `research/scripts/exp42_{run_cell,drive_all}.sh`).

### Dense-run weight-behavior v2 (deeper GPU-free follow-up)

`reports/exp42-dense-deep-analysis.html` (builder `research/scripts/build_dense_report_v2.py`) adds
five studies on the regime-A dense decoder sketch (196 matrices, 160 ticks, k=4096; rel
std ~1.6%). One-line read: the dense GRPO trajectory is globally near-linear and
low-rank, but the look-ahead's two-point slope overshoots, so a damped coefficient
is the lever.

- **Low-rank (RLVR claim, temporal sense) SUPPORTED.** Per-tick update subspace
  participation ratio ~7.6 of a 159 ceiling (~26 components for 90% energy);
  cumulative displacement is near rank-1 (PR ~1.2, top direction holds ~69% of the
  centered energy). A like-for-like global straight-line fit gives R^2 ~0.85
  (through-origin), reconciling the prior "local R^2 decays to 0.32" with the cited
  paper's global ~0.9: one slow drift plus per-step noise, local metric sees the
  noise, global sees the drift. NOT computable, not claimed: matrix-native (LoRA)
  rank of a single weight matrix (flatten+sketch destroys it); embeddings / norms /
  biases (not collected).
- **Per-matrix crossover is tight:** h* 9 to 14 ticks (median 11), ratio@10 in
  [0.956, 0.985]. Attention v_proj / o_proj (mid-to-late layers) project furthest;
  MLP and k/q_proj least. Projectability is decoder-wide.
- **A better-than-naive coefficient exists (actionable).** The naive alpha = h/Delta
  overshoots. A damped alpha (~0.53 at h=10, ~0.74 at h=20) cuts the median ratio
  0.972 to 0.836 at h=10 and keeps h=20 below 1 (1.173 to 0.897). The optimal alpha
  is stable (~0.5 to 0.75), so a fixed damped coefficient near 0.5 is a deployable
  change to the look-ahead rule (validate in the compressed regime, h*=5 there). The
  two-point slope over-states the persistent drift because it captures per-step
  noise; damping corrects the over-step.
- **More-linear means more-projectable** (Spearman +0.45 vs h*, -0.51 vs ratio@10;
  significant at n=196). Real but loose.
- **Learned residual is inert** on the dense run (max |resid| 2.7e-9, ratio change
  <= 6.5e-5, below the 1.6% sketch floor): a scalar mean-shift barely moves a
  high-dimensional displacement and the per-matrix mean drifts smoothly.

## Bottom line

The baseline is a comm-eff trainer that is **stable at low anchor latency but
collapses at the realistic high latency** (single slow anchor serving a fast
swarm). The two open priorities both target the anchor ↔ fast-circuit coupling:
**(1)** fix the k-collapse by extrapolating the anchor's weights forward
(Priority 1), and **(2)** reduce the compression-induced train–inference
mismatch (Priority 2). Both next steps are **GPU-free offline kill-gates** — see
the two reports. Stay on EMA-family mergers; everything else is locked
(`FIXED_CONTROL_SURFACE.md`).
