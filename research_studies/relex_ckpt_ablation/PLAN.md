# PLAN - How many exact checkpoints does the two-circuit anchor need to forecast the fast network?

> Copy-paste this whole file into a fresh Claude Code session to run the study.
> It is self-contained. It needs **no GPU** and **no verl install**. Everything
> runs on CPU with `torch + safetensors + huggingface_hub + matplotlib`.
> The final deliverable is an **HTML report with plots** (path given at the end).

---

## 0. Mission (one paragraph)

The comm-efficient GRPO trainer has two circuits: a compressed **fast** learner
(PowerSGD) and an isolated **anchor** that never optimizer-steps. The anchor is
fed the fast network's own **exact but delayed** checkpoints and must project them
FORWARD to estimate where the fast network is *now*, so it can hand back a fresh Q
basis and an M gradient-sign reference. We already ran this live and found a
**surprising result**: a dead-simple 2-checkpoint per-tensor **secant** (window
W=2) forecasts the current weights far better (probe skill **0.78**, direction
cosine **0.91**) than the paper-style rank-1 OLS over a 4-checkpoint window (W=4:
skill **0.17**, cosine **0.57**) - even though W=4's in-window fit looks pristine
(EVR 0.97, R^2 0.99). **This study explains why, and finds the best (window,
horizon) operating point**, by replaying the projector on the open-sourced RELEX
checkpoint trajectory and scoring whole-tensor forecast skill. Output: an HTML
report answering H1-H6 below.

---

## 1. The problem in detail

### 1.1 The two circuits and what the anchor projects
- **Fast circuit**: compressed PowerSGD forward/backward + optimizer (rank 77).
- **Anchor circuit**: uncompressed, never optimizer-steps, never reads immediate
  fast weights. It supplies (a) **Q** = activation-energy basis for PowerSGD, and
  (b) **M** = EMA of the clean anchor gradient, used as a sign reference:
  `G_corr = 0.25 G_fast + 0.75 |G_fast| sign(M_anchor)` (M covers the 196 decoder
  matrices).
- The anchor's weights are set by **projecting** a short window of exact delayed
  checkpoints forward to the current fire step. That projector is the object under
  study.

### 1.2 Cadence / delay / window (validate the mental model)
Config defaults (`verl/workers/config/comm_eff.py`): `cadence=20`, `delay_K=20`
optimizer ticks, `lookahead_window_snapshots=4`, `lookahead_strength=1.0`. There
are **2 optimizer ticks per global step**, so:

- The anchor fires **every 10 global steps**.
- The newest exact checkpoint it holds is **10 global steps stale** (t - 10).
- **Your mental model is correct.** When the fast net is at step 50 and the anchor
  fires: newest exact = step **40**, history = **{10, 20, 30, 40}** (that is W=4,
  gap 10). It then **projects to step 50** ("current fast", horizon = 1 gap) and can
  also project to step **60** ("twice a fast", horizon = 2 gaps), etc.
- Definitions used throughout: **gap G** = spacing between source checkpoints
  (= cadence, default 10 steps); **window W** = number of source checkpoints;
  **horizon h** = gaps ahead to predict (h=1 current-fast, h=2 twice-a-fast).

### 1.3 The surprising finding to explain (REAL data, already pulled)
Weights-and-Biases `shamanework-pl/verl_compression_research`:

| Run (id) | W | deltas | in-window EVR | in-window R^2 | **probe skill** | **dir cos** | proj RMSE | stale RMSE | MATH |
|---|---|---|---|---|---|---|---|---|---|
| **vqe9554z** W=2 secant a=1 | 2 | 1 | 1.00 | 1.00 | **0.780** | **0.906** | 6.47e-6 | 1.38e-5 | 67.89% |
| **lzl4vlcr** W=4 rank1 OLS a=1 | 4 | 3 | 0.969 | 0.991 | **0.173** | **0.571** | 8.74e-6 | 9.62e-6 | 63.61%@50 |
| kvgtcs07 dense control | - | - | - | - | - | - | - | - | 67.41% |

The paradox: **great in-window fit does not imply a good forecast**. W=4's rank-1
line fits its own history almost perfectly (R^2 0.99) yet forecasts the next
checkpoint barely better than doing nothing (skill 0.17), while the trivial W=2
secant nearly halves the error (skill 0.78).

### 1.4 Why this is a DIFFERENT regime from the RELEX paper
RELEX (arXiv 2605.21468) validated rank-1 + linear over a **dense ~50-checkpoint
prefix** extrapolating **~20x forward**, and reported "neither increasing rank nor
non-linear modeling yields further gains." **Our regime is the opposite**: very
few checkpoints (W=2..8), spaced by the cadence, projecting only **1-2 gaps
ahead**. In that short-horizon few-sample regime the recent tangent matters more
than a denoised long-run direction, which is exactly why W=2 can beat W=4. Also,
**our fast network does no dense training** to be replaced: the projection is only
a short-horizon *catch-up of an auxiliary reference*, not a training shortcut, so
the accuracy bar is "beat the stale checkpoint" (skill > 0), not "match a fully
trained model." This reframing is important: **we only ever need short horizons.**

---

## 2. The exact method under test (do not re-derive it)

Source of truth: `verl/workers/comm_eff/lookahead.py`.
- `project_rank1_tensor` (lines ~731-909): the per-tensor projector.
- `Rank1RelexProjector.project` (lines ~912-1044): applies it to every floating tensor.
- `compute_theta_hat` / `LookaheadProjector` (lines ~282-499): the `fixed_linear`
  (decoder-only, 2-point) variant.

Mechanics (mirror these exactly; a byte-identical CPU port is already provided):
1. For a window of W checkpoints `snapshots[0..W-1]` (oldest..newest) at ticks
   `t[0..W-1]`, form **cumulative deltas vs the window base**:
   `D_i = snapshots[i] - snapshots[0]`, i = 1..W-1  (W checkpoints -> W-1 deltas).
2. Rank-1 via Gram: `G = D D^T` (small, [W-1, W-1]); take top eigenpair; temporal
   coefficients `c = u1 * sigma`; spatial direction `v1 = (u1 . D)/sigma`.
3. **W = 2 (one delta) special case**: cannot fit a slope from one point, so the
   known base coordinate `c(t0)=0` is added as a second fit point. Result is the
   exact per-tensor **secant**:
   `theta_hat = latest + alpha * (h/g) * (latest - base)`  (`fit_kind = two_checkpoint_secant`).
4. **W >= 3**: OLS `c_i ~ slope*t_i + intercept` over `t[1:]` (base NOT added)
   (`fit_kind = rank1_ols`).
5. **Prediction is PINNED TO `latest`** (the newest exact checkpoint), adding only
   the incremental rank-1 motion:
   `theta_hat = latest + (alpha * slope * horizon) * v1`, `horizon = target - t[-1]`.
   This PRESERVES the newest checkpoint's off-subspace residual. **This is the key
   difference from the RELEX paper**, which rebuilds the whole delta from the base
   (`theta_hat = base + c_pred . V_r^T`) and discards that residual.

Consequence proven during setup: **at W=2 the pinned and rebuild-from-base
predictors are identical** (both reduce to the secant). They only diverge for
W >= 3, which is where the study's contrast lives.

---

## 3. What is already done (starting state for the new session)

All under the git worktree **`/Users/shamane/Documents/new-harness/verl-relex-ckpt-ablation`**
on branch **`exp/relex-ckpt-ablation`** (branched from `exp/rank1-relax`; the two
protected branches `autonomous-harness-v1` and `exp/rank1-relax` were NOT touched).

Files created in `research_studies/relex_ckpt_ablation/`:
- `harness_projector.py` - CPU port of `project_rank1_tensor` + `Rank1RelexProjector`
  + `fixed_linear` + the `relex_from_base` contrast + `stale` baseline.
  **PROVEN byte-identical** to the live harness (`maxdiff 0.00e+00` for W=2,3,4,6):
  run `python3 research_studies/relex_ckpt_ablation/harness_projector.py` from the
  worktree root (loads the leaf `lookahead.py` via importlib; no verl install).
- `download_subset.py` - fetches ONLY the needed RELEX step revisions + base to a
  temp dir (sparse; ~3.1 GB/checkpoint).
- `run_forecast_ablation.py` - the grid sweep; whole-tensor metrics -> CSV + JSON.
- `make_plots.py` - result plots + a `results.html`.
- `run_study.sh` - one-command end-to-end (deps check, equivalence proof, download,
  ablation, plots) writing everything under a **temp** `STUDY_ROOT`.
- `PLAN.md` - this file.

Verified during setup:
- The port equals the live projector (numerically identical).
- The runner + plots run end-to-end on synthetic checkpoints (CSV/JSON/PNGs
  produced), and W=2 pin-to-latest == W=2 from-base (identical), as expected.
- WandB numbers in section 1.3 are the real pulled values (runs vqe9554z, lzl4vlcr,
  kvgtcs07; also xcu56scj running, ltc94vtu legacy-invalid).

Nothing has been downloaded to disk yet (no checkpoints), per the "temp only"
instruction. The new session does the downloading (to temp) and runs the study.

---

## 4. Substrate and constraints

- **Substrate**: `relex-rlvr/RLVR-Qwen2.5-Math-1.5B` on the HF Hub. Each training
  step is its own branch `revision="step_N"`, N in [1, 500]. Base model is
  `Qwen/Qwen2.5-Math-1.5B` - the **same base the harness MATH track uses**, so the
  proxy is well matched. (RELEX also saves one checkpoint per 2 optimizer steps,
  mirroring the harness's 2-ticks-per-global-step, so RELEX step index is a fair
  time axis.)
- **No GPU**: pure CPU weight-space math. (End-to-end MATH accuracy of a
  reconstructed checkpoint WOULD need a GPU/vLLM; it is out of scope here and
  listed as a follow-up.)
- **No verl install**: standalone scripts. The optional equivalence proof loads the
  single file `lookahead.py` directly.
- **Temp only**: never write checkpoints or outputs into the repo. Use
  `STUDY_ROOT` (defaults to `$TMPDIR/relex_ckpt_study`); point it at an external
  SSD if `$TMPDIR` is small.
- **Disk**: Tier-1 core set (base + steps 10,20,...,100) is ~35 GB. Tier-2
  (consecutive steps for gap sensitivity) adds more; only run if disk allows.
- **RAM**: peak ~8 GB with the embedding tensor at W=8. On a machine with < 24 GB
  add `--skip_embedding` to `run_forecast_ablation.py`.

---

## 5. Scientific questions (what the report must answer)

- **H1 - How many checkpoints?** Forecast skill vs window W at horizon 1. Hypothesis:
  skill peaks at W=2 (or small W) and falls as W grows, because older deltas bias
  the fitted slope away from the most recent tangent.
- **H2 - Why is W=2 so good?** Because the 2-point secant uses the most recent
  tangent exactly; W>=3 OLS averages in older, slower directions. Test by comparing
  secant vs full-window OLS slope, and by using only the 2 newest checkpoints inside
  a larger window.
- **H3 - Is rank-1 right for 1 delta?** Trivially yes (a single delta spans a 1-D
  subspace, EVR=1). The real test for W>=3: does EVR/R^2 (looks great) predict
  forecast skill (may not)? Show the decoupling. Also sweep rank r in {1,2,3}: does
  higher rank help? (Paper says no; confirm in our regime.)
- **H4 - Horizon (current-fast vs twice-a-fast).** Skill vs horizon h in {1,2,3}.
  Determines whether the anchor can fire half as often (h=2) and still beat stale,
  which halves anchor communication.
- **H5 - Pin-to-latest vs rebuild-from-base.** Compare `rank1_relex` (pinned) vs
  `relex_from_base` (paper) for W>=3. Hypothesis: pinning wins because the newest
  exact checkpoint carries real off-subspace content.
- **H6 - Which tensors benefit?** Per tensor-type skill (embedding, q/k/v/o_proj,
  gate/up/down_proj, biases, norms). The live probe hinted norms win big and
  q_proj less so; confirm across all 338 tensors.

---

## 6. What to run

### 6.1 One command (recommended)
```bash
cd /Users/shamane/Documents/new-harness/verl-relex-ckpt-ablation/research_studies/relex_ckpt_ablation
# optional: point at a big scratch disk
export STUDY_ROOT="$TMPDIR/relex_ckpt_study"
bash run_study.sh                 # Tier-1 (base + steps 10..100 by 10), full grid
# TIER=2 bash run_study.sh        # also gap-sensitivity (heavier download)
```
`run_study.sh` will: check/install deps, prove the port == live projector, download
the sparse checkpoint set to `$STUDY_ROOT/checkpoints`, run the ablation to
`$STUDY_ROOT/outputs`, and render plots to `$STUDY_ROOT/plots` (open `results.html`).

### 6.2 Manual steps (if you want control)
```bash
export STUDY_ROOT="$TMPDIR/relex_ckpt_study"
# (0) prove the projector is the real thing
python3 harness_projector.py            # run from the worktree root for the live check

# (1) download only what we need, to TEMP
python3 download_subset.py --with_base \
    --steps 10,20,30,40,50,60,70,80,90,100 \
    --output_dir "$STUDY_ROOT/checkpoints"

# (2) run the ablation (add --skip_embedding on < 24 GB RAM)
python3 run_forecast_ablation.py \
    --ckpt_dir "$STUDY_ROOT/checkpoints" --out_dir "$STUDY_ROOT/outputs" \
    --windows 2,3,4,5,6,8 --horizons 1,2,3 --gap 10 \
    --ranks 1,2,3 --strengths 1.0 \
    --methods rank1_relex,relex_from_base,fixed_linear

# (3) plots
python3 make_plots.py --in_dir "$STUDY_ROOT/outputs" --out_dir "$STUDY_ROOT/plots"
```

### 6.3 The ablation grid
| Axis | Values | Meaning |
|---|---|---|
| window W | 2,3,4,5,6,8 | number of source checkpoints (H1/H2) |
| horizon h | 1,2,3 gaps | current-fast / twice-a-fast / 3x (H4) |
| gap G | 10 (Tier-1); 5,20 optional | source spacing = cadence |
| anchor | auto (each on-disk step that fits) | many instances -> mean +/- std |
| rank r | 1,2,3 | is rank-1 enough (H3) |
| strength alpha | 1.0 (default); 0.5 optional | horizon damping |
| method | rank1_relex, relex_from_base, fixed_linear (+ stale=skill 0) | H5 |

### 6.4 Metrics computed (per combo, per tensor, then aggregated)
Whole-tensor port of the live probe (`rank1_probe.projection_sample_metrics`),
computed over ALL elements of every one of the 338 floating tensors:
- **skill** = 1 - projected_SSE / stale_SSE  (> 0 means it beats the stale checkpoint)
- **direction cosine** between predicted update (proj - latest) and actual update (actual - latest)
- projected RMSE, stale RMSE
- in-window **EVR** and **R^2** from the projector (to expose the H3 paradox)
- aggregated two ways: **energy-pooled** (magnitude-weighted) and **macro**
  (mean of per-tensor), overall and per tensor-type.

---

## 7. FINAL DELIVERABLE - the HTML report

Produce a single self-contained HTML report at:
`/Users/shamane/Documents/new-harness/verl-relex-ckpt-ablation/docs/experiments/relex_ckpt_ablation_report.html`

Start from the `results.html` that `make_plots.py` emits (it embeds the six result
PNGs) and expand it into a proper report. Match the house style of the existing
report `docs/experiments/relex_rank1_report.html` (dark theme, cards, tables).
**Do not use em-dashes anywhere** (project standing rule; use hyphens/colons/parens).

The report must contain, with the actual numbers from this run:
1. TL;DR: the answer to "how many checkpoints" (winning W) and "how far ahead can we
   fire" (max horizon that still beats stale), stated as a config recommendation.
2. The six plots (from `$STUDY_ROOT/plots`, copied next to the HTML or inlined):
   - `skill_vs_W_pooled.png` / `skill_vs_W_macro.png` (H1)
   - `skill_vs_horizon.png` (H4)
   - `fit_vs_skill.png` (H3 paradox: EVR/R^2 high, skill can collapse)
   - `method_compare.png` (H5)
   - `skill_by_type.png` (H6)
   - `rank_ablation.png` (H3 rank sweep)
3. A results table: for each (method, W, horizon) the pooled skill, macro skill,
   direction cosine, fraction of tensors that beat stale, EVR, R^2 (mean +/- std
   over anchor positions).
4. The real live-run anchor (section 1.3 table) next to the study's whole-tensor
   estimates, to show they agree (the live 16-sample probe vs the whole-tensor
   truth).
5. An interpretation + DECISION section (see 8).
6. Caveats (see 9).

---

## 8. Interpretation / decision guide (write this into the report)
- If skill(W=2) >= skill(W>=3) at h=1 (expected): recommend **W=2 secant** as the
  default projector; more checkpoints add cost and readiness delay for no forecast
  gain. Explain via H2 (recent-tangent argument) and H3 (EVR/R^2 decoupling).
- If some W>=3 wins on specific tensor types (H6), consider a **mixed** policy
  (secant for fast-moving matrices, small-W OLS for slow norms).
- Report the largest horizon h with pooled skill > 0: if h=2 still beats stale, the
  anchor can fire **half as often** (cadence 40/20 ticks) for the same benefit,
  a direct communication saving. State the exact skill(h) numbers.
- rank sweep (H3): if rank>1 does not improve skill, confirm the harness's hard
  rank-1 choice.
- pin-vs-base (H5): recommend keeping the pinned-to-latest increment if it wins.

---

## 9. Caveats to state in the report (do not oversell)
- **Dense proxy vs compressed fast.** The RELEX trajectory is a DENSE (uncompressed)
  RLVR run; the live fast network trains under PowerSGD compression, so its real
  trajectory may be noisier and less rank-1-linear. This study establishes the
  projector's behavior on a clean same-base-model trajectory; it upper-bounds what
  to expect live. The live probe numbers (section 1.3) are the reality check.
- **Weight-space, not MATH.** Skill/cosine are weight-space forecast quality, not
  downstream accuracy. The MATH link is the existing live result (W=2 67.89% vs
  W=4 63.61%). A GPU follow-up could reconstruct a predicted checkpoint and eval it.
- **Step-index mapping.** We treat RELEX step index as the trajectory time axis and
  gap G as a free parameter chosen to bracket the harness's 10-global-step cadence.
  Absolute spacing is a proxy, not an exact match to optimizer ticks.
- **Sparse anchors.** Tier-1 gives a handful of anchor positions; report mean +/-
  std and do not over-interpret single-anchor wins. Run Tier-2 or more steps for
  tighter error bars if disk allows.

---

## 10. Appendix

**Direct answers to the operator's questions**
- No GPU needed: correct. No verl install needed: correct.
- "fast=50 -> anchor gets step 40, has 30/20/10": correct; that is W=4, gap 10,
  predict 50 (h=1) and 60 (h=2).
- "rank-1 for 1 delta": trivially exact (EVR=1); the open question is whether more
  deltas (W>=3) with rank-1 OLS help, and the live data says they hurt here (H3).
- "why is window 2 so good": H2 (recent tangent) + H3 (fit != forecast); this study
  quantifies it across all 338 tensors and multiple anchors/horizons.

**Config defaults** (`verl/workers/config/comm_eff.py`): cadence=20, delay_K=20
optimizer ticks (fire every 10 global steps), lookahead_window_snapshots=4,
lookahead_strength=1.0, powersgd rank 77. Tensor scope: 338 floating tensors
(196 decoder matrices + 84 q/k/v biases + 57 norms + 1 tied embedding); M gradient
correction targets the 196 decoder matrices.

**WandB run ids** (`shamanework-pl/verl_compression_research`): vqe9554z (W=2,
skill 0.78), lzl4vlcr (W=4, skill 0.17), kvgtcs07 (dense 67.41%), xcu56scj
(qboot-v2 alpha=0 running), ltc94vtu (legacy pre-fix, invalid comparator).

**Paper**: RELEX, arXiv 2605.21468. Findings: RLVR updates are ~rank-1; the rank-1
coefficient evolves ~linearly; fit on ~15% prefix, extrapolate 10-20x; higher rank
and non-linear fits do not help (validated in the many-checkpoint long-horizon
regime, which is NOT ours).

**File map** (branch `exp/relex-ckpt-ablation`, worktree
`/Users/shamane/Documents/new-harness/verl-relex-ckpt-ablation`):
- `research_studies/relex_ckpt_ablation/PLAN.md`  (this file)
- `research_studies/relex_ckpt_ablation/harness_projector.py`
- `research_studies/relex_ckpt_ablation/download_subset.py`
- `research_studies/relex_ckpt_ablation/run_forecast_ablation.py`
- `research_studies/relex_ckpt_ablation/make_plots.py`
- `research_studies/relex_ckpt_ablation/run_study.sh`
- output HTML report -> `docs/experiments/relex_ckpt_ablation_report.html`
