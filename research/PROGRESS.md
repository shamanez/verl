# Progress — focused on two priorities (2026-06-25)

Repo de-bloated to the two active fronts. Durable record:
`runs/SUMMARY.md` · `runs/FIXED_CONTROL_SURFACE.md` · the two summaries in `reports/` · W&B · git history.
North-star + "done" definition: `.claude/GOAL.md`.

## Basic setup (operating base for both priorities)

The **EMA merger** — `signed_ema` (α=0.25, β_anc=0.50) — on the **fast 1K surface**: response 1024,
dynamic-bsz, rollout TP=1, gpu_mem 0.55, ppo_max_token 24576, 50 steps, val@25/50, at HIGH anchor
latency (cadence/delay_K = **20/20**, the k-collapse regime), on the locked **PowerSGD r=77 anchor
substrate** (anchor owns `Q`, clean=0, paired replay, `disable_custom_all_reduce=true`). A bare run
reproduces it: `examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`. Exact
values live in `runs/FIXED_CONTROL_SURFACE.md` (not duplicated here).

The baseline runs at high latency on purpose — that is where the method collapses (Priority 1). At
LOW latency (5/5) the same merger reached parity (val@50 ≈ 0.736 vs dense ≈ 0.766, n=1, older 2K
surface), so parity is reachable; the open problem is holding it at realistic high latency.

## The two priorities

### 1 — Solve the k-collapse by projecting the weights (milestone M4)

The anchor gradient is taken on `delay_K`-stale weights and **rotates to orthogonal by k≈10–20**
(GSM8K cos `0.51 → 0.18@k5 → 0.02@k10 → −0.01@k20`; norm ratio ≈ 1.0 ⇒ *pure rotation*, magnitude
intact; sign → coin-flip). **Fix — extrapolate the anchor's _weights_ forward, not its gradient**
(Nesterov-style): predict the future weights θ̂≈θ_t and compute the gradient **at θ̂**, so g(θ̂)≈g(θ_t)
for free. Two upgrades over AsyncPP's fixed-linear look-ahead (arXiv:2505.01099): **(1) linear →
learned, per-block weight-projection** (captures curvature → the only route that can *surpass* dense,
beyond a diagonal Adam rescale), **(2) supervision from the fast circuit** — the true weights θ_t that
arrive at each sync are ground truth; the residual θ_t−θ̂ trains the projector online so it sharpens.

- **Next step (GPU-free) — the kill-gate:** on stored `(θ, g)` pairs, can a per-block projector predict
  θ_t, and does the gradient at the predicted weights lift cos@k5 from 0.18 to **≥ 0.40** **off-diagonal**
  (not the diagonal trap)? No → STOP, zero GPU.
  ⚠️ The EXP-38 captures that feed this gate were de-bloated — **re-import from backup before running it.**
- Summary: `reports/priority-1-anchor-staleness-k-collapse.html`.

### 2 — Reduce the compression-induced train–inference mismatch (milestone M6)

The codec's forward-pass distortion ("Gap A") makes the recomputed log-probs differ from vLLM's.
Verdict (2026-06-23): Gap A is a **bounded ~0.04 tax, constant in stable and collapsing runs, and GRPO
absorbs it** — not the cause of collapse. The real blocker is "Gap B" = anchor staleness (= Priority 1).

- **Lever:** shrink the forward distortion / switch on the **truncated-IS corrector** (available but
  unused — `old_log_prob` is recomputed common-mode, not vLLM-referenced). A planned FP8 rollout-only probe
  isolates the precision component.
- Summary: `reports/priority-2-compression-train-inference-mismatch.html`.

## Settled background (do not relitigate)

- **Substrate locked:** PowerSGD r=77 + a mandatory anchor that owns `Q`; the two-circuit structure is mandatory.
- **Goals 1–3 met at low latency:** stable / parity / savings (≈5% gradient-comm). Goal 4 (one canonical launcher) is open.
- **Merger family settled:** `signed_ema`; prior anchor-usage + β_anc sweeps were all null beyond eval noise.
