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
[2026-06-25T14:51:50+10:00] [research-planner #41] plan written
[2026-06-25T14:52:22+10:00] [triage] dispatched 1 planners, 0 issues already planned
[2026-06-25T15:04:59+10:00] [research-planner #41] plan revised — closed 7 code-grounded review must-fixes (alignment-lift telemetry spec, new look-ahead snapshot ring, fire-forcing probe, source-snapshot canary, corrected verification keys, off-diagonal demoted to diagnostic); all file:line claims re-verified vs source
[2026-06-25T15:35:27+10:00] [experiment-runner #41] MANUAL_REVIEW_NEEDED: EXP-41 — vast_account=team but the locked research template 'verl-research-vllm020' (hash 3b0f8b726ac3036d6c007bfa13b6d75f, id 447527) is owned by the PRIVATE account (538739) and is NOT accessible/shared to the TEAM account (558152): vastai create returns {"status_code":400,"msg":"invalid template hash or id or template not accessible by user"}. Confirmed the team account's 'search templates' (2048 public templates) contains no verl/vllm020 template. NO instance was created (0 live on BOTH accounts — no leak, no ledger row, nothing to tear down). The look-ahead patch is COMPLETE + durable: branch exp/41-lookahead-anchor @ 649594ae committed + PUSHED to origin (shamanez/verl) + bundled to runs/EXP-41/exp.bundle; run scaffold (config.yaml/launch.sh/probe.sh/commit-hotfix.sh) materialized. OPERATOR DECISION NEEDED: either (a) share/recreate the verl-research-vllm020 template on the team account (then re-dispatch EXP-41 with vast_account=team — provisioning will work), or (b) re-dispatch with vast_account=private (the template's owner account). Re-dispatch goes straight to provisioning; no code work remains.
[2026-06-25T16:06:24+10:00] [orchestrator EXP-41] probe PASSED all 10 hard invariants (probe-invariants.md); launched cells A→B in tmux exp-41-145_241_107_153 on external team box 42465843 (4×H200); dispatched training-log-monitor (bg). running=[41] analyzing=[] blocked=[]
