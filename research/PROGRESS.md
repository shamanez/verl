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

## EXP-41 Monitor Run (2026-06-25T06:06:36Z to 06:25:50Z, ~20 min wall)
Cell A (cadence 5/5, lookahead DISABLED) at step 31/100 at monitor exit. Clean training: no errors, anchor isolation clean, val@25=0.6998. anchor_align_cos baseline (raw stale anchor): range [-0.059, 0.046], mean ~+0.012 over 12 fires. Score at step 28: 0.816. Cell B not started. Monitor exited TIMEOUT; run continues autonomously on box.
[2026-06-25T16:27:16+10:00] [orchestrator EXP-41] monitor#1 returned: cell A HEALTHY step 31/100, val@25=0.6998, no collapse, anchor counters 0; cellA anchor_align_cos baseline mean~+0.012 (range -0.059..+0.046). Cell B not started. Re-dispatching monitor. running=[41]
[2026-06-25T17:10:47+10:00] [orchestrator EXP-41] HANG-CHECK: NOT hanging — cell A DONE (done_A.flag), cell B at step 4/100 ~22s/it, step advanced 2->4 in 22s, GPUs cycling 79-100%, tmux alive. Watch: cell B step4 grad_norm=200 (early cold-M transient, no anchor fire yet).
[2026-06-25T17:23:30+10:00] [orchestrator EXP-41] WandB tail-drop CONFIRMED+FIXED for cell A (7tbzm9kl): online run committed only to step 99, val@100 dropped (had 25/50/75). Resume-relogged step 100 (158 keys) from authoritative train.log -> WandB now has val@100=0.7066, summary global_step=100. Reusable /workspace/runs/EXP-41/backfill_wandb.py on box. MUST backfill cell B (g6dt6bza) step 100 BEFORE teardown.

[2026-06-25T07:51:00Z] [training-log-monitor EXP-41] DONE_AGGREGATE (exit): cell A 100 steps CLEAN (val@25=0.6998, val@50=0.7255, val@75=0.7233, val@100=0.7066, 39 anchor fires, anchor_align_cos mean~+0.012 range [-0.059,+0.052]); cell B 100 steps DEGRADED (val@25=0.3616, val@50=0.4981, val@75=0.1145, val@100=0.0478, 9 anchor fires, 8 true lookahead fires, anchor_align_cos mean~+0.033 for true-lookahead fires [+0.131 peak at fire6 step60], 2 soft-collapse peaks steps 47-59 and 90-92 [max resp_len 552 not catastrophic], no NaN/OOM/crash). ALL artifacts rsynced. Recommend dispatch_analyst.
[2026-06-25T17:56:18+10:00] [orchestrator EXP-41] cells A+B DONE. WandB backfilled both (A val@100=0.7066, B val@100=0.0478). Artifacts local (verl_train_{A,B}.log + resolved_params). Box 42465843 TORN_DOWN (team), verified 0 live. Cell B: lift PRESENT (+0.021, 6/8 fires pos, peak +0.131), NO catastrophic crash, BUT val collapsed 0.36->0.50->0.11->0.048 + soft-collapse resp_len oscillations (552@s59). Dispatching analyst for verdict. analyzing=[41]
[2026-06-25T18:02:24+10:00] [analyst #41] verdict=STOP
[2026-06-25T18:05:25+10:00] [log-writer #41] logged verdict=STOP milestone=M4
[2026-06-25T19:29:04+10:00] [orchestrator EXP-42] provisioned PRIVATE 4xH200 i_42488295 ($12.39/hr) after tearing down key-injection-broken i_42485119; shipped exp/42 bundle (lookahead_strength alpha knob, commit 08bc6d96); launched probe.sh (P1 fixed-alpha=0.5 + P2 learned). running=[42]
[2026-06-25T20:02:55+10:00] [orchestrator EXP-42] both probes PASS (P1 fixed-α knob strength=0.5000 plumbed; P2 learned coeff cross_rank_max_rel_dev=0.0, no-leak 3-pt ring, no OOM). Launched α-sweep cells A25/A50/A75/L (delay_K=20, 100 steps) in tmux exp-42-210_157_233_100 on private i_42488295. running=[42]
2026-06-25T10:21:53Z EXP-42 A25 monitor: collapse_skip_cell at step 38-39 (resp_len 519/665 > 488 threshold); val@25=0.5724; lookahead_fires=1 at step 30 (align_cos 0.004→0.016 lift); A50/A75/L not yet started. Waiting for orchestrator action.
[2026-06-25T20:25:15+10:00] [orchestrator EXP-42] cell A25 (alpha=0.25, 5-ahead) COLLAPSED@step38 (resp_len 244->665, clip 0.585, score 0.56->0.09; val@25=0.5724) — EARLIER than EXP-41 alpha=1.0 (~step55). Per operator early-collapse directive: SIGKILLed A25 + ray cleanup (GPUs freed), launched launch_remaining.sh (A50/A75/L) fresh. A25 lookahead DID lift cos (0.0041->0.0157 at first true fire). running=[42] cell=A50
2026-06-25T10:43:11Z EXP-42 resumed monitor: TIMEOUT at 40min. A50 step=39 HEALTHY (no collapse), val@25=0.6459, score=0.663, resp_len=130 (stable). Second lookahead at step 40. A75+L not started. Recommend dispatch_analyst for A50 final result.
[2026-06-25T20:44:27+10:00] [orchestrator EXP-42] monitor#2 (40min cap): A50 (alpha=0.50) HEALTHY @step39, val@25=0.6459 (>A25 0.5724), resp_len stable ~130, NO collapse (survived A25's step-38 window). NON-MONOTONIC: alpha0.25 collapse@38 < alpha0.50 stable < alpha1.0 collapse@~55 => stability sweet spot ~alpha0.5. A75/L pending. Re-resuming monitor. running=[42] cell=A50
2026-06-25T11:00:01Z EXP-42 A50 monitor: collapse_skip_cell at steps 83-84 (resp_len 543/698 > 516 threshold). val@25=0.6459 val@50=0.5694 val@75=0.3124. Anchor cosine series oscillating. Recommend dispatch_analyst.
[2026-06-25T21:03:22+10:00] [orchestrator EXP-42] cell A50 (alpha=0.50) COLLAPSED@step83 (survived epoch1, degraded epoch2: val@25/50/75=0.6459/0.5694/0.3124, then resp_len explosion 211->843). MECHANISM: oscillating anchor_align_cos — extrapolated fires alternate +/- alignment (step40 -0.016, step50 -0.030, recovering). Killed+cleaned (pgrep self-match scare resolved), launched A75_L.sh fresh (A75 alpha=0.75 + L learned). Sweep so far: alpha0.25@38 < alpha1.0@55 < alpha0.50@83 all COLLAPSE. running=[42] cell=A75
2026-06-25T11:16:20Z EXP-42 A75 monitor: collapse_skip_cell val@25=0.1873 (step 25, onset step 20 negative cosine alignment). anchor_cos: step10=+0.004 step20=-0.013. Score declining 0.55→0.20. resp_len 83 (shrinking). Advancing to cell L.
[2026-06-25T21:19:54+10:00] [orchestrator EXP-42] A75 (alpha=0.75) val@25 CRASHED to 0.187 (cos negative@step20, earliest/worst); skipped. ALL fixed-alpha collapse via oscillating extrapolated-anchor cos (amplitude scales w/ alpha). Launched final LEARNED cell L (confirmed Qwen2.5-1.5B + lookahead_mode=learned + strength=1.0, no OOM, A75 residual reclaimed). L = decisive test: do adaptive coeffs dampen the oscillation? running=[42] cell=L
2026-06-25T11:39:32Z EXP-42 L monitor: collapse at steps 44-45 (resp_len 561/603 > 560 threshold). val@25=0.3965. Cosine series: +0.013/+0.001/+0.031(raw) then -0.014(extrapolated). Learned cell dampened raw-stale oscillation but NOT the extrapolated overshoot. recommendation=collapse_skip_cell.
[2026-06-25T21:42:14+10:00] [orchestrator EXP-42] cell L (LEARNED/adaptive) COLLAPSED@44 — learned coeffs dampened raw-stale oscillation (3 consec + fires) but could NOT prevent sign-flip at first true extrapolation (step40 cos -0.014). ALL 4 cells collapse. Box 42488295 TORN_DOWN (private, 0 live verified). Data captured. Starting agent-team HTML report.
[2026-06-29T11:51:41+10:00] [planner EXP-42] REFRAME (operator): EXP-42 now measures WEIGHT-projection accuracy (does θ̂ land closer to θ_now than raw-stale θ[t−K]) vs steps-ahead, in 2 regimes (plain GRPO / +activation-compression), fixed vs learned — this is upstream of and gates the gradient claim. Motivated by the prior collapses' fingerprint (extrapolated-anchor cos sign-flip = weight-space OVERSHOOT, lines 73-82). Cost-minimised per operator: ordinary training on ONE 1×H200 emits a tiny per-tick weight SKETCH (count-sketch k=4096, ~320 MB/regime); look-ahead replayed OFFLINE on the MacBook across every method×horizon. Operating point = K=10 (test at 10, NOT 20). Gradient-accuracy follow-up DEFERRED to a separate future session — no 43 plan kept (would pollute agents; deleted). Rewrote .claude/plans/42.md + runs/EXP-42/RUNBOOK.md + STATUS. Single-GPU operator-AUTHORIZED (1×H200 primary, 1×B200 only on OOM) with full permission to fit to H200. No GPU spend yet.
