# Progress (updated 2026-06-30)

Durable record: `runs/SUMMARY.md` · `runs/FIXED_CONTROL_SURFACE.md` · the two `reports/*.html`
summaries · W&B · git. North-star + "done": `.claude/GOAL.md`. Per-experiment verdicts: `LOG.md`.
This file holds only the current operating base + the two active priorities. Closed-experiment
tick logs were pruned (recoverable from git + LOG.md); the harness appends new ticks below.

## Operating base (both priorities start here)
EMA merger `signed_ema` (alpha=0.25, beta_anc=0.50) on the fast 1K surface (response 1024,
dynamic-bsz, rollout TP=1, gpu_mem 0.55, 50 steps, val@25/50) at HIGH anchor latency
(cadence/delay_K = 20/20, the k-collapse regime), on the locked PowerSGD r=77 anchor substrate
(anchor owns Q, clean=0, paired replay, disable_custom_all_reduce=true). Reproduce:
`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`; exact values in
`runs/FIXED_CONTROL_SURFACE.md`. The baseline runs at high latency on purpose: that is where the
method collapses (Priority 1). At LOW latency (5/5) the same merger reached parity (val@50 ~0.736 vs
dense ~0.766), so parity is reachable; holding it at realistic high latency is the open problem.

## Two priorities (one knob at a time, from the base above)
1. M4: solve the k-collapse by PROJECTING THE WEIGHTS, not the gradient (Nesterov-style: predict
   theta_hat ~ theta_t, take the gradient at theta_hat). The stale anchor gradient rotates
   ~orthogonal by k~10-20 (GSM8K cos 0.51 -> 0.02@k10 -> -0.01@k20; pure rotation). EXP-42 measured
   weight-projectability directly (LOG + `reports/exp42-*.html`): in the clean run linear projection
   helps only out to ~5 steps (crossover h*=10 ticks), the trajectory is locally linear (R^2 ~0.80
   at 1 tick) but curves by K (R^2 ~0.32), and a DAMPED coefficient (alpha ~0.5, below the naive
   h/Delta) is the lever. The compressed-regime measurement must use an ADAPTIVE basis Q
   (owns_q=false); a frozen Q is a degenerate codec. Summary:
   `reports/priority-1-anchor-staleness-k-collapse.html`.
2. M6: reduce the compression train-inference mismatch (Gap A): a bounded ~0.04 tax that GRPO
   absorbs, NOT the collapse cause; the real blocker is Gap B = anchor staleness (= Priority 1).
   Lever: the truncated-IS corrector / an FP8 rollout-only probe. Summary:
   `reports/priority-2-compression-train-inference-mismatch.html`.

## Settled background (do not relitigate)
PowerSGD r=77 + a mandatory anchor that owns Q; the two-circuit structure is mandatory. Goals 1-3
met at low latency (stable / parity / ~5% gradient-comm). Merger family settled: `signed_ema`;
prior anchor-usage + beta_anc sweeps were null beyond eval noise. Do not import old anchor-gradient
claims or run labels into new plans.
[2026-06-30T15:42:58+10:00] [research-planner #43] plan written
[2026-06-30T15:43:21+10:00] [triage] dispatched 1 planner, 0 issues already planned
[2026-06-30T15:54:06+10:00] [orchestrator] tick1: EXP-43 READY_TO_RUN → dispatched experiment-runner (attach external 1xH200 i_43190371 team, launch regimeA collection cell). box SSH-probed healthy. awaiting RUNNING report → monitor next.
[2026-06-30T15:56:42+10:00] [experiment-runner #43] launched regimeA on 1 external H200 (inst 43190371, team acct) tmux=exp43a exp_name=exp42-regimeA-exp43 dph=0 max_gpu_hr=14
[2026-06-30T15:59:46+10:00] [orchestrator] tick1 cont: runner returned RUNNING (exp43a ALIVE, code_change=false @ ded0762, WandB exp42-regimeA-exp43; gates 1+4 pre-confirmed in resolved cmd). training-log-monitor dispatched (bg). teardown sweep clean — healthy box survived. awaiting monitor terminal report.
[2026-06-30T16:04:43+10:00] [orchestrator] OPERATOR STOP: user halted EXP-43 mid-run (killed tmux exp43a, GPU freed 0%, monitor stopped). Run had written only manifest.jsonl + sketch_tick_1_0.npz. Reason: the probe.weight_traj.* config saves k=4096 count-SKETCHES (lossy, non-invertible), NOT full weight matrices; operator wants ALL weight matrices saved each step for unconstrained offline analysis. Box 43190371 KEPT UP (not torn down) pending reconfig decision (precision/storage/cadence). Heartbeat refreshed to defer auto-reap.
[2026-06-30T16:36:16+10:00] [orchestrator] CODE CHANGE (operator-directed, this session): removed the count-sketch/calib weight-compression path; WeightTrajObserver now dumps FULL weight matrices once per training step (bf16, full/step_*.pt + full_manifest.jsonl). Files: capture.py/comm_eff.py/actor.yaml/transformer_impl.py + exp42_run_cell.sh + new verify_full_weight_dump.py; removed weight_proj_sweep.py + 3 sketch report builders. py_compile OK; observer + verifier CPU-smoked PASS. Plan 43 reconciled (code DONE on vast-ai-workload, new session fetch+launch).
[2026-06-30T16:39:17+10:00] [orchestrator] EXP-43 row RUNNING->ABORTED (operator stop, study redefined full-weights). Code DONE+pushed vast-ai-workload@12202b0e. Issue->status:approved. Box 43190371 KEPT UP (teardown sweep intentionally skipped per operator keep-up). New session: attach 43190371, fetch vast-ai-workload, smoke, launch exp42_run_cell.sh regimeA.
[2026-06-30T17:46:19+10:00] [orchestrator] tick1(new-session): EXP-43 ABORTED row (operator stop, study redefined to FULL-weight per-tick R2) → READY_TO_RUN on NEW external box 43197578 (team). SSH-probed healthy (H200 idle, aws-cli 1.45.37, R2_*+HF+WandB in $HOME/.verl_auth.env, /workspace/verl on vast-ai-workload, 400G 1%). experiment-runner dispatched (bg) to attach + R2 smoke + launch 80-step regimeA collection. Old box 43190371 confirmed DESTROYED via vastai (no leak). Stale handle 43190371.json removed. Teardown sweep DEFERRED to next turn (avoid runs.jsonl write-race with runner promotion; budget proven safe — one healthy box $3.66/hr under $8).
[2026-06-30T17:59:03+10:00] [experiment-runner #43] launched EXP-43 dense regime-A FULL-weight per-tick collection on external 1xH200 43197578 (team, dph=0): smoke PASSED (2 ticks uploaded+verified to R2, n_matrices=338, down_proj [1536,8960], verify --r2 PASS); real 80-step run live in tmux exp43a, exp_name=exp42-regimeA-exp43-full, R2 prefix verl-research/EXP-43/regimeA/weights/full/ (upload-then-delete)
[2026-06-30T18:05:40+10:00] [orchestrator] tick1 cont: runner returned RUNNING — smoke PASSED (tick_0/1 uploaded+verified to R2, verify --r2 PASS, n_matrices=338, down_proj [1536,8960], bf16); 80-step regimeA LIVE (tmux exp43a, WandB a51waqza, codec OFF, per_tick+r2 on). First-tick upload-then-delete CONFIRMED in production. training-log-monitor dispatched (bg). R2 WRITE TEST (user req): box+laptop BOTH upload OK to shamane-pluralis (bucket-guarded, probes cleaned up); earlier 'malformed laptop cred' was WRONG — R2_ACCESS_KEY_ID is 32-char, failure was aws needing AWS_* names not R2_* (sink maps internally, r2_sink.py:130-133). Teardown sweep ran (no-op, healthy box). GPU idles ~60-90s/upload = EXPECTED not stall.
[2026-06-30T18:22:08+10:00] [orchestrator] tick: EXP-43 RUNNING healthy at 5/80 (global_step:5), 11 R2 objects climbing, 0 NaN/crash, local staging .pt=1 (upload-then-delete OK), disk 376G free. Prior monitor died on API stream-idle-timeout (agent failure, run unaffected); heartbeat refreshed manually + monitor RE-DISPATCHED (45s cadence, 25-min cap, box-side manifest counts, upload-bound GPU-idle flagged as normal). Teardown sweep ran (no-op, heartbeat fresh, healthy box $3.66/hr). Pace ~4 min/step => ~5h to step 80 within 16 GPU-hr cap.
[2026-06-30T18:40:43+10:00] [orchestrator] tick: EXP-43 RUNNING healthy @ 11/80 (~205s/step, ETA ~3.5h). 23 ticks dumped / 22 R2-verified, local staging=0 (upload-then-delete OK), disk 376G free, 0 crash/NaN, comm_eff counters all 0 (codec OFF confirmed). Monitor returned clean HEALTHY_RETURN (robustness tweaks fixed the stream-idle death); re-dispatched. Heartbeat refreshed + sweep no-op (healthy box preserved). TEARDOWN GATE holds: no verdict.md, no analyst until done.flag+~160 R2 objects+verified.
[2026-06-30T19:00:52+10:00] [orchestrator] tick: EXP-43 healthy @ 17/80 (~200s/step, ETA ~3.4h). 36 ticks / 35 R2-verified (each ~3.09GB, local==remote bytes), staging 0-1 (upload-then-delete OK), disk 376G free, 0 crash/NaN, codec-OFF counters all 0, reward 0.668→0.753 rising, grad_norm 0.36-0.41 stable. Monitor clean HEALTHY return; re-dispatched. Heartbeat refreshed + sweep no-op. GATE holds. Manifests on box: /workspace/runs/EXP-43/regimeA/weights/{full,r2}_manifest.jsonl.
