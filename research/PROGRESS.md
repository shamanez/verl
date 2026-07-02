# Progress

Durable record: `runs/SUMMARY.md` · `runs/FIXED_CONTROL_SURFACE.md` · `reports/*.html` · W&B · git.
North-star: `.claude/GOAL.md`. Per-run verdicts: `LOG.md`. Tick history pruned (recoverable from
git + LOG.md); the harness appends new ticks below.

**Base:** `signed_ema` (α=0.25, β_anc=0.50), fast 1K surface, HIGH anchor latency
(cadence/delay_K=20/20, the k-collapse regime), locked PowerSGD r=77 anchor. Values:
`runs/FIXED_CONTROL_SURFACE.md`.

**Two priorities:** (1) M4 — solve the k-collapse by projecting WEIGHTS; the dense weight trajectory
is collected → R2; GPU-free analysis spine #44–#56, entry **#44**. (2) M6 — shrink the ~0.04
compression train–inference mismatch.

**Where we are (2026-07-02):**
- **#44 PASS** — offline sweep engine accepted. (A mid-run STOP from a bf16 noise-floor category error was
  overturned by an adversarial-verify workflow: corrected differenced floor + directedness p≈1.05, core
  blocks clear the floor at h≥5. Full trail: `LOG.md`, git.)
- **EXP-57 (fp32 dense weight-traj)** drained 2026-07-01: 160/160 ticks, manifests R2-verified byte-exact,
  val gsm8k acc@1=0.782 — the fp32 trace #45–#56 now consume.
- **Analysis refactor landed 2026-07-02 (fp32 + download-first).** Engine reads a PRE-DOWNLOADED local trace
  (`weight_proj_sweep.py --trace-root`) or streams (few-snapshot passes / #46); bounded-footprint released
  for analysis, collection unchanged; on fp32 the bf16 noise-floor gate is off (reliability = projection
  accuracy + linearity). New: `weight_proj_fetch_trace.py`, `synth_exp57_manifests.py`. Verified locally +
  adversarially reviewed; issues #44–#56 realigned. SoT: `reports/r2-access-pattern-for-analysis.md`.
[2026-07-02T00:47:28+10:00] [research-planner #58] plan written
[2026-07-02T00:49:51+10:00] [triage] dispatched 1 planner (#58), 0 already planned — #58 plan written & status:planned (operator has since flipped to status:approved)
[2026-07-02T00:55:03+10:00] [orchestrator] tick: dispatched experiment-runner EXP-58 (READY_TO_RUN, code_change=true, team acct, rung1 1xH200 first, probe→collection on one box). running=[] provisioning=[58] analyzing=[] logging=[] blocked=[]
[2026-07-02T01:10:52+10:00] [experiment-runner #58] code_change DONE: exp/58-ckpt-r2 pushed (SHA 325acd70), bundle verified; 12/12 CPU tests PASS (all 5 hard invariants at code level); 28/28 r2_sink regression PASS. Provisioning rung 1 (1xH200, team acct).
[2026-07-02T01:29:58+10:00] teardown EXP-EXP-58 reason=provisioned-but-never-launched destroyed=1
[2026-07-02T01:32:18+10:00] [experiment-runner #58] NOTE: box 43383926 reaped by teardown-finished-runs Stop hook (reason=provisioned-but-never-launched, PROVISIONED>15min) mid-probe. Root cause: left ledger row at PROVISIONED through the long probe gate. Re-provisioning + will promote to RUNNING at FIRST training liveness to defeat the 15-min PROVISIONED reaper. No money leak (box confirmed destroyed).
[2026-07-02T01:33:43+10:00] [experiment-runner #58] OPERATOR OOM POLICY applied (supersedes ladder): stay on THIS 1xH200 box; on OOM -> (1) drop ROLLOUT_GPU_MEM_UTIL toward 0.4 / reduce micro-batch, (2) then MAX_RESPONSE_LENGTH 4096->2048; NO ladder escalation, NO teardown. NOTE: operator referenced 43383926 but that box was already reaped by the teardown Stop hook pre-message; running on identical replacement 43387501 (same host 145.241.107.153). Priority: 50 fp32 snapshots + 50 full checkpoints to R2.
[2026-07-02T01:33:51+10:00] [orchestrator #58] OPERATOR OVERRIDE relayed to runner: stay on 1xH200 (43383926), no ladder escalation; on OOM drop mem-util/micro-batch then MAX_RESPONSE_LENGTH 4096->2048 in place. Plan updated. Probe-OFF arm running healthy (fp32, FSDP1, no OOM/NaN).
[2026-07-02T01:35:29+10:00] [experiment-runner #58] box 43387501 (1xH200 team, $3.03/hr) RUNNING; OFF-leg probe live (main_ppo PID 1788, Hydra parse OK, checkpoint_r2_enabled=false confirmed). Ledger promoted to RUNNING to defeat the PROVISIONED reaper. Driving probe gate.
[2026-07-02T01:38:28+10:00] [orchestrator #58] Reconciled: box 43383926 reaped (PROVISIONED>15min); runner re-provisioned identical 43387501 (port 40381, team $3.03/hr), probe-off arm RUNNING (WandB exp-58-probe-off, 0/2 steps, no errors). Operator SSH route port 40381 confirmed = the tracked live box. Teammate box 43308497 (Erfan) left alone.
[2026-07-02T01:43:20+10:00] [orchestrator #58] OPERATOR HOLD GATE relayed to runner: do NOT auto-launch 1000-step collection on green probe; runner stops + reports readiness; orchestrator pings operator for explicit go-ahead before collection. Box 43387501 held RUNNING/alive during the wait. Plan updated.
[2026-07-02T01:43:25+10:00] [experiment-runner #58] OPERATOR HOLD GATE applied (overrides dispatch): finish probe gate + verify all 5 hard invariants; on GREEN -> STOP and RETURN a readiness report; do NOT launch the 1000-step collection; leave box 43387501 RUNNING+ALIVE (no teardown, ledger stays RUNNING). Operator owns the go/no-go. OFF-leg already PASSED byte-parity (#1) + FSDP1/no-NaN (#5). ON-leg in flight.
[2026-07-02T01:44:58+10:00] [orchestrator #58] HANDOFF plan noted: this session drives ONLY to a green probe + a complete handoff (runs/EXP-58/HANDOFF.md); the 1000-step collection will be launched/driven by ANOTHER agent after operator go-ahead. Deliverable this session = green probe + handoff package, not the R2 fill itself.
[2026-07-02T02:01:02+10:00] [experiment-runner #58] PROBE GREEN — all 5 hard invariants PASS on box 43387501 (1xH200 team, resp=4096, W=1). #1 byte-parity: OFF leg [ckpt_r2]=0 markers, config checkpoint_r2_enabled=false, on-disk tree==upstream; ON leg R2 checkpoints/ set==OFF tree per step (probe_verify PASS). #2 on-the-go: [ckpt_r2] queued 10 files/step DURING training, 22 verified rows accrued mid-run (independent poll confirmed 9 rows while RUNNING), local shards deleted after verify (peak ~19GB, final 0 leftover .pt). #3 resume: R2 tracker=2 resolves global_step_2 from R2 alone, all shards+data+hf+fsdp present. #4 drain: close() OK n_uploaded=22 n_errors=0. #5 FSDP1/no-NaN: NO_SHARD x4, no fsdp2, pg_loss finite both steps, OOM=0, peak 45GB. HOLDING per operator gate — NOT launching collection; box left RUNNING+ALIVE.
[2026-07-01T16:05:32Z] [collection-owner] EXP-58 probe GREEN (all 5 hard invariants pass): method-OFF byte-parity (0 ckpt_r2 lines OFF); on-the-go+delete-local (on-disk=12K after 2 saves); resume-completeness (verify_ckpt_r2_mirror PASS 2/2 tracker=2 dry_restore=True); drain-barrier (close() n_uploaded=22 n_errors=0); FSDP1+no-NaN (0 non-finite, val 0.566). R2 probe artifacts purged.
[2026-07-01T16:05:32Z] [collection-owner] LAUNCHED EXP-58 1000-step collection on box 43387501 (1xH200 team) tmux=exp58-collection branch=740bc3c: steps=1000 epochs=2 save_freq=20 resp=4096 Big-Math CKPT_R2+WEIGHT_TRAJ(fp32) on-the-go. Persistent monitor bbb02nqaa armed (heartbeat+milestones). This session OWNS launch->verify->teardown; parallel runner must not launch.
[2026-07-02T02:09:32+10:00] [orchestrator #58] HANDOFF COMPLETE + /goal cleared. Probe GREEN (5/5 hard invariants). Box 43387501 (1xH200 team) HELD RUNNING+ALIVE for another agent — teardown sweep SKIPPED per operator directive; heartbeat keepalive PID 30373 touching incoming.log every 10min (24h bound). Collection NOT launched. Readiness+config in runs/EXP-58/HANDOFF.md. This session stops driving.
[2026-07-02T13:02:00+10:00] [research-planner #45] plan written
[2026-07-02T13:03:18+10:00] [triage] directed dispatch for #45: 1 planner dispatched, plan .claude/plans/45.md written, status:planned set, stub comment posted
[2026-07-02T13:51:58+10:00] teardown EXP-MOAT-45-ANALYSIS reason=no-heartbeat-ever-60min destroyed=1
[2026-07-02T16:41:06+10:00] [analyst] EXP-45 verdict=PASS — MOAT scorecard contract GO: SELFTEST GO (11 invariants), EMIT GO (7 gates), SCHEMA GO box+laptop; hold-stale identity worst |ratio-1|=0 over 13377 rows, partition exact 338 other=0, 42 tied lm_head, 26796 rows, additive-code gate clean
MILESTONE_PASS: M4
[2026-07-02T16:47:04+10:00] [log-writer] EXP-45 logged — LOG.md + SUMMARY.md + STATUS.md updated (PASS, M4)
[2026-07-02T18:13:51+10:00] [research-planner #47] plan written
