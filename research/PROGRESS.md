# Progress

Historical tick-by-tick orchestration output has been de-bloated. The next phase
starts from the compact handoff:

- `research/runs/SUMMARY.md`
- `research/.claude/plans/SUMMARY.md`
- `research/runs/FIXED_CONTROL_SURFACE.md`

Current working state:

- The **accelerated comm-eff loop** is the locked base: `signed_ema(α=0.25, β_anc=0.50)`,
  accel surface @ gpu_mem 0.55, `diagnostics=false`, PowerSGD r=77 anchor substrate.
  Launcher: `examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`.
- ~25 min train / ~28 min wall per 50-step run. Reference val@50 (n=1): comm-eff 0.7362
  (EXP-36B) vs dense 0.7657 (EXP-36C).
- The vLLM speed knobs (gpu_mem 0.75 / chunked_prefill / forward_prefetch) were tried and
  dropped — no speedup, added noise.
- Prior bulky run directories and execution plans are removed.
[2026-06-18T21:33:50+10:00] [research-planner #37] plan written
[2026-06-18T21:34:12+10:00] [triage] dispatched 1 planner (#37), skipped #35 (status:done)
[2026-06-18T21:43:33+10:00] [orchestrator #37] EXP-37 launched DIRECTLY on operator-provisioned team box i_41475643 (4×H200, $12.17/hr); cadence/delay 20/20 verified (trailing Hydra args win over bare-export 5); 100 steps, signed_ema(0.25,0.50). tmux exp-37-104_202_252_41. Dispatching background monitor.
EXP-37 exp-37-cad20-delay20 DONE 2026-06-18T12:40Z 100steps val@25=0.592 val@50=0.648 val@75=0.490 val@100=0.444 anchor_fires=10 stability=STOP(val@50<0.6862,late_len_excursion_steps93-100) wandb=fxo8chsv
[2026-06-18T22:48:22+10:00] [orchestrator] EXP-37 COMPLETE (100 steps, val@50 0.6482 < gate 0.6862, back-half length-escalation s93-100 → STOP-class; anchor_backwards=10 latency realized). Dispatched: EXP-37 analyst + 2-member delay-failure analysis team (staleness-theorist, cadence-analyst); report-lead pending.
[2026-06-18T22:50:32+10:00] [analyst #37] verdict=STOP
[2026-06-18T22:54:44+10:00] [orchestrator] Team box i_41475643 torn down 12:46Z by the vast-teardown SKILL invoked from a CONCURRENT /loop session 2d9239b8 (PID 68399), which reaped the finished EXP-37 box ~2min after this session had reused it. Lesson: box reuse across concurrent /loop sessions is a teardown race (memory: box-reuse-race-across-loop-sessions). Ledger: 41475643 rows -> TORN_DOWN.
[2026-06-18T23:01:39+10:00] [orchestrator] OPERATOR: switch off all H200s. VERIFIED 0 live instances on BOTH team + private accounts (i_41475643 already destroyed by concurrent session). No teardown needed; no money leaking. Ledger clean (no RUNNING/PROVISIONED rows). Delivered delay-failure HTML report.
[2026-06-20T00:49:38+10:00] [orchestrator/operator] EXP-37B LAUNCHED on team box 41680420 (4xH200, 84.8.106.109:40206). Single cell exp-37b-cad5-delay5-100step: signed_ema(0.25,0.50) accel base, anchor cadence=5/delay_K=5 (NOT 20/20), 100 steps, test_freq=25, project verl_compression_research_accel_rebaseline. Banner verified 5/5 + 100 steps + epochs=2. Monitor dispatched (bg).
[2026-06-20T01:39:19+10:00] [operator-directive] DO NOT teardown box 41680420 after EXP-37B. CHAIN EXP-37C (same specs, beta_anc=0.0) on the SAME box. Sequence: EXP-37B done -> backfill tail -> flip EXP-37B COMPLETE + register EXP-37C RUNNING (same box, fresh heartbeat) -> launch EXP-37C (trailing hydra spectral.beta_anc=0.0; verify via resolved cmd NOT banner) -> dispatch EXP-37C monitor -> EXP-37B analyst (parallel). Teardown ONLY after EXP-37C done, with TEAM account. EXP-37C staged+md5-verified on box.
[2026-06-20T01:47:07+10:00] [operator-directive] EXP-37D ADDED to the chain: DENSE (comm-eff OFF) 100 steps on the accel surface, same box 41680420. FULL CHAIN now: EXP-37B (running, signed_ema beta0.50) -> EXP-37C (signed_ema beta_anc=0.0) -> EXP-37D (dense, comm_eff.enabled=false). All back-to-back, reuse-on-COMPLETE, NO teardown between. Teardown ONLY after EXP-37D done, TEAM account. 37D launch=accel base + trailing comm_eff.enabled=false (banner LIES 'master:true'; verify enabled=false via resolved cmd+WandB). 37D apples-to-apples w/ EXP-36C (yabc92t5, dense@50 0.7657, same surface ppo_max_token 24576/TP1/mem0.55/resp2048). 37D staged+md5-verified on box. Purpose: does GSM8K epoch-2 revisit (~step58->100) wobble dense too, isolating dataset-revisit from compressed-merger instability.
[2026-06-19T15:47:28Z] [monitor/EXP-37B] EXP-37B COMPLETE. 100/100 steps. val@25=0.7384 val@50=0.6808 val@75=0.6983 val@100=0.7346. anchor_backwards=40 (target met). back-half=STABLE (transient length excursion steps79-85, fully recovered by step88). No errors. tmux dead, train.log synced to runs/EXP-37B/. Dispatch EXP-37C.
[2026-06-20T01:58:53+10:00] [operator-correction] EXP-37C latency 5/5 -> 20/20 (cadence=20, delay_K=20) + beta_anc=0.0. Killed the 5/5 attempt (was only at vLLM init, no steps), freed GPUs (0 MiB verified), relaunched. Resolved cmd confirms beta_anc=0.0/cadence=20/delay_K=20/comm_eff.enabled=true (banner lies). anchor_backwards target=10 (not 40) at 20/20. WandB name=exp-37c-cad20-delay20-beta0-100step. EXP-37B closed out: WandB pns1le3x backfilled to step100 (val@100=0.7346, was underselling at val@75 0.6982); analyst dispatched. Chain unchanged: 37C(20/20,beta0) -> 37D(dense) -> teardown(team).
[2026-06-20T02:03:10+10:00] [analyst #37] verdict=PASS
## EXP-37C monitor: 2026-06-19 (session c7a130ef)
Run: exp-37c-cad20-delay20-beta0-100step | config: signed_ema, beta_anc=0.0, cadence=20, delay_K=20, 100 steps
Status at TIMEOUT (21min wall, step 52/100): tmux ALIVE, training in progress, val@25=0.6808, val@50=0.5368
Collapse: onset step 35, peak step 41 (resp_mean=779, clip=0.32), PARTIAL RECOVERY by step 50 (resp_mean=136, clip=0.0)
anchor_backwards: 5 fires by step 50 (steps 10,20,30,40,50) = exactly cadence=20 target
Verdict (first-half): COLLAPSE at step 35 (earlier than EXP-37's ~step 61), with PARTIAL SELF-RECOVERY; val@25→50 degraded 68%→54%
[2026-06-20T02:58:04+10:00] [chain] EXP-37C DONE (20/20, beta_anc=0): OSCILLATING instability val@25/50/75/100=0.6808/0.5368/0.7013/0.3457 (collapse@35->recover@75 0.70->RE-COLLAPSE@100 0.346); anchor_backwards=10 (20/20 realized). WandB u16ui4vx backfilled to step100 (was overselling at val@75 0.70). 37C analyst dispatched. EXP-37D (dense) LAUNCHED on same box (comm_eff.enabled=false verified in resolved cmd; monitor dispatched). Chain: 37B(PASS,stable) -> 37C(oscillating collapse) -> 37D(dense, running) -> teardown(team) after 37D. NOTE: pkill -f self-match footgun hit during 37C->37D handoff (my ssh shell matched 'ray::'/'main_ppo' patterns); box was actually clean, relaunched 37D OK.
[2026-06-20T03:01:38+10:00] [analyst #37C] verdict=STOP (sibling of #37; 20/20-latency signed_ema beta_anc=0 OSCILLATING instability: collapse 35-41 -> recover val@75=0.70 -> re-collapse val@100=0.3457; contrasts EXP-37 terminal-monotonic & EXP-37B 5/5 STABLE; high-latency not viable for either beta; issue #37 label left status:pass)
[2026-06-20T03:06:15+10:00] [operator-directive] EXP-37E ADDED (LAST run): ef_powersgd best-proven (EXP-26 B_ef_r2 config: clip=1.0/decay=0.9/beta_anc=0.95, M6 record 0.7210) at latency 20/20, 100 steps. FULL CHAIN: 37B(5/5,b0.5,PASS) -> 37C(20/20,b0,STOP oscillate) -> 37D(dense,running) -> 37E(EF 20/20) -> END. END-OF-CHAIN (operator-requested, separate agents): (1) WandB-completeness agent — verify 37B/37C/37D/37E all have 100 steps, backfill last 1-2 from local train.log if missing; (2) teardown box 41680420 via vast-teardown skill TEAM account; (3) teardown-VERIFY agent — confirm instance gone via 'vastai show instances' (team key) + /tmp/teardown.err per [[vast-teardown-ledger-can-lie]]. 37E staged+md5-verified on box.
[2026-06-20T04:57:25+10:00] [chain-END] Box 41680420 REMOVED BY OPERATOR (verified gone, team live count 0). EXP-37D (dense) COMPLETE before removal: val@25/50/75/100=0.7521/0.7665/0.7771/0.7832 — STABLE + MONOTONIC through epoch-2 boundary => instability is COMPRESSION-SPECIFIC, not epoch/dataset-revisit (answers the original Q). 37D train.log LOST (monitor killed pre-rsync) but SALVAGED from incoming.log heartbeat (runs/EXP-37D/train-salvaged.log) + WandB 04uozfpx backfilled to step100 val@100=0.7832. Ledger: 37D TORN_DOWN. End-of-chain AUDIT agent spawned (teardown-verify + WandB-completeness for 37B/37C/37D). EXP-37E (EF best-proven @20/20) STAGED but NOT RUN — box gone, needs a new box. AWAITING operator decision on 37E.
