# PROGRESS — append-only audit log

> **Reset 2026-06-15.** Prior ticks (EXP-25 → EXP-31 close-out) are in git history. The orchestrator
> appends one tick per loop iteration below; `STATUS.md` is the derived dashboard (rewritten each tick).

[2026-06-15] [reframe] **EXP-31 re-scoped** to the OPEN-ENDED **4-lever anchor-gradient-usage tournament**
— L4 perturbation [already built] / L2 δ-momentum / L3 adaptive-dose / L1 control-variate — target greedy
val@50 → **0.80**. **B2 frozen as the comm-eff SOTA** (ground truth migrated to `runs/EXP-31/B2_baseline/`);
the sub-basis bet is frozen as a parity-only null. Dose λ/β_anc unblocked (EXP-27 ignition watch back on).
**Async-realism constraint:** anchor = single SLOW node serving a fast SWARM ⇒ always lagging, never leads
(delay-compensation ruled out). Housekeeping: `runs/` de-bloated 64G→16M; old plans removed (only `31.md`
+ `TEMPLATE.md`); plan `.claude/plans/31.md` rewritten + labeled `status:planned`.
**AWAITING operator `status:approved`** → provision a box → Cell A (B2 reproduce) → L4 → L2/L3/L1.
[2026-06-15T22:10:51+10:00] [orchestrator/goal] EXP-31 tournament START on operator box 46.243.55.155:40276 (i_41048644, 4xH200, separate Vast acct=operator teardown). Cell A (A_b2_reproduce) LAUNCHED tmux exp-31-46_243_55_155: bitwise B2 verified (delayed_ef λ=1 β_anc=0 r=77 anchor c5/dK5/owns_q/replay, KL off, dynbsz off, 18432, dcar=true, 50 steps/val@25). GPUs busy 45-62%. Monitor(bg)+heartbeat alive. L2+L3 code-build runner(bg) on exp/31-anchor-usage-levers. Tournament: A->L4(σ.01/.03,env-only)->L2->L3->L1(gated). Target greedy val 0.80.

[2026-06-15T12:17:00Z] [training-log-monitor/EXP-31/A_b2_reproduce] TIMEOUT at step 8/50 (healthy). val@0=0.0910, bytes_ratio=0.0504, anchor fires at steps 3+5+8 (5-tick cadence confirmed), resp_len/mean~280 (no explosion), grad_norm settling 480->0.79->2.32->1.28->2.55 (anchor-step spikes expected), reward 0.12->0.33->0.19->0.28 (GRPO fluctuation normal), WandB running historyLines=8. val@25 expected ~12:46 UTC. No errors. RECOMMENDATION: continue.
[2026-06-16T00:15:11+10:00] [orchestrator/goal] EXP-31: Cell A DONE B2_live val 0.0910/0.7202/0.7354 (WandB fy920fty). Full autonomous pipeline ARMED: chain1 A->L4(.01 running)->L4(.03); chain2 waits CHAIN_L4_DONE then checkout exp/31@1d9077d (FETCH_HEAD SHA-asserted) -> L2(mu.9/.5 age_decay) + L3(ratio/cos x kappa .5/1.0 cap2) = 8 lever cells, zero GPU idle. Gate vs B2_live@50=0.7354: EXTEND>.7594/BANK .7114-.7594/KILL<.7114. Monitor on L4_s001. Operator: autonomous, no questions, GPU always busy.
[2026-06-16T04:19:46+10:00] [analyst #31] verdict=STOP
[2026-06-16T04:25:57+10:00] [log-writer #31] logged verdict=STOP milestone=M6
[2026-06-16T13:11:44+10:00] [research-planner #32] plan written
[2026-06-16T13:12:07+10:00] [triage] dispatched 1 planners, 0 issues already planned
