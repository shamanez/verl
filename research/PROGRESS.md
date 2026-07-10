# Progress — THE one local session file (capped tick echo + end-of-session checklist)
STANDING (operator 2026-07-05): team-account boxes labeled `erfan-*` belong to ANOTHER user — NEVER tear down; vast-cost's "possible LEAK" flag on them is a false positive. (Pin by LABEL PATTERN, not instance id — ids rotate.)
Role (operator 2026-07-08): an agent reads this file ONCE before ending its window — any unresolved MANUAL_REVIEW_NEEDED / TEARDOWN_FAILED / READY_FOR_GPU / STUCK / OPERATOR_STOP line means the session is NOT done. Durable record lives elsewhere (issue close comments · report pages at https://com-eff-rlvr.pages.dev/runs/ · runs/SUMMARY.md · git); /close's cleanup sweeps a finished issue's ticks from here. LOG.md is retired.
[2026-07-09T12:33:21+10:00] #63 operator re-plan 2026-07-09: all arms -> 102 steps, ckpt @ step 100 (SAVE_FREQ 100). dense finishes to 102 (no ckpt); launch2.sh staged for the 3 comm-eff arms. run.json + plan SSOT updated.
[2026-07-09T13:35:03+10:00] #63 transition DONE 2026-07-09T03:33Z: dense-control stopped at step 103 (AIME@100=0.254, reward~0.526, no ckpt per operator). tmux run-63 relaunched with launch2.sh -> 3 comm-eff arms @ 102 steps, save@100. verl re-checked-out to 40e05b3a (training code identical to eeafc2a5). Monitoring resumes on signed-ema-b50.
[2026-07-09T15:16:54+10:00] #63 signed-ema-b50 OOM'd at first anchor refresh (tick20/step10, replay_paired_batch extra fwd+bwd, offload OFF, 143GB ceiling). Operator fix 2026-07-09: offload ON for the 3 comm-eff arms (param+optimizer_offload=True via trailing Hydra override; NO numeric change). Stale flags cleared, OOM log archived, launch2.sh relaunched. Watching b50 through step 20.
[2026-07-09T16:50:53+10:00] #63 signed-ema-b50 OOM'd AGAIN at anchor refresh (step10) under offload-ON alone (218MiB free, needed 8.21GiB) — offload marginal per the memory playbook. Auto-recovery: launch2b.sh = offload ON + ppo_max_token_len 30000->16000 (cuts token-linear anchor surcharge ~30GB), relaunched 06:50Z. Playbook logged to memory; definitive anchor-clone-ckpt fix filed as follow-up task.
[2026-07-09T17:05:25+10:00] #63 b50 relaunch #4 at 07:04Z: token-len 16000 tripped AssertionError (below the 18432 max-seq floor). Corrected launch2b.sh to token-len 20000 (offload ON). Watching for step-11 clear = anchor-refresh fit proven.
[2026-07-09T18:56:22+10:00] #63 FIX PROVEN: signed-ema-b50 cleared the step-10 anchor refresh (anchor_q_updates=1, 339/339 stale params loaded, peak mem 124GB/140, 0 OOM). grad_norm 45->1.4 post-anchor (correction working), reward tracking dense. offload ON + token-len 20000 is the working comm-eff config for all 3 arms. Resuming long-horizon monitoring (each arm ~102 steps; ckpt@100 -> R2).
[2026-07-10T10:53:34+10:00] [research-planner #64] plan published (tier=fast)
[2026-07-10T11:10:08+10:00] [approve] #64 approved (edits: vast_account=team, VAL_BEFORE_TRAIN=False both, Big-Math on accel surface @resp4096, S_base operator-supplied)
[2026-07-10T11:47:48+10:00] MANUAL_REVIEW_NEEDED: #64 READY FOR GPU: #64 prepared (branch exp/64-middle-block-freeze-grpo @ 36cc5fc0 pushed to origin; payload launch.sh + run.json + exp.bundle written; CPU gates green — off-path parity + freeze indexing L11–15 = 5/28 decoder layers ≈15% trainable asserted). No box, no ledger row, status unchanged (still approved).

Two cells, sequential, one box, team account, budget max_gpu_hr=10 / max_dph=24: (1) freeze-block-l11-15-gsm8k [accel surface, resp=1024, C≥0.80 vs S_full=0.7657], (2) freeze-block-l11-15-bigmath [resp=4096, dyn-bsz, fallback S_block−S_base>+0.03].

Resume with:
  /execute 64 --attach <instance-id>   # your own box (login registered via vast-attach)
  /execute 64 --gpu auto               # harness provisions (default ladder)

Note for /analyze: C(block) needs operator-supplied S_base (base Qwen2.5-1.5B-Instruct GSM8K + Big-Math val-core scores) — VAL_BEFORE_TRAIN=False, so S_base is not measured at step 0.
[2026-07-10T12:28:35+10:00] LAUNCHED: #64 run-64 on box 44365338 (1xH200, team) — GitHub-clone bootstrap (no bundle upload); 2 cells gsm8k->bigmath
