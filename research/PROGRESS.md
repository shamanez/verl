# Progress — THE one local session file (capped tick echo + end-of-session checklist)
STANDING (operator 2026-07-05): team-account boxes labeled `erfan-*` belong to ANOTHER user — NEVER tear down; vast-cost's "possible LEAK" flag on them is a false positive. (Pin by LABEL PATTERN, not instance id — ids rotate.)
Role (operator 2026-07-08): an agent reads this file ONCE before ending its window — any unresolved MANUAL_REVIEW_NEEDED / TEARDOWN_FAILED / READY_FOR_GPU / STUCK / OPERATOR_STOP line means the session is NOT done. Durable record lives elsewhere (issue close comments · report pages at https://com-eff-rlvr.pages.dev/runs/ · runs/SUMMARY.md · git); /close's cleanup sweeps a finished issue's ticks from here. LOG.md is retired.
[2026-07-10T10:53:34+10:00] [research-planner #64] plan published (tier=fast)
[2026-07-10T11:10:08+10:00] [approve] #64 approved (edits: vast_account=team, VAL_BEFORE_TRAIN=False both, Big-Math on accel surface @resp4096, S_base operator-supplied)
[2026-07-10T11:47:48+10:00] MANUAL_REVIEW_NEEDED: #64 READY FOR GPU: #64 prepared (branch exp/64-middle-block-freeze-grpo @ 36cc5fc0 pushed to origin; payload launch.sh + run.json + exp.bundle written; CPU gates green — off-path parity + freeze indexing L11–15 = 5/28 decoder layers ≈15% trainable asserted). No box, no ledger row, status unchanged (still approved).

Two cells, sequential, one box, team account, budget max_gpu_hr=10 / max_dph=24: (1) freeze-block-l11-15-gsm8k [accel surface, resp=1024, C≥0.80 vs S_full=0.7657], (2) freeze-block-l11-15-bigmath [resp=4096, dyn-bsz, fallback S_block−S_base>+0.03].

Resume with:
  /execute 64 --attach <instance-id>   # your own box (login registered via vast-attach)
  /execute 64 --gpu auto               # harness provisions (default ladder)

Note for /analyze: C(block) needs operator-supplied S_base (base Qwen2.5-1.5B-Instruct GSM8K + Big-Math val-core scores) — VAL_BEFORE_TRAIN=False, so S_base is not measured at step 0.
[2026-07-10T12:28:35+10:00] LAUNCHED: #64 run-64 on box 44365338 (1xH200, team) — GitHub-clone bootstrap (no bundle upload); 2 cells gsm8k->bigmath
[2026-07-10T14:50:47+10:00] teardown 64-middle-block-freeze-grpo reason=no-heartbeat-30min destroyed=1
