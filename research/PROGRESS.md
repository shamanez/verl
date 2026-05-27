# PROGRESS — research orchestration audit log

Append-only. Subagents (triage, planner, runner, analyst, log-writer, codex-bridge) and the Stop hook (`teardown-finished-runs.sh`) write one line per significant event in the format:

```
[<ISO timestamp>] [<agent-name> #<exp-id-if-applicable>] <one-line description>
```

The orchestrator greps this file each tick for these structured markers to route work:

| Marker | Meaning | Routed to |
|---|---|---|
| `STUCK: <ctx>` | runner hit verl-internal it can't resolve | `codex-bridge --mode=code-rescue` |
| `RESCUE_REQUEST: math <ctx>` | planner/analyst needs derivation review | `codex-bridge --mode=math-rescue` |
| `MILESTONE_PASS: M<X>` | log-writer wrote a SUMMARY for a milestone | `codex-bridge --mode=adversarial` |
| `VERIFY_TIMEOUT:` / `BROKER_DIED:` | codex-bridge hit the watchdog | demote plan, ping human |
| `MANUAL_REVIEW_NEEDED: <ctx>` | anything that needs a human (no offers, codex down, ...) | flagged in STATUS, no auto-action |
| `LAUNCH_FAILED: EXP-<N>` / `LAUNCH_FAILED_TIER:` | experiment-runner couldn't launch | depends; usually walks to next tier or stops |
| `BUDGET_EXCEEDED: EXP-<N>` | run blew through `max_gpu_hr` | teardown hook destroys, orchestrator notes |
| `TEARDOWN_FAILED` | Stop hook couldn't destroy an instance | loud warning — operator must check `vastai show instances` |
| `PR_SKIPPED: EXP-<N>` | log-writer refused to draft PR (wrong default repo) | flagged, no auto-action |
| `NEEDS_PLAN: #<N>` | orchestrator saw a `research:claim` issue with no plan | triage owes a plan |

Anything else here is informational. The hook that used to spam `[session-id] Edit` per file write has been removed — only meaningful events are recorded.

---
[2026-05-26T17:05:00+10:00] [research-planner #3] plan written
[2026-05-26T17:05:12+10:00] [triage] dispatched 1 planners, 0 issues already planned
[2026-05-26T17:32:05+10:00] [experiment-runner #3] launched on 1 instances dph=$16.0540
[2026-05-26T17:33:20+10:00] [orchestrator] tick: verify=[] running=[3] analyzing=[] logging=[] blocked=[]
[2026-05-26T18:08:41+10:00] [triage] no claimable issues — ALL_PLANNED
[2026-05-26T18:14:42+10:00] [hf-watcher #3] step_100 pushed to gshasiri/qwen25-1p5b-grpo-gsm8k-baseline-step100 (private)
[2026-05-26T18:17:16+10:00] [user #3] issue closed via gh issue close; val=0.789 (+0.702 from baseline) at step 100
[2026-05-26T18:18:27+10:00] [vast-teardown #3] destroyed instance 37881404 (reason: EXP-3 closed at global_step_100 — checkpoints preserved on HF)
[2026-05-27T22:32:42+10:00] [research-planner #4] plan written
[2026-05-27T22:33:08+10:00] [triage] dispatched 1 planners, 0 issues already planned
[2026-05-27T22:58:59+10:00] [codex-bridge --mode=verify] result=fail
[2026-05-27T23:01:24+10:00] [orchestrator] tick: verify=[4→FAIL] running=[] analyzing=[] logging=[] blocked=[] demoted=[4] — EXP-4 codex VERIFY:FAIL @ pre-impl gate, label→status:planned, critique posted, zero GPU spend
[2026-05-27T23:21:15+10:00] [codex-bridge --mode=verify] result=concerns
[2026-05-27T23:33:27+10:00] [orchestrator] EXP-4 OPERATOR-CLEARED — operator waived codex plan-verify (raw: FAIL rev-1 / CONCERNS rev-2, judged out-of-scope for M2 disabled-scaffolding smoke). Reverted plan to rev-1; deleted FAIL demotion comment from #4; posted VERIFY:PASS (operator) comment + artifact runs/EXP-4/verify/20260527T233157Z.md; label status:approved. State=VERIFIED. NO runner dispatched (stop per operator). Next session: dispatch experiment-runner for EXP-4 WITHOUT re-verifying the plan (plan-level gate only; code-level verify of exp/4 diff not waived).
[2026-05-28T00:07:44+10:00] [experiment-runner #4] launched on 1 instance (id=38088784, 4xH200, tier_idx=1) dph=14.738212719298245 — 3 back-to-back smokes (m2-commeff-noop-disabled/-default/-reference); code-level verify.log VERIFY:PASS pre-launch; cgroup pids.max=23552
[2026-05-28T00:09:59+10:00] [orchestrator] tick: verify=[] running=[4] analyzing=[] logging=[] blocked=[] dispatched=[experiment-runner→EXP-4] — EXP-4 VERIFIED (operator-cleared plan gate) → runner launched i_38088784 (4xH200, tier_idx=1, dph=14.74), branch exp/4-commeff-noop pushed, pre-launch unit tests 10/10 PASS, status:running. #5-11 backlog skipped (no status/claim).
[2026-05-28T00:19:30+10:00] [orchestrator] EXP-4 CHAIN ABORTED after Run A (1/3). ROOT CAUSE: baseline launcher line 196 `touch /workspace/verl/runs/qwen25_1p5b_grpo_gsm8k_baseline/done.flag` is a HARDCODED path (default exp-name, not $EXPERIMENT_NAME) that does not exist under SAVE_FREQ=-1 -> touch fails -> launcher exits 1 (set -euo pipefail) -> run_smoke returned rc=1 -> launch.sh set -e aborted before Run B/C. Run A SCIENCE CLEAN: global_step=2 (2 steps), comm_eff counters all 0.0 (mask/anchor/spectral), grad_norm=3.07e-4 finite <5, no NaN/Inf, entropy 0.357. Runs B(default)+C(reference) NOT executed -> A-vs-B parity + criterion-7 rel-tol 1e-4 UNTESTED. Box i_38088784 (4xH200, $14.74/hr) now IDLE (GPU 0%, no procs, tmux gone) + still billing; no done.flag. STUCK: launcher done.flag path bug -> needs fix + relaunch; teardown pending operator confirm.
