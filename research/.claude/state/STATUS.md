# Research Status — 2026-06-01T14:17:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 17 | M3 — Long-horizon masked GRPO, clean_cadence K=20 over 2 epochs (CORE single-run diagnostic) | **RUNNING** (monitor active) | 1×4H200 (i_38877541, $9.29/hr) | — | status:running. Launched on operator PRE-PROVISIONED box (runner skipped vast-provision). tmux `exp-17-210_157_233_86`, WandB `grpo_mask_channel_p0p9_rescale_clean_every20_2epoch`. 116-step schedule confirmed live (`Total steps: 116, num_warmup_steps: 0`); clean steps at 20/40/60/80/100, vals at 0,10,…,110. exp branch `exp/17-masked-clean-every20` pushed (marker-only, target_modules:[]). Env hygiene PASSED (no VAST_API_KEY in container). training-log-monitor running in background. |
| 16 | Short-run stability matrix (mask/rescale/clean-cadence/spectral) | DONE | 1×4B200 (manual, i_38454090) TORN_DOWN | PASS (manual) | Completed manually on operator 4×B200 2026-05-30 (5 rescale cells: dense strict-no-op proven, clean@5/50 ≈ dense parity 0.729 vs 0.741). PR #10 draft → vast-ai-workload. Evidence in runs/EXP-16/. Issue closed. EXP-17 depends_on:[16] → SATISFIED. |
| 11 | M3 — 100-step M95+AP GRPO vs dense (K=20) | NOT_CLAIMED (superseded) | — | — | kind:experiment, milestone:M3; no research:claim/status/plan. SUPERSEDED by #17 (its anchor+spectral K=20 hypothesis is falsified by EXP-16 — spectral fails by orthogonality). Out of orchestrator scope. |
| 10 | M3 — DP gradient compression (PowerSGD-64 + Streaming-DiLoCo) scope | NOT_CLAIMED | — | — | kind:experiment, milestone:M3; gated behind M95+AP smoke. No status/plan. Out of orchestrator scope. |

`baseline` (dense control, `.claude/plans/baseline.md`) is a design template, not a gating EXP-run.

## EXP-17 launch detail (this tick)

- **Pre-provisioned reuse**: operator stood up i_38877541 (4×H200, 143 GB ea., $9.29/hr, region JP, image verlai/verl:vllm020.dev1) created 2026-06-01T03:38:55Z. Orchestrator verified reachable (SSH OK via `~/.ssh/vast_ai_name`, 4×H200 confirmed, /workspace/verl @ 8304419 on vast-ai-workload). Runner SKIPPED vast-provision, registered PROVISIONED→RUNNING from the pre-written handle.
- **Config (verbatim issue env)**: COMM_EFF p=0.9, rescale ON, mask_recompute ON, clean_cadence=20, anchor+spectral OFF, 2 epochs / 116 steps, TEST_FREQ=10, dynamic_bsz, max_token_len=98304 (EXP-16-proven perf knob). no-KL no-entropy, rollout IS/RS strictly OFF, calculate_log_probs=True (read-only train-inference diagnostic).
- **Liveness at launch**: FSDP 1.54B params sharded (5.05 GB/GPU — >2× H200 headroom), Ray init + dataset filtering OK, no error signatures, NIXL warnings benign.

## Last tick
2026-06-01T14:17:00+10:00 · running=[17] · analyzing=[] · logging=[] · blocked=[] · skipped=[11,10 not-claimed]

## Budget
$/hr now: $9.29 (i_38877541 training) · est. spend so far this run: ~$6 (≈38 min idle pre-launch + ramp) · account credit remaining: ~$1018 · max_gpu_hr cap 96 (4 GPU × ~1.5 hr ≈ 6 gpu-hr expected, far under cap) · max_dph cap $24 (under).

## Notes
- Kill switch clear (`~/.claude-kill-switch` absent).
- Stale `.claude/state/supply-poll.sh` (from 2026-05-30 EXP-16 supply-block) is NOT running as a process — harmless leftover, left in place.
- gh default repo: `shamanez/verl-compression-research` (issue queue). Code PRs target `shamanez/verl` base `vast-ai-workload`.
- Next: training-log-monitor (background) returns a terminal report → on done.flag dispatch analyst (predicate in plan §Analyst predicate) → on PASS dispatch log-writer.
