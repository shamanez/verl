# PROGRESS — append-only audit log

Live: **#25** (RES-133) — realistic anchor-circuit PowerSGD GRPO trainer. One line per action.

[2026-06-05] #25 planned + codex-reviewed; 4 silent landmines hardened into plan (see `runs/UNWANTED_HOOKS_AND_SILENT_FAILURES.md`).
[2026-06-05] #25 R1 (DP-reduce + full-coverage M) / R2 (anchor-owns-Q) / R3 (signed_ema merger) + dead-spectral cleanup built; 128 CPU tests green. Provisioning blocked by Vast SSH key-injection bug (operator-owned).
[2026-06-05] #25 consolidated onto `vast-ai-workload` (`exp/25-anchor-default` merged `--no-ff` + deleted); doc/flag cleanup pushed; stale run scaffold deleted → re-materialize from plan.
[2026-06-05] #25 ready to resume on existing box 39602487 (4×H200) — verify SSH, reuse, don't provision.
[2026-06-06T00:47:43+10:00] [orchestrator] #25 resumed on operator warm box 39613656 (4×H200, direct -p 40872 root@46.243.55.155). Box synced vast-ai-workload@107ca01 (R1/R2/R3 present, editable install), secrets.env pushed (HF+WANDB), RUNNING ledger row registered (budget clock start, 12h headroom). Dispatching experiment-runner for id-0/id-1 probes + α sweep. DO-NOT-PROVISION.
[2026-06-06T00:50:39+10:00] [orchestrator] tick: running=[25] analyzing=[] logging=[] blocked=[24 dep]. experiment-runner (bg) driving id-0/id-1 probes + α sweep on reused box 39613656.
[2026-06-06T01:01:46+10:00] [orchestrator] id-0 probe (exp25_id0_anchorM) CRASHED on first launch — 2 backend bugs caught from log: (A) device-mismatch cuda/cpu in powersgd_activation.orthonormalize:127 degenerate-col repair (q_fix on CPU vs q on GPU); (B) anchor staleness off-by-one transformer_impl:1262 (step>=delay_K should be > ; step0 never pushed). Diagnosis+exact fixes sent to experiment-runner for on-box hotfix. Flags all verified correct in launcher banner.
