# PROGRESS — append-only audit log

Live: **#25** (RES-133) — realistic anchor-circuit PowerSGD GRPO trainer. One line per action.

[2026-06-05] #25 planned + codex-reviewed; 4 silent landmines hardened into plan (see `runs/UNWANTED_HOOKS_AND_SILENT_FAILURES.md`).
[2026-06-05] #25 R1 (DP-reduce + full-coverage M) / R2 (anchor-owns-Q) / R3 (signed_ema merger) + dead-spectral cleanup built; 128 CPU tests green. Provisioning blocked by Vast SSH key-injection bug (operator-owned).
[2026-06-05] #25 consolidated onto `vast-ai-workload` (`exp/25-anchor-default` merged `--no-ff` + deleted); doc/flag cleanup pushed; stale run scaffold deleted → re-materialize from plan.
[2026-06-05] #25 ready to resume on existing box 39602487 (4×H200) — verify SSH, reuse, don't provision.
