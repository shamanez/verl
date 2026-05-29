# Research Status — 2026-05-29 — investigation phase

## Permanent reference runs

- **baseline** (`runs/baseline/`) — dense GRPO on Qwen2.5-1.5B + GSM8K, verl
  unmodified. 100 steps, val 0.087 → 0.789. The control, ran before any
  comm-eff code change.
- **communication-baseline** (`runs/communication-baseline/`) — comm-eff M90+AP
  smoke proof (p=0.9, α=0.5, τ=0.01, β_anc=0.9, anchor cadence=5/delay=5,
  mask_recompute=true), no KL, no entropy. 20-step verification at smoke
  rollout shape. PASS — all comm-eff guards held, visible learning. Formerly
  EXP-9 iter2; promoted to a permanent reference alongside the dense
  baseline.

## Active experiment

- **EXP-13** (`runs/EXP-13/`) — paper-scale extension of communication-baseline
  (TRAIN_BATCH=128, ROLLOUT_N=8, MAX_RESPONSE=16384). 58 of 100 steps reached
  (dataset-epoch limit; TOTAL_EPOCHS=1 ⇒ 58.4 batches per epoch). Verdict:
  **PASS at-risk** — infrastructure healthy (counters scale, memory bounded,
  guards hold, +26% val gain), but `grad_norm` starts at 1134 at step 1 and
  `entropy` collapses 6.4 → 0.023 by step 58. Investigation queued.

## Investigation queued

- `notes/investigation-prompt-grad-norm.md` — paste into a fresh session to
  draft the GitHub issue. Documents 9 candidate root causes and a 4-test
  discriminating plan. KL stays off across all tests (operator constraint).

## Repo hygiene

- **De-bloated**: EXP-4, EXP-5, EXP-6, EXP-7, EXP-8, EXP-12 (folded into
  `runs/SUMMARY.md`).
- **Renamed**: EXP-9 → `communication-baseline` (run dir, plan file, finding).
- **Notes** (`research/notes/`):
  - `anchor-memory-cost.md` — the EXP-12 anchor clone's ~3 GB explanation
  - `fast-circuit-vs-anchor-pass.md` — masking semantics across the 5 GRPO
    forward passes
  - `investigation-prompt-grad-norm.md` — the next investigation issue draft

## Vast.ai

No instances running. All ledger rows TORN_DOWN. `runs.jsonl` carries
historical records of every provisioning event; handle files have been
cleaned up.

## Git

Local `vast-ai-workload` synced to `origin/vast-ai-workload`. PRs merged
this research cycle: `#1` (comm_eff scaffolding), `#2` (actor-only mask),
`#3` (mask contamination guard), `#4` (spectral correction + FSDP discovery),
`#5` (anchor backward isolation), `#6` (mask_recompute extension + EXP-9
iter2 PASS lineage), `#7` (paper-scale EXP-13 launcher + notes + findings).
All `exp/*` branches deleted local + remote.
