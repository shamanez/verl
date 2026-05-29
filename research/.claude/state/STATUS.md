# Research Status — fresh research cycle

## Permanent reference runs

- **baseline** (`runs/baseline/`) — dense GRPO on Qwen2.5-1.5B + GSM8K,
  verl unmodified. 100 steps, val 0.087 → 0.789. The dense control.
  Launcher: `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`.
- **communication-baseline** (`runs/communication-baseline/`) — comm-eff
  method smoke verification: PRF mask `p=0.9` on both gradient-feeding
  forwards, hookless K-stale anchor (cadence=5/delay=5), two-sided Tikhonov
  spectral correction (`α=0.5, τ=0.01, β_anc=0.9`); no KL, no entropy.
  20-step smoke PASS — all comm-eff guards held, visible learning.
  Launcher: `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`.

## Active

- **Investigation queued**: `notes/investigation-prompt-grad-norm.md`.
  Paste into a fresh session to draft the GitHub issue. Documents 9
  candidate root causes (IS variance under independent PRF masks, spectral
  conditioning on empty M_anchor, FSDP integration audit, anchor harvest,
  variance amplification from smaller mini-batch + token wedge, etc.) and
  a four-test discriminating plan. KL stays off in all tests.

## Implementation locus

- `verl/workers/config/comm_eff.py` — Hydra config schema
- `verl/workers/comm_eff/{state.py, activation_mask.py, anchor.py, spectral_filter.py}` — runtime
- `verl/workers/engine_workers.py` — `compute_log_prob` `mask_active` stamp
- `verl/workers/engine/fsdp/transformer_impl.py` — `_comm_eff_mask_active` gating
- `tests/workers/comm_eff/` — CPU unit tests

## Conceptual notes

- `notes/anchor-memory-cost.md` — why the anchor clone takes ~3 GB
- `notes/fast-circuit-vs-anchor-pass.md` — masking semantics across the 5 GRPO forwards
- `notes/investigation-prompt-grad-norm.md` — the investigation issue draft

## Vast.ai

No instances running. All ledger rows TORN_DOWN.

## Git

Local `vast-ai-workload` synced to `origin/vast-ai-workload`. All prior
experiment branches deleted local + remote. The two canonical launchers
live in `examples/grpo_trainer/`; there are no duplicate launcher scripts
under `runs/*/`.
