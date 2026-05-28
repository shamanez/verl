# Research runs — summary

Concise record of what has run on this harness. Full per-experiment artifacts were
pruned to keep the repo lean; the durable record is here + git history + the merged code.

| id | milestone | what | result | merged |
|---|---|---|---|---|
| EXP-8 | M2 | M2 — anchor circuit: same-process K-stale unmasked GRPO-actor-loss refresh + two-step smoke | REVISE | — |
| EXP-7 | M2 | Spectral correction filter (anchor-EMA → thin SVD → Tikhonov → two-sided projection → α-blend) + FSDP gradient-application-point discovery | PASS: FSDP1 full 2D `Tensor` via `use_orig_params`, correction AFTER FSDP reduction / BEFORE grad clipping; `spectral_corrections` fired (8→16 mask-off / 8→80 combined 10-step), per-target rel_change in (0,1], grad_norm finite, disabled cell dense no-op | PR #4 → `vast-ai-workload` |
| **baseline** | M1 | Dense GRPO, Qwen2.5-1.5B-Instruct on GSM8K — verl unmodified (the control) | `val/test_score` 0.0872 → 0.7892 over 100 steps, 4×H200 | — (no code change) |
| EXP-4 | M2 | `comm_eff` no-op scaffolding: config group + disabled-by-default integration hooks | Run A no-op parity validated (disabled == dense) | PR #1 → `vast-ai-workload` |
| EXP-5 | M2 | Actor-only PRF activation masking (in-graph `h*mask`, no rescale) | PASS: mask_ratio tracks p (p95→0.950, p90→0.900, ±0.02); confined to actor-train path; disabled holds EXP-4 no-op; grads finite, no NaN/Inf | PR #2 → `vast-ai-workload` |
| EXP-6 | M2 | Mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS: per-path counters train=28/all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean (0 leaked keys/4 shards), no NaN/Inf | PR #3 (draft) → `vast-ai-workload` |

**M2 milestone summary:** `findings/M2/SUMMARY.md` (EXP-4/5/6 synthesis).

**Baseline run dir:** `runs/baseline/` (config, launcher, REPRODUCIBILITY.md, handle).
**Baseline plan:** `.claude/plans/baseline.md`.

**EXP-5 headline numbers** (the masking proof): boundaries `[3,7,11,15,18,21,24]` (L=28,
from model.config); p95 mask_ratio 0.9498/0.9502, p90 0.8999/0.9002; grad_norm finite on
every substep; disabled cell all `comm_eff` counters 0. Verdict PASS, issue #5 `status:pass`.

**EXP-6 headline numbers** (the contamination guard): per-path counters train=28,
rollout/old_logprob/ref_logprob/val/infer/ckpt=0; old/ref log-prob bit-equal within 1e-6
mask-on vs mask-off (unit test); validation ran (val-core/openai/gsm8k/acc/mean@1 parity
within noise); checkpoint save/load clean (0 leaked keys / 4 shards). 35 unit tests passed.
Verdict PASS, issue #6 `status:pass`.

The verl implementation lives in the merged code: `verl/workers/comm_eff/` (+ `activation_mask.py`),
`verl/workers/config/comm_eff.py`, `tests/workers/comm_eff/test_activation_mask.py`.

_Carryover follow-up:_ launcher `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
hardcodes a `done.flag` path that fails under `SAVE_FREQ=-1`, aborting multi-cell smoke chains
under `set -e`. EXP-5 + EXP-6 worked around it on-box (pre-created the dir); a real fix
(`$EXPERIMENT_NAME` + `mkdir -p`) still belongs in the launcher.
_EXP-6 caveat:_ mask_off cell auto-resumed mask_on's checkpoint (shared `experiment_name`) —
pin a unique `experiment_name` per cell for pristine val-parity reruns. Plans grep
`val/test_score` but verl emits `val-core/openai/gsm8k/acc/mean@1`; update plan templates.
