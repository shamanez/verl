# Verdict — Issue #62: Add RLVR-paper models + math datasets to comm-eff GRPO

**VERDICT: PASS** (2026-07-07)

Additive enablement. All 4 stage gates evaluated, none skipped, all GREEN. No
algorithm path forked; every dataset routes through the locked
`DigitalLearningGmbH/MATH-lighteval` → `math_reward` key; no new reward function
added. Dense control byte-identical.

## Stage gates
| stage | gate | result |
|---|---|---|
| 1 registry+schema | 8/8 tokenizers load + chat-template preserves `\boxed{}`; 5/5 dataset schemas + parquet round-trip | ✓ GREEN (CPU) |
| 2 reward-preflight (MONEY GATE) | 25/25 model×dataset pairs pass checks 1–5 (nested balanced-brace; correct→1.0/wrong→0.0) | ✓ GREEN (CPU) — GPU unlocked |
| 3 full 5×5 smoke matrix | 25/25 cells: 100% non-null train+val rewards, no NaN, no `EARLY_STOP_SIGNAL` | ✓ GREEN (GPU) |
| 4 dense-control-parity | off-path byte-identical (reward routes + 3 launchers = 0 changed lines vs base); GSM8K step-0 val in-band | ✓ GREEN (GPU) |

## Stage 3 — 25 smoke cells (train/val reward @ 5 steps, all PASS, all train_rc=0 except one benign wandb-teardown rc=1)
| dataset\model | qwen25-math-1p5b | qwen3-1p7b-base | r1-distill-qwen-1p5b | open-nemotron-1p5b | qwen3-4b-base |
|---|---|---|---|---|---|
| math        | .432/.585 | .274/.510 | .558/.590 | .030/.040 | .288/.470 |
| numina-cot  | .237/.347 | .159/.296 | .291/.367 | .024/.041 | .149/.173 |
| deepscaler  | .171/.330 | .101/.210 | .211/.280 | .002/.035 | .136/.300 |
| skywork-or1 | .136/.245 | .056/.185 | .167/.220 | .004/.010 | .088/.220 |
| dapo-math   | .131/.235 | .058/.160 | .188/.180 | .002/.000 | .101/.245 |

## Stage 4 — dense-control-parity
Qwen2.5-1.5B-Instruct, GSM8K, comm-eff OFF: train_rc=0, train reward 0.438, val 0.697 — sane/in-band.
Off-path byte-identical: `verl/utils/reward_score/__init__.py`, `vast_comm_eff_accel_base_*.sh`,
`vast_comm_eff_baseline_*.sh`, `run_qwen3_4b_fsdp.sh` all 0 changed lines vs base branch.

## Recorded negatives (clean negatives = gate PASS, per plan)
- **open-nemotron-1p5b** scores near-zero on all 5 datasets (0.00–0.04) — weak at math-`\boxed{}`
  output. NOT a failure: emits valid non-null, NaN-free rewards; the reward loop works. Kept in
  the registry as a documented low-capability entry, not dropped.
- **R1-Distill** cells ran at `MAX_RESPONSE_LENGTH=4096` (plan q5) — no truncation-driven null
  rewards observed; all 5 passed.
- No OOM on any model (incl. Qwen3-4B-Base on 1×H200). No model dropped from the registry.

## 7B/8B integrate-only (not GPU-smoked, per plan)
r1-distill-qwen-7b, r1-distill-llama-8b, qwen3-8b-base: registry + CPU tokenizer/chat-template
load only (Stage 1, 8/8). Verified load a chat template preserving the boxed instruction.

## Infra notes (not gate-affecting)
- First full-matrix run filled the box's 200G disk with per-step FSDP checkpoints (~20G/cell,
  verl force-saves final step). Fixed: driver now runs `SAVE_FREQ=-1` + `rm -rf checkpoints`
  per cell; `resume` mode re-ran only the 18 failed cells. Peak disk after fix: 169G free.
- Box: operator-provided 1×H200 NVL (EXTERNAL). Model downloads amortized (each of 5 models
  pulled once, reused across its 5 datasets).
