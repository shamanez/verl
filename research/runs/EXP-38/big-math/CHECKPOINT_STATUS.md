# EXP-38 ARM B — CHECKPOINT · dataset = **Big-Math** · analysis HELD for operator "go"

> This arm is **Big-Math ONLY** (`gshasiri/Big-Math-RL-Verified-filtered`). Model Qwen2.5-1.5B-Instruct,
> accel surface, **dense GRPO (comm_eff OFF)**, 75 global steps, **resp 16384**, **validation OFF**.
> Three deltas vs the GSM8K arm (ARM A): dataset → Big-Math, MAX_RESPONSE_LENGTH → 16384, val → OFF.
> **The two datasets must never be mixed** — Big-Math lives ONLY here; GSM8K lives ONLY in `../gsm8k/`.
> Full contract + the strict capture taxonomy: plan `.claude/plans/38.md` → "✅✅ ARM B — EXECUTION RESULT".

---

## ✅ FINISHED

| # | Step | Evidence |
|---|------|----------|
| 1 | **Fresh team box** provisioned (operator) + branch `exp/38-dense-drift-probe@ef0a3e7` checked out | `41779517` · 4×H200 · 84.8.116.228 · $12.88/hr · team account |
| 2 | **Big-Math pre-staged** | train 123,602 / val 6,506 → `/root/data/bigmath/{train,test}.parquet` (20000/500, seed 42); p99 prompt 211 tok ⇒ MAX_PROMPT_LENGTH 1024; reward → `math_reward` (`\boxed{}`) |
| 3 | **B1 off-parity gate PASS** (dense, 16K, flags OFF, 2 steps) | Hydra-resolved comm_eff.enabled=false / max_response_length=16384 / test_freq=0; **ALL comm_eff counters 0**; no drift dir / `[exp38]` lines; HBM 31 GB; no NaN |
| 4 | **B2 75-step dense capture COMPLETE** | 75/75, `train_rc=0`, no NaN; reward 0.43→0.52→0.62 (rising; model learns on Big-Math); resp_len max ~6K |
| 5 | **All artifacts synced + VERIFIED local** | **1071/1071 `.pt`, 0 missing, 16.15 GB**; `verify_bigmath.py` ALL HARD CHECKS PASS (5 roles at counts; spot-load finite; lags k∈{1,2,5,10,20,40} sampleable 5/4/2/14/13/11) |
| 6 | **Lag sampleability confirmed** | ticks {0,1,2,3,4,5,8,18,…,148} (21); k1:5 · k2:4 · k5:2 · k10:14 · k20:13 · k40:11 pairs. *(No epoch-2 crossing — Big-Math < 1 epoch at cap 20000; expected.)* |
| 7 | **WandB complete** | run `gos2wfpj`; async tail-drop of steps 74–75 backfilled from local train.log → max=75 |
| 8 | **GPU torn down** | team box `41779517` `vastai destroy` → **API-verified 0 team instances live** |

## ⏳ TO DO NEXT (HELD — do NOT start until operator says "go"; analyze GSM8K + Big-Math jointly)

1. **B3 — Big-Math offline analysis + HTML** (laptop, free): `python3 scripts/exp38_drift_analysis.py runs/EXP-38/big-math` → `research/reports/comm-eff-grpo/exp38-dense-drift-big-math.html` (+ `_findings.json`).
2. **ARM A (GSM8K) analysis** — same engine on `../gsm8k` (also HELD; see `../CHECKPOINT_STATUS.md`).
3. **B4 — JOINT GSM8K↔Big-Math comparative HTML** (the headline science): does the dense-gradient staleness budget / activation low-rank-ness / nature-of-learning DIFFER between an easy (GSM8K) and a hard (Big-Math) task? Dataset-tagged, tensors NEVER merged.
4. analyst verdict → log-writer → draft PR (`exp/38-dense-drift-probe` → `vast-ai-workload`).

**The full COLD-START step-by-step is the plan's "⏭️ NEXT BIG THING — ANALYSIS RUNBOOK (cold-start, self-contained)" section** (`.claude/plans/38.md`): scientific frame + exact commands + the Big-Math epoch-2 gotcha + the B4 joint-report recipe + deliverable acceptance + hard rules. A fresh agent can run the whole analysis from it. Everything is laptop-only and free — the capture phase is DONE.

---

## 📂 WHERE THE DATA IS (all local — `research/runs/EXP-38/big-math/`)

| path (under `runs/EXP-38/big-math/`) | what | size |
|---|---|---|
| `captures/rank0/manifest.jsonl` | 1 row / dumped tensor (role, target, tick, global_step, shape, norm, path) | 1071 rows |
| `captures/rank0/tick_<G>_<T>/<role>/*.pt` | fp32 tensors: `theta` / `g_dense` / `update_vector` (15 matrices) + `boundary_h` / `boundary_grad_h` (3 boundaries: layers 6/13/20) | **16.15 GB** |
| `captures/sidecar_layernorms.jsonl` | per-layer weight+grad L2 norms (all 28 decoder layers) | small |
| `sidecar_grpo.jsonl` (+`.config.json`) | per-step GRPO scalars (reward, resp_len, entropy, pg_clipfrac, ppo_kl, advantages, rollout-vs-actor logprob gap) | 75 rows |
| `train.log` | authoritative 75-step log + resolved Hydra config | 500 KB |
| `DATASET.json` | dataset stamp = **big-math** (anti-mixing guard) | — |
| `handles/41779517.json` | the (destroyed) Vast box handle | — |
| `monitor.log` | per-poll monitor snapshots | — |

**Naming convention:** `tick_<global_step>_<optimizer_tick>/<role>/<target>.pt`. 2 optimizer ticks/global step (ppo_mini 64 / batch 128) ⇒ first tick of step G is T=2(G−1). Steps 1–3 dump both ticks (small lags k=1,2); thereafter every 5th step's first tick (k=5,10,20,40). 21 ticks × 51 rows = 1071. Full taxonomy: plan `.claude/plans/38.md` → "WHAT WE CAPTURED".

**WandB (cloud):** `shamanework-pl/verl_compression_research_accel_rebaseline/runs/gos2wfpj`.
**⚠️ The 16 GB `captures/` are gitignored and are the SOLE copy** (box destroyed) — do not `git clean -fdx` this tree.

---

## 🧪 TWO-DATASET PLAN (keep STRICTLY separate)

- **GSM8K** — `../gsm8k/` (ARM A) — DONE through capture; analysis HELD. `../CHECKPOINT_STATUS.md`.
- **Big-Math** — `./` (ARM B) — **DONE through capture+sync+verify (this doc)**; analysis HELD.
- **Joint report** (B4) — once both analyses run, a combined dataset-tagged HTML. **Never mix tensors/curves across datasets** (enforced by per-arm run dirs + the `DATASET.json` stamp + dataset-tagged output filenames).
