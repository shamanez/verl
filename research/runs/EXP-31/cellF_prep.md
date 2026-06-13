# Cell F prep — seed-replicate certification (pin dense + B2 bands)

**Runs LAST, after Cell D production. Pins both bands so the surpass claim is band-vs-band.**
Gated on Cell D's val@50 (decides whether to also run the winner ×3).

## Seed knob (resolved from verl source)
- **`data.seed`** (Hydra; `data/legacy_data.yaml:68` default `null`) → `main_ppo.py:368` seeds the train-dataloader generator + `rl_dataset.py:154` the shuffle RNG. **This is the training/data-draw seed to vary.**
- Codec seeds are INDEPENDENT and HELD: `powersgd.seed=0`, `mask.seed=0` (the launcher pins them). The merger SVD seed is per-target off `powersgd.seed`, so also fixed.
- Cell A / EXP-30 ran at `data.seed=null` (the default) = "seed 0". New seeds → `data.seed=1`, `data.seed=2`.

## Run list (model launchers on launch_A.sh / launch_D.sh — they carry the disable_custom_all_reduce fix + the self-patching actor.yaml)
| arm | config | seeds | have | new runs |
|---|---|---|---|---|
| **B2** | exp/31 + `delta_subbasis_rank=0` (== B2 bitwise) | 0,1,2 | seed0 = Cell A 0.7400 | `data.seed=1`, `data.seed=2` |
| **dense** | `COMM_EFF_ENABLED=false` (byte-identical dense) | 0,1,2 | EXP-30 0.7839 was a DIFFERENT box/config (no disable_custom_all_reduce) | **re-run all 3 on THIS config** for a clean band (the flag shifted B2 0.7528→0.7400 ≈0.013, so re-run dense-0 too rather than reuse 0.7839) |
| **winner (Cell D)** | exp/31 + `delta_subbasis_rank=2 family=tail` | 0,1,2 | seed0 = this Cell D run | `data.seed=1`, `data.seed=2` — ONLY if Cell D@50 ≥ 0.79 |

Dense launcher = launch_A.sh with `COMM_EFF_ENABLED=false` (drop the comm_eff Hydra overrides; keep `+...disable_custom_all_reduce=true`, 50 steps, test_freq=25, `data.seed=<s>`). All arms: identical batch128/mini64/lr1e-6/n=8/resp16384/2-epoch/test_freq25.

## Decision (plan §F)
SURPASS ESTABLISHED iff `mean_D − mean_dense > pooled-SE (0.020)`; PARITY iff `|mean_D − mean_dense| ≤ 0.020`.

## Note on cost (one box, sequential)
~5 new dense/B2 runs + (3 winner if D passes) × ~2.5h each = a long tail (~12-20h). Prefer sequential (Cell D peaked 31.8GB transient — 2-per-box would OOM). Keep launchers staged for gap-free handoff; finalize concrete launchers when Cell D's val@50 is known.
