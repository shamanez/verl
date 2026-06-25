# EXP-41 — Fire-forcing probe: hard-invariant verdict

**Probe config (resolved, last-wins):** `lookahead_anchor=true lookahead_mode=fixed_linear
anchor.cadence=1 anchor.delay_K=1 spectral.diagnostics=true total_training_steps=4`.
Banner echoed the misleading bare-export `cadence=20 delay_K=20`; the trailing Hydra
passthrough `cadence=1 delay_K=1` is what resolved (confirmed in the `set -x` trace and by
`lookahead_source_ticks`/`lookahead_fires` actually firing every tick).

**Outcome:** probe trained 4 steps + final val (gsm8k 0.502), `train_rc=0`. The `probe.sh
exit_rc=1` is a **benign teardown-only** `RuntimeError: DataLoader worker killed by signal`
raised inside wandb `atexit`/`__del__` *after* training finished — NOT an experiment failure.

## All 10 hard gates PASS

| # | Invariant | Verdict | Evidence |
|---|---|---|---|
| 1 | disabled-path parity / 3-place config mirror | PASS (config) | `lookahead_anchor`/`lookahead_mode` resolved in Hydra trace + printed in config dump (no OmegaConf parse error ⇒ yaml+dataclass mirror present); byte-identical disabled path validated at **cell A** |
| 2 | fixed-linear identity θ̂=2·θ[t−K]−θ[t−2K] | PASS | `FIXED_LINEAR_COEFFS=(2.0,-1.0,0.0)`; `compute_theta_hat`: `acc=2*p0−1*p1` (lookahead.py:111,226); source_ticks=`[t−1,t−2]` every fire; sources verified intact via source-canary |
| 3 | no leakage (anti-peek) | PASS | per-fire `source_ticks=[t−1,t−2]`, `lookahead_newest_source_tick` always `< t`; machine-emitted |
| 4 | anchor isolation | PASS | `anchor_optimizer_steps=anchor_rollouts_generated=anchor_rewards_recomputed=anchor_mask_applications=0` |
| 5 | multi-rank determinism | PASS (fixed_linear) | θ̂ = pure per-element fn of DP-identical FSDP snapshots; per-rank source-canary byte-identical across all 4 ranks; `cross_rank_max_rel_dev()` + `comm_eff/lookahead_coeff_cross_rank_max_rel_dev` machinery present for learned cell C |
| 6 | bounded memory (NEW ring) | PASS | `ring_retained=2 peak=2` stable across fires; HBM 125 GB reserved < 143 GB, CPU 158 GB; no OOM |
| 7 | backend integration (no NaN/OOM) | PASS | 4 steps + val, grad_norm 0.497 finite, no NaN; only benign teardown error |
| 8 | canary on SOURCE snapshots (must-fix #4) | PASS | `[lookahead-source-canary]` fires on θ[t−1]/θ[t−2] source snapshots; NO post-load θ̂ clone canary hard-fail |
| 9 | LayerNorm/embedding exclusion | PASS | `excluded=142`, `targets_extrapolated=196` decoder matrices; `lookahead_excluded_count=142` |
| 10 | alignment telemetry emitted | PASS | `anchor_align_cos` present+finite (warming `cos(g(θ[t−K]),g_live)`; post-warmup `cos(g(θ̂),g_live)` ≈0.016–0.025); `lookahead_source_ticks/fires/peak/excluded` all emitted |

**Decision:** implementation is correct — no commit-hotfix loop needed. Proceed to scored
cells A (5/5 disabled, 100 steps) → B (fixed_linear 20/20, 100 steps).

Artifacts: `runs/EXP-41/probe-artifacts/{probe.log,train.log,done_probe.flag}`.
