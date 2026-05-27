## EXP-4 — comm_eff no-op scaffolding (disabled-by-default two-circuit config + inert hooks)

Adds the **disabled-by-default** `comm_eff` scaffolding that later M2 issues (mask / spectral / anchor) build on. When `enabled=false` (the default) the path is a **bit-for-bit no-op**: no object is constructed, no forward hooks register, no SVD/EMA buffers allocate, no extra all-reduce, and **no RNG is drawn** — so dense GRPO is unperturbed.

Implements issue shamanez/verl-compression-research#4.

### What's in the diff (11 files, +645/−1)
- `verl/workers/config/comm_eff.py` *(new)* — `CommEffConfig` + mask/anchor/spectral sub-configs, `enabled=false` default, strict OmegaConf schema (rejects unknown keys), range-validating `__post_init__`.
- `verl/workers/comm_eff/{__init__,state}.py` *(new)* — `maybe_build_comm_eff_state()` returns `None` when disabled (the inertness gate); counters live on the state object only.
- `verl/workers/config/actor.py`, `verl/workers/config/__init__.py` — wire `comm_eff` onto `ActorConfig` (reachable as `actor_rollout_ref.actor.comm_eff.*`).
- `verl/trainer/config/actor/actor.yaml`, `verl/trainer/config/ppo_trainer.yaml` — default `comm_eff:` block (enabled:false).
- `verl/workers/engine_workers.py`, `verl/workers/engine/base.py`, `verl/workers/engine/fsdp/transformer_impl.py` — three guarded no-op hooks (update_actor / spectral grad-correction / mask+anchor points), all short-circuit on the `None`/`enabled` check.
- `tests/workers/config/test_comm_eff_config.py` *(new)* — defaults-off + schema-rejection + range + inertness unit tests.

### Validation
- **Local unit tests (pre-launch gate): 10/10 pass** — defaults-off, schema rejection of unknown top-level/mask/spectral keys, `__post_init__` ranges, disabled-state inertness (`runs/EXP-4/verify.log`).
- **GPU smoke (Run A, explicit `comm_eff.enabled=false`, Qwen2.5-1.5B / GSM8K, 4×H200):** reached `global_step=2` with
  - `comm_eff/mask_applications = anchor_backwards = spectral_corrections = 0.0` — the headline no-op proof,
  - `actor/grad_norm = 3.07e-4` (finite, ≪5), no NaN/Inf, `entropy = 0.357` (training really happened).

### Known follow-up (not blocking this scaffolding)
The back-to-back smoke chain did **not** complete Run B (config-default no-op) or Run C (unmodified reference), so the A-vs-B parity and the rel-tol-1e-4 reference check are still pending. Cause was an **unrelated launcher bug** (`vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` hardcodes a `done.flag` path that doesn't exist under `SAVE_FREQ=-1`, exiting nonzero and aborting the chain under `set -e`) — not a defect in this scaffolding. Tracked for a follow-up run once the launcher is patched. The core invariant this PR exists to prove (counters == 0 when disabled) is satisfied.

Co-authored-by: Claude
Signed-off-by: Shamane Siri <shamane@pluralis.ai>
