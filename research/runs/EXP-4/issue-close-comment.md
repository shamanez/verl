## EXP-4 closed — comm_eff no-op scaffolding merged ✅

The disabled-by-default `comm_eff` scaffolding is **merged into `vast-ai-workload`** via shamanez/verl#1 (merge `1f376733`, branch tip `b036c656`). Local checkout is on par with origin.

### What was validated
- **Headline no-op proof (Run A, explicit `comm_eff.enabled=false`, Qwen2.5-1.5B / GSM8K, 4×H200):** reached `global_step=2` with `comm_eff/mask_applications = anchor_backwards = spectral_corrections = **0.0**`, `actor/grad_norm = 3.07e-4` (finite, ≪5), **no NaN/Inf**, `entropy = 0.357`. Disabled ⇒ bit-for-bit inert path confirmed.
- **Local unit tests: 10/10** — defaults-off, strict-schema rejection of unknown keys, range validation, disabled-state inertness (`runs/EXP-4/verify.log`).
- Pre-launch code-level gate honored cheaply (no GPU spent on unvetted code); plan-level gate was operator-cleared.

### Known gap (deferred to follow-up — not blocking closure)
Runs **B** (config-default no-op) and **C** (unmodified reference) did **not** execute, so the A-vs-B parity and the rel-tol-1e-4 reference check are still pending. Root cause was an **unrelated launcher bug**, not the scaffolding:

> `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh:196` runs a hardcoded `touch .../qwen25_1p5b_grpo_gsm8k_baseline/done.flag` that doesn't exist under `SAVE_FREQ=-1`, exiting nonzero → `launch.sh`'s `set -e` aborted the back-to-back chain after Run A.

**Follow-up:** patch the launcher's `done.flag` to use `$EXPERIMENT_NAME` + `mkdir -p` (or guard the touch), then re-run the 3-cell smoke to complete the B-vs-A parity + rel-tol-1e-4 checks. The core invariant this issue exists to prove (counters == 0 when disabled) is satisfied.

Closed per operator decision. Instance `i_38088784` torn down after this closure.
