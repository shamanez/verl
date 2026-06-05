# Research Status — 2026-06-05

## Active
No active experiments. 0 live Vast.ai instances.

## Next
- **issue #25** (`research:claim`, M6 — GitHub `shamanez/verl-compression-research#25`, Linear RES-133): make the PowerSGD communication-efficient GRPO trainer **realistic** via the anchor circuit. (1) Fix the anchor `M_anchor` update — globally DP-reduce the raw anchor gradient + cover all compressed targets. (2) Move the projection basis `Q` to the anchor (computed from stale, delayed weights) and broadcast it to the fast net. (3) Add the signed-EMA gradient merger `α·G + (1−α)·|G|·sign(M_anchor)`. The anchor refresh replaces any periodic dense step. Awaiting `research-planner` (no plan file yet) → human `status:approved`.
- **issue #24** (`research:claim`, M6): error-feedback on the PowerSGD residual + basis-aligned anchor. Blocked on #25.

## Hyperparameters
- **Core (held constant):** Qwen2.5-1.5B-Instruct + GSM8K, vanilla GRPO (no-KL/no-entropy), lr 1e-6, train_batch 128, ppo_mini 64, micro 1, rollout.n 8, rollout TP 2, max_prompt 1024, max_response 16384, seed 0, val_before_train True.
- **Run control (realistic setting):** total_training_steps **50** (extend to **100** once 50 trains cleanly); validation every **25** steps; anchor refresh every **5** steps from stale (delayed) weights; **no periodic dense clean step** (the anchor refresh is the realistic substitute).
- **Codec:** PowerSGD activation compression (the chosen method).
- Full surface: `runs/FIXED_CONTROL_SURFACE.md` + the launcher `${VAR:-default}` defaults (`examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`) + `project.yaml.fixed_control_surface`.

## Budget
$0/hr (0 live instances).
