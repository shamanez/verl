# EXP-6 live checkpoint leakage scan — 2026-05-28T03:21+10:00

Target: `checkpoints/.../m2-mask-invariants/global_step_2/actor` (saved by the
mask_on cell, comm_eff.mask.enabled=true, p=0.95) on instance 38107546.

Scan: loaded all 4 FSDP model shards (`model_world_size_4_rank_0..3.pt`) on CPU
and searched every state_dict key for `comm_eff | mask_applications | path_tag |
anchor | spectral`.

Result: **shards scanned: 4 · LEAKED KEYS: NONE (clean)**

=> The mask-on checkpoint's actor weights carry no comm_eff/mask/anchor/spectral
state. Closes the live half of the criterion "checkpoint contains no
comm_eff/mask tensors". The reload-bit-identity half is covered by the unit
tests `test_checkpoint_guard_passes_on_clean_state_dict` +
`test_checkpoint_guard_rejects_leaked_comm_eff_state` (35 passed,
runs/EXP-6/verify/unit_tests.log).
