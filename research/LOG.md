# Research Log (newest first)

## EXP-12 · 2026-05-28T05:15:00+00:00 · M2 · PASS
REVISE child of EXP-8 — anchor backward graph isolation (cloned-no-hook module / no_sync+summon_full_params)
- hypothesis: same-process anchor refresh every cadence steps from delay_K-stale weight snapshot runs unmasked GRPO-actor-loss forward/backward on a cloned-no-hook module (detached from FSDP registration) to populate live anchor EMA/SVD and cache basis across fast mini-batches; all anchor-semantics guards upheld; cells reach global_step >= 5
- result: four on-box hot-fix iterations closed FSDP autograd-hook collision (iter01 wired missing call site; iter02 config-rebuild fallback for HF monkey-patch unpicklability; iter03 DTensor materialization via .full_tensor(); iter04 cached anchor clone + empty_cache for vLLM sleep_replicas hygiene). Both anchor-enabled cells (faithful HBM EMA + full SVD / lean CPU EMA + low-rank SVD) reached global_step:10 with all 6 anchor-semantics guards held; anchor_backwards:20.0, anchor_mask_applications:0, anchor_grad_corrected:0, anchor_rollouts_generated:0, anchor_rewards_recomputed:0, anchor_optimizer_steps:0 on both cells; within-run anchor-off regression reproduced EXP-7 spectral path (cell 2 step:5 spectral_corrections:40, no anchor activity). Criterion-13 regression test test_fsdp_anchor_backward_no_collision added.
- run dir: runs/EXP-12/
- verdict: runs/EXP-12/verdict.md

## EXP-7 · 2026-05-28T06:01:00+10:00 · M2 · PASS
M2 — spectral correction filter (paper formula) + FSDP gradient-application-point discovery
- hypothesis: spectral filter (anchor-EMA → full thin SVD → Tikhonov spectral weights → two-sided projection → alpha blend) is a no-op when alpha=1.0, equals pure two-sided Tikhonov when alpha=0, preserves shape, and is deterministic; when wired into FSDP actor path after backward and before optimizer.step(), reaches global_step=2 with finite actor/grad_norm, logs gradient representation (full Tensor/DTensor/FlatParameter) and correction point relative to FSDP reduction and clipping, with per-target ||G_proj - G_mask||/||G_mask|| in (0, 1] for alpha=0.3, and enabled=false regression matches dense no-op
- result: unit test VERIFY:PASS (alpha=1 no-op ≤1e-6, alpha=0 = pure Tikhonov, shape preserved, deterministic); FSDP discovery: FSDP1 use_orig_params=true yields full logical 2D Tensor (not DTensor/FlatParameter), correction applied AFTER FSDP gradient reduction and BEFORE grad clipping, world_size=4; spectral_corrections 8→16 (2-step isolation) / 8→80 (10-step combined mask+spectral); rel_change q≈0.0037/k≈0.646/v≈0.642/o≈0.0036 all in (0,1]; actor/grad_norm finite (54.8–300.1, no NaN/Inf); param deltas step0→step10 confirmed; disabled cell = true dense no-op (corrections=0); three fixed defects: entropy_coeff=0.001 for gradient signal, FSDP1 use_orig_params=true for unsharded matrix access, stable sha256 anchor seed for cross-rank determinism
- run dir: runs/EXP-7/
- verdict: runs/EXP-7/verdict.md

## EXP-6 · 2026-05-28T03:25:00+10:00 · M2 · PASS
M2 — mask contamination guard: invariants for rollout / old-logprob / ref-logprob / validation / checkpoint / infer_batch
- hypothesis: per-path mask counter strictly 0 on rollout, old-logprob, ref-logprob, validation, checkpoint, infer paths; strictly > 0 only on actor-train forward/backward; old/ref log-probs equal within 1e-6 regardless of mask config; validation unchanged vs masking-off; checkpoints contain no comm_eff/mask state
- result: key-prefix grep yields only `actor/comm_eff/mask_applications`; 35 unit tests pass (incl 1e-6 log-prob equality + checkpoint guard); live 2-step GRPO smoke with per-path counters train=28/all-RL-paths=0; val ran with parity within noise (0.0508 vs 0.0485-0.0652); checkpoint leak-scan clean; no NaN/Inf
- run dir: runs/EXP-6/
- verdict: runs/EXP-6/verdict.md

## EXP-5 · 2026-05-28T01:45:00+10:00 · M2 · PASS
M2 actor-only PRF activation masking smoke
- hypothesis: deterministic PRF masks applied in-graph (h*mask, no rescale) at boundary decoder layers confined to actor-train forward/backward; measured mask ratio tracks configured p within ±0.02; no NaN/Inf in losses/grads; ≥1 param changed between step 0 and 2; disabled cell regresses EXP-4 dense no-op contract
- result: p95/p90 mask_ratio 0.9498/0.8999 ± 0.0002 from configured p; all 7 boundaries [3,7,11,15,18,21,24] masked; zero mask applications on non-actor-train paths; actor/grad_norm finite on all substeps (p95: 42.15/19.86, p90: 18.11/95.18, disabled: 1.13/0.37); no NaN/Inf in any loss/grad/reward/log_prob/kl field; disabled cell all comm_eff counters 0 (dense no-op matched); tests/workers/comm_eff/test_activation_mask.py VERIFY:PASS
- run dir: runs/EXP-5/
- verdict: runs/EXP-5/verdict.md

The `log-writer` subagent prepends one entry per PASS or STOP verdict. Each entry links the experiment id, the verdict file, and the run dir for forensics. Empty until the first verdict.
