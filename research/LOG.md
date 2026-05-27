# Research Log (newest first)

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
