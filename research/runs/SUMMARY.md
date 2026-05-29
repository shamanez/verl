# Research runs — summary

Permanent references + a folded history of the experiments whose heavy
artifacts have been pruned to keep the repo lean.

## Permanent reference runs

| id | what | result | dir | launcher |
|---|---|---|---|---|
| **baseline** | Dense GRPO (the control), Qwen2.5-1.5B-Instruct on GSM8K — verl unmodified, **no-KL no-entropy** (pg_loss only). EXP-14 reconfirmed no-KL dense learns cleanly (val 0.083 → 0.721 in 10 steps); launcher standardized to no-KL. | `val/test_score` 0.087 → 0.789 over 100 steps, 4×H200 | `runs/baseline/` | `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` |
| **communication-baseline** | Comm-eff **smoke-scale** reference (historical, old config): PRF mask `p=0.9` + anchor (cadence=5/delay=5) + two-sided Tikhonov spectral (`α=0.5, τ=0.01, β_anc=0.9`), no-KL. ⚠️ At paper scale **EXP-14 (#14)** showed the masked path does NOT yet learn: step-1 grad_norm explosion (~771) = the mask's **magnitude collapse** (h*mask, no rescale, RMS→0.32×), tamed by inverted-dropout **`rescale`** (771→1.5) but val stays flat; naive `clean_cadence` is unsustainable (PPO `pg_clipfrac` saturates 0.26→0.44). Launcher now **DEFAULTS to mask + rescale + per-channel** (anchor/spectral OFF, cadence off) as the #15 mask-rate-sweep starting point. | PASS 20-step smoke (old config); paper-scale learning unresolved → #15 | `runs/communication-baseline/` | `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh` |

## Comm-eff knob surface (EXP-14, in `vast_comm_eff_baseline_*.sh`)

All independently env-toggleable; defaults = the mask-only comm-eff baseline.

| knob | default | meaning |
|---|---|---|
| `COMM_EFF_ENABLED` | true | master switch (false ⇒ byte-identical dense) |
| `COMM_EFF_MASK_ENABLED` | true | activation mask on boundary blocks |
| `COMM_EFF_MASK_P` | 0.9 | masked fraction (sweep 0.9→0.5→0.1, #15) |
| `COMM_EFF_MASK_GRANULARITY` | channel | per-channel (packing-invariant ⇒ cross-pass IS consistency) vs `element` (legacy) |
| `COMM_EFF_MASK_RESCALE` | true | inverted-dropout `h*mask/(1-p)` — tames magnitude-collapse grad_norm (NOT a learning fix alone) |
| `COMM_EFF_CLEAN_CADENCE` | 0 (OFF) | naive periodic full-(unmasked)-grad step — **unsustainable** (PPO clip saturation), opt-in only |
| `COMM_EFF_ANCHOR_ENABLED` | false | K-stale anchor circuit (re-enable later; ~3 GB clone) |
| `COMM_EFF_SPECTRAL_ENABLED` | false | two-sided Tikhonov spectral correction (re-enable later) |

## Folded history (artifacts pruned, durable record below)

The communication-efficient implementation arrived through a sequence of
incremental experiments, each merged to `vast-ai-workload` via its own PR.
The two permanent baselines above subsume the result; the table below is
the audit trail.

| what | result | PR |
|---|---|---|
| `comm_eff` no-op scaffolding (config group + disabled-by-default integration hooks) | dense parity validated — `comm_eff.enabled=false` is bit-identical to unmodified verl | #1 |
| Actor-only PRF activation masking (in-graph `h*mask`, no `1/(1-p)` rescale) | mask_ratio tracks configured `p` (p95→0.950, p90→0.900, ±0.02); confined to actor-train path; grads finite, no NaN/Inf | #2 |
| Mask contamination guard (explicit `path_tag` + assert + per-path counters + checkpoint mask-free) | per-path counters train>0 / all-other-paths=0; 35 unit tests including 1e-6 logprob equality + checkpoint guard | #3 |
| Spectral correction filter (anchor-EMA → full thin SVD → Tikhonov → two-sided projection → α-blend) + FSDP gradient-application-point discovery | correction applied AFTER FSDP all-reduce / BEFORE grad clip; FSDP1 with `use_orig_params=true` surfaces full 2D Tensor (not DTensor/FlatParameter); per-target `rel_change` in (0,1]; `α=1.0` is exact no-op | #4 |
| Anchor backward graph isolation (hookless clone, no FSDP wrap, K-stale snapshot) | resolves the FSDP1 `_post_backward_hook` collision a second backward on the live wrapped module would hit; anchor produces full gradient without re-triggering FSDP all-reduce | #5 |
| Mask extension to `compute_log_prob` (`mask_recompute=true`) — the mask now fires on BOTH gradient-feeding forwards | smoke PASS at p=0.9 / α=0.5 / τ=0.01 / β_anc=0.9; +82% second-half reward; all 13 success criteria green | #6 |
| Paper-scale dry run + the two conceptual notes (anchor memory cost + fast-circuit vs anchor-pass) | revealed the grad-norm + entropy-collapse symptoms now queued for investigation in `notes/investigation-prompt-grad-norm.md`; no algorithmic regression — symptoms classed under variance amplification + IS / FSDP audit hypotheses | #7 |
| **EXP-14** — paper-scale grad_norm explosion: peel-and-fix diagnosis (GitHub #14, closed) | Root cause = mask **magnitude collapse** (`h*mask` no rescale, RMS→0.32×) → step-1 grad_norm ~771. `rescale` (inverted-dropout `h*mask/(1-p)`) tames it 771→1.5 but masked path does NOT learn (val flat at p=0.9); `consistent_across_forwards` refuted; naive `clean_cadence` unsustainable (PPO `pg_clipfrac` saturates 0.26→0.44). Added env-toggleable knobs (clean_cadence / rescale / granularity[**channel** default] / consistent / mask.seed); anchor + spectral default OFF. Open → #15 (mask-rate sweep; bar = stable low pg_clipfrac + sustained val/score). | #8 |

## Implementation locus

The verl implementation lives on `vast-ai-workload`:
- `verl/workers/config/comm_eff.py` — Hydra config schema
- `verl/workers/comm_eff/{state.py, activation_mask.py, anchor.py, spectral_filter.py}` — runtime
- `verl/workers/engine_workers.py` — `compute_log_prob` `mask_active` stamp
- `verl/workers/engine/fsdp/transformer_impl.py` — `_comm_eff_mask_active` gating
- `tests/workers/comm_eff/` — CPU unit tests

## Conceptual notes

- `notes/anchor-memory-cost.md` — why the anchor clone takes ~3 GB
- `notes/fast-circuit-vs-anchor-pass.md` — which of the 5 GRPO forwards get masked
- `notes/investigation-prompt-grad-norm.md` — the investigation issue draft

## Carryover follow-ups

- Launcher `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
  inherits a `done.flag` path that fails under `SAVE_FREQ=-1`, aborting
  multi-cell smoke chains under `set -e`. A real fix
  (`$EXPERIMENT_NAME` + `mkdir -p`) belongs in the launcher.
- Plan templates grep for `val/test_score` but verl emits
  `val-core/openai/gsm8k/acc/mean@1` — update plan templates accordingly.
- `TOTAL_EPOCHS=1 × 7473 / 128 = 58 batches per epoch` is the paper-scale
  step ceiling on the current dataset; use `TOTAL_EPOCHS=2` to reach a
  full 100-step horizon.
