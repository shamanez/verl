# Research runs — summary

Permanent references + a folded history of the experiments whose heavy
artifacts have been pruned to keep the repo lean.

## The two reference points

Run artifacts are pruned (de-bloated). The durable record is here + git
history + the merged PRs on `vast-ai-workload`.

### Baseline = dense GRPO == comm-eff OFF

There is **one** baseline: the dense control, which *is* the comm-eff launcher
with the master switch off. `COMM_EFF_ENABLED=false` is byte-identical to
unmodified verl (validated PR #1). No-KL, no-entropy (pg_loss only).

- **Proof the codebase trains dense-perfect:** EXP-14 `test1_cellA`
  (comm-eff OFF), **10 steps**, 4×H200 — `val/test_score` **0.083 → 0.721**,
  clean monotone improvement. This is the dense-correctness proof; we **no
  longer keep the old 100-step baseline run** (artifacts pruned). 10 steps is
  enough to see clear learning, so it's the standing control horizon.
- Launcher: `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
  with `COMM_EFF_ENABLED=false`, or the convenience dense launcher
  `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` (identical
  no-KL objective).

### Comm-eff method = implementation correct, masking still under test

- **Implementation is correct.** comm-eff OFF ⇒ byte-identical dense (PR #1);
  with masking ON the mask fires on exactly the gradient-feeding forwards,
  `mask_ratio` tracks `p`, grads stay finite (PRs #2–#6, 127 unit tests).
- **The masking side still needs a lot of testing.** EXP-14 (#14) showed that
  at paper scale pure masked GRPO does **NOT yet learn**: mask magnitude
  collapse explodes grad_norm (~771); inverted-dropout `rescale` tames it
  (→1.5) but val stays flat; naive `clean_cadence` is unsustainable (PPO
  `pg_clipfrac` saturates 0.26→0.44). Open question → **#15**: can masked GRPO
  learn at all, and at what mask rate? Default config = mask + rescale +
  per-channel — the #15 mask-rate-sweep (p=0.9→0.5→0.1) starting point.
- **Anchor + spectral correction are layered fixes for LATER** — the
  corrections you bring in *only once the plain masked path is understood*
  (i.e. when masking alone is not enough). They **default OFF and must stay
  OFF to start** (`COMM_EFF_ANCHOR_ENABLED=false`,
  `COMM_EFF_SPECTRAL_ENABLED=false`).

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
The reference points above subsume the result; the table below is the audit
trail. The old dense **100-step baseline** run (`val/test_score`
0.087→0.789) and the **communication-baseline** smoke run (old
mask+anchor+spectral config, 20-step PASS) have both had their artifacts +
plan + finding pruned — the dense proof is now EXP-14 `test1_cellA` (above)
and the comm-eff result is EXP-14 (#14) / row below.

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
