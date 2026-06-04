## §9 RESOLVED — Dense baseline results (`ce_dense_50s_gsm8k`, WandB [`5e2jpho9`](https://wandb.ai/shamanework-pl/verl_compression_research/runs/5e2jpho9))

The same-config **dense control** (comm-eff OFF — byte-identical dense GRPO — identical lr/batch/2-epoch surface, `test_freq=10`) completed all 50 steps cleanly (no NaN/OOM/stall; all `comm_eff/*` counters = 0). **`dense@50 = 0.7536`.**

Full dense val trajectory (`val-core/openai/gsm8k/acc/mean@1`):

| step | 0 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| **dense** | 0.0849 | **0.7324** | 0.7377 | 0.7415 | 0.7483 | **0.7536** |

(train reward `critic/score/mean`: 0.085 → **0.737 by step 10** → 0.776 @20 → 0.840 @48 — rapid early convergence, then a slow grind.)

---

### Comparison 1 — dense@10 vs compressed@50: **"10 full grads ≈ the ceiling" CONFIRMED**
`dense@10 = 0.7324` ≈ the compressed arms' **val@50** (mask 0.7384 / r=77 0.7415 / r=102 0.7437). **Ten consecutive full-rank gradient steps already reach the compressed ceiling**, then dense only creeps +2.1 pp over the next 40 steps.

**But this does NOT collapse to "the compressed 0.74 is just its 10 clean steps"** (the §4 nuance, now with the dense number): the compressed arms' 10 clean steps are *spread across steps 5–50 and interleaved* with 40 compressed steps — they are **not** 10 consecutive full steps. At step 10 the compressed arms sit at train-reward ~0.31, nowhere near 0.73. And §4 showed compressed steps carry **57–95%** of the within-run climb. Both paths reach ~0.74 — **dense via ~10 full steps; compressed via compressed-steps-doing-the-work + periodic flushes.**

### Comparison 2 — dense@50 vs compressed@50: a **small but consistent ~1–1.5 pp compression tax** (not quite "free")

| | dense | r=102 | r=77 | mask p=0.95 |
|---|---|---|---|---|
| val@50 | **0.7536** | 0.7437 | 0.7415 | 0.7384 |
| gap vs dense | — | **−0.0099** | **−0.0121** | **−0.0152** |

Dense sits **~1.0–1.5 pp above every compressed arm**, ordered sensibly (r=102 closest, mask furthest). That gap is **~2–3× the 0.53 pp inter-codec spread**, so it reads as a **real (if small) compression cost, not "accuracy-free."** This *sharpens* [#22](https://github.com/shamanez/verl-compression-research/issues/22): the gap is exactly what error feedback / a stale-tolerant correction would target — and it's measured **with the fresh `clean@5` crutch in place**, so the realistic (stale / no-clean) gap is plausibly *larger*, not smaller. (Single seed — treat the magnitude as directional, but all three arms land below dense, consistently.)

### Comparison 3 — shape: dense preserves the diminishing-returns curve
dense post-10 slope ≈ **+0.0005/step** (0.732 → 0.754 over steps 10–50) — the same steep-to-~step-10 then-flat shape as the compressed arms (val@25→50 ≈ +0.0007/step). **Compression preserves the learning *dynamics*, not just the endpoint.**

---

**Net.** dense@10 ≈ compressed@50 (≈10 full grads suffice to reach the plateau); dense@50 ≈ **+1–1.5 pp** over the compressed arms (a small, consistent tax); the trajectory *shape* is preserved. The within-run decomposition (§4 — compressed steps carry the learning) **stands**; the dense ceiling adds that the compressed arms pay a **modest, error-feedback-addressable cost** — and that this whole comparison was run with the unrealistic fresh-clean-step crutch (see [#22](https://github.com/shamanez/verl-compression-research/issues/22) for why that matters and how to test the real regime).

*Box `39409362` torn down on completion; artifacts in `research/runs/EXP-20/ce_dense_50s_gsm8k.log`.*
