# Research Runs Summary

Durable record after de-bloat. Full run directories are intentionally gone; use
this file, `research/LOG.md`, W&B, git history, and merged code for provenance.

## Evidence Boundary

**The paired-replay fix is the validity boundary for anchor-circuit claims.**
Current comm-efficient PP/GRPO claims use only valid-M measurements where `M_rep`
is paired with the retained fast gradient at the same `(batch, theta)`. The old
clean-step result is kept only as clean-step history and is not a current floor
or anchor-gradient result.

## Current Best Method

| rank | method | status | best read |
|---|---|---|---|
| 1 | **B2 `delayed_ef` error feedback** | **confirmed SOTA comm-eff method** | val@50 **0.7528** in the first valid-M proof; reproduced in the 0.735-0.754 band |
| 2 | `signed_ema` | candidate/legacy merger, not promoted | alpha=0.5 is the only signed-EMA setting worth tracking; no durable post-#29 verdict promoted it |
| control | no merger, PowerSGD+Q | realistic floor | val@50 **0.6300** |
| dense | full gradient | reference band | val@50 about **0.75-0.78** |

**Interpretation:** B2 reaches dense parity within single-draw eval noise at about
5% fast-path gradient communication. No tested anchor-usage or beta lever gives a
credible dense surpass.

## Canonical B2 Settings

| knob | value |
|---|---|
| codec | PowerSGD, rank `r=77`, activation basis |
| anchor | enabled, owns `Q`, full target coverage |
| replay fix | `replay_paired_batch=true`, `snapshot_device=cpu` |
| cadence | `cadence=delay_K=5` |
| merger | `correction_mode=delayed_ef` |
| residual | `G_corr = G_comp + lambda * (M_rep - G_comp_ring)` |
| `lambda` | `1.0` |
| `beta_anc` default | `0.0` |
| clean step | `clean_cadence=0` |
| comm | bytes ratio about `0.0504-0.0506` |

## Parameters Tested

| axis | values tested | result |
|---|---|---|
| `beta_anc` on B2 | `0.00`, `0.25`, `0.50`, `0.75`, `1.00` | flat for `0.00-0.75`; `1.00` cold-M collapse |
| nominal beta draw | `0.50` | highest single draw: **0.75284**, but only +0.0144 over beta=0 control, inside +/-0.024 noise |
| default beta | `0.00` | still default; beta=0.5 is a nominal tie, not a promotion |
| delta momentum | `mu=0.5`, `mu=0.9` | null/regress |
| adaptive lambda | ratio `k=1`, cosine `k=1` | null |
| perturbation | `sigma=0.01` | null |
| control variate | gated | skipped/null; covariance gate failed |
| sub-basis correction | rank-2 tail variants | early boost, no surpass |
| signed EMA | `alpha=0.5` | keep as the signed-EMA reference setting; not promoted over B2 |

## Run Index

| id | role | durable result |
|---|---|---|
| paired-replay fix | validity | paired replay + CPU snapshots made valid-M measurements possible |
| B2 proof | method proof | `delayed_ef`, lambda=1, beta=0 reached **0.7528** and became SOTA |
| anchor-usage tournament | lever sweep | L2/L3/L4/L1 levers failed to beat B2 |
| beta sweep | beta axis | beta=0.5 was nominal best draw, beta=0 remains default |

## Bottom Line

Use **B2 `delayed_ef` with lambda=1, beta=0** as the reference. It is okay to
mention beta=0.5 as the best single beta draw, but only with the caveat that it is
a noise-bounded tie and did not replace beta=0. `signed_ema alpha=0.5` is the only
signed-EMA setting worth keeping in future comparisons, but it is not the current
best method.
