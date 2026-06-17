# Post-Experiment Summary Plan

Use this instead of the deleted execution plans. Those plans were run artifacts;
this file is the compact handoff for future planning.

## Default Method

| item | value |
|---|---|
| method | **B2 `delayed_ef` error feedback** |
| codec | PowerSGD `r=77`, activation basis |
| anchor | owns `Q`, paired replay, CPU snapshot |
| cadence | `delay_K=5` |
| merger | `G_corr = G_comp + lambda * (M_rep - G_comp_ring)` |
| `lambda` | `1.0` |
| default `beta_anc` | `0.0` |
| communication | bytes ratio about `0.0505` |

B2 is the confirmed comm-efficient reference: dense-parity within eval noise, no
length ignition, and no promoted successor.

## Tested Knobs

| knob family | tested values | keep in mind |
|---|---|---|
| `beta_anc` | `0`, `0.25`, `0.5`, `0.75`, `1` | `0.5` is the nominal best draw, but inside noise; default stays `0` |
| signed EMA | `alpha=0.5` | the only signed-EMA point worth comparing; not promoted over B2 |
| delta momentum | `mu=0.5`, `mu=0.9` | null/regress |
| adaptive lambda | ratio/cosine `k=1` | null |
| perturbation | `sigma=0.01` | null |
| sub-basis | rank-2 tail variants | early boost only, no surpass |

## Planning Rule

New experiments should start from **B2 unchanged** and vary one knob. Do not
rebuild old plan files or import invalid anchor-gradient claims. If a future run
uses beta=0.5, label it as a nominal tie candidate, not as the default.
