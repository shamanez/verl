# ROUND A COMPLETE: no winner under the blind-committed criteria. Operator decision required, GPU idle by protocol.

Five arms, 120 steps each (a2 killed at 60 as pre-authorized), 20.1 GPU-h, about $67. Nothing is launching. The pre-registered ladder has **no matching row**, which is the registered judgment-fallback case, so this escalates rather than auto-proceeds.

## 1. The five-arm table, scored on criteria committed before a4 and a5 ran

| arm | wire | V1 drift @100-120 | V2 grad max/mean | V3 score level | gap slope 61-120 | gap level | E[rho] |
|---|---|---|---|---|---|---|---|
| dense (uncompressed) | n/a | 0.000016 | 0.07/0.05 | 0.6587 | -0.000000 | 0.0002 | 1.0000 |
| incumbent PRF exact-k | 1232 | 0.002176 | 4.17/1.65 | 0.6577 | +0.001867 | 14.2458 | 0.0021 |
| a1 1-bit SR | 2304 | 0.003887 | 0.70/0.65 | 0.6529 | +0.003265 | 13.7511 | 0.0025 |
| a2 1-bit RN | 2304 | killed@60, 6.86x, z=+15 | 9.15/8.20 | n/a | n/a | n/a | n/a |
| a3 parity hybrid | 1232.5 | **0.001616 PASS 0.74x** | pass | pass | +0.001300 | 14.9924 | 0.0017 |
| a4 PRF + CVC-CE | 1232 | 0.003955 FAIL 1.82x | pass | pass | +0.001685 | 14.2473 | 0.0025 |
| **a5 FRLR + token-IS** | **1232** | **0.004584 FAIL 2.11x** | **0.20/0.14 best** | **0.5908 FAIL** | **-0.002049 PASS** | **4.4842 PASS** | **0.3985 IN quadrant** |

**No winner.** E leaves {incumbent, a4, a5}; V1 eliminates a4 and a5. a3 dies at E on 0.5 bits, but that technicality changes nothing because a3 also fails both objective clauses, so its exclusion is overdetermined. The incumbent survives its own vetoes by construction but fails the objective at 3.7x S-bar.

**a3 and a5 are complementary failures**: a3 is safe but moves nothing; a5 moves the objective decisively and fails two vetoes.

## 2. a5 is the scientific result of the round, and it is genuinely not promotable as-is

a5 is the **only arm in the program's history to reverse the gap**: -6.18 nats from step 1, where every other compressed arm rises +0.45 to +0.77. It is the only arm to satisfy registered success criterion 1, at exact wire parity, with E[rho] 190x the incumbent inside the registered quadrant and the best gradient behaviour in the matrix.

**But its drift is real, not a measurement artifact, and it extrapolates into collapse inside round C.**

Fitting a power law to a5's reference-KL levels (0.0028 at step 24, 0.1993 at step 120) gives exponent **p = 2.65**. The model is self-consistent: it predicts a relative slope at step 120 of p/120 = 0.02209/step against 0.02300 observed, **agreeing to 4 percent**. Extrapolated:

| ref-KL | extrapolated step |
|---|---|
| 3 nats (collapse band floor) | **334** |
| 8 nats (collapse band ceiling) | **483** |
| projection at step 600 | **14.19 nats** |

The historic 3-8 nat collapse band falls squarely **inside** the registered 600-step round-C horizon.

**The drift is real.** A zero-GPU discriminator settles it. If the reference-KL rise were view contamination it would decay as the codec view converged. Instead:

| window | d(ref-KL)/dt | d(gap)/dt |
|---|---|---|
| 21-40 | +0.000225 | +0.004627 |
| 41-60 | +0.000510 | +0.001902 |
| 61-80 | +0.001640 | +0.001784 |
| 81-100 | +0.002982 | **-0.003168** |
| 101-120 | +0.004619 | **-0.007951** |

The gap **reverses sign and falls faster and faster while reference KL rises faster and faster**. They are anti-correlated in the second half. View-motion contamination is refuted; this is real policy drift.

## 3. The theory became predictive, which is the most important thing round A produced

Round A's cleanest law, from the a1/a2 single-knob factorial, is that **bias rather than magnitude gates drift** (2.7x noise-energy variation at zero bias moved drift not at all; a bias flip at identical wire moved it 6.9x, z = +15).

Token-IS at threshold 2.0 downweights about 87 percent of tokens, which is a strong **constant-direction estimator bias**. So a5's drift failure is **predicted by a law derived on entirely different arms**, and three quantitative checks agree:

- **Exponent.** Constant-direction bias gives KL growing as t^2; unbiased noise gives t^1; step size moves the coefficient, never the exponent. Observed 2.65 is bias-like.
- **Drift per unit step.** a5 takes roughly 12x smaller effective steps (grad_norm 0.121 vs the incumbent's 1.65, ESS 0.24 to 0.28) yet shows 2.11x the drift slope, i.e. about **25x the incumbent's drift per unit gradient norm**. Near-impossible under unbiased noise, near-mandatory under directional bias.
- **Sign structure.** Updates dominated by the ~13 percent agreement-region tokens explain *both* observations at once: the training view sharpens exactly where the views already agree (gap falls), while the update direction is systematically biased (drift accelerates out of proportion to step size).

This resolves the paradox I flagged earlier and could not explain: smaller steps producing more drift.

## 4. Correction to my own analysis

My discriminator script printed "H1 not excluded (they co-move)". That was wrong: it compared the **absolute value** of the gap slope, which hid a sign reversal. The signed slopes are anti-correlated and H1 is refuted. The conclusion above is the corrected one.

## 5. Recommended next step, and it is not round B

**Recommendation: one pre-registered 120-step probe cell at a widened token-IS threshold, about 4 GPU-h (about $14), before committing 28 h to rounds B and C.**

Config: a5 exactly, with the IS threshold raised one octave from 2.0 to 4.0. Three riders that fix round A's structural gaps:

1. **Save checkpoints at 0/60/120.** Round A saved none, so none of these five arms can ever be re-analysed. That is the single biggest self-inflicted limitation of the round.
2. **Log actor-vs-ref KL through the anchor's existing paired dense replay** (it already runs every 20 optimizer ticks). This gives a **true-view drift channel essentially for free** and permanently retires the codec-view ambiguity that round A could not resolve retrospectively.
3. **Run step-120 validation plus 2 OOD benchmarks** (about 1 h). This is the first time anything in this program touches capability.

**Promote-to-round-B bar, registered now, before launch:** dense-channel V1 <= 3.264e-3 AND score at 100-120 >= 0.6248 AND gap slope <= +5.0e-4 AND gap level <= 14.2458 AND wire = 1232.

**Pre-registered falsifiers.** Tunability is dead if, at a threshold lifting ESS to >= 0.5, any of: (i) score at 100-120 is still below 0.6248, meaning the deficit is FRLR-caused not IS-caused; (ii) gap slope turns positive above +5e-4, meaning the gap result was an agreement-region artifact; (iii) V1 worsens in the dense channel, meaning the bias story is wrong. Any one kills a5 as a promotable line while leaving the mechanism findings intact.

Under the bias hypothesis, widening the threshold should improve V1 **and** V3 together rather than trading them, because less clipping means less directional bias. That is a sharp, falsifiable prediction.

**Why not the alternatives.** Round B on a5 (6.7 h) puts a controller on a knob whose sign of effect on drift is untested, and automation on top of an unresolved mechanism can manufacture any outcome. Round B on a3 (6.7 h) is the worst buy available: a3 is veto-clean but moved the objective nowhere and is ineligible anyway, so it would spend 6.7 h confirming a null. Stopping now is premature by exactly one experiment, since it abandons the round's only positive result on an untested one-knob confound.

**Strongest objection to this recommendation:** it is post-hoc knob-tuning after unblinding, which is precisely what pre-registration exists to prevent. Mitigations: the bar and the falsifiers are registered above **before** launch; a negative result is still decisive because the probe doubles as the mechanism discriminator; and the validation rider folds the cheapest capability experiment into the same 4 hours.

## 6. Proposed criteria amendment, for your yes/no

Round A showed two committed bars are mis-formed, and one program-level objective is probably measuring the wrong thing.

- **V1 should be re-formed to the dense-view channel, on slope and curvature**, not a codec-view relative slope. Round A's own science says reference KL read through the codec is a codec-view metric, and rider 2 above makes the dense channel available.
- **V3 should be re-formed to a trajectory or extended-horizon parity test**, not a fixed-horizon level bar. As written it mechanically punishes an arm whose design intent is smaller effective steps, and it imports a reward-competitiveness requirement that criterion 1 never demanded, in a program whose cardinal rule ranks preservation above reward.
- **The gap-level clause should be demoted to a diagnostic.** This is the deepest point and it is substantially correct: this program's own earlier OOD work found a compressed arm matching dense on all 10 benchmarks at roughly 1000x the reference KL, with damage appearing only at collapse. So gap and drift **levels are not capability measures pre-collapse**. They are defensible as **collapse lead indicators**, which is exactly how the t^2.65 extrapolation is used above. The defensible re-anchoring is from gap level to **predicted time-to-collapse** (drift curvature in the dense channel) subject to the wire budget, with step-0 and terminal validation plus saved checkpoints made mandatory in every future cell so a capability dose-response accumulates for free.

## 7. Budget and the idle GPU

Ledger **20.1 of 100 GPU-h**, about **$67** spent. The box is **idle right now at $3.34/h** because the protocol forbids launching past an unmatched ladder row. Costs of the options: probe cell 4 h plus 1 h val (about $17); round B 6.7 h (about $22); round C 21 h plus 1 h OOD (about $74).

**Awaiting your decision. Nothing launches until then.** If you want the GPU busy immediately regardless, the one item that is useful under every branch is the OOD read on the incumbent's R2 checkpoints (about 1 h), which round C needs anyway and which ladder row 6 already authorizes as the STOP-path action.
