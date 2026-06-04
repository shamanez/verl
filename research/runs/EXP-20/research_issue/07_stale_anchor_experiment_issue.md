## Summary

An additional, runnable test for EXP-20 (#21) that makes the periodic full-gradient re-anchor **realistic for decentralized pipeline-parallel training**: instead of a *fresh* full-gradient "clean step" (which a slow community-GPU interconnect cannot deliver — see #22), the full-gradient re-anchor is **stale** (computed from weights `θ_{t−5}`) and **combined with the live PowerSGD-compressed gradient**. Does PowerSGD-GRPO still reach the fresh-clean result when its full refresh is stale?

This is the concrete first experiment from the #22 staleness direction, applied to the PowerSGD codec, reusing existing knobs.

## Background (settled, do not re-litigate)

- EXP-20: PowerSGD r=77 ≥ PRF mask at equal budget (val@50 **0.7415** vs 0.7384); dense ceiling **0.7536**.
- The compressed gradient does **real** optimization — 57–95% of the within-run reward gain is on the compressed (non-clean) steps (#21 §4), confirmed on the **training-reward** curve (validation is saturated and not discriminative here).
- But EXP-20 leaned on a **fresh** full-gradient clean step every 5 steps. In real async PP that refresh is (a) the expensive full-`H` transfer compression avoids, and (b) **stale** — gradients crossing stage boundaries are delayed. This test isolates (b).

## Mechanism already in the codebase (reuse — do not rebuild)

- `comm_eff.anchor.delay_K` — runs a full GRPO forward/backward from a `delay_K`-stale weight snapshot → `G_anchor` → EMA `M_anchor`. **Set `delay_K=5`** = "full gradient from `θ_{t−5}`".
- `comm_eff.spectral.correction_mode` combines `M_anchor` (stale full) with the live post-codec gradient `G` and writes the result to `p.grad` **before** the AdamW step (optimizer-agnostic):
  - `inject`: `G_corr = G + γ·scale·(M_anchor − proj_G(M_anchor))` — ADD the missing (off-subspace) component of the stale full gradient. *(= delayed error feedback.)*
  - `blend`: `G_corr = (1−η)·G + η·scale·M_anchor` — convex combine toward the stale full gradient.
- Prior anchor+spectral tests were on the **mask** codec only (`reweight` inert; `blend` live) and on the now-fixed buggy anchor — **never on PowerSGD**.

## Hypothesis

A `delay_K=5` **stale** full-gradient re-anchor (via `inject` or `blend`), with the fresh clean step OFF, recovers most of the fresh-clean benefit for PowerSGD:
> PowerSGD + stale-anchor@5  ≈  PowerSGD + fresh-clean@5 (0.7415)  ≫  PowerSGD + no refresh.

If true → the method survives the realistic (stale) refresh constraint, not just the unrealistic fresh one.

## Arms (codec fixed = PowerSGD r=77; the only axis is the refresh)

| arm | refresh mechanism | knobs |
|---|---|---|
| **A0** (ref, done) | fresh clean@5 | EXP-20 r=77 = **0.7415** |
| **A1** (floor) | none | `clean_cadence=0`, anchor OFF |
| **A2** | stale full, **inject** | `anchor.enabled=true delay_K=5 cadence=5`, `spectral.enabled=true correction_mode=inject`, `clean_cadence=0` |
| **A3** | stale full, **blend** | as A2 with `correction_mode=blend` (`blend_eta` default) |
| dense (ref) | — | ceiling **0.7536** |

A2/A3 are the test (combine compressed + stale full); A1 is the no-refresh floor; A0/dense bracket the target/ceiling.

## Fixed control surface

Per `research/runs/FIXED_CONTROL_SURFACE.md` — Qwen2.5-1.5B-Instruct, GSM8K, vanilla GRPO no-KL/no-entropy, lr 1e-6, train_batch 128, ppo_mini 64, rollout.n 8, max_response 16384, total_training_steps 50, total_epochs 2, `test_freq=10`, 4×H200. `compression_type=powersgd`, `powersgd.rank=77`, `sync_basis=true`. The **only** variable is the refresh mechanism (the table above).

## Success / falsification

- **Success:** A2 or A3 val@50 within ~1 pt of A0 (0.7415) **and** clearly above the A1 floor ⇒ a stale full-gradient re-anchor substitutes for the fresh clean step ⇒ the method is viable under realistic async-PP staleness.
- **Falsified:** A2/A3 ≈ A1 (no-refresh floor) ⇒ staleness breaks the re-anchor; the fresh clean step was load-bearing in a way `delay_K=5` already kills (⇒ need error feedback instead, or a smaller delay).
- Judge on **both** curves: val@50 (headline) **and** the training-reward trajectory + the clean-vs-compressed-style decomposition (val saturates, so train-reward is the sensitive signal).

## First gate (before the full arms)

The inject/blend math is codec-agnostic (operates on the post-codec gradient), so it *should* compose with PowerSGD — but the anchor+spectral circuit was only ever exercised with `prf_mask`. **On-box smoke (≤2 steps):** with `compression_type=powersgd` + anchor + spectral, verify the anchor forward fires, `M_anchor` populates, the correction applies to the PowerSGD-path gradient, and `q_cond≈1` / `q_cross_rank=0` / no NaN/OOM still hold. Patch the integration if needed (exp/* branch). Only then run A1–A3.

## Compute

1×(4×H200) (chain → 8×H100), `max_dph=24`. ~3 arms × ~2 h + the smoke. One warm box, reused across arms.

## Out of scope (keep this experiment minimal)

`delay_K` sweep (>5), error-feedback codec, rank sweep, harder task — those are #22 follow-ups, not this issue.
