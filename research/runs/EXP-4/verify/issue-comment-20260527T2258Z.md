**[orchestrator] EXP-4 demoted `status:approved` → `status:planned`** — `codex-bridge --mode=verify` (pre-implementation structural gate) returned **`VERIFY: FAIL`** before any Vast.ai launch, so **no GPU was spent**. The human gate re-opens: tighten the parity-smoke falsification method per the critique below, then flip back to `status:approved` to re-enter the verify → run pipeline. (Genuine verdict — wrapper exit 0, not a timeout/broker-death.)

Suggested remediation, distilled from the critique:
- Replace `rel-tol=1e-4` on `actor/{pg_loss,kl,entropy}` with **exact/bitwise** comparison of the deterministic scalar logs against the unmodified-launcher reference.
- Add ≥1 **stronger state comparison vs the unmodified reference** — e.g. `actor/grad_norm`, a parameter-delta hash, or an optimizer/state-dict hash after the fixed steps — not just step0-vs-step2 self-comparison.
- Validate the 3 counters' "absent ⇒ 0" with an **explicit disabled marker + initialized-zero counters**, since the counters are blind to perturbations outside mask/anchor/spectral ops.
- Note that the A-vs-B cell check (explicit-disabled vs no-override) only proves the two configs agree with each other, **not** that either equals dense — the dense reference comparison is the load-bearing check.

---

# Codex Verify — 2026-05-27T22:58:55+10:00
VERIFY: FAIL

The disabled-by-default guard design is sound in principle: short-circuiting before hook registration, state/buffer allocation, collectives, and RNG use is the right scaffold for a dense-path no-op. But the falsification method cannot establish the stated bit-for-bit hypothesis. A `rel-tol=1e-4` check on only `actor/pg_loss`, `actor/kl`, and `actor/entropy` would explicitly allow real numerical perturbations to pass, and it may miss RNG-state changes if construction/import order is later re-seeded or if nondeterminism masks the effect. For a bit-for-bit no-op, require exact checks for deterministic scalar logs and at least one stronger state comparison against the unmodified reference, such as `grad_norm`, parameter-delta hash, or optimizer/state-dict hash after fixed steps. The counters are useful but insufficient because perturbations outside mask/anchor/spectral ops remain invisible, and "counter absent" should not pass unless absence is itself validated by an explicit disabled marker plus initialized zero counters. The smoke shape likely exercises the named actor path, but the A-vs-B config check only proves default and explicit-disabled agree, not that either equals dense. As written, the plan risks spending GPU time on a parity smoke that cannot prove the claimed no-op property.
