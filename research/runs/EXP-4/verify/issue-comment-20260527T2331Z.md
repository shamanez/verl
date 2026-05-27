## ✅ EXP-4 plan-verify: PASS — cleared to proceed (operator-approved)

The operator reviewed codex's plan-level structural feedback and judged the **bit-for-bit / SHA-256 hardening out of scope for this M2 disabled-scaffolding parity smoke**. The plan is reverted to its original rel-tol parity-smoke form and is **approved to proceed to implementation** — `status:approved`, cleared for `experiment-runner`.

**`VERIFY: PASS`** (operator-cleared). No further **plan-level** codex verification is required for EXP-4; the orchestrator treats this as VERIFIED and dispatches the runner on the next session.

Codex's raw advisory output is preserved in `runs/EXP-4/verify/` for the record (CONCERNS on the hardened rev-2, FAIL on rev-1). Code-level verification of the eventual `exp/4-commeff-noop` diff remains a separate later gate and is **not** waived here.
