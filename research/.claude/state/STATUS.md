# Research Status — 2026-06-01T21:10:00+10:00 (complete: runs + theory report + cleanup)

## Compute: ALL TORN DOWN — $0/hr. No live boxes.

## Headline result (masked+clean@20 GRPO vs dense, GSM8K vs Big-Math)
The activation mask is **not a policy** — it's a biased, high-variance **estimator of the true GRPO
gradient** on the unmasked policy; the deployed/validated policy is always the unmasked weights. The
20M× train-inference ppl gap is a diagnostic that never enters the loss; learning needs only positive
projection of the masked gradient on the true gradient. `clean@20` = error-feedback/Local-SGD re-sync
(clean-resettable sawtooth). Whether it learns is an **SNR question**:

| Qwen2.5-1.5B | base (no RL, \boxed) | masked+clean@20 | dense |
|---|---|---|---|
| **GSM8K** | **0.715** | **0.735** (EXP-17) | 0.741 (EXP-16) |
| **Big-Math** | **0.480** | **~0.55 flat** (EXP-19) | **~0.59–0.61** (EXP-20) |

GSM8K is *easy* (base 0.715) → RL just elicits → 95%-compressed gradient reaches dense parity.
Big-Math is *hard* (base 0.48) → dense learns +0.06, the lossy masked gradient stalls (g_true small/
sparse, drowned by the mask's fixed bias+noise floor). Headroom exists (dense finds it) → the stall is
a gradient-fidelity limit, not a missing ceiling. Honest alt: 2 of 7 mask boundaries (L18,L21) coincide
with the Qwen memorization circuit → may be retrieval not reasoning (arXiv 2601.11061). Decisive next
experiment: **p-sweep** (does lowering p unlock Big-Math?).

## Deliverables (all committed + pushed, vast-ai-workload @ ae5769d1a, remote↔local 0/0)
- `findings/theory/REPORT.md` — operator-facing synthesis (the goal).
- `findings/theory/{theory,literature,empirical_check,base_capability_eval,CONTEXT}.md` — supporting.
- `runs/SUMMARY.md` — Big-Math added to inventory; outdated "clean_cadence unsustainable" claim corrected.
- `scripts/{bigmath_dapo,base_capability_eval,verify_bigmath_reward,wandb_rollout_corr}.py`.

## Cleanup (operator-directed) — DONE
- ✅ EXP-20 stopped + captured; base-capability eval run before teardown.
- ✅ Box i_38877541 torn down ($0/hr).
- ✅ Big-Math in inventory (runs/SUMMARY.md).
- ✅ Git synced: remote↔local 0/0; all work committed + pushed.
- ✅ Branch cleanup: exp/16 (already auto-deleted on PR#10 merge) + exp/17 removed from origin; local
  exp/17 + worktree-agent-* + stale worktree deleted. Branches now: local main + vast-ai-workload only;
  no exp/* on origin.
- ⏳ Team cleanup: shutdown requests sent to theorist/lit-scout/empiricist; TeamDelete after they exit.

## Runs ledger (final)
EXP-16 TORN_DOWN(PASS) · EXP-17 COMPLETE(PASS, GSM8K parity) · EXP-18 SUPERSEDED(confounded reward) ·
EXP-19 COMPLETE(Big-Math masked, flat) · EXP-20 COMPLETE(Big-Math dense, modest climb). All on the
since-destroyed i_38877541.

## Notes
- Kill switch clear. gh default: shamanez/verl-compression-research. Code PRs → shamanez/verl base vast-ai-workload.
- verl/ `math_bigmath` route (@265fca825) is UNUSED + buggy (pred=None crashes val) — documented, kept; do not route through it. Use `DigitalLearningGmbH/MATH-lighteval` → math_reward for \boxed prompts.
- EXP-19 step-50 checkpoint discarded with the box (operator: "enough training").
