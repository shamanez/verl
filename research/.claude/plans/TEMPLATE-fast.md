# Plan <N> — <title>

<!-- FAST tier: single hypothesis, one launch round, ≤ ~6 cells. Target ≤ 4 KB.
     The plan LIVES IN THE GITHUB ISSUE BODY between the plan:start/plan:end
     markers (_lib.sh plan_publish installs it; plan_fetch caches it locally
     under .claude/state/plan-cache/ — gitignored).
     The yaml block below is the ONLY machine-read part (flat keys — parsed by
     skills/_lib.sh plan_field). Everything else is for humans: plain prose,
     no boilerplate, no (n/a) sections — delete what you don't need. -->

```yaml
issue: <N>
slug: <kebab-slug>            # becomes runs/<N>-<slug>/, branch exp/<N>-<slug>, WandB group
wandb_project: <N>-<kebab-slug>   # REQUIRED = run_id: every issue gets its OWN WandB project (never the shared legacy project) — runner exports PROJECT_NAME from this
title: <issue title>          # one line — SUMMARY.md rows and the run-report page read it
kind: experiment              # experiment|ablation|implementation|brainstorm|literature|analysis
tier: fast
code_change: false
target_modules: []            # verl paths, only when code_change: true
baseline_run: baseline        # the dense control (comm-eff OFF) unless stated
depends_on: []                # issue numbers that must be status:pass|stop first
milestone: none
gpu_mode: default             # 'default' = project.yaml default_compute.gpu_mode (auto = hands-off provision; ask = pause READY-FOR-GPU for an operator box). CLI --gpu beats this.
gpu_filter_chain: default     # 'default' = project.yaml ladder (1×H200 → 1×B200 → 2×H200)
max_dph: 24.0
max_gpu_hr: 24                # HARD cap — reaper enforces via the ledger row
max_parallel: 1
attach_box: none              # instance id to attach instead of provisioning
vast_account: private         # private|team
promote_launcher_as: none     # vast_*.sh name to promote on PASS, or none
iterations: 2                 # max REVISE depth on this lineage
```

## What & why
<3–6 sentences, plain language: what question this answers, why it matters for
the north-star (GOAL.md), and what we'll do differently depending on the answer.>

## Hypothesis
<ONE falsifiable sentence with numeric thresholds,
e.g. "signed_ema(α=0.25) at cadence 10/10 reaches val/score ≥ 0.72 at step 50
(dense ref ≈ 0.7657), with no NaN.">

## Cells
<Every cell name says method + key knob. Banned: c1, c2, armA-….
 Config delta = env/Hydra overrides on the canonical launcher only.>

| cell | config delta vs `vast_comm_eff_accel_base_…sh` | passes when |
|---|---|---|
| dense-control | `COMM_EFF_ENABLED=false` | val@50 ≈ 0.7657 ± noise |
| <method-knob> | `<VAR=value …>` | <numeric bar> |

## Success criteria
<Machine-checkable boxes; the analyst greps runs/<N>-<slug>/metrics/ for these.
 Default predicate applies: PASS iff all ✓; STOP if falsified/budget/depth;
 REVISE (≤ iterations) with next_actions. A clean symmetric negative is PASS.>

- [ ] every cell reaches step <S> with no NaN / non-finite grads
- [ ] <headline metric vs numeric target>
- [ ] <comparison vs baseline_run within tolerance>

## Verification commands
<!-- CWD is research/ — paths are scripts/… and runs/…, never research/… -->
```bash
python scripts/analyze.py runs/<N>-<slug> --emit verdict.md
```

## Notes (optional)
<Gotchas for the runner/analyst: dataset prep, env vars, known failure surfaces.>
