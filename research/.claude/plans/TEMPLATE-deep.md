# Plan <N> — <title>

<!-- DEEP tier: multi-stage / days-long research. Target ≤ 15 KB. Same flat
     yaml contract as the fast tier — and the same home: the plan LIVES IN THE
     GITHUB ISSUE BODY between the plan:start/plan:end markers (plan_publish /
     plan_fetch). Deep adds staged execution with gates, correctness
     invariants, and a session hand-off. All heavy deliberation (judge panels,
     adversarial review, open questions) happens HERE, before approval —
     never during execution. -->

```yaml
issue: <N>
slug: <kebab-slug>
title: <issue title>          # one line — SUMMARY.md rows and LOG entries read it
kind: experiment
tier: deep
code_change: false
target_modules: []
baseline_run: baseline
depends_on: []
milestone: none
gpu_filter_chain: default
max_dph: 24.0
max_gpu_hr: 72
max_parallel: 1
attach_box: none
vast_account: private
promote_launcher_as: none
iterations: 3
```

## What & why
<A paragraph a colleague could read cold: the question, why now, what changes
depending on the answer. Link prior LOG.md / runs/SUMMARY.md entries instead
of restating them.>

## Hypothesis
<One falsifiable sentence with numeric thresholds. Symmetric: state what a
clean negative looks like — it is a PASS too.>

## Stages
<Sequential gated stages. Each stage's `gate` is numeric and decides whether
the next stage launches. on_fail: stop | revise | skip-to:<stage>. One box
serves consecutive stages — never teardown/reprovision between them.>

```yaml
stages:
  - id: 1
    name: <e.g. probe-correctness>
    cells: [<cell>, <cell>]
    gate: "<numeric condition that unlocks stage 2>"
    on_fail: stop
  - id: 2
    name: <e.g. main-sweep>
    cells: [<cell>, <cell>, <cell>]
    gate: "<numeric condition>"
    on_fail: revise
```

## Cells
| cell | config delta vs canonical launcher | passes when |
|---|---|---|
| <method-knob-value> | `<VAR=… hydra.key=…>` | <numeric bar> |

## Correctness invariants (code_change only — the pre-sweep probe)
<Cheap machine-checkable gates run as a 1–2 step probe cell BEFORE the sweep.
 gate: hard = abort plan on failure (after /monitor's ≤ 3 bounded on-box fix
 attempts); soft = record and continue. CPU checks (import/shape/off-path
 parity) run ONCE locally before launch — no CPU verification loops; numerics
 under training are validated on the GPU probe, not the MacBook.>

```yaml
invariants:
  - {name: off-path-parity, check: "feature OFF ⇒ byte-identical to baseline", gate: hard}
  - {name: probe-trains-clean, check: "2 probe steps, no NaN/OOM", gate: hard}
```

## Success criteria
- [ ] <stage gates all evaluated (✓ or ✗ — none skipped)>
- [ ] <headline metric vs numeric target>
- [ ] <controlled variables held equal across compared arms>

## Verification commands
<!-- CWD is research/ — paths are scripts/… and runs/…, never research/… -->
```bash
python scripts/analyze.py runs/<N>-<slug> --emit verdict.md
```

## Open questions (must be empty at approval)
<Everything uncertain goes here during planning; /approve refuses while
anything remains. Judge-panel / codex-verify outcomes get folded in here.>

## Progress / session hand-off
<Append-only ticks so a fresh session can resume mid-plan:
 - [ ] stage 1 launched · - [ ] stage 1 gate ✓ · …
 Update as stages complete; /go reads labels+ledger first, this second.>

## Notes
<Runner/analyst gotchas: dataset prep commands, env vars, failure surfaces,
 monitor threshold adjustments for known-idle phases.>
