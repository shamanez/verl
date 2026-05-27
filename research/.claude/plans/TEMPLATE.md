# Plan EXP-<ID>

The `research-planner` subagent fills this template by parsing the GitHub issue body. Humans may edit before flipping the label to `status:approved`. After approval, the orchestrator treats every field below as the contract for `experiment-runner`, `analyst`, and `log-writer`. Do not delete sections; mark unused ones with `(n/a)`.

## Experiment
- id:            EXP-<NUMBER>
- title:         <verbatim from issue title>
- issue:         <gh issue URL>
- kind:          experiment | ablation | implementation | brainstorm | literature   # routes state machine — see §"Kind" below
- milestone:     M<N>            # routing tag only; user-defined semantics
- created_at:    <ISO-8601>
- baseline_run:  EXP-<NN> | none           # for ablation, this is the parent experiment

## Kind (drives orchestrator routing)

| kind | code_change | runs on Vast | analyst | output |
|---|---|---|---|---|
| `experiment` (default) | maybe | yes | runs the plan's predicate | verdict + LOG entry; draft PR on PASS+code_change |
| `ablation` | maybe | yes | runs the plan's predicate | same as experiment; requires `depends_on:` parent EXP that PASSED |
| `implementation` | **true** (required) | NO | n/a | codex-bridge verify only; draft PR on VERIFY:PASS |
| `brainstorm` | no | NO | n/a | plan is the deliverable; iterate as comments; promote to `experiment` later by editing kind |
| `literature` | no | NO | n/a | codex-bridge math-rescue writes `findings/derivations/<topic>.md` |

If `kind:` is missing in the issue body, the planner defaults to `experiment`.

## Hypothesis
<One paragraph. Falsifiable. Contains numeric thresholds. Example: "At setting X with knob Y, observed metric M_target / M_baseline <= 0.10 within `wall_clock_hr` hours, while validation loss stays within 0.05 of baseline at step 5000.">

## Background pointers
- prior findings: findings/M<N>/EXP-NN.md, findings/M<N>/EXP-MM.md   # or "(none)"
- referenced docs: <optional paths/URLs the issue body called out; planner does NOT read any fixed background doc by default>

## Experiment design
```yaml
sweep_grid:
  <knob_a>: [<v1>, <v2>, <v3>]
  <knob_b>: [<v1>, <v2>]
baselines:
  - dense                          # always include unless the plan justifies omission
  - <method>_only
ablations:
  - disable_<component>
seed_replicates:  3
fanout_max:       6                # cap on parallel cells across the sweep
```

## Compute budget (HARD CAPS)
```yaml
gpu_count:        1                       # number of Vast.ai instances (single-node default)
gpu_filter_chain:                         # runner tries each in order; first tier with ≥1 offer ≤ max_dph wins
  - "num_gpus=4 gpu_name=H200 gpu_ram>=140 reliability>=0.95 rentable=true verified=true"   # preferred: most VRAM per $
  - "num_gpus=8 gpu_name=H100 gpu_ram>=80 reliability>=0.95 rentable=true verified=true"
  - "num_gpus=4 gpu_name=H100 gpu_ram>=80 reliability>=0.95 rentable=true verified=true"
max_dph:          24.0                    # per-instance $/hr ceiling
max_gpu_hr:       96                      # total across all cells (runner aborts past this)
max_parallel:     1                       # how many cells may run concurrently
wall_clock_hr:    12                      # soft deadline for analyst's first read
iterations:       3                       # max REVISE child-experiment depth on this lineage
```

The chain above is the planner default. The runner walks it in order and stops at the first tier with ≥1 offer ≤ `max_dph`; it records the chosen tier in the PROVISIONED ledger row so the analyst can attribute results. `per_node_gpus` is implicit in each tier's `num_gpus=` clause — the runner reads the actual count from each handle JSON to set `NGPUS_PER_NODE`.

Per-experiment plans override only when justified — e.g. M0 smoke chains with cheaper SKUs (RTX 3090/4090), multi-node fan-out for a large sweep, or a single-tier chain when a specific GPU is the experimental variable.

## Vast.ai training footprint (KEEP MINIMAL — this is real money)

State the *smallest* run that can still falsify the hypothesis. Every training
step on the Vast.ai box burns GPU-hours; fewer cells × fewer steps is always
better. Count ONLY what actually launches on Vast — codex-verify / local
assertions are free and do not belong here.

```yaml
vast_cells:        <N>          # how many training cells actually launch on Vast
steps_per_cell:    <S>          # global_step target per cell (the smallest that answers the hypothesis)
total_train_steps: <N*S>        # cross-check against max_gpu_hr — if this climbs, re-justify
justification:     "<why this is the minimum number of cells × steps that still falsifies the hypothesis>"
```

**What the plan does NOT specify.** The docker image, container `--shm-size` / `--cap-add`, onstart script (clone fork + pip-install verl `--no-deps`), recommended disk default, and CUDA driver filter all live in the locked Vast.ai Template referenced by `research/.claude/skills/vast-provision/templates.json`. The `vast-provision` skill auto-reads that file and pins the Template; plans MUST NOT name a `template_hash` or `image`. If a future plan needs a different runtime (new vllm major, different image), update `templates.json` and the Vast.ai Template record together — never bypass.

## Success criteria
- [ ] every sweep cell reaches `>= <step_target>` training steps without NaN or non-finite gradients
- [ ] dense baseline reproduces published reference within `eval_<metric> <= <bound>`
- [ ] best cell satisfies <method-specific metric threshold>
- [ ] best cell `<quality_metric> - dense_<quality_metric> <= <delta>` at step `<step>`
- [ ] <method-specific staleness/communication/efficiency metric within target>

Criteria must be **machine-checkable** — no "looks good" or "directionally correct". The analyst greps `runs/EXP-<ID>/metrics/*.jsonl` for the values referenced here.

## Verification commands
The analyst runs exactly these commands and captures stdout to `runs/EXP-<ID>/analysis.log`.
```bash
python research/scripts/analyze.py runs/EXP-<ID> --emit verdict.md
python research/scripts/check_budget.py runs/EXP-<ID>
python research/scripts/diff_against_baseline.py runs/EXP-<ID> --baseline <baseline_run>
```

## Analyst predicate
- **PASS** iff every box in `## Success criteria` is checked.
- **REVISE** if at most `iterations` boxes are unchecked AND the analyst can name a concrete next-action knob change for each. Output `next_actions:` as a yaml list of `{knob, from, to, rationale}` objects.
- **STOP** if the hypothesis is falsified (e.g. method underperforms baseline on the headline metric) OR budget exhausted OR `iterations` REVISE cycles already consumed on this lineage.

## Code change
```yaml
code_change: false                 # if true, runner branches exp/<ID>-<slug> from origin/vast-ai-workload (NEVER from main — main tracks upstream)
target_modules:                    # only meaningful when code_change=true
  - verl/<path>/<module>.py
```

If `code_change: true`, the orchestrator routes this plan through `codex-bridge --mode=verify` before any Vast.ai launch. The runner is the only agent allowed to write under `verl/`, and only while on an `exp/*` branch inside its worktree.

## Dependencies
```yaml
depends_on: [<EXP-N>, <EXP-M>]     # other experiments that must PASS or STOP first
```

The orchestrator refuses to dispatch the runner until every `depends_on` entry has VERDICT in {PASS, STOP}.

## Rescue triggers
```yaml
escalate_to_codex_if:
  - <pattern the runner or analyst might emit in PROGRESS.md>
  - <another pattern>
```

The orchestrator greps PROGRESS.md each tick for these patterns and routes to `codex-bridge` in the appropriate mode.

## Notes for runner
<Anything the planner discovered that the runner should not have to rediscover: verl-internal API gotchas, dataset paths to rsync, environment variables to set on the Vast.ai node, etc. Free-form prose; keep it tight.>

## Notes for analyst
<Optional. Anything specific the analyst should weight more heavily — e.g. "p95 staleness is the proxy for the headline claim; eval_loss is secondary".>
