# parallel-runs — fan out READ-ONLY orchestration across several runs

**Status:** opt-in template (operator-launched). Saved-workflow lane for the
"many sequential / parallel runs" future. See `project.yaml workflows:` for the policy.

## What it is

A dynamic-workflow **template** for the moment when several `status:approved` plans (or
several finished runs) need attention at once and you want fan-out instead of one-at-a-time
ticks. It is launched EXPLICITLY by the operator or the orchestrator session (never via
session-wide `/effort ultracode`).

## HARD safety contract (non-negotiable — workflow workers auto-approve edits)

A workflow's worker agents run in `acceptEdits` and don't prompt. Therefore this workflow is
**READ-ONLY orchestration only**:

- ✅ MAY: read `runs/<ID>/`, `runs.jsonl`, plans, WandB, `verdict.md`; produce reports /
  candidate verdicts under `runs/<ID>/`.
- ❌ MUST NOT: call `vast-provision` / `vast-attach` / `vast-teardown`, write `runs.jsonl`,
  flip any `status:*` label, open a PR, or edit `verl/`. Those are money-spending / durable
  state mutations and **stay gated single-shot subagent dispatches** in the orchestrator —
  the `status:approved` human gate and budget caps depend on it.

So: this workflow **decides what to do and drafts the read-only artifacts**; the orchestrator
session then performs any provisioning / ledger / label / PR action through the normal gated
path (experiment-runner / analyst label-write / log-writer PR).

## args

```jsonc
// pass actual JSON, not a stringified list
{ "exp_ids": ["EXP-43", "EXP-44", "EXP-45"],   // approved plans or finished runs to process
  "mode": "analyze" }                           // "analyze" (fan-out verdicts) | "triage-status" (read-only status sweep)
```

## Script skeleton (run via the Workflow tool / `ultracode`)

```javascript
export const meta = {
  name: 'parallel-runs',
  description: 'READ-ONLY fan-out across several runs: per-run multi-dimension analysis -> adversarial verify -> candidate verdict. Mutations stay with the gated orchestrator.',
  phases: [{ title: 'Analyze' }, { title: 'Verify' }],
}
const EXP_IDS = (args && args.exp_ids) || []
const DIMS = ['reward', 'length', 'entropy', 'grad-cosine', 'train-infer-gap']

// One independent chain per run: fan out dimensions -> adversarially verify -> synthesize.
const results = await pipeline(
  EXP_IDS,
  (id) => parallel(DIMS.map(d => () =>
    agent(`READ-ONLY. For ${id}, analyze the "${d}" dimension from runs/${id}/metrics + WandB. `
        + `Do NOT write runs.jsonl, flip labels, provision, or open PRs. Return findings as text.`,
      { label: `analyze:${id}:${d}`, phase: 'Analyze' })
  )),
  (dimFindings, id) => agent(
    `READ-ONLY. Adversarially reconcile these per-dimension findings for ${id} into ONE candidate `
  + `verdict (PASS/REVISE/STOP + evidence). Write it to runs/${id}/verdict.candidate.md ONLY. `
  + `Do NOT flip the status label or touch the ledger — the orchestrator does that through the gate.\n\n`
  + JSON.stringify(dimFindings),
    { label: `verify:${id}`, phase: 'Verify' })
)
return { exp_ids: EXP_IDS, candidates: results.filter(Boolean) }
```

> The output is `runs/<ID>/verdict.candidate.md` per run. The orchestrator reviews each
> candidate and, through the normal gated path, has the `analyst` finalize `verdict.md` +
> flip the label, and `log-writer` open any PR. The workflow never crosses the gate.

## How to save it as a `/parallel-runs` command

After running it once via the Workflow tool, use `/workflows` → `s` to save the run's script
as a reusable `/<name>` command (project scope = this dir). Re-verify the on-disk format
matches your Claude Code version; this `.md` is the human-readable spec + skeleton.
