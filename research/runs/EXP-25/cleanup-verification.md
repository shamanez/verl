# EXP-25 legacy-spectral cleanup — empirical verification

Date: 2026-06-05. Read-only verification of the LIVE working tree at
`/Users/shamane/Documents/verl`. Verdict based solely on observed git/rg output.

## VERDICT: CLEAN

The legacy spectral-correction path (reweight / SVD / Tikhonov / seeded-anchor)
is gone from all LIVE source code on the currently checked-out local branch
`vast-ai-workload`. The prior report is **STALE**.

## Git state

```
$ git rev-parse --abbrev-ref HEAD
vast-ai-workload
$ git rev-parse HEAD
98a9583a0dd230bd27f622addb8a0fa8a4461c85
$ git log --oneline -3
98a9583a0 [autosave] research session 0b8a70b7-5cb7-4687-960d-0c655dfd4fba stop
30621b48a [EXP-25] remove dead reweight/SVD/Tikhonov/seeded-anchor spectral path (signed_ema is the live merger)
6a8906114 [autosave] research session ... stop
```

The cleanup is included in `vast-ai-workload` as `30621b48a`
("remove dead reweight/SVD/Tikhonov/seeded-anchor spectral path"). The branch tip is
the later autosave commit `98a9583a0`.

### Commit-identity note (resolving the apparent contradiction)

Three commits share the cleanup subject line; they are NOT the same patch:

| sha | where | spectral_filter.py | transformer_impl.py | state.py |
|---|---|---|---|---|
| `9cf5f8c47` | ORPHAN — on no branch (`branch --contains` empty) | — | — | — |
| `e9931b23e` | tip-2 of `origin/exp/25-anchor-default` | 386 lines | 8 lines | 29 lines |
| `30621b48a` | included in `vast-ai-workload` (`HEAD~1` before this note commit) | 461 lines | 301 lines | 56 lines |

- `9cf5f8c47` (named in the task and the git-status snapshot) is now **orphaned** —
  reachable by sha but on no branch tip. The branch was amended/rebased after the
  snapshot, replacing it with `30621b48a`. `9cf5f8c47` differs from HEAD only by 26
  added / 1 deleted lines in `state.py`.
- `30621b48a` is a locally-evolved variant of `e9931b23e`, NOT a clean
  cherry-pick (transformer_impl.py: 301 vs 8 lines; spectral_filter.py: 461 vs 386).
  Same intent, different/larger patch. Provenance differs but the END STATE is what
  matters, and the end state is clean (below).

### Working tree

```
$ git status --porcelain
(empty)
```

CLEAN. The 4 files flagged dirty in the start-of-session snapshot
(`.last-sync`, `STATUS.md`, `PROGRESS.md`, `runs/EXP-25/launch.sh`) have all been
committed since; `git diff research/runs/EXP-25/launch.sh` is empty (byte-identical
to HEAD). No uncommitted edits reintroduce dead code.

## Token grep — LIVE SOURCE vs docs/comments/tests

### Removed-mode tokens in live .py — ZERO

- `rg 'seed_anchor_cache|svd_mode|basis_cache' verl/ --glob '*.py'` → **empty**.
- `rg '[Tt]ikhonov|svd_mode' verl/ --glob '*.py'` → only `spectral_filter.py:39`,
  which is a **docstring stating the path was removed**, not code.
- No torch SVD / QR / pinv / lstsq in the correction path. The only `torch.linalg`
  in `spectral_filter.py` is `.norm()` (scale matching). QR/`svdvals` exist solely in
  `powersgd_activation.py` — the LIVE PowerSGD codec, unrelated to the dead path.

### `reweight` hits — all benign

- `verl/trainer/ppo/core_algos.py` + `ray_trainer.py` + `ppo_trainer.yaml`:
  upstream `pf_ppo` SAMPLE reweighting (`reweight_method`: pow/max_min/max_random).
  Unrelated to spectral correction. NOT a regression.
- `verl/workers/comm_eff/spectral_filter.py:39`: docstring noting removal.
- `tests/workers/comm_eff/test_spectral_filter.py:22`: comment noting removal.

### `max_targets`

Value is **-1** (canonical correct value, full coverage), confirmed in both:
- `verl/workers/config/comm_eff.py:217` — dataclass default `max_targets: int = -1`.
- `verl/trainer/config/actor/actor.yaml:410` — `max_targets: -1`.
No `max_targets: 4` anywhere in live config. `max_targets` itself is a legitimate
surviving cap parameter (caps both anchor extraction and merger).

## Live merger = signed_ema (confirmed)

- `spectral_filter.py` dispatch (lines ~391-401) accepts only
  `inject` / `blend` / `signed_ema`; default is `signed_ema`; anything else raises
  `ValueError(... expected one of (inject, blend, signed_ema))`. Constructor asserts
  the same set (line 126).
- `comm_eff.py` config validation (lines 450-454) raises `ValueError` for any
  `correction_mode` outside `{inject, blend, signed_ema}`. So a removed mode like
  `reweight` **fails fast**, it does not silently no-op.

## Config / launcher checks (PROGRESS claims)

- `verl/trainer/config/actor/actor.yaml`: NO leftover dead struct fields. The
  spectral block carries only live keys (enabled, cadence, beta_anc, target_substr,
  max_targets=-1, ema_device). **One STALE COMMENT** at `actor.yaml:384` still
  describes the old "anchor-EMA -> full thin SVD -> Tikhonov -> two-sided projection
  -> alpha blend" formula — cosmetic only, no code behind it.
- `research/runs/EXP-25/launch.sh`: PROGRESS claims confirmed. `MAX_TARGETS=-1`
  (line 65) and all stages hardcode `COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema`
  (probe0 L88, probe1 L105, arm L124). No override forwards a removed key
  (`SEED_ANCHOR_CACHE` export at L67 is inert — the launcher does NOT forward it to
  any Hydra key, confirmed by grep of the comm_eff baseline launcher).

## One non-blocking spec/doc inconsistency (NOT a cleanup failure)

`research/runs/EXP-25/config.yaml:35` (cell id-0 `probe0-anchorM`) still lists
`correction_mode: reweight` in its flag set. This is a **stale spec/materialized-plan
artifact**, NOT live code: `launch.sh`'s `probe0()` does NOT read these cell flags —
it hardcodes `signed_ema @ alpha=1.0` as the identity merger and even documents that
"the old 'reweight' mode was removed in e9931b23e" (launch.sh:86). So the run will
NOT crash from this. It is purely a documentation drift in the run's config.yaml.

## Operator action required

**None for the cleanup.** signed_ema is the live merger; reweight/SVD/Tikhonov/
seeded-anchor are removed from all live source; the cleanup is already merged into
`vast-ai-workload` as `30621b48a`. No need to merge or cherry-pick `e9931b23e`.

Optional cosmetic tidy (does not affect runtime): the stale SVD/Tikhonov comment at
`actor.yaml:384`, and the `correction_mode: reweight` flag at `config.yaml:35` in the
EXP-25 materialized config.
