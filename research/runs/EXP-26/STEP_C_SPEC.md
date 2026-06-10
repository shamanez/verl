# EXP-26 Step C spec — RLVR-native Q-content sweep (orchestrator, 2026-06-10)

Closes the design gap left by the Step-A analyst: the `next_actions` named the
`q_basis` families but not their sketch constructions. The runner implements THIS
spec on `exp/26-geometry-audit-ef-powersgd` (branch @ `5a35fa96c`, already pushed).
Plan: `.claude/plans/26.md` · Decision: `runs/EXP-26/stepA_decision.md`
(`go_C_then_B`) · Stage verdict (renamed): `runs/EXP-26/stepA_verdict.md`.

## Why C runs first (one line)

`Q_act` activation-capture 0.9985 but UPDATE-energy capture 0.318 (off-principal
share 0.682) — the basis CONTENT misses ~68% of the dense GRPO update energy (H2).

## C1 — single-cell PASSIVE screen (the cheap part; replaces a 6-arm sweep)

All families differ ONLY in what feeds the sketch `V` inside the ANCHOR. So ONE
short run can build ALL family sketches simultaneously, passively, while the live
training path stays `q_basis=act` (the control). Per-family candidate bases are
orthonormalized + dumped at cadence; the judge metrics are computed against the
SAME captured reference grads. 6 geometry arms collapse into 1 cell.

- Live config: the LOCKED substrate exactly as Step A's A1 arm (plain PowerSGD
  r77, `anchor.owns_q=true`, cadence=5, `delay_K=5`, `clean_cadence=0`, spectral
  OFF / no merger), capture flags ON, `delay_K=0` fresh-anchor MEASUREMENT probe
  ON (same as Step A; probe never feeds the optimizer — hard gate).
- Length: ~15-18 optimizer ticks so there are ≥2 post-warm capture ticks
  (Step A precedent: ticks 10/15 post-warm, `COMM_EFF_CAPTURE_MIN_TICK` wired).
- Passive family sketches accumulate ONLY inside the anchor's stale-weight pass
  (`_anchor_sketch_mode`), in `no_grad`, never touching the live `Q`, the fast
  path, or the optimizer — the off-path-parity and probes-don't-feed-optimizer
  hard gates from the plan bind this code too.
- Cross-DP: each family's `V_f` is all-reduced over the DP group exactly like
  `V_act` BEFORE orth. **Collective safety:** iterate a FIXED
  `sorted(boundaries) × FIXED family order` on every rank or it deadlocks.
- Dumps (small): per boundary, per cadence tick: `Q_f` (1536×77 fp32 ≈ 0.5 MB)
  per family + the Step-A-style per-target `G_fresh_anchor` / `G_comp` captures.

## Family sketch constructions (H=1536, r=77 FIXED for every family)

Operands available inside the anchor pass: boundary activation `M ∈ R^{N×H}`
(rmpad flat), boundary activation GRADIENT `G_b = dL_anchor/dh ∈ R^{N×H}` (tensor
hook on the boundary output of the anchor's backward — the anchor already runs a
full uncompressed backward, `anchor_backwards>0`), per-token GRPO advantage
`a ∈ R^N` (plumb the mini-batch `advantages` into the compressor context, aligned
to the rmpad row layout), and the LIVE act basis `Q_act` (H×77).

| family | sketch fed to power iteration / construction |
|---|---|
| `act` (control) | `V += Mᵀ(M Q)` — unchanged, byte-identical (the live basis) |
| `grad` | `V_g += G_bᵀ(G_b Q_g)` — grad second moment: the activation-space directions that carry GRPO update energy |
| `adv` | `V_a += (wM)ᵀ((wM) Q_a)`, `w = diag(|a_t|/mean|a_t|)` — advantage-magnitude-weighted activation energy |
| `tail` | `V_t += G_tᵀ(G_t Q_t)`, `G_t = G_b − (G_b Q_act)Q_actᵀ` — grad energy DEFLATED of the act-principal subspace (targets the missing 0.68 off-principal share directly) |
| `hybrid` | `Q_h = orth([Q_act[:, :39], Q_grad_deflated[:, :38]])` — 39 act columns + 38 grad columns deflated against them, joint re-orth; preserves forward fidelity AND adds update directions |
| `ticket` | axis-aligned: `Q_tk = I[:, S]`, `S` = top-77 coordinates of `diag(Σ G_b ⊙ G_b)` (per-dim grad second moment) — the "winning-ticket" coordinate basis; comm-cheapest (indices, no dense H×r broadcast) |

Implementation notes:
- Replace the fail-loud `NotImplementedError` ONLY for families actually
  implemented; keep fail-loud for anything else.
- `tail`'s deflator on the screen is the live `Q_act`. If `tail` wins and goes
  to a training arm, it needs an INTERNAL act-basis maintained alongside (note
  in code; only build it if tail actually wins).
- A LIVE (training-path) family arm only needs the anchor to feed `V` from the
  family's statistic — the fast path stays a read-only consumer (owns_q
  invariant untouched).

## Judge metrics (computed per family, per target, post-warm ticks; analyst ranks)

Reference: `G_fresh = G_fresh_anchor@delay_K=0` (validated faithful in Step A,
`cos(G_fresh_anchor, G_dense)=0.985` on the dense arm). The clean-PG-vs-PPO-clip
operand confound is CONSTANT across families ⇒ the RANKING is confound-free even
though absolute capture values inherit the Step-A caveat.

1. **Update-capture** `UC_f = ‖proj_{Q_f}(G_fresh)‖²/‖G_fresh‖²` (same projection
   convention as the Step-A audit; median over targets). Step-A baseline:
   `UC_act = 0.318`.
2. **Off-principal preservation** `OPP_f = ‖proj_{Q_f}(G_off)‖²/‖G_off‖²`,
   `G_off = G_fresh − proj_{Q_act}(G_fresh)` (act-deflated reference grad).
3. **Activation-capture guardrail** `AC_f = ‖M Q_f Q_fᵀ‖²/‖M‖²` — NOT a judge
   criterion (plan: judge by update geometry, not reconstruction), but `Q` does
   double duty (forward reconstruction + grad routing), so flag any family with
   `AC_f < 0.9` as a forward-fidelity risk for its training arm. `tail`/`grad`
   may legitimately be low here — the training arm decides viability.

**Winner rule (plan gate):** family `f` beats the control iff `UC_f > UC_act`
AND `OPP_f > OPP_act` on the median. Analyst writes
`runs/EXP-26/stepC_screen_report.md` with the full table + winner (or
`none_beats_act`). **Do NOT write `verdict.md`** — that path is reserved for the
TERMINAL whole-issue verdict (and the teardown Stop hook triggers on it).

Optional (should-have, only if cheap): an anchor probe-loss mode `ppo_clip`
(ratio vs the batch's `old_log_probs`, clip as the fast path) for the `delay_K=0`
probe — removes the loss-mismatch confound entirely and gives Step B a clean
`cos(G_fresh_ppo, G_corr)` vs `cos(G_fresh_ppo, G_comp)` improvement test. If
hairy, skip: the fallback discriminators below suffice.

## Decision tree after C1

- Some family beats act → **C2**: ONE 50-step training arm, plain PowerSGD r77 +
  winner `q_basis` LIVE (substrate locked, no merger). Gate: `val@50 >= 0.7414`,
  no length/clip collapse.
  - C2 PASSES → Step B arms: `{ef_powersgd + winner Q, plain PowerSGD r77 + act,
    dense}`.
  - C2 FAILS → Step B arms: `{ef_powersgd + act, plain PowerSGD r77 + act,
    dense}` (C outcome recorded as REVISE-grade finding, not a STOP for the issue).
- No family beats act → skip C2 → Step B with `q_basis=act` (H2's implication
  retired empirically; record it).

Never vary Q content AND merger simultaneously unless the earlier gate passed
(plan rule) — `ef + winner Q` is allowed ONLY after C2 passes.

## Step B arms (after C resolves) — 50 steps each, val@25, sequential

| arm | correction_mode | q_basis | purpose |
|---|---|---|---|
| B-ef | `ef_powersgd` (live `ef_decay≈0.9`, `ef_clip≈1.0` — runner confirms defaults vs config comments; NO sign term) | winner-or-act | the method under test |
| B-plain | none (spectral merger off) | act | compression-benign control on the LOCKED substrate (this 50-step reference does NOT yet exist — Step A's A1 was ~15 ticks) |
| B-dense | comm-eff OFF | — | dense control; must reproduce W&B `0.7536 ± 0.01`; also validates off-path parity of all new code |

Step B direction gate (carry the Step-A caveat): the literal
`cos(G_dense, G_corr)` is not cleanly measurable. Use confound-free
discriminators: `cos(G_comp, G_corr)` for the ef arm must be HIGH (direction
preserved; signed_ema's was 0.717 ≈ 44° rotation — ef must materially exceed it),
plus `cos(G_fresh_ppo, ·)` improvement if the ppo_clip probe landed. Headline
gate stays `val@50 >= 0.7414` + no collapse.

## Step E — measured comm volume (NO extra runs)

Wire byte counters NOW so every training arm logs, per optimizer tick and per
boundary: tokens `N`, payload elements actually sent under the codec
(`N·r` for Y + amortized `H·r/cadence` for the Q broadcast + anchor M/Q refresh
traffic at its cadence), and the dense-equivalent (`N·H`). Log to metrics jsonl
(e.g. `comm/bytes_compressed`, `comm/bytes_dense_equiv`). The terminal analyst
reports the measured ratio — Step E satisfied from Step B/C2's own runs.

## Box / ledger / budget (operator-authorized)

- Warm box **40242796** (4×H200, $12.88/hr) ALIVE; direct route verified:
  `ssh -i ~/.ssh/vast_ai_name -p 40280 root@145.241.108.98` (Vast also lists
  proxy `ssh8.vast.ai:12796`). `/workspace/verl` already on
  `exp/26-geometry-audit-ef-powersgd@5a35fa96c`; 143G free.
- The old EXP-26 ledger row stays COMPLETE (Step-A record; keeps the teardown
  hook off the box during code work). Register the NEW RUNNING row (id `EXP-26`,
  same instance_id, fresh clock, `max_gpu_hr=60`, note "stage C/B/E") ONLY
  immediately before launching the first cell. `runs/EXP-26/verdict.md` was
  RENAMED to `stepA_verdict.md` so the hook's verdict-written trigger is
  disarmed; nobody writes `verdict.md` until the terminal verdict.
- Budget estimate: probe+screen ≈ 4-6 GPU-hr; 3-4 training arms ≈ 12-13 wall-hr
  ≈ 48-52 GPU-hr. Fresh `max_gpu_hr=60` covers it; `max_parallel=1` (sequential
  cells, one box).

## Hard gates before the screen launches (from the plan, unchanged)

Off-path parity · probes-never-feed-optimizer (now including the passive family
sketches + `G_b` hook + advantage plumbing) · anchor-owns-Q counters · full pass
only in anchor · `delay_K>=5` on the training path · fp32 dump fidelity · backend
integration (FSDP1 + grad-ckpt + bf16, no NaN/OOM in a 1-3 tick probe) · EF
limiting-case identity + no-sign-term static check (probe it NOW — it gates
Step B and the code already exists on the branch).
