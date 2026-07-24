# Issue 93: plan of execution, written 2026-07-25 06:53 AEST

Operator going offline. This is the complete plan I will follow unattended,
including exactly where I stop and wake them. Companion files:
`PROGRAM_STATE.md` (facts and fixes), `preread-a1.md` (a1 analysis),
`kl_target_table.txt` (controller setpoints).

## 0. Where things stand right now

- Ledger elapsed **3.4 h of the 100 GPU-h cap**. Box 45725398, $3.344/h.
- Cell **a1 at step 98 of 120**, 112 s/step, healthy, about 41 min to finish.
- Round A cells a2 to a5 config-verified and ready to fire.
- Round C fully unblocked: aws CLI installed, R2 buckets fixed, all 10 OOD
  benchmark parquets built, val cadence 0/300/600 committed.
- Setpoint table complete from the dense reference (which finished 600/600).
- Branch commits so far: `70c2f08` (val cadence), `acdd1a7` (analysis docs).

## 1. The sequence I will execute

Strictly one cell at a time on the one box. On a non-STOP verdict the next cell
launches immediately, target under 15 min of GPU gap. Fable consults happen
AFTER the next cell is already burning, never in front of it.

| # | cell | steps | cost | cumulative | ETA (AEST) |
|---|---|---|---|---|---|
| 1 | a1 finish | 120 | in flight | 4.1 h | 07:35 Sat |
| 2 | a2 `srq-b1-rn` | 120 or 60 | 3.8 h / 1.9 h | 7.9 h | 11:25 Sat |
| 3 | a3 `srq-parity-k493` | 120 | 3.8 h | 11.7 h | 15:15 Sat |
| 4 | a4 `prf-exactk-cvc-ce` | 120 | 3.8 h | 15.5 h | 19:05 Sat |
| 5 | a5 `frlr-r48k28-tis` | 120 | 3.8 h | 19.3 h | 22:55 Sat |
| - | **A to B money read + Fable winner pick** | - | 0.4 h | 19.7 h | 23:20 Sat |
| 6 | b1 winner + control plane | 200 | 6.3 h | 26.0 h | 05:40 Sun |
| - | **B to C money read + Fable** | - | 0.4 h | 26.4 h | 06:05 Sun |
| 7 | c winner 600 + val 0/300/600 | 600 | 20.5 h | 46.9 h | 02:35 Mon |
| 8 | OOD suite on the round-C checkpoint | - | 1.0 h | 47.9 h | 03:35 Mon |
| 9 | verification pass, report, close | - | 0 | 47.9 h | Mon morning |

**Projected total about 48 GPU-h against the 100 h cap, about $160 of rent.**
Comfortable headroom, which also covers the two contingency arms (a1-prime if
RN beats SR, b2 if b1 is ambiguous) at 3.8 h and 6.3 h respectively.

OOD is 1.0 h rather than 1.8 h because the dense comparison column is being
produced by the other session on its own box, so I only evaluate my own
round-C checkpoint.

## 2. Per-cell loop (what happens without me being asked)

1. Watcher fires on terminal (tmux gone), crash markers, stall (no step for
   12 min), or ssh loss for 12 min.
2. `python3 research/scripts/gate93.py --run <cell> --gate-lo 100 --gate-hi 120`
   with `WANDB_API_KEY` from the laptop secrets.
3. Analyst subagent writes ONE verdict to `verdict-<arm>.md`: PASS, REVISE or
   STOP. No re-verification of its own verdict.
4. Non-STOP: launch the next cell immediately via
   `runs/93-long-horizon-stability/launch_cell.sh <arm>`, which re-points the
   `/workspace/train.log` heartbeat symlink and refuses if a session is live.
5. Post the gate table to issue 93. Then, and only then, any Fable consult.

## 3. Pre-registered decision rules I will apply

**a2's early kill (made well posed, since a1's slope varies 2.5x by window).**
Fit a2's `actor/kl_loss` over steps 2 to 60 and compare against a1's matched
steps 2 to 60 slope of 0.002707/step. Threshold **0.005414/step**.
Per-arm HAC standard error is about 0.00023, so:
- a2 slope above 0.0063: confident kill at step 60, saves 1.9 h.
- a2 slope below 0.0045: confident acquittal, run to 120.
- in between: inconclusive, decide on reward slope and gap corroboration, and
  default to running to 120 rather than killing on noise.

**A1 versus A2 is the mechanism question.** SR winning means precision
allocation is the axis and I carry the better of a1 and a3 forward. RN winning
falsifies the mechanism model and triggers the in-stage contingency
`a1-prime` (sr_quant plus PRF Hadamard pre-rotation), which is a REVISE inside
round A, not a new round.

**a4** passes if its gap slope goes negative and stabilizes with reward slope
intact; CVC then joins the round-B config. If it trips the uniformization guard
(rollout perplexity, reward slope, or val proxy degrading) I switch I4 to DC
mode for round B.

**a5** is a first-class candidate if it survives with E[rho] in [0.2, 2].

**Winner rule.** Lowest train-inference gap subject to reference KL at or below
baseline and reward-slope parity; tie-break on higher E[rho].

## 4. The one thing I expect to have to wake you for

The stage-1 gate as written requires an arm with `rollout_corr/kl < 10` nats.
**The incumbent baseline itself sits at 14.24, so the gate demands a large
absolute improvement, and a1 is coming in near 13.65.** If a3, a4 and a5 all
land in the 13 to 14 band, then on the strict reading round A fails and the
program STOPs at stage 1, even though the winner rule would happily name a
relative winner.

I will not quietly reinterpret that gate in either direction. If it happens:
1. I compute the full five-arm gate table and post it.
2. I consult Fable with the table and the decision-tree rules.
3. I flag `needs:human` with a concrete recommendation and the cost of each
   option, and **nothing launches** while it is flagged.

That is the one plausible outcome where you would wake to a paused GPU. Every
other branch continues autonomously.

## 5. What I will do without asking

- Launch each next round-A cell on a non-STOP verdict.
- Kill a2 at step 60 if it clears the 0.0063 confident-kill bar.
- Fire the a1-prime contingency if RN beats SR (it is in-stage and pre-approved
  within the iterations budget).
- Launch b1 with the winner config, probe, controller and the setpoint table.
- Launch round C on a clean b1 PASS within budget.
- Run the OOD suite on the round-C checkpoint.
- Fix and relaunch anything that crashes for environment reasons, since
  fix-then-relaunch preempts all deferrable work.
- Commit deliverables to `93-mismatch-control-kit` and post progress to issue 93.

## 6. What I will NOT do without you

- **Tear down the box.** There is no standing authorization and I will not
  assume one. On completion or STOP I flag `needs:human` and leave it to you.
- Spend beyond the 100 h cap, or launch anything past a STOP verdict.
- Amend the pre-registered matrix in a way that costs new GPU time without
  escalating first.
- Reinterpret the stage-1 gate threshold (see section 4).
- Touch the other session's dense box beyond read-only.

## 7. Money note you may want to act on when you wake

Ledger row `90-prf-exactk-600` (box **45621340**, $3.97/h) belongs to the other
session. Its dense-600 training is FINISHED (600/600) and it is now running the
dense OOD eval. Once that eval completes, that box has no remaining work that I
know of, and it is the more expensive of the two. It is not mine to tear down,
but it is worth a look: at $3.97/h an idle H200 is about $95/day.

## 8. Verification and close-out

ONE bounded verification pass over round C only, reproducing the headline
numbers from the pulled WandB history plus the on-box train.log, since WandB has
already proved it drops late steps on this project. Then: round verdicts posted
to issue 93, `runs/93-long-horizon-program.html` extended in
`~/Documents/com-eff-RLVR` and pushed, memory updated, R2 artifacts confirmed,
`needs:human` raised for teardown, and a self-contained handoff prompt printed.

## 9. Standing hazards I am carrying forward

- WandB silently truncates late steps on this project. Always cross-check the
  on-box `train.log`. It cost the dense reference 192 steps of apparent history.
- `research/runs/` is blanket-gitignored at `.gitignore:175`; run deliverables
  need `git add -f`, which is the existing convention (issue 89 did the same).
- The Bash tool runs zsh, where an unquoted variable does NOT word-split. This
  silently broke my first watcher into a false "box unreachable" alarm. Inline
  ssh invocations in monitor scripts rather than storing them in a variable.
- Round C uses `delete_local` on the R2 sink, so a broken upload path is
  destructive. It is verified working now; re-verify the `[ckpt_r2]` lines at
  the first save (step 100) rather than assuming.
