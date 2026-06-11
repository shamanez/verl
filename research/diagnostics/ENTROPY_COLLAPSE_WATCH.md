# Standing Check: Entropy-Collapse Watch (ALL runs)

A **reusable, mechanical** monitor any analyst/monitor applies to **every** run
(dense control, comm-eff, anchor/merger arms). Born from EXP-25 α=0
(`|G_noisy|·sign(M)` merger), which collapsed entropy 5.69 → 0.06, exploded
response length to the 16 384 cap, and crashed reward 0.79 → 0.32 — while the
4 anchor-OFF reference runs converged cleanly. See
`research/runs/EXP-25/ENTROPY_COLLAPSE_FINDINGS.md` for the full mechanism.

**Why this matters:** GRPO no-KL / no-entropy has *no explicit entropy
regularizer*. It relies on the true-gradient geometry for implicit
regularization. Any intervention that distorts the gradient direction (sign
mergers, aggressive masking, stale-anchor corrections) can silently induce
entropy collapse. The collapse is **monotonic and visible early** (by ~step 10
in EXP-25), so you never need to wait for the run to finish to call it.

---

## ⚠️ 2026-06-11 RE-CENTERING (EXP-27 post-mortem — READ THIS FIRST)

The EXP-27 3-run comparison (`runs/EXP-27/RUN_COMPARISON.md` +
`comparison_metrics/scorecard.csv`; dense `5e2jpho9` vs signed_ema α0.5
`1wulaelw` vs damped-ef `qa6sll3h`) **falsified entropy-as-trigger**:

- **Entropy is a FOLLOWER of the length spiral, not its cause.** Dense is the
  *lowest*-entropy run of the three (0.12–0.16 from step 36) and the *most*
  stable. **T1-RED (`entropy<0.3`) fires on the healthiest run we have** — an
  absolute-entropy trigger is invalid on its own. (This sharpens the EXP-25
  finding "length-hack, not low entropy, is the killer" —
  `runs/EXP-25/COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` — into the operational rule.)
- **PRIMARY kill triggers are the length-spiral precursors** (these called
  EXP-27's ignition and retro-dict α0.5's near-miss):
  - **P1 — consecutive cap-pins:** `response_length/max` = cap (16384) on
    **≥2 consecutive** steps. Six-run evidence (RUN_COMPARISON.md §7: dense,
    plain, ef_r2, α0.5, exp27, ef_r1): EVERY run touches 16384 at least once —
    a lone spike is NOT diagnostic; max consecutive streak = 1 for all
    survivors, 2 for α0.5 (censored mid-onset at step 50), 7 for both igniters.
    **Operational kill rule: trip on the 2nd consecutive pin** — retro-dicts
    exp27 at step 62 (actual kill ~66) and α0.5 at 48. RED, kill-early.
  - **P2 — mean slope:** trailing-10-step slope of `response_length/mean`
    > **+2 tok/step** sustained = YELLOW; combined with any pin = RED.
    (EXP-27 pre-ignition: +0.10/step; α0.5@47-50: **+5.92/step**; ignition:
    +50-100/step.)
  - **P3 — the old T4 absolute:** mean > 2× first-10-step average — confirmed
    in EXP-27 (~509 crossed at step 66, after P1/P2 had fired ~5 steps earlier).
- **Gradient quality is the cross-run discriminator** (probable upstream cause):
  dense `actor/grad_norm` ≈ 0.35 clean/stable; comm-eff arms 3–7 noisy;
  20–60 during ignition. Treat the grad_norm *trajectory* as a **run-health
  baseline metric across arms** (which arm is closer to dense-quality
  gradients), NOT as a within-run instantaneous trigger (it stayed O(1–16)
  through EXP-25's collapse — the caveat below still holds within-run).
- **50-step runs are CENSORED observations.** α0.5 "survived" 50 steps but was
  already in the early spiral (consecutive cap-pins at steps 47–48); EXP-27 was
  clean at 50 and ignited at ~61. A clean step-50 endpoint does NOT certify
  stability (GOAL.md criterion 1 needs a longer horizon).

**Trigger precedence from now on:** P1/P2/P3 (length spiral) are the RED
kill-early triggers. T1–T3 (entropy) and T6 (IS gap) are demoted to
**corroborators** — meaningful only WITH a P-trigger; never alert on entropy
alone. T5 (reward peak-then-degrade) unchanged but note EXP-27 ignited with
score still 0.73–0.84 (reward-preserving length-hack) — do not wait for reward
to fall.

---

## Metrics to watch every run (the 6 core signals)

Pull these per training step. WandB keys (and the matching `train.log` field names):

| # | Signal | Metric key | Healthy | Collapse pattern |
|---|---|---|---|---|
| 1 | **Entropy (abs)** | `actor/entropy` | decays gently, plateaus > ~0.5–1.0 for Qwen2.5-1.5B/GSM8K | monotone toward 0; < 0.3 is danger |
| 2 | **Entropy slope** | Δ`actor/entropy` over a 5-step window | small, decelerating | large, *accelerating* drop |
| 3 | **Response length explosion** | `response_length/mean`, `response_length/clip_ratio` | mean roughly stable or shrinking (200–350 tok on GSM8K); clip_ratio ≈ 0 | mean climbs hundreds→thousands; clip_ratio rises off 0 toward the cap |
| 4 | **Reward peak-then-degrade** | `critic/score/mean` (or `critic/rewards/mean`) | rises monotonically, plateaus | rises, **peaks, then falls** |
| 5 | **IS gap shrink** | `training/rollout_probs_diff_mean` | stays moderate (~0.5–0.8) | shrinks toward 0 (policy → deterministic) while `..._max` stays 1.0 |
| 6 | **Train/rollout pearson rise** | `training/rollout_actor_probs_pearson_corr` | low/stable | rises (both distributions sharpening together) |

Secondary corroborators: `actor/ppo_kl` (~0 throughout when rollout correction
is OFF — confirms PPO clipping is NOT braking the drift), `actor/pg_clipfrac`,
`rollout_corr/kl` (rollout→training KL; *drops* under collapse as the rollout
distribution itself sharpens), `actor/grad_norm` (noisy; not a reliable collapse
signal — it stayed O(1–16) throughout EXP-25's collapse).

Merger-specific (only when `correction_mode=signed_ema`): per-step
`rel_change` median ≈ **√2 ≈ 1.414** is the fingerprint of the α=0 sign-flip
(≈50 % of coords sign-flipped at full magnitude every step). `rel_change=0`
means the cold-M guard is no-op'ing (M not yet warm). `actor/comm_eff/merger_coldM_fallbacks`
should be `==corrected` on step 1 then drop to 0 once M warms.

---

## TRIGGER thresholds — fire an "ENTROPY-COLLAPSE ALERT"

Evaluate these per step. **Any ONE red trigger ⇒ alert the operator immediately**
(don't wait for the timeout/end of run). Tune the absolute entropy bands per
model/task; the *patterns* (slope, peak-then-degrade, length explosion) are
model-agnostic.

| Trigger | Condition | Severity |
|---|---|---|
| **T1 entropy abs** | `actor/entropy < 0.5` | YELLOW; `< 0.3` RED |
| **T2 entropy slope** | entropy drops `> 40 %` from its value 5 steps earlier, **and** the per-step drop is *increasing* (accelerating) | RED |
| **T3 entropy early-drop** | entropy falls `> 50 %` of its step-1 value within the first 10 steps (EXP-25: 5.69→2.08 by s10) | RED — the canonical early warning |
| **T4 response length** | `response_length/mean` exceeds **2×** its first-10-step average, **or** `response_length/clip_ratio > 0.05` and rising | YELLOW; `clip_ratio > 0.20` RED |
| **T5 reward peak-degrade** | `critic/score/mean` falls `> 10 %` below its running max over the run (it had been rising) | RED |
| **T6 IS gap collapse** | `training/rollout_probs_diff_mean` falls below `0.15` while `..._max` stays ≈ 1.0 | YELLOW (deterministic-policy symptom) |
| **T7 merger sign-flip** | (signed_ema only) `rel_change` median `> 1.2` for a sustained window AND any of T1–T5 firing | RED — implicates the merger as the driver |

**Composite RED (highest confidence):** T1/T3 (entropy) **+** T4 (length) **+** T5
(reward) firing together = the EXP-25 collapse signature. Recommend pausing the
arm and (a) raising the merger α toward ≥0.5, (b) adding a KL/entropy floor, or
(c) capping response length — see the FINDINGS mitigations section.

---

## grep / WandB recipes (compute mechanically)

### A. From a local `train.log` (verl console logger — single-line `key:value - key:value` dicts)

Drop-in extractor — prints the 6 core signals per step:

```python
# usage: python3 watch.py /path/to/train.log
import re, sys
KEYS = ["actor/entropy","response_length/mean","response_length/clip_ratio",
        "critic/score/mean","training/rollout_probs_diff_mean",
        "training/rollout_probs_diff_max","training/rollout_actor_probs_pearson_corr",
        "actor/ppo_kl","actor/pg_clipfrac","actor/grad_norm"]
rows={}
for line in open(sys.argv[1]):
    m=re.search(r"step:(\d+)\s*-", line)
    if not m or "actor/entropy" not in line:        # the verl metric line carries actor/entropy
        continue
    s=int(m.group(1)); d={}
    for kv in line.split(" - "):
        k,_,v=kv.strip().partition(":")
        if k.strip() in KEYS:
            try: d[k.strip()]=float(v)
            except ValueError: pass
    rows[s]=d
ent0=rows[min(rows)].get("actor/entropy")
peak=-1
for s in sorted(rows):
    d=rows[s]; e=d.get("actor/entropy"); sc=d.get("critic/score/mean")
    if sc is not None: peak=max(peak,sc)
    flags=[]
    if e is not None and e<0.5: flags.append("T1:ent<0.5")
    if e is not None and ent0 and s<=10 and e<0.5*ent0: flags.append("T3:early-50%")
    rl=d.get("response_length/clip_ratio",0)
    if rl and rl>0.05: flags.append(f"T4:clip{rl:.2f}")
    if sc is not None and peak>0 and sc<0.9*peak and peak>0.3: flags.append("T5:reward-degrade")
    isg=d.get("training/rollout_probs_diff_mean")
    if isg is not None and isg<0.15: flags.append("T6:IS-collapse")
    print(f"step {s:3d}  ent={e}  resp={d.get('response_length/mean')}  "
          f"score={sc}  ISgap={isg}  {' '.join(flags)}")
```

Quick one-liners:

```bash
# entropy trajectory only
grep -oE "step:[0-9]+ .*actor/entropy:[0-9.eE+-]+" train.log \
  | grep -oE "step:[0-9]+|actor/entropy:[0-9.eE+-]+"

# merger sign-flip fingerprint: rel_change median (signed_ema arms)
grep -oE "rel_change=\|\|G_proj-G_mask\|\|/\|\|G_mask\|\|=[0-9.]+" train.log \
  | grep -oE "[0-9.]+$" | sort -n | awk '{a[NR]=$1} END{print "median rel_change =", a[int(NR/2)]}'

# cold-M warm-up proof (should be 196->0 once M warms)
grep "\[comm_eff\]\[merger\]" train.log | grep -oE "merger_coldM_fallbacks=[0-9]+" | uniq -c
```

### B. From WandB (any run, no SSH to the box)

```python
import os, wandb
# load WANDB_API_KEY from ~/.config/verl-research/secrets.env (never print it)
api = wandb.Api(timeout=30)
run = api.run("shamanework-pl/verl_compression_research/<RUN_ID>")
keys = ["actor/entropy","response_length/mean","response_length/clip_ratio",
        "critic/score/mean","training/rollout_probs_diff_mean",
        "training/rollout_actor_probs_pearson_corr","actor/ppo_kl"]
ent0=None; peak=-1
for r in run.scan_history(keys=["_step"]+keys):
    s=r["_step"]; e=r.get("actor/entropy"); sc=r.get("critic/score/mean")
    if e is not None and ent0 is None: ent0=e
    if sc is not None: peak=max(peak,sc)
    # apply T1/T3/T4/T5/T6 thresholds from the table above
# validation: run.summary["val-core/openai/gsm8k/acc/mean@1"]
```

Reference comparison (anchor/merger OFF, should NOT collapse): runs
`5e2jpho9` (dense, val@50=0.7536), `kqozxfr0` (0.7437), `oquyeic3` (0.7415),
`3yxzzwn3` (0.7384) — all `actor/comm_eff/anchor_backwards=0`, val rises s25→s50.
If a comm-eff arm's val/reward *falls* s25→s50 while these rise, suspect collapse.

---

## How to read each pattern (one line each)

- **Entropy ↓ monotone toward 0** → policy going deterministic; with no KL/entropy reg there is nothing pulling it back.
- **Response length ↑ + clip_ratio ↑** → low-entropy degenerate/repetitive generations that don't emit EOS; also a direct compute-waste signal (rollouts pinned at the cap).
- **Reward peaks then degrades** → the sharpening overshot; the policy locked onto a mode that stops solving the task (length-degeneration feedback loop).
- **IS gap (`rollout_probs_diff_mean`) ↓ while max stays 1.0** → training policy now ≈deterministic on sampled tokens; a *symptom* of collapse, not a cause.
- **pearson(train,rollout) ↑** → both distributions sharpening onto the same peaks together.
- **`ppo_kl`≈0 + small clipfrac the whole run** → if rollout correction is OFF, `old_log_prob` is recomputed by the training policy (ratio≈1), so PPO clipping is NOT braking the drift. Consider enabling rollout-correction IS to restore the brake.
- **rel_change median ≈ √2 (signed_ema)** → the α=0 merger is sign-flipping ~half of every gradient at full magnitude every step — the direct driver.
