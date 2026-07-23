# Issue #89 final report: PRF codec autoresearch (benign-KL search)

index.html is the single self-contained deliverable for research issue #89
(https://github.com/shamanez/verl-compression-research/issues/89): an
autonomous 8-candidate search that tried to drive the PRF activation codec
toward the dropout-benign regime (low codec-view entropy, non-climbing
reference KL) at fixed 95% compression, 77 values per token.

Contents of the report (all numbers parsed from run artifacts, none invented):

1. Header: run id, 1x H200, ~41.5 box-hours at $3.97/hr (~$165), 11 cells.
2. Executive summary: clean negative under the strict gate; entropy is
   structurally reducible by 63% via FRLR but every entropy win re-inflates
   reference KL; shipped config stays PRF-constant plus exact-k.
3. Reference frame: dense / dropout-p10 / prf-constant step-40 table, control
   KL curves (Figure 1, log y), entropy bars (Figure 2), and the
   metric-geometry caveat (codec view vs eval view).
4. Search trajectory: the 8-candidate gate table plus headline charts,
   kl_loss (Figure 3) and entropy (Figure 4) for incumbent + all candidates,
   with the slowq sawtooth and the r48k28 budget cut at step 21 visible, and
   score small multiples (Figure 5) showing capability unaffected.
5. Mechanism findings (i to v): structural entropy inflation, FRLR 63%
   reduction proof, KL level/slope decomposition, monotone rank ladder,
   within-step ppo_kl identity never broken.
6. Verdict: PASS (symmetric clean negative), recommended gate-legal config
   box, goal-nearest FRLR r48k28 box, what was not run, future work.
7. Run record: 11-cell table, budget timeline (attach 2026-07-21T10:34Z to
   budget-exceeded teardown 2026-07-23T03:05Z, ledger cap never raised),
   health record (62 monitor cycles, zero real environment incidents).

Data sources: metrics/train_<cell>.log (per-step series), monitor-detail.log
cycles 58 to 62 (r48k28 samples), verdict.md (analyst SSOT), run.json, ledger.
Charts are inline SVG built from the parsed series; no external resources.
