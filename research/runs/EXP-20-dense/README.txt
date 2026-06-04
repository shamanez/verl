EXP-20-dense is the 4th arm of EXP-20 (the pure-dense GRPO control, comm-eff OFF).
Per operator directive its training ARTIFACTS live alongside the other three arms in:
    research/runs/EXP-20/
  - log:     research/runs/EXP-20/ce_dense_50s_gsm8k.log   (rsync target; sync-metrics pulls it)
  - handles: research/runs/EXP-20/handles/39409362.json
  - launch:  research/runs/EXP-20/launch_dense.sh
This EXP-20-dense/ dir holds ONLY the ledger-id-keyed heartbeat (metrics/incoming.log) that the
teardown + sync-metrics hooks resolve from the runs.jsonl row id.
Box-local authoritative training log: /workspace/verl/runs/ce_dense_50s_gsm8k/train.log
WandB: https://wandb.ai/shamanework-pl/verl_compression_research/runs/5e2jpho9
