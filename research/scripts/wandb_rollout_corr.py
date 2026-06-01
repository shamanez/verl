"""Pull and tabulate rollout_corr/* (train-inference consistency) for a WandB run."""
import sys
import wandb

run_id = sys.argv[1] if len(sys.argv) > 1 else "zejoupvf"
api = wandb.Api()
r = api.run(f"shamanework-pl/verl_compression_research/{run_id}")
print("RUN NAME:", r.name, "| STATE:", r.state)
cfg = r.config
for k in sorted(cfg):
    if "comm_eff" in k and any(x in k for x in ("enabled", "clean_cadence", "mask.p", "mask.enabled")):
        print("  cfg", k, "=", cfg[k])

keys = [
    "training/global_step", "actor/comm_eff/clean_steps", "actor/comm_eff/mask_ratio",
    "training/rollout_probs_diff_mean", "training/rollout_probs_diff_max",
    "training/rollout_actor_probs_pearson_corr",
    "rollout_corr/training_log_ppl", "rollout_corr/rollout_log_ppl",
    "rollout_corr/kl", "rollout_corr/k3_kl", "rollout_corr/log_ppl_diff",
    "rollout_corr/ppl_ratio", "rollout_corr/chi2_token", "rollout_corr/chi2_seq",
    "critic/score/mean",
]
hist = list(r.scan_history(keys=keys))
print("rows:", len(hist))
want = {1, 2, 5, 10, 19, 20, 21, 40, 41, 60, 61, 80, 100}


def g(row, k):
    v = row.get(k)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


for row in hist:
    s = row.get("training/global_step", -1)
    try:
        si = int(s)
    except Exception:
        continue
    if si not in want:
        continue
    print(
        f"step {si:>3} clean={g(row,'actor/comm_eff/clean_steps')} mask_ratio={g(row,'actor/comm_eff/mask_ratio')} "
        f"| diff_mean={g(row,'training/rollout_probs_diff_mean')} pearson={g(row,'training/rollout_actor_probs_pearson_corr')} "
        f"| kl={g(row,'rollout_corr/kl')} k3_kl={g(row,'rollout_corr/k3_kl')} log_ppl_diff={g(row,'rollout_corr/log_ppl_diff')} ppl_ratio={g(row,'rollout_corr/ppl_ratio')} "
        f"| train_logppl={g(row,'rollout_corr/training_log_ppl')} roll_logppl={g(row,'rollout_corr/rollout_log_ppl')} "
        f"| chi2_tok={g(row,'rollout_corr/chi2_token')} chi2_seq={g(row,'rollout_corr/chi2_seq')} | score={g(row,'critic/score/mean')}"
    )
