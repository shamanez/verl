import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Panel A: grad_norm by config (real 4xB200 / FSDP / GRPO) ----
cfgs = ["dense\n(comm-eff OFF)", "mask + RESCALE", "mask, NO rescale"]
gn = [0.38, 4.5, 2700.0]
mult = ["1x", "~12x", "~7100x"]
colors = ["#2e7d32", "#1565c0", "#c62828"]
bars = ax[0].bar(cfgs, gn, color=colors, width=0.6)
ax[0].set_yscale("log")
ax[0].set_ylabel("actor/grad_norm  (log scale)")
ax[0].set_title("Gradient norm — real 4xB200 / FSDP / GRPO (Qwen2.5-1.5B)")
for b, v, m in zip(bars, gn, mult):
    ax[0].text(b.get_x() + b.get_width()/2, v*1.25, f"{v:g}\n({m})", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax[0].set_ylim(0.1, 1e4)
ax[0].axhline(0.38, ls="--", lw=1, color="#2e7d32", alpha=0.6)
ax[0].text(2.4, 0.45, "dense floor", color="#2e7d32", fontsize=8, ha="right")

# ---- Panel B: the mechanism — RMSNorm backward gain ~ 1/RMS ----
rms = np.array([0.02, 1.0, 58.0])
grad = np.array([15642.0, 313.0, 5.4])
xs = np.logspace(-2, 2, 100)
ref = 313.0 * (1.0/xs)          # reference 1/RMS line anchored at the dense point
ax[1].loglog(xs, ref, "k--", lw=1.2, label=r"reference  $\propto 1/\mathrm{RMS}$")
ax[1].scatter(rms, grad, s=130, c=["#c62828", "#2e7d32", "#1565c0"], zorder=5, edgecolors="k")
labels = ["no-rescale\n(RMS collapses ->\ngrad BLOWS UP)", "dense\n(pretrained RMS)", "rescale\n(RMS restored/\novershoot)"]
offs = [(1.6, 1.3), (1.4, 2.2), (0.5, 2.0)]
for r, g, lab, (ox, oy) in zip(rms, grad, labels, offs):
    ax[1].annotate(lab, (r, g), textcoords="offset points", xytext=(ox*8, oy*8), fontsize=8.5,
                   ha="center", arrowprops=dict(arrowstyle="->", lw=0.8))
ax[1].set_xlabel("boundary activation RMS  (input to downstream RMSNorm)")
ax[1].set_ylabel("gradient passed upstream by RMSNorm")
ax[1].set_title(r"Why: pre-RMSNorm backward $\propto 1/\mathrm{RMS}$  (Qwen2.5 AND Llama-3.2)")
ax[1].legend(loc="upper right", fontsize=9)
ax[1].grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("Masked GRPO grad-norm: no-rescale RMS collapse -> RMSNorm 1/RMS blow-up;  rescale fixes the cause, leaves a ~12x variance gap",
             fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig("/Users/shamane/Documents/verl/research/runs/EXP-16/grad_sketch.png", dpi=130, bbox_inches="tight")
print("saved grad_sketch.png")
