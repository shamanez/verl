from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

RUNS = {
    "K=20 cadence/delay (fxo8chsv)": {
        "path": DATA / "fxo8chsv_history.jsonl",
        "color": "#c2410c",
        "cadence": 20,
        "delay_K": 20,
        "total_steps": 100,
    },
    "K=5 cadence/delay (rsvo7y1p)": {
        "path": DATA / "rsvo7y1p_history.jsonl",
        "color": "#2563eb",
        "cadence": 5,
        "delay_K": 5,
        "total_steps": 50,
    },
}

Q_ERR = "actor/comm_eff/powersgd_reconstruction_rel_error"
Q_COND = "actor/comm_eff/powersgd_q_cond"
Q_UPDATES = "actor/comm_eff/anchor_q_updates"
FAST_UPDATES = "actor/comm_eff/powersgd_basis_updates"
ANCHOR_BACK = "actor/comm_eff/anchor_backwards"
COLD_M = "actor/comm_eff/merger_coldM_fallbacks"
SPECTRAL_STEP = "actor/comm_eff/spectral_step"
CLIP = "actor/pg_clipfrac"
ENTROPY = "actor/entropy"
RESPONSE = "response_length/mean"
SCORE = "critic/score/mean"


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    for col in [Q_ERR, Q_COND, Q_UPDATES, FAST_UPDATES, ANCHOR_BACK, COLD_M, SPECTRAL_STEP, CLIP, ENTROPY, RESPONSE, SCORE]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
    return df


def first_update_step(df: pd.DataFrame) -> int:
    hit = df[df[Q_UPDATES] > 0]
    if hit.empty:
        return -1
    return int(hit.iloc[0]["step"])


def update_events(df: pd.DataFrame, label: str) -> list[dict]:
    out = []
    prev = None
    for _, row in df.iterrows():
        cur = int(row[Q_UPDATES])
        if prev is None or cur != prev:
            out.append(
                {
                    "run": label,
                    "global_step": int(row["step"]),
                    "optimizer_tick_seen": int(row[SPECTRAL_STEP]) if pd.notna(row[SPECTRAL_STEP]) else None,
                    "anchor_q_updates": cur,
                    "q_reconstruction_rel_error": float(row[Q_ERR]),
                    "q_cond": float(row[Q_COND]),
                    "pg_clipfrac": float(row[CLIP]),
                    "entropy": float(row[ENTROPY]),
                    "response_length_mean": float(row[RESPONSE]),
                    "score_mean": float(row[SCORE]),
                }
            )
        prev = cur
    return out


def write_tables(frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    events = []
    for label, df in frames.items():
        spec = RUNS[label]
        pre = df[df[Q_UPDATES] == 0]
        post = df[df[Q_UPDATES] > 0]
        fstep = first_update_step(df)
        rows.append(
            {
                "run": label,
                "cadence": spec["cadence"],
                "delay_K": spec["delay_K"],
                "rows": len(df),
                "first_q_update_global_step": fstep,
                "first_q_update_optimizer_tick_seen": int(df.loc[df["step"] == fstep, SPECTRAL_STEP].iloc[0]) if fstep > 0 else None,
                "final_anchor_q_updates": int(df[Q_UPDATES].iloc[-1]),
                "final_fast_powersgd_basis_updates": int(df[FAST_UPDATES].iloc[-1]),
                "pre_update_rows": len(pre),
                "pre_update_q_error_mean": float(pre[Q_ERR].mean()),
                "pre_update_q_error_min": float(pre[Q_ERR].min()),
                "pre_update_q_error_max": float(pre[Q_ERR].max()),
                "post_update_q_error_mean": float(post[Q_ERR].mean()),
                "post_update_q_error_min": float(post[Q_ERR].min()),
                "post_update_q_error_max": float(post[Q_ERR].max()),
                "final_q_error": float(df[Q_ERR].iloc[-1]),
                "final_q_cond": float(df[Q_COND].iloc[-1]),
                "pre_update_coldM_fallbacks_last": int(pre[COLD_M].iloc[-1]) if len(pre) else None,
                "post_update_coldM_fallbacks_first": int(post[COLD_M].iloc[0]) if len(post) else None,
            }
        )
        events.extend(update_events(df, label))
    pd.DataFrame(rows).to_csv(ROOT / "q_basis_summary.csv", index=False)
    pd.DataFrame(events).to_csv(ROOT / "q_update_events.csv", index=False)


def setup_ax(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_q_error_and_updates(frames: dict[str, pd.DataFrame]) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for label, df in frames.items():
        color = RUNS[label]["color"]
        axs[0].step(df["step"], df[Q_UPDATES], where="post", label=label, color=color, linewidth=2.2)
        axs[0].step(df["step"], df[FAST_UPDATES], where="post", color=color, linestyle=":", linewidth=1.4)
        axs[1].plot(df["step"], df[Q_ERR], label=label, color=color, linewidth=2.2)
        axs[2].plot(df["step"], df[Q_COND] - 1.0, label=label, color=color, linewidth=2.2)
        fstep = first_update_step(df)
        if fstep > 0:
            for ax in axs:
                ax.axvline(fstep, color=color, alpha=0.18, linewidth=2)
            axs[1].annotate(
                f"first Q update\nstep {fstep}",
                xy=(fstep, float(df.loc[df["step"] == fstep, Q_ERR].iloc[0])),
                xytext=(fstep + 2, 0.55 if RUNS[label]["cadence"] == 20 else 0.72),
                arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.7},
                fontsize=9,
                color=color,
            )
    setup_ax(axs[0], "Anchor-owned Q refresh counter", "cumulative updates")
    setup_ax(axs[1], "Q projection reconstruction error", "||A - A_hat|| / ||A||")
    setup_ax(axs[2], "Q orthogonality health", "q_cond - 1")
    axs[1].set_ylim(0, 1.03)
    axs[2].set_ylim(0, 7e-7)
    axs[2].set_xlabel("global training step")
    axs[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Observable Q-basis behavior in the two W&B runs", fontsize=14, fontweight="bold")
    fig.savefig(FIGS / "q_error_and_update_counters.png", dpi=180)
    plt.close(fig)


def plot_pre_first_update(frames: dict[str, pd.DataFrame]) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for label, df in frames.items():
        color = RUNS[label]["color"]
        fstep = first_update_step(df)
        zoom = df[df["step"] <= max(12, fstep + 3)]
        axs[0].plot(zoom["step"], zoom[Q_ERR], color=color, linewidth=2.2, marker="o", ms=4, label=label)
        axs[1].plot(zoom["step"], zoom[COLD_M], color=color, linewidth=2.2, marker="o", ms=4, label=label)
        for ax in axs:
            ax.axvspan(1, fstep - 0.5, color=color, alpha=0.06)
            ax.axvline(fstep, color=color, alpha=0.25, linewidth=2)
    setup_ax(axs[0], "Before the first anchor Q update", "Q reconstruction error")
    setup_ax(axs[1], "Cold-M fallback disappears at the same first anchor fire", "cold-M fallbacks")
    axs[0].set_ylim(0, 1.03)
    axs[1].set_xlabel("global training step")
    axs[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Bootstrap-Q interval: K=20 waits much longer before the first learned Q", fontsize=14, fontweight="bold")
    fig.savefig(FIGS / "pre_first_q_update_zoom.png", dpi=180)
    plt.close(fig)


def plot_q_error_by_window(frames: dict[str, pd.DataFrame]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for ax, (label, df) in zip(axs, frames.items()):
        color = RUNS[label]["color"]
        d = df.copy()
        d["window"] = d[Q_UPDATES].astype(int)
        windows = sorted(d["window"].unique())
        data = [d.loc[d["window"] == w, Q_ERR].to_numpy() for w in windows]
        bp = ax.boxplot(data, positions=windows, widths=0.55, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
            patch.set_edgecolor(color)
        for key in ["whiskers", "caps", "medians"]:
            for artist in bp[key]:
                artist.set_color(color)
        means = [np.mean(x) for x in data]
        ax.plot(windows, means, color=color, marker="o", linewidth=2)
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Q update window (0 = before first update)")
        ax.grid(True, axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[0].set_ylabel("Q reconstruction error")
    axs[0].set_ylim(0, 1.03)
    fig.suptitle("Error distribution within each held-Q window", fontsize=14, fontweight="bold")
    fig.savefig(FIGS / "q_error_by_update_window.png", dpi=180)
    plt.close(fig)


def plot_layer_heatmaps(frames: dict[str, pd.DataFrame]) -> None:
    layer_keys = [
        "actor/comm_eff/powersgd_reconstruction_rel_error/layer_3",
        "actor/comm_eff/powersgd_reconstruction_rel_error/layer_7",
        "actor/comm_eff/powersgd_reconstruction_rel_error/layer_11",
        "actor/comm_eff/powersgd_reconstruction_rel_error/layer_15",
        "actor/comm_eff/powersgd_reconstruction_rel_error/layer_18",
        "actor/comm_eff/powersgd_reconstruction_rel_error/layer_21",
        "actor/comm_eff/powersgd_reconstruction_rel_error/layer_24",
    ]
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    for ax, (label, df) in zip(axs, frames.items()):
        mat = df[layer_keys].to_numpy(dtype=float).T
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.02, vmax=0.1)
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(layer_keys)))
        ax.set_yticklabels([k.rsplit("_", 1)[-1] for k in layer_keys])
        ticks = np.linspace(0, len(df) - 1, min(8, len(df)), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(df.iloc[i]["step"])) for i in ticks])
        ax.set_ylabel("boundary layer")
        ax.set_xlabel("global training step")
        # Mark Q refreshes after bootstrap.
        prev = None
        for i, (_, row) in enumerate(df.iterrows()):
            cur = int(row[Q_UPDATES])
            if prev is not None and cur != prev:
                ax.axvline(i, color="white", alpha=0.35, linewidth=0.8)
            prev = cur
    cbar = fig.colorbar(im, ax=axs, shrink=0.9)
    cbar.set_label("per-boundary reconstruction error")
    fig.suptitle("Per-boundary Q projection error after learned-Q bootstrap", fontsize=14, fontweight="bold")
    fig.savefig(FIGS / "per_boundary_q_error_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_policy_context(frames: dict[str, pd.DataFrame]) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for label, df in frames.items():
        color = RUNS[label]["color"]
        axs[0].plot(df["step"], df[Q_ERR], color=color, linewidth=2, label=label)
        axs[1].plot(df["step"], df[CLIP], color=color, linewidth=1.8, label=label)
        axs[2].plot(df["step"], df[RESPONSE], color=color, linewidth=2, label=label)
        prev = None
        for _, row in df.iterrows():
            cur = int(row[Q_UPDATES])
            if prev is not None and cur != prev:
                for ax in axs:
                    ax.axvline(int(row["step"]), color=color, alpha=0.08, linewidth=1)
            prev = cur
    setup_ax(axs[0], "Q reconstruction error", "Q error")
    setup_ax(axs[1], "Clip fraction spikes at Q-refresh steps", "pg_clipfrac")
    setup_ax(axs[2], "Late response-length instability is K=20-only", "response length mean")
    axs[0].set_ylim(0, 1.03)
    axs[2].set_xlabel("global training step")
    axs[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Q telemetry with minimal training context", fontsize=14, fontweight="bold")
    fig.savefig(FIGS / "q_error_with_clipfrac_and_length.png", dpi=180)
    plt.close(fig)


def main() -> None:
    frames = {label: load_jsonl(spec["path"]) for label, spec in RUNS.items()}
    write_tables(frames)
    plot_q_error_and_updates(frames)
    plot_pre_first_update(frames)
    plot_q_error_by_window(frames)
    plot_layer_heatmaps(frames)
    plot_policy_context(frames)
    print(f"wrote figures to {FIGS}")
    print(f"wrote summary tables to {ROOT}")


if __name__ == "__main__":
    main()
