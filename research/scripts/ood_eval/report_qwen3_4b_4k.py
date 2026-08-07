#!/usr/bin/env python3
"""Render the Qwen3-4B-Base 4k dense-vs-compressed comparison as one HTML file.

    python3 research/scripts/ood_eval/report_qwen3_4b_4k.py \
        --results /workspace/runs/ood-eval-4b/results.json \
        --out     /workspace/runs/ood-eval-4b/qwen3-4b-4k-comparison.html

Input is the results.json written by collect_qwen3_4b_4k.py. Output is a single
self-contained page: no external stylesheet, no script, every figure an inline
SVG, every number also present as text in a table. Missing cells render as n/a
rather than disappearing, so a partial eval produces an honest partial report.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from typing import Optional

DENSE = "#2563eb"
PRF = "#16a34a"
BASE = "#94a3b8"
INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"

# Theme-following SVG attribute groups. Figures inherit the page colour, so a
# published page reads correctly in both light and dark.
AXIS = 'stroke="currentColor" stroke-opacity="0.55" stroke-width="1.5"'
GRIDLINE = 'stroke="currentColor" stroke-opacity="0.16" stroke-width="1"'
DIM = 'fill="currentColor" fill-opacity="0.66"'
FAINT = 'fill="currentColor" fill-opacity="0.55"'


def esc(text) -> str:
    return html.escape(str(text))


def pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{100.0 * v:.1f}"


def signed_pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{100.0 * v:+.1f}"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def grouped_bars(data: dict, final: int) -> str:
    """Dense against compressed at the final step, one pair per benchmark.

    In-domain sits left of a dashed divider, out of domain to its right, so the
    two questions (did it learn, did it keep what it had) read separately.
    """
    benches = data["benches"]
    pretty = data["pretty"]
    n_in = len(data["in_domain"])
    ce = data["eval"].get(f"commeff{final}", {})
    de = data["eval"].get(f"dense{final}", {})
    bs = data["eval"].get("base", {})

    values = [v for b in benches for v in (bs.get(b), de.get(b), ce.get(b)) if v is not None]
    top = max(values) if values else 1.0
    ymax = min(1.0, max(0.1, top * 1.18))

    left, right, topm, bottom = 62, 18, 26, 108
    group_w, plot_h = 92, 300
    plot_w = group_w * len(benches)
    w, h = left + plot_w + right, topm + plot_h + bottom

    def y(v: float) -> float:
        return topm + plot_h - (v / ymax) * plot_h

    aria = "Accuracy by benchmark, dense against compressed"
    out = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{aria}">'
    ]
    out.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="none"/>')

    ticks = 6
    for i in range(ticks + 1):
        v = ymax * i / ticks
        yy = y(v)
        out.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" {GRIDLINE}/>'
        )
        out.append(
            f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" {DIM}>{100 * v:.0f}%</text>'
        )

    if 0 < n_in < len(benches):
        xd = left + n_in * group_w
        out.append(
            f'<line x1="{xd}" y1="{topm - 8}" x2="{xd}" y2="{topm + plot_h}" '
            f'stroke="currentColor" stroke-opacity="0.5" '
            f'stroke-width="1.5" stroke-dasharray="6 5"/>'
        )
        out.append(
            f'<text x="{left + 6}" y="{topm - 12}" font-size="12" {DIM}>in domain</text>'
        )
        out.append(
            f'<text x="{xd + 8}" y="{topm - 12}" font-size="12" {DIM}>out of domain</text>'
        )

    bar_w, gap = 22, 5
    for gi, b in enumerate(benches):
        gx = left + gi * group_w
        trio = [(bs.get(b), BASE), (de.get(b), DENSE), (ce.get(b), PRF)]
        span = 3 * bar_w + 2 * gap
        x0 = gx + (group_w - span) / 2
        for bi, (val, color) in enumerate(trio):
            bx = x0 + bi * (bar_w + gap)
            if val is None:
                out.append(
                    f'<text x="{bx + bar_w / 2:.1f}" y="{topm + plot_h - 6}" text-anchor="middle" '
                    f'font-size="10" {FAINT}>n/a</text>'
                )
                continue
            by = y(val)
            out.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{topm + plot_h - by:.1f}" '
                f'fill="{color}" rx="2"><title>{esc(pretty.get(b, b))} {100 * val:.1f}%</title></rect>'
            )
            out.append(
                f'<text x="{bx + bar_w / 2:.1f}" y="{by - 5:.1f}" text-anchor="middle" font-size="10" '
                f'fill="currentColor">{100 * val:.0f}</text>'
            )
        label = pretty.get(b, b)
        out.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{topm + plot_h + 20}" text-anchor="end" font-size="12" '
            f'fill="currentColor" transform="rotate(-35 {gx + group_w / 2:.1f} '
            f'{topm + plot_h + 20})">{esc(label)}</text>'
        )

    out.append(
        f'<line x1="{left}" y1="{topm + plot_h}" x2="{left + plot_w}" y2="{topm + plot_h}" {AXIS}/>'
    )
    ly = h - 22
    legend = [(BASE, "untrained"), (DENSE, "dense"), (PRF, "compressed")]
    lx = left + plot_w / 2 - 150
    for color, label in legend:
        out.append(f'<rect x="{lx}" y="{ly - 11}" width="14" height="14" fill="{color}" rx="2"/>')
        out.append(f'<text x="{lx + 20}" y="{ly}" font-size="13" fill="currentColor">{esc(label)}</text>')
        lx += 105
    out.append("</svg>")
    return "\n".join(out)


def line_chart(series: dict, title: str, ylabel: str, ypercent: bool = False) -> str:
    """One panel, one line per arm, x is the training step."""
    pts = {k: sorted((int(s), float(v)) for s, v in v_.items()) for k, v_ in series.items() if v_}
    if not any(pts.values()):
        return f'<p class="muted">{esc(title)}: no data in the training logs.</p>'

    xs = [p[0] for v in pts.values() for p in v]
    ys = [p[1] for v in pts.values() for p in v]
    xmin, xmax = min(xs), max(max(xs), 1)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymin, ymax = ymin - max(abs(ymin) * 0.05, 0.5), ymax + max(abs(ymax) * 0.05, 0.5)
    pad = (ymax - ymin) * 0.12
    ymin, ymax = ymin - pad, ymax + pad
    if ypercent or min(ys) >= 0:
        ymin = max(0.0, ymin)

    left, right, topm, bottom = 68, 20, 20, 56
    plot_w, plot_h = 620, 230
    w, h = left + plot_w + right, topm + plot_h + bottom

    def X(v):
        return left + (v - xmin) / max(1, (xmax - xmin)) * plot_w

    def Y(v):
        return topm + plot_h - (v - ymin) / (ymax - ymin) * plot_h

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(title)}">']
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        yy = Y(v)
        lab = f"{100 * v:.0f}%" if ypercent else (f"{v:.3g}")
        out.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" {GRIDLINE}/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" {DIM}>{lab}</text>'
        )

    colors = {"dense": DENSE, "commeff": PRF}
    for name, series_pts in pts.items():
        color = colors.get(name, MUTED)
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v):.1f}" for i, (s, v) in enumerate(series_pts))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for s, v in series_pts:
            out.append(
                f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{color}">'
                f'<title>step {s}: {v:.4g}</title></circle>'
            )

    uniq = sorted(set(xs))
    stride = max(1, round(len(uniq) / 6))
    ticks_x = uniq[::stride]
    if uniq[-1] not in ticks_x:
        ticks_x.append(uniq[-1])
    for s in ticks_x:
        out.append(
            f'<text x="{X(s):.1f}" y="{topm + plot_h + 18}" text-anchor="middle" font-size="11" {DIM}>{s}</text>'
        )
    out.append(
        f'<line x1="{left}" y1="{topm + plot_h}" x2="{left + plot_w}" y2="{topm + plot_h}" {AXIS}/>'
    )
    out.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{h - 16}" text-anchor="middle" font-size="12" {DIM}>training step</text>'
    )
    out.append(
        f'<text x="14" y="{topm + plot_h / 2:.0f}" font-size="12" {DIM} '
        f'transform="rotate(-90 14 {topm + plot_h / 2:.0f})" text-anchor="middle">{esc(ylabel)}</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def matrix_table(data: dict, final: int) -> str:
    benches = data["benches"]
    pretty = data["pretty"]
    steps = data["steps"]
    ev = data["eval"]

    head = ['<tr><th class="l">benchmark</th><th class="l">protocol</th><th>untrained</th>']
    for s in steps:
        head.append(f"<th>dense {s}</th><th>compressed {s}</th>")
    head.append("<th>gap at final</th></tr>")

    rows = []
    for b in benches:
        klass = "indomain" if b in data["in_domain"] else ""
        cells = [
            f'<tr class="{klass}"><td class="l">{esc(pretty.get(b, b))}</td>',
            f'<td class="l muted">{esc(data["protocol"].get(b, ""))}</td>',
            f"<td>{pct(ev.get('base', {}).get(b))}</td>",
        ]
        for s in steps:
            cells.append(f"<td>{pct(ev.get(f'dense{s}', {}).get(b))}</td>")
            cells.append(f'<td class="hl">{pct(ev.get(f"commeff{s}", {}).get(b))}</td>')
        ce, de = ev.get(f"commeff{final}", {}).get(b), ev.get(f"dense{final}", {}).get(b)
        gap = (ce - de) if (ce is not None and de is not None) else None
        cls = "" if gap is None else ("pos" if gap >= 0 else "neg")
        cells.append(f'<td class="{cls}">{signed_pct(gap)}</td></tr>')
        rows.append("".join(cells))

    # Mean over the out-of-domain block, computed only where both arms have a number.
    def block_mean(tag: str, names: list) -> Optional[float]:
        vals = [ev.get(tag, {}).get(b) for b in names]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if len(vals) == len(names) else None

    foot = [
        '<tr class="total"><td class="l">out-of-domain mean</td><td></td>',
        f"<td>{pct(block_mean('base', data['ood']))}</td>",
    ]
    for s in steps:
        foot.append(f"<td>{pct(block_mean(f'dense{s}', data['ood']))}</td>")
        foot.append(f'<td class="hl">{pct(block_mean(f"commeff{s}", data["ood"]))}</td>')
    mce, mde = block_mean(f"commeff{final}", data["ood"]), block_mean(f"dense{final}", data["ood"])
    mgap = (mce - mde) if (mce is not None and mde is not None) else None
    foot.append(f'<td class="{"" if mgap is None else ("pos" if mgap >= 0 else "neg")}">{signed_pct(mgap)}</td></tr>')

    return (
        '<div class="scroll"><table><thead>'
        + "".join(head)
        + "</thead><tbody>"
        + "".join(rows)
        + "".join(foot)
        + "</tbody></table></div>"
    )


def config_table(data: dict, cfg: dict) -> str:
    rows = [
        ("model", cfg["model"], cfg["model"]),
        ("training data", "MATH", "MATH"),
        ("context", "1024 prompt + 3072 response", "1024 prompt + 3072 response"),
        ("prompts per step", cfg["train_batch"], cfg["train_batch"]),
        ("rollouts per prompt", cfg["rollout_n"], cfg["rollout_n"]),
        ("optimizer", f"AdamW {cfg['lr']}", f"AdamW {cfg['lr']}"),
        ("reference KL", f"low_var_kl {cfg['kl']}", f"low_var_kl {cfg['kl']}"),
        ("steps", cfg["steps"], cfg["steps"]),
        ("checkpoints", cfg["save_every"], cfg["save_every"]),
        ("boundary transport", "full precision activations", cfg["codec"]),
        ("bits per token per boundary", f"{cfg['dense_bits']}", f"{cfg['prf_bits']}"),
        ("share of the dense wire", "100%", cfg["wire_pct"]),
        ("slow correction circuit", "inactive", cfg["anchor"]),
    ]
    body = "".join(
        f'<tr><td class="l">{esc(k)}</td><td>{esc(a)}</td><td class="hl">{esc(b)}</td></tr>' for k, a, b in rows
    )
    return (
        '<div class="scroll"><table><thead><tr><th class="l">setting</th>'
        "<th>dense control</th><th>compressed</th></tr></thead><tbody>" + body + "</tbody></table></div>"
    )


# --------------------------------------------------------------------------- #
def build(data: dict, cfg: dict) -> str:
    steps = data["steps"]
    final = steps[-1] if steps else 0
    ev = data["eval"]
    tr = data.get("training", {})

    val_series = {arm: {int(k): v for k, v in tr.get(arm, {}).get("val", {}).items()} for arm in ("dense", "commeff")}

    def series(name):
        return {
            arm: {int(k): v for k, v in tr.get(arm, {}).get("series", {}).get(name, {}).items()}
            for arm in ("dense", "commeff")
        }

    ce_final = ev.get(f"commeff{final}", {})
    de_final = ev.get(f"dense{final}", {})
    base_row = ev.get("base", {})
    ind = data["in_domain"][0] if data["in_domain"] else None

    def headline(b):
        return base_row.get(b), de_final.get(b), ce_final.get(b)

    b0, d0, c0 = headline(ind) if ind else (None, None, None)
    ood_vals = [(de_final.get(b), ce_final.get(b)) for b in data["ood"]]
    ood_pairs = [(d, c) for d, c in ood_vals if d is not None and c is not None]
    ood_gap = (sum(c for _, c in ood_pairs) - sum(d for d, _ in ood_pairs)) / len(ood_pairs) if ood_pairs else None

    cards = [
        ("in domain, untrained", pct(b0) + "%"),
        ("in domain, dense", pct(d0) + "%"),
        ("in domain, compressed", pct(c0) + "%"),
        ("out of domain, mean gap", signed_pct(ood_gap) + " pts"),
        ("wire per boundary", cfg["wire_pct"]),
    ]
    card_html = "".join(
        f'<div class="card"><div class="cv">{esc(v)}</div><div class="ck">{esc(k)}</div></div>' for k, v in cards
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{esc(cfg["model"])} at 4k, compressed against dense</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#f8fafc; color:{INK};
  font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 40px 22px 80px; }}
h1 {{ font-size: 27px; margin: 0 0 6px; letter-spacing:-.01em; }}
h2 {{ font-size: 19px; margin: 44px 0 10px; letter-spacing:-.01em; }}
.sub {{ color:{MUTED}; margin: 0 0 26px; }}
.muted {{ color:{MUTED}; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:22px 0 8px; }}
.card {{ background:#fff; border:1px solid {GRID}; border-radius:10px; padding:14px 16px; }}
.cv {{ font-size:23px; font-weight:650; letter-spacing:-.02em; }}
.ck {{ font-size:12.5px; color:{MUTED}; margin-top:3px; }}
figure {{ margin:14px 0 0; background:#fff; border:1px solid {GRID}; border-radius:10px; padding:18px 16px; }}
figcaption {{ color:{MUTED}; font-size:13px; margin-top:10px; }}
.scroll {{ overflow-x:auto; background:#fff; border:1px solid {GRID}; border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th,td {{ padding:8px 11px; text-align:right; border-bottom:1px solid {GRID}; white-space:nowrap; }}
th {{ background:#f1f5f9; font-weight:600; font-size:12.5px; color:#334155; }}
td.l, th.l {{ text-align:left; }}
th.l:first-child, td.l:first-child {{ position:sticky; left:0; background:#fff; z-index:1; }}
thead th.l:first-child {{ background:#f1f5f9; }}
td.hl {{ background:rgba(22,163,74,.07); }}
tr.indomain td {{ font-weight:600; }}
tr.total td {{ border-top:2px solid {INK}; font-weight:650; }}
td.pos {{ color:{PRF}; font-weight:600; }}
td.neg {{ color:#dc2626; font-weight:600; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:14px; }}
.note {{ background:#fff; border:1px solid {GRID}; border-left:3px solid {PRF};
  border-radius:8px; padding:12px 16px; margin-top:14px; }}
@media (prefers-color-scheme: dark) {{
  body {{ background:#0b1120; color:#e2e8f0; }}
  .card, figure, .scroll, .note {{ background:#111827; border-color:#1f2937; }}
  th {{ background:#182031; color:#cbd5e1; }}
  th.l:first-child, td.l:first-child {{ background:#111827; }}
  thead th.l:first-child {{ background:#182031; }}
  th,td {{ border-bottom-color:#1f2937; }}
  .cv {{ color:#f1f5f9; }}
}}
</style></head><body><main>

<h1>{esc(cfg["model"])} at 4k, compressed against dense</h1>
<p class="sub">Two reinforcement-learning runs on MATH, identical in every respect except how much of the
activation at each shard boundary is put on the wire. {esc(cfg["steps"])} steps each, checkpoints every
{esc(cfg["save_every"])}, evaluated on one in-domain set and nine held-out ones.</p>

<div class="cards">{card_html}</div>

<h2>Where the two runs end up</h2>
<figure>{grouped_bars(data, final)}
<figcaption>Accuracy at step {final}. Left of the dashed line is the distribution the models trained on.
Right of it is everything they did not see. Bars are absent where a benchmark has no result yet.</figcaption>
</figure>

<h2>Every number</h2>
{matrix_table(data, final)}
<p class="muted" style="font-size:13px">Values are percentages. The final column is compressed minus dense at
step {final}, in points. The highlighted columns are the compressed run.</p>

<h2>How they got there</h2>
<div class="grid2">
<figure>{line_chart(val_series, "in-domain validation", "accuracy", ypercent=True)}
<figcaption>In-domain validation accuracy through training.</figcaption></figure>
<figure>{line_chart(series("score"), "training reward", "reward")}
<figcaption>Mean training reward.</figcaption></figure>
<figure>{line_chart(series("response_length"), "response length", "tokens")}
<figcaption>Mean response length. A run drifting toward the 3072 cap is the early sign of the
truncation feedback loop that ended the previous long-context attempt.</figcaption></figure>
<figure>{line_chart(series("grad_norm"), "gradient norm", "norm")}
<figcaption>Gradient norm. Flat is the healthy shape.</figcaption></figure>
<figure>{line_chart(series("kl_loss"), "reference divergence", "nats")}
<figcaption>Divergence from the frozen reference. Read the compressed line only against itself,
never against the dense one, because the two are measured through different views.</figcaption></figure>
<figure>{line_chart(series("step_seconds"), "seconds per step", "seconds")}
<figcaption>Wall clock per step on the same four-GPU host.</figcaption></figure>
</div>

<h2>What the two runs were</h2>
{config_table(data, cfg)}

<div class="note">Each boundary sends {esc(cfg["prf_bits"])} bits per token instead of {esc(cfg["dense_bits"])},
so {esc(cfg["wire_pct"])} of the traffic crosses the link. Nothing about the selection is transmitted:
both ends derive the same coordinate set from a shared key, so the saving is real rather than moved elsewhere.</div>

<p class="muted" style="margin-top:34px;font-size:12.5px">Generated from {esc(cfg["source"])}.</p>
</main></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/workspace/runs/ood-eval-4b/results.json")
    ap.add_argument("--out", default="/workspace/runs/ood-eval-4b/qwen3-4b-4k-comparison.html")
    args = ap.parse_args()

    with open(args.results) as fh:
        data = json.load(fh)

    hidden = int(os.environ.get("EXPECT_HIDDEN", "2560"))
    keep = int(round(0.05 * hidden))
    cfg = {
        "model": data.get("model", "Qwen/Qwen3-4B-Base"),
        "steps": os.environ.get("TOTAL_TRAINING_STEPS", "500"),
        "save_every": os.environ.get("SAVE_FREQ", "100"),
        "train_batch": os.environ.get("TRAIN_BATCH_SIZE", "128"),
        "rollout_n": os.environ.get("ROLLOUT_N", "8"),
        "lr": os.environ.get("ACTOR_LR", "1e-6"),
        "kl": os.environ.get("KL_LOSS_COEF", "0.001"),
        "codec": f"{keep} of {hidden} coordinates per token, keyed selection",
        "dense_bits": hidden * 16,
        "prf_bits": keep * 16,
        "wire_pct": f"{100.0 * keep / hidden:.1f}%",
        "anchor": "paired replay every 20 updates, rank-1 projection",
        "source": os.path.abspath(args.results),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(build(data, cfg))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
