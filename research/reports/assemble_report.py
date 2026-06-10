#!/usr/bin/env python3
"""Assemble the comm-eff program retrospective from the 5 team fragments.

Reads research/reports/fragments/tab{1..5}_*.html (each a single
<section class="tab-content" id="tabN">...</section> fragment written by one
teammate) and inlines them into the tabbed page shell. Idempotent — re-run any
time a fragment changes.
"""
import glob
import os
import re
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
FRAG = os.path.join(HERE, "fragments")
OUT = os.path.join(HERE, "comm_eff_program_report.html")

TABS = [
    ("tab1", "EXP-25 · Core Idea"),
    ("tab2", "EXP-26 · Core Idea"),
    ("tab3", "EXP-25 · Results & the Sign Math"),
    ("tab4", "EXP-26 · Math, Geometry & Papers"),
    ("tab5", "Where We Are & What's Next"),
]

SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Communication-Efficient GRPO — Program Retrospective (EXP-25 → EXP-26)</title>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['$$', '$$']] }},
                   options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }} }};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root {{
    --bg: #f7f8fa; --card: #ffffff; --ink: #1f2430; --muted: #5d6678;
    --accent: #2456d6; --accent-ink: #ffffff; --line: #e3e7ee;
    --callout: #eef4ff; --callout-line: #b9cdf5;
    --warn: #fff6e8; --warn-line: #eccf95;
    --win: #edf9f0; --win-line: #a9dcb8;
    --code: #f0f2f6;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif; }}
  header.page {{ background: linear-gradient(135deg, #15224a 0%, #2456d6 100%);
                 color: #fff; padding: 34px 24px 26px; }}
  header.page .wrap {{ max-width: 1020px; margin: 0 auto; }}
  header.page h1 {{ margin: 0 0 6px; font-size: 26px; line-height: 1.3; }}
  header.page p.sub {{ margin: 0; opacity: .85; font-size: 14.5px; max-width: 860px; }}
  nav.tabs {{ position: sticky; top: 0; z-index: 10; background: var(--card);
              border-bottom: 1px solid var(--line); box-shadow: 0 1px 4px rgba(20,30,60,.06); }}
  nav.tabs .wrap {{ max-width: 1020px; margin: 0 auto; display: flex; flex-wrap: wrap; }}
  nav.tabs button {{ appearance: none; background: none; border: 0; cursor: pointer;
                     padding: 13px 16px 11px; font: 600 13.5px/1.2 inherit; color: var(--muted);
                     border-bottom: 3px solid transparent; }}
  nav.tabs button:hover {{ color: var(--ink); }}
  nav.tabs button.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  main {{ max-width: 1020px; margin: 26px auto 80px; padding: 0 20px; }}
  section.tab-content {{ display: none; background: var(--card); border: 1px solid var(--line);
                         border-radius: 12px; padding: 30px 36px 38px;
                         box-shadow: 0 1px 6px rgba(20,30,60,.05); }}
  section.tab-content.active {{ display: block; }}
  h2 {{ margin: 4px 0 10px; font-size: 23px; line-height: 1.3; }}
  h3 {{ margin: 28px 0 8px; font-size: 18px; border-bottom: 1px solid var(--line); padding-bottom: 5px; }}
  h4 {{ margin: 20px 0 6px; font-size: 15.5px; }}
  p {{ margin: 10px 0; }}
  p.lede {{ font-size: 17.5px; color: var(--muted); border-left: 4px solid var(--accent);
            padding: 4px 0 4px 14px; margin: 14px 0 20px; }}
  ul, ol {{ padding-left: 26px; }}
  li {{ margin: 5px 0; }}
  ol.flow li {{ margin: 10px 0; }}
  code {{ background: var(--code); border-radius: 4px; padding: 1px 5px; font-size: 13.5px; }}
  pre {{ background: var(--code); border: 1px solid var(--line); border-radius: 8px;
         padding: 14px 16px; overflow-x: auto; font-size: 13.5px; line-height: 1.5; }}
  pre code {{ background: none; padding: 0; }}
  table.data {{ border-collapse: collapse; margin: 16px 0; width: 100%; font-size: 14.5px; }}
  table.data th, table.data td {{ border: 1px solid var(--line); padding: 8px 12px; text-align: left; }}
  table.data th {{ background: #f2f5fa; font-weight: 600; }}
  table.data tr.hl td {{ background: var(--win); font-weight: 600; }}
  div.callout, div.warn, div.win {{ border-radius: 10px; padding: 14px 18px; margin: 16px 0; }}
  div.callout {{ background: var(--callout); border: 1px solid var(--callout-line); }}
  div.warn {{ background: var(--warn); border: 1px solid var(--warn-line); }}
  div.win {{ background: var(--win); border: 1px solid var(--win-line); }}
  span.metric {{ font-weight: 700; color: var(--accent); white-space: nowrap; }}
  footer {{ max-width: 1020px; margin: 0 auto 40px; padding: 0 20px; color: var(--muted);
            font-size: 13px; }}
  @media (max-width: 640px) {{ section.tab-content {{ padding: 20px 18px; }} }}
</style>
</head>
<body>
<header class="page">
  <div class="wrap">
    <h1>Communication-Efficient GRPO — Program Retrospective</h1>
    <p class="sub">EXP-25 → EXP-26 on the shamanez/verl fork · Qwen2.5-1.5B-Instruct + GSM8K ·
    fast compressed circuit + stale anchor circuit (rank-77 PowerSGD boundary compression, ≈19.8× less
    boundary traffic) · five-agent synthesis, {today}</p>
  </div>
</header>
<nav class="tabs"><div class="wrap">
{tab_buttons}
</div></nav>
<main>
{sections}
</main>
<footer>Generated by the comm-eff-retrospective agent team (5 members). Sources: research/runs/EXP-25/,
research/runs/EXP-26/, .claude/plans/26-27, CODE_WALKTHROUGH.md, W&amp;B project verl_compression_research.
Numbers are val@50 GSM8K greedy accuracy unless stated otherwise.</footer>
<script>
  const tabs = document.querySelectorAll('nav.tabs button');
  const panes = document.querySelectorAll('section.tab-content');
  const typeset = new Set();
  function activate(id, push) {{
    tabs.forEach(b => b.classList.toggle('active', b.dataset.tab === id));
    panes.forEach(p => p.classList.toggle('active', p.id === id));
    if (push) history.replaceState(null, '', '#' + id);
    const pane = document.getElementById(id);
    if (window.MathJax && MathJax.typesetPromise && !typeset.has(id)) {{
      typeset.add(id);
      MathJax.typesetPromise([pane]).catch(() => typeset.delete(id));
    }}
    window.scrollTo({{ top: 0 }});
  }}
  tabs.forEach(b => b.addEventListener('click', () => activate(b.dataset.tab, true)));
  const initial = (location.hash || '#tab1').slice(1);
  window.addEventListener('load', () => activate(document.getElementById(initial) ? initial : 'tab1', false));
</script>
</body>
</html>
"""


def main():
    sections, missing = [], []
    for tab_id, _ in TABS:
        matches = sorted(glob.glob(os.path.join(FRAG, f"{tab_id}_*.html")))
        if not matches:
            missing.append(tab_id)
            continue
        frag = open(matches[0], encoding="utf-8").read().strip()
        # strip accidental wrappers, keep the <section> only
        m = re.search(r"<section\b.*</section>", frag, re.S)
        if m:
            frag = m.group(0)
        else:
            frag = f'<section class="tab-content" id="{tab_id}">\n{frag}\n</section>'
        if f'id="{tab_id}"' not in frag:
            frag = re.sub(r"<section\b", f'<section id="{tab_id}"', frag, count=1)
        if 'class="tab-content"' not in frag and "tab-content" not in frag:
            frag = re.sub(r"<section\b", '<section class="tab-content"', frag, count=1)
        sections.append(frag)
    if missing:
        sys.exit(f"missing fragments for: {missing} — not assembling")
    active_attr = ' class="active"'
    buttons = "\n".join(
        f'  <button data-tab="{tid}"{active_attr if i == 0 else ""}>{label}</button>'
        for i, (tid, label) in enumerate(TABS)
    )
    html = SHELL.format(today=date.today().isoformat(), tab_buttons=buttons,
                        sections="\n\n".join(sections))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"assembled {OUT} ({os.path.getsize(OUT)} bytes, {len(sections)} tabs)")


if __name__ == "__main__":
    main()
