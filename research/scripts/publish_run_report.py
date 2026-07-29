#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E501  (inline report-template CSS lines are long by design)

"""publish_run_report.py — publish a finished run's report to the Cloudflare
Pages repo (project.yaml `reports:` block) and its bulk artifacts to R2.

    python3 scripts/publish_run_report.py --issue 62 --run-id 62-rlvr-models-datasets [--no-push] [--dry-run]

What it does (all paths from project.yaml — nothing hardcoded):
  1. Gathers the run's record: the GitHub issue (title/labels/close comment) +
     runs/<id>/{verdict.md, run.json, resolved_params.txt} when they still exist.
  2. Renders ONE self-contained HTML page → <reports.repo_dir>/<runs_dir>/<id>.html
     (site-styled, linked to the issue, WandB group, PR, R2 prefix).
  3. Inserts a card into <runs_dir>/index.html at the `<!-- runs:insert -->`
     marker (idempotent — skips if the run already has a card).
  4. Copies SMALL artifacts (≤ --small-mb, default 20 MB; excludes handles/ and
     provision logs) → <repo>/<artifacts_dir>/<id>/  (gitignored, local-only).
  5. Uploads LARGE artifacts to R2: s3://$R2_BUCKET/<r2.prefix>/<id>/…
     via `aws --endpoint-url $R2_ENDPOINT` (creds from env or the secrets file;
     values are never printed). Missing creds/CLI → warn and continue.
  6. Commits the report repo; pushes unless --no-push (a push IS the Cloudflare
     Pages deploy — reports.push_on_publish).

Run by /close's log-writer AFTER the verdict is final; safe to re-run.
Degrades gracefully: a deleted runs/<id>/ still yields a page built from the
issue's close comment alone (labels+GitHub are the state, files are evidence).
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


def _resolve_research() -> Path:
    """Anchor to the PRIMARY checkout's research/ (same rule as _lib.sh
    lib_research_dir) — /close may run in a per-issue worktree whose runs/
    is empty while the artifacts live in the primary checkout."""
    here = Path(__file__).resolve().parent.parent  # this file's research/
    try:
        out = subprocess.run(
            ["git", "-C", str(here), "worktree", "list", "--porcelain"], capture_output=True, text=True, timeout=15
        )
        first = next((ln.split(" ", 1)[1] for ln in out.stdout.splitlines() if ln.startswith("worktree ")), None)
        if first and (Path(first) / "research").is_dir():
            return Path(first) / "research"
    except Exception:
        pass
    return here


RESEARCH = _resolve_research()  # primary checkout's research/
PROJECT_YAML = RESEARCH / ".claude" / "project.yaml"
SECRETS_FILE = Path(os.path.expanduser("~/.config/verl-research/secrets.env"))
MARKER = "<!-- runs:insert -->"


# ── tiny line-based YAML reads (stdlib-only; flat keys under a known block) ──
def yaml_block_value(block: str, key: str, default: str = "") -> str:
    in_block, in_sub = False, False
    for line in PROJECT_YAML.read_text().splitlines():
        if re.match(rf"^{block}:", line):
            in_block = True
            continue
        if in_block and re.match(r"^[a-zA-Z_]", line):  # next top-level block
            break
        if in_block:
            m = re.match(rf"^  {key}:\s*(.*?)\s*(?:#.*)?$", line)
            if m and not in_sub:
                return m.group(1).strip().strip('"')
            if re.match(r"^  r2:", line):
                in_sub = True
            elif re.match(r"^  [a-zA-Z_]", line):
                in_sub = False
            m = re.match(rf"^    {key}:\s*(.*?)\s*(?:#.*)?$", line)
            if m and in_sub:
                return m.group(1).strip().strip('"')
    return default


def load_secrets_env() -> dict:
    """R2_* creds from the environment, falling back to the secrets file.
    Values are returned for subprocess env only — NEVER printed or logged."""
    keys = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_ENDPOINT", "R2_ACCOUNT_ID"]
    out = {k: os.environ.get(k, "") for k in keys}
    if all(out[k] for k in keys[:4]):
        return out
    if SECRETS_FILE.exists():
        for line in SECRETS_FILE.read_text().splitlines():
            m = re.match(r"^(?:export\s+)?([A-Z_0-9]+)=(.*)$", line.strip())
            if m and m.group(1) in keys and not out.get(m.group(1)):
                out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


# ── minimal markdown → html (headers, bold, code, fences, tables, lists, links) ──
def md_to_html(md: str) -> str:
    def inline(s: str) -> str:
        s = html.escape(s, quote=False)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
        s = re.sub(r"(?<![\"'>=])(https?://[^\s<)]+)", r'<a href="\1">\1</a>', s)
        return s

    out, i, lines = [], 0, md.splitlines()
    in_list = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):  # code fence
            fence = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                fence.append(lines[i])
                i += 1
            fence_html = html.escape("\n".join(fence))
            out.append(f"<pre><code>{fence_html}</code></pre>")
            i += 1
            continue
        # a GFM separator row must carry BOTH '-' and '|' — a bare '---' hr or a
        # whitespace-only line must not turn the preceding prose into a table
        if (
            "|" in ln
            and i + 1 < len(lines)
            and re.match(r"^\s*[|: -]+$", lines[i + 1])
            and "-" in lines[i + 1]
            and "|" in lines[i + 1]
        ):
            if in_list:
                out.append("</ul>")
                in_list = False
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]
            out.append(
                '<div class="tablewrap"><table><thead><tr>'
                + "".join(f"<th>{inline(c)}</th>" for c in hdr)
                + "</tr></thead><tbody>"
            )
            i += 2
            while i < len(lines) and "|" in lines[i]:
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table></div>")
            continue
        if in_list and not re.match(r"^\s*[-*] ", ln):
            out.append("</ul>")
            in_list = False
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl = min(len(m.group(1)) + 1, 5)  # h1 in md → h2 on page
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
        elif re.match(r"^\s*[-*] ", ln):
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = re.sub(r"^\s*[-*] ", "", ln)
            out.append(f"<li>{inline(item)}</li>")
        elif ln.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


# ── data gathering ──
def gh_issue(repo: str, n: int) -> dict:
    try:
        raw = subprocess.run(
            ["gh", "issue", "view", str(n), "--repo", repo, "--json", "title,url,labels,closedAt,comments,body"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
        return json.loads(raw)
    except Exception as e:  # offline degradation
        print(f"warn: gh issue view failed ({e}) — building from local files only", file=sys.stderr)
        return {}


def find_verdict_comment(issue: dict) -> str:
    for c in reversed(issue.get("comments", [])):
        if "VERDICT" in c.get("body", ""):
            return c["body"]
    return ""


def parse_verdict(*texts: str) -> str:
    for t in texts:
        m = re.search(r"VERDICT:?\**\s*\**(PASS|REVISE|STOP)", t or "")
        if m:
            return m.group(1)
    return "DONE"


# ── page rendering ──
PAGE_CSS = """
:root{--ink:#162025;--muted:#52636c;--line:#d8e0df;--paper:#fff;--bg:#f5f7f4;
--green:#0f6d58;--blue:#195e9f;--amber:#8b5a13;--rose:#9e3443;
--green-soft:#e3f2eb;--blue-soft:#e6eff8;--amber-soft:#f7ecd7;--rose-soft:#f8e5e8;--maxw:980px}
*{box-sizing:border-box}html{background:var(--bg)}
body{margin:0;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased}
.shell{width:min(var(--maxw),calc(100% - 32px));margin:0 auto;padding:28px 0 60px}
a{color:var(--blue)}
h1{font-size:clamp(26px,4vw,40px);line-height:1.1;margin:10px 0 6px}
h2{font-size:22px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:17px;margin:22px 0 8px}
.crumb{color:var(--muted);font-size:14px}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.badge{display:inline-flex;align-items:center;min-height:26px;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:.03em}
.badge.pass{color:var(--green);background:var(--green-soft)}
.badge.stop{color:var(--rose);background:var(--rose-soft)}
.badge.revise{color:var(--amber);background:var(--amber-soft)}
.badge.info{color:var(--blue);background:var(--blue-soft)}
.links{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 6px}
.links a{display:inline-flex;align-items:center;min-height:36px;padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:var(--paper);text-decoration:none;font-weight:700;font-size:14px}
section{margin:18px 0;padding:20px 22px;border:1px solid var(--line);border-radius:8px;background:var(--paper)}
pre{overflow-x:auto;background:#f2f5f4;border:1px solid var(--line);border-radius:8px;padding:12px;font-size:13px}
code{background:#f2f5f4;border-radius:4px;padding:1px 4px;font-size:.92em}
pre code{background:none;padding:0}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}
th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em;background:#f6f8f7}
ul{padding-left:22px}
details{margin:12px 0}summary{cursor:pointer;font-weight:700}
.footer{color:var(--muted);font-size:13px;margin-top:34px}
"""

VERDICT_CLASS = {"PASS": "pass", "STOP": "stop", "REVISE": "revise", "DONE": "info"}


def render_page(
    cfg, run_id, issue_n, issue, verdict, close_md, verdict_md, resolved_txt, wandb_url, r2_url, pr_url, artifacts_rel
):
    title = issue.get("title", run_id)
    closed = (issue.get("closedAt") or "")[:10] or date.today().isoformat()
    kind = next((lb["name"].split(":", 1)[1] for lb in issue.get("labels", []) if lb["name"].startswith("kind:")), "")
    parts = []
    parts.append(f'<p class="crumb"><a href="index.html">← All runs</a> · {run_id}</p>')
    parts.append(f"<h1>{html.escape(title)}</h1>")
    badges = [
        f'<span class="badge {VERDICT_CLASS.get(verdict, "info")}">{verdict}</span>',
        f'<span class="badge info">closed {closed}</span>',
    ]
    if kind:
        badges.append(f'<span class="badge info">kind: {html.escape(kind)}</span>')
    parts.append('<div class="badges">' + "".join(badges) + "</div>")
    links = []
    if issue.get("url"):
        links.append(f'<a href="{issue["url"]}">GitHub issue #{issue_n}</a>')
    if pr_url:
        links.append(f'<a href="{pr_url}">Code PR</a>')
    if wandb_url:
        links.append(f'<a href="{wandb_url}">WandB group</a>')
    if r2_url:
        links.append(f'<a href="{r2_url}">R2 artifacts</a>')
    parts.append('<div class="links">' + "".join(links) + "</div>")
    if artifacts_rel:
        parts.append(
            f'<p class="crumb">Small artifacts (local only, not deployed): '
            f"<code>{html.escape(artifacts_rel)}</code></p>"
        )
    if close_md:
        parts.append("<section><h2>Close-out verdict (issue record)</h2>" + md_to_html(close_md) + "</section>")
    if verdict_md:
        parts.append("<section><h2>Analyst verdict (verdict.md)</h2>" + md_to_html(verdict_md) + "</section>")
    if resolved_txt:
        parts.append(
            "<section><h2>Provenance</h2><details><summary>resolved_params.txt "
            "(what actually ran)</summary><pre>" + html.escape(resolved_txt) + "</pre></details></section>"
        )
    parts.append(
        '<p class="footer">Generated by the research harness at /close '
        "(publish_run_report.py) · source of truth: the GitHub issue thread.</p>"
    )
    body = "\n".join(parts)
    return (
        f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(run_id)} · run report</title>\n"
        f'<style>{PAGE_CSS}</style>\n</head>\n<body>\n<div class="shell">\n'
        f"{body}\n</div>\n</body>\n</html>\n"
    )


def index_card(run_id, issue_n, issue, verdict, one_liner):
    closed = (issue.get("closedAt") or "")[:10] or date.today().isoformat()
    title = html.escape(issue.get("title", run_id))
    return f"""      <a class="run-card" href="{run_id}.html" data-run="{run_id}">
        <div class="card-top">
          <span class="date">{closed}</span>
          <span class="status {VERDICT_CLASS.get(verdict, "info")}">{verdict}</span>
        </div>
        <h3>#{issue_n} · {title}</h3>
        <p>{html.escape(one_liner)}</p>
        <div class="link-row">Open run report</div>
      </a>
"""


# ── main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--small-mb",
        type=float,
        default=20.0,
        help="files ≤ this size copy to the local artifacts dir; larger go to R2",
    )
    args = ap.parse_args()
    n, run_id = args.issue, args.run_id

    repo_dir = Path(yaml_block_value("reports", "repo_dir"))
    site_url = yaml_block_value("reports", "site_url")
    runs_dir = repo_dir / yaml_block_value("reports", "runs_dir", "runs")
    artifacts_dir = repo_dir / yaml_block_value("reports", "artifacts_dir", "artifacts") / run_id
    r2_prefix = yaml_block_value("reports", "prefix", "autonomous-harness-rlvr-compression")
    push_on_publish = yaml_block_value("reports", "push_on_publish", "true") == "true"
    research_repo = yaml_block_value("github", "research_repo")
    wandb_entity = yaml_block_value("wandb", "entity")
    wandb_project = yaml_block_value("wandb", "project")  # noqa: F841  (parsed for parity with entity)
    if not repo_dir.is_dir():
        sys.exit(f"REFUSED: reports.repo_dir {repo_dir} does not exist")

    run_dir = RESEARCH / "runs" / run_id
    issue = gh_issue(research_repo, n)
    close_md = find_verdict_comment(issue)
    verdict_md = (run_dir / "verdict.md").read_text() if (run_dir / "verdict.md").exists() else ""
    resolved = (run_dir / "resolved_params.txt").read_text() if (run_dir / "resolved_params.txt").exists() else ""
    verdict = parse_verdict(close_md, verdict_md)
    pr_m = re.search(r"https://github\.com/[\w./-]+/pull/\d+", close_md or "")
    pr_url = pr_m.group(0) if pr_m else ""
    # WandB uses a PER-ISSUE project == run_id (the launcher exports PROJECT_NAME=$RUN_ID
    # and WANDB_RUN_GROUP=$RUN_ID). project.yaml wandb.project is the GLOBAL default, NOT
    # what per-issue runs log to, so link the per-issue project view directly.
    wandb_url = f"https://wandb.ai/{wandb_entity}/{run_id}" if wandb_entity else ""
    one_liner = ""
    if close_md:
        for ln in close_md.splitlines():
            if ln.strip() and "VERDICT" not in ln and not ln.startswith(("|", "#")):
                one_liner = ln.strip()[:220]
                break

    # R2: upload large files under runs/<id>/ (skip handles/ + provision logs).
    # Any failed/skipped bulk upload is a PROBLEM: cleanup would delete the only
    # copy — the script must exit nonzero so /close flags instead of cleaning.
    problems = []
    r2 = load_secrets_env()
    r2_console = yaml_block_value("reports", "console")
    r2_url = f"{r2_console}?prefix={r2_prefix}%2F{run_id}%2F" if r2_console else ""
    uploaded, copied = [], []
    if run_dir.is_dir():
        for f in sorted(run_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(run_dir)
            if rel.parts[0] == "handles" or re.match(r"provision\..*\.log", rel.name):
                continue
            size_mb = f.stat().st_size / 1e6
            if size_mb <= args.small_mb:
                dest = artifacts_dir / rel
                if not args.dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
                copied.append(str(rel))
            else:
                key = f"s3://{r2['R2_BUCKET']}/{r2_prefix}/{run_id}/{rel}"
                if args.dry_run:
                    uploaded.append(str(rel))
                elif r2["R2_ACCESS_KEY_ID"] and shutil.which("aws"):
                    env = dict(
                        os.environ,
                        AWS_ACCESS_KEY_ID=r2["R2_ACCESS_KEY_ID"],
                        AWS_SECRET_ACCESS_KEY=r2["R2_SECRET_ACCESS_KEY"],
                    )
                    try:
                        rc = subprocess.run(
                            ["aws", "s3", "cp", str(f), key, "--endpoint-url", r2["R2_ENDPOINT"]],
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=1800,
                        )
                        err = rc.stderr[-300:] if rc.returncode != 0 else ""
                    except (subprocess.TimeoutExpired, OSError) as e:
                        rc, err = None, str(e)[-300:]
                    if rc is not None and rc.returncode == 0:
                        uploaded.append(str(rel))
                    else:
                        problems.append(f"R2 upload failed: {rel}")
                        print(f"warn: R2 upload failed for {rel}: {err}", file=sys.stderr)
                else:
                    problems.append(f"R2 skipped (no creds/aws CLI): {rel} ({size_mb:.0f} MB)")
                    print(f"warn: no R2 creds/aws CLI — skipped {rel} ({size_mb:.0f} MB)", file=sys.stderr)

    # page + index card
    page = render_page(
        cfg=None,
        run_id=run_id,
        issue_n=n,
        issue=issue,
        verdict=verdict,
        close_md=close_md,
        verdict_md=verdict_md,
        resolved_txt=resolved,
        wandb_url=wandb_url,
        r2_url=r2_url,
        pr_url=pr_url,
        artifacts_rel=f"artifacts/{run_id}/" if copied else "",
    )
    page_path = runs_dir / f"{run_id}.html"
    index_path = runs_dir / "index.html"
    if args.dry_run:
        print(f"[dry-run] write {page_path}\n[dry-run] card into {index_path}")
    else:
        runs_dir.mkdir(parents=True, exist_ok=True)
        page_path.write_text(page)
        if index_path.exists():
            idx = index_path.read_text()
            if f'data-run="{run_id}"' in idx:
                print(f"index: card for {run_id} already present — page refreshed only")
            elif MARKER in idx:
                idx = idx.replace(MARKER, MARKER + "\n" + index_card(run_id, n, issue, verdict, one_liner), 1)
                index_path.write_text(idx)
            else:
                print(f"warn: {index_path} lacks the {MARKER} marker — card not inserted", file=sys.stderr)
        else:
            print(f"warn: {index_path} missing — page written, no index update", file=sys.stderr)

    # commit (+ push = deploy) the report repo — every failure is a PROBLEM
    if not args.dry_run:
        to_add = [str(p) for p in (page_path, index_path) if p.exists()]
        try:
            rc = subprocess.run(
                ["git", "-C", str(repo_dir), "add", *to_add], capture_output=True, text=True, timeout=60
            )
            if rc.returncode != 0:
                problems.append("git add failed in report repo")
                print(f"warn: report-repo add failed: {rc.stderr[-200:]}", file=sys.stderr)
            rc = subprocess.run(
                ["git", "-C", str(repo_dir), "commit", "-q", "-m", f"run report: {run_id} (#{n}) {verdict}"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if rc.returncode != 0 and "nothing to commit" not in (rc.stdout + rc.stderr):
                problems.append("git commit failed in report repo")
                print(f"warn: report-repo commit failed: {(rc.stdout + rc.stderr)[-200:]}", file=sys.stderr)
        except (subprocess.TimeoutExpired, OSError) as e:
            problems.append("git add/commit errored in report repo")
            print(f"warn: report-repo git error: {str(e)[-200:]}", file=sys.stderr)
        if push_on_publish and not args.no_push:
            try:
                rc = subprocess.run(["git", "-C", str(repo_dir), "push"], capture_output=True, text=True, timeout=120)
                if rc.returncode == 0:
                    print("report repo pushed — Cloudflare Pages will deploy")
                else:
                    problems.append("push failed (report committed locally, NOT deployed)")
                    print(f"warn: push failed: {rc.stderr[-200:]}", file=sys.stderr)
            except (subprocess.TimeoutExpired, OSError) as e:
                problems.append("push errored (report committed locally, NOT deployed)")
                print(f"warn: push errored: {str(e)[-200:]}", file=sys.stderr)

    tail = (
        f" · small_artifacts={len(copied)} → {artifacts_dir if copied else '—'}"
        f" · r2_uploads={len(uploaded)} → {r2_prefix}/{run_id}/"
    )
    if problems and not args.dry_run:
        # nonzero exit so /close flags a human instead of running the cleanup
        # sweep — a partial publish must never look like success.
        print(f"REPORT_PUBLISH_PARTIAL: {site_url}/runs/{run_id}.html{tail} · problems: {'; '.join(problems)}")
        sys.exit(2)
    print(f"REPORT_PUBLISHED: {site_url}/runs/{run_id}.html{tail}")


if __name__ == "__main__":
    main()
