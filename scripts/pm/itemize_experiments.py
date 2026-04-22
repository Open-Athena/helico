"""Regenerate docs/experiments/index.md from gh issues + notebook frontmatter.

Usage:
    uv run python scripts/pm/itemize_experiments.py
    uv run python scripts/pm/itemize_experiments.py --check   # exit non-zero if stale

Design choices:

- Source of truth for WHICH experiments exist is the GitHub issue list
  filtered by the `experiment` label. Each experiment's notebook dir is
  named `experiments/exp<issue#>_<slug>/`, so we can cross-reference by
  issue number.
- Per-experiment metadata (title, branch) comes from the notebook's
  frontmatter when present; falls back to the issue title otherwise.
- Experiment dirs without a matching issue are listed under "Orphans"
  so we notice and fix the naming.

Requires `gh` CLI authenticated against Open-Athena/helico.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SLUG = "Open-Athena/helico"


def list_experiment_issues() -> list[dict]:
    out = subprocess.check_output(
        [
            "gh", "issue", "list",
            "--repo", REPO_SLUG,
            "--label", "experiment",
            "--state", "all",
            "--limit", "200",
            "--json", "number,title,state,url,createdAt,closedAt",
        ],
        text=True,
    )
    return json.loads(out)


def read_frontmatter(readme_md: Path) -> Optional[dict]:
    if not readme_md.exists():
        return None
    content = readme_md.read_text()
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None


def scan_experiment_dirs() -> dict[int, Path]:
    """Map issue number -> experiments/exp<N>_*/ directory."""
    out: dict[int, Path] = {}
    exp_root = REPO_ROOT / "experiments"
    if not exp_root.is_dir():
        return out
    for p in exp_root.iterdir():
        if not p.is_dir() or not p.name.startswith("exp"):
            continue
        # Filename convention: exp<N>_<slug>. Anything else (TEMPLATE.md,
        # AGENTS.md) is skipped.
        head = p.name.split("_", 1)[0]
        if not head[3:].isdigit():
            continue
        try:
            num = int(head[3:])
        except ValueError:
            continue
        out[num] = p
    return out


def _render_row(issue: dict, exp_dir: Optional[Path]) -> str:
    number = issue["number"]
    issue_url = issue["url"]
    # Escape pipe in titles so table doesn't break
    title = issue["title"].replace("|", "&#124;")

    if exp_dir is not None:
        slug = exp_dir.name
        nb_url = f"https://github.com/{REPO_SLUG}/blob/main/experiments/{slug}/README.md"
        fm = read_frontmatter(exp_dir / "README.md") or {}
        branch = (fm.get("helico_experiment") or {}).get("branch") or "?"
        fm_title = (fm.get("helico_experiment") or {}).get("title")
        display_title = fm_title or title
        notebook_cell = f"[`{slug}`]({nb_url})"
    else:
        display_title = title
        branch = "—"
        notebook_cell = "_no notebook yet_"

    return f"| [#{number}]({issue_url}) | {display_title} | `{branch}` | {notebook_cell} |"


def _render_orphan_row(exp_dir: Path) -> str:
    slug = exp_dir.name
    fm = read_frontmatter(exp_dir / "README.md") or {}
    hel = fm.get("helico_experiment") or {}
    title = hel.get("title", slug)
    issue_num = hel.get("issue")
    branch = hel.get("branch", "?")
    nb_url = f"https://github.com/{REPO_SLUG}/blob/main/experiments/{slug}/README.md"
    issue_cell = f"#{issue_num}" if issue_num else "—"
    return f"| {issue_cell} | {title} | `{branch}` | [`{slug}`]({nb_url}) |"


TABLE_HEADER = "| Issue | Title | Branch | Notebook |\n|---|---|---|---|"


def render_index(issues: list[dict], exp_dirs: dict[int, Path]) -> str:
    open_rows = []
    closed_rows = []
    matched_issue_numbers: set[int] = set()
    for issue in sorted(issues, key=lambda i: -int(i["number"])):
        matched_issue_numbers.add(issue["number"])
        exp_dir = exp_dirs.get(issue["number"])
        row = _render_row(issue, exp_dir)
        if issue["state"].lower() == "open":
            open_rows.append(row)
        else:
            closed_rows.append(row)

    orphan_rows = [
        _render_orphan_row(p)
        for n, p in sorted(exp_dirs.items())
        if n not in matched_issue_numbers
    ]

    lines = [
        "# Experiments",
        "",
        "Every research question is documented as a GitHub issue tagged ",
        "`experiment`. The experiment's notebook lives at ",
        "`experiments/exp<N>_<slug>/README.md` and is a self-contained record:",
        "prose, Modal invocations (training, benchmarks), analysis, and plots.",
        "Raw result CSVs are committed alongside the notebook under `data/`",
        "so every plot is re-plottable without rerunning Modal.",
        "",
        "_This page is generated by `scripts/pm/itemize_experiments.py` from_",
        "_`gh issue list --label experiment` + each notebook's frontmatter._",
        "",
        "## Open",
        "",
    ]
    if open_rows:
        lines.append(TABLE_HEADER)
        lines.extend(open_rows)
    else:
        lines.append("_(No open experiments.)_")
    lines.extend(["", "## Closed", ""])
    if closed_rows:
        lines.append(TABLE_HEADER)
        lines.extend(closed_rows)
    else:
        lines.append("_(No closed experiments yet.)_")

    if orphan_rows:
        lines.extend([
            "",
            "## Orphans",
            "",
            "Experiment directories that don't match any open or closed issue "
            "labelled `experiment`. Fix the name or file an issue.",
            "",
            TABLE_HEADER,
        ])
        lines.extend(orphan_rows)

    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "docs" / "experiments" / "index.md",
        help="Output path",
    )
    ap.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if the existing file is out of date; don't write.",
    )
    args = ap.parse_args(argv)

    issues = list_experiment_issues()
    exp_dirs = scan_experiment_dirs()
    content = render_index(issues, exp_dirs)

    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current.strip() != content.strip():
            print(f"STALE: {args.output}", file=sys.stderr)
            print("Regenerate with: uv run python scripts/pm/itemize_experiments.py",
                  file=sys.stderr)
            return 1
        print(f"up to date: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
