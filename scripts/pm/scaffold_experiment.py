"""Scaffold a new experiment directory from a GitHub issue.

Usage:
    uv run python scripts/pm/scaffold_experiment.py --issue 7
    uv run python scripts/pm/scaffold_experiment.py --issue 7 --slug my_slug
    uv run python scripts/pm/scaffold_experiment.py --issue 7 --branch exp/7-foo

Reads the issue via `gh api` (must be authenticated), derives a slug
from the title, and creates experiments/exp<N>_<slug>/README.md from
TEMPLATE.md with frontmatter pre-filled (issue, title, branch). The
notebook body is seeded with the issue's Question/Hypothesis/Background
sections if they exist; unfilled sections keep the template placeholders.

Does NOT overwrite an existing directory — re-run with --force to
scaffold again (will clobber README.md).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SLUG = "Open-Athena/helico"
TEMPLATE_PATH = REPO_ROOT / "experiments" / "TEMPLATE.md"


def fetch_issue(number: int) -> dict:
    out = subprocess.check_output(
        ["gh", "api", f"/repos/{REPO_SLUG}/issues/{number}"], text=True,
    )
    return json.loads(out)


def title_to_slug(title: str) -> str:
    # Strip common prefixes
    t = title.strip()
    for prefix in ("exp:", "experiment:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
            break
    # kebab-case-ish -> snake_case
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"[\s-]+", "_", t)
    t = t.strip("_").lower()
    # Cap to ~5 words
    parts = t.split("_")
    return "_".join(parts[:5]) or "experiment"


def extract_section(body: str, header_names: list[str]) -> Optional[str]:
    """Return the body under the first matching `## <name>` header.

    Stops at the next `## ` (or `#`) header or end of document.
    """
    lines = body.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("## "):
            continue
        label = stripped[3:].strip().lower()
        if any(label.startswith(h.lower()) for h in header_names):
            # Collect until next header
            out: list[str] = []
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("## ") or lines[j].strip().startswith("# "):
                    break
                out.append(lines[j])
            return _strip_template_placeholders("\n".join(out).strip())
    return None


def _strip_template_placeholders(text: str) -> str:
    """Drop the `<!-- placeholder -->` comment blocks and HTML comments
    that the issue template leaves behind if the user didn't edit them."""
    out = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return out.strip()


def render_notebook(
    *,
    issue: dict,
    slug: str,
    branch: str,
) -> str:
    template = TEMPLATE_PATH.read_text()
    # The template's jupytext/kernelspec frontmatter is fixed; we only
    # replace the `helico_experiment:` block and the title-ish first
    # heading. Simpler to regen the frontmatter by hand.
    frontmatter = (
        "---\n"
        "jupyter:\n"
        "  jupytext:\n"
        "    text_representation:\n"
        "      extension: .md\n"
        "      format_name: markdown\n"
        "      format_version: '1.3'\n"
        "  kernelspec:\n"
        "    name: python3\n"
        "    display_name: Python 3\n"
        "helico_experiment:\n"
        f"  issue: {issue['number']}\n"
        f"  title: \"{issue['title'].replace('\"', '\\\"')}\"\n"
        f"  branch: {branch}\n"
        "  baselines: []\n"
        "---\n\n"
    )

    title_line = f"# {issue['title']}\n\n"
    link_line = (
        f"**Issue:** [#{issue['number']}]({issue['html_url']}) · "
        f"**Branch:** `{branch}`\n\n"
    )

    body = issue.get("body") or ""

    question = extract_section(body, ["Question"]) or ""
    hypothesis = extract_section(body, ["Hypothesis"]) or ""
    background = extract_section(body, ["Background"]) or ""

    sections = [
        ("## Question", question or "_(Copy from the issue.)_"),
        ("## Hypothesis", hypothesis or "_(Copy from the issue.)_"),
    ]
    if background:
        sections.append(("## Background", background))

    prose = ""
    for header, content in sections:
        prose += f"{header}\n\n{content}\n\n"

    # Keep the setup/run/analyze/conclusion scaffolding from TEMPLATE.md
    # by slicing out everything after the first `## Setup` header.
    try:
        setup_idx = template.index("## Setup")
    except ValueError:
        setup_idx = None
    if setup_idx is not None:
        prose += template[setup_idx:]
    else:
        prose += (
            "## Setup\n\n"
            "```python\n"
            "from helico.experiment import ensure_bench_run, experiment_dir, set_experiment\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            f"set_experiment(\"exp{issue['number']}_{slug}\")\n"
            "DATA = experiment_dir() / \"data\"\n"
            "PLOTS = experiment_dir() / \"plots\"\n"
            "DATA.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)\n"
            "```\n\n"
            "## Run\n\n"
            "_(Add ensure_bench_run / ensure_training_run calls here.)_\n\n"
            "## Conclusion\n\n"
            "_(Fill in after the run completes.)_\n"
        )

    return frontmatter + title_line + link_line + prose


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", type=int, required=True,
                    help="GitHub issue number")
    ap.add_argument("--slug", default=None,
                    help="Override the auto-derived slug")
    ap.add_argument("--branch", default="main",
                    help="Branch the experiment lives on (default: main)")
    ap.add_argument("--force", action="store_true",
                    help="Clobber an existing README.md")
    args = ap.parse_args(argv)

    issue = fetch_issue(args.issue)
    slug = args.slug or title_to_slug(issue["title"])

    # Check for any existing exp<N>_* directory; otherwise a slug change
    # would silently create a second notebook for the same issue.
    existing = sorted(
        p for p in (REPO_ROOT / "experiments").glob(f"exp{args.issue}_*")
        if p.is_dir()
    )
    if existing and not args.force:
        existing_names = ", ".join(p.name for p in existing)
        print(f"Experiment for issue #{args.issue} already exists: {existing_names}",
              file=sys.stderr)
        print("Re-run with --force to clobber README.md, or edit the existing notebook.",
              file=sys.stderr)
        return 1

    exp_dir = REPO_ROOT / "experiments" / f"exp{args.issue}_{slug}"
    readme = exp_dir / "README.md"

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)

    readme.write_text(render_notebook(issue=issue, slug=slug, branch=args.branch))

    print(f"Scaffolded {readme.relative_to(REPO_ROOT)}")
    print(f"Next steps:")
    print(f"  1. Edit {readme.relative_to(REPO_ROOT)}: set the ensure_* calls, success criteria, baselines")
    print(f"  2. HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py {exp_dir.relative_to(REPO_ROOT)}/")
    print(f"  3. uv run python scripts/pm/run_experiment.py {exp_dir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
