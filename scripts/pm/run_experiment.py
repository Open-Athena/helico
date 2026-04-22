"""Execute a jupytext-markdown experiment notebook and export HTML.

Usage:
    uv run python scripts/pm/run_experiment.py experiments/exp1_protenix_baseline/
    uv run python scripts/pm/run_experiment.py <dir>/README.md

    # Dry run: no Modal dispatch; prints cost estimates + total.
    HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py <dir>

Pipeline:
    README.md  --(jupytext)-->  .cache/README.ipynb  --(nbconvert)-->  README.html

The paired .ipynb lives under .cache/ so it's gitignored. The executed
.ipynb carries outputs; README.html is the rendered artifact published to
the site. README.md itself is never modified.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _find_notebook(target: Path) -> Path:
    if target.is_file():
        return target
    if target.is_dir():
        readme = target / "README.md"
        if readme.exists():
            return readme
    raise SystemExit(
        f"Could not find a notebook at {target}. Pass an experiment dir or a "
        f"path to a jupytext-markdown .md file."
    )


def _slug_from_notebook(nb: Path) -> str:
    """Infer experiment slug from a notebook path at experiments/<slug>/README.md."""
    parts = nb.resolve().parts
    try:
        i = parts.index("experiments")
    except ValueError:
        raise SystemExit(f"Notebook {nb} is not under an experiments/ tree")
    if i + 1 >= len(parts):
        raise SystemExit(f"Notebook {nb} is not inside an experiments/<slug>/ dir")
    return parts[i + 1]


def _run(cmd: list[str], env: dict | None = None, cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help="Experiment dir or path to a .md notebook")
    ap.add_argument(
        "--no-html", action="store_true",
        help="Skip HTML export (useful for dry runs that only need costs).",
    )
    args = ap.parse_args(argv)

    notebook_md = _find_notebook(args.target).resolve()
    exp_dir = notebook_md.parent
    slug = _slug_from_notebook(notebook_md)

    cache_dir = exp_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    notebook_ipynb = cache_dir / "README.ipynb"
    notebook_executed = cache_dir / "README.executed.ipynb"
    notebook_html = exp_dir / "README.html"

    env = os.environ.copy()
    env["HELICO_EXPERIMENT"] = slug

    # Step 1: md -> ipynb
    _run(
        ["uv", "run", "jupytext", "--to", "ipynb",
         "-o", str(notebook_ipynb), str(notebook_md)],
        env=env,
    )

    # Step 2: execute. Notebooks build artifact paths via experiment_dir()
    # (see the TEMPLATE), which resolves absolutely regardless of where
    # the kernel's cwd is set.
    _run(
        ["uv", "run", "jupyter", "nbconvert",
         "--to", "notebook", "--execute",
         "--output", notebook_executed.name,
         "--output-dir", str(cache_dir),
         str(notebook_ipynb)],
        env=env,
        cwd=exp_dir,
    )

    if args.no_html or os.environ.get("HELICO_DRY_RUN"):
        print(f"[run_experiment] executed {notebook_md.name}; HTML skipped.")
        return 0

    # Step 3: executed ipynb -> html
    _run(
        ["uv", "run", "jupyter", "nbconvert",
         "--to", "html",
         "--output", notebook_html.name,
         "--output-dir", str(exp_dir),
         str(notebook_executed)],
        env=env,
    )
    print(f"[run_experiment] wrote {notebook_html.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
