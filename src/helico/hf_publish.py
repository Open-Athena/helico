"""Publish an experiment's artifacts to HuggingFace.

For the Wave 2 MVP this uploads a bench run's outputs
(summary.csv, per-category CSVs, plots, a generated model card) to a
HuggingFace Bucket. Training-checkpoint publishing is a planned
followup (needs ensure_training_run first).

CLI:
    helico-publish bench --experiment exp4_baseline_protenix_v1 \\
                         --name protenix-v1-default

    helico-publish bench --experiment exp4_baseline_protenix_v1 \\
                         --name protenix-v1-default \\
                         --bucket timodonnell/helico-experiments \\
                         --include-plots
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


_MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _MODULE_DIR.parent.parent
DEFAULT_BUCKET = "timodonnell/helico-experiments"


def _load_meta(cache_dir: Path) -> dict:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path} — run hasn't completed yet?")
    with open(meta_path) as f:
        return json.load(f)


def _render_model_card(
    *,
    experiment: str,
    name: str,
    meta: dict,
    summary_csv: Path,
    issue: Optional[int],
    wandb_url: Optional[str],
) -> str:
    """Write a README.md-style card that HF renders as the bucket entry's page."""
    lines: list[str] = []
    lines.append(f"# {experiment} / {name}")
    lines.append("")

    header = []
    if issue is not None:
        header.append(f"**Issue:** [#{issue}](https://github.com/Open-Athena/helico/issues/{issue})")
    git_sha = meta.get("git_sha")
    if git_sha:
        short = git_sha[:8]
        header.append(f"**Commit:** [`{short}`](https://github.com/Open-Athena/helico/commit/{git_sha})")
    if wandb_url:
        header.append(f"**WandB:** [{wandb_url}]({wandb_url})")
    if header:
        lines.append(" · ".join(header))
        lines.append("")

    lines.append("## Bench configuration")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for k in ("checkpoint", "gpu", "workers", "n_samples", "max_tokens",
              "n_cycles", "cutoff_date", "categories"):
        if k in meta:
            lines.append(f"| `{k}` | `{meta[k]}` |")
    if meta.get("est_cost_usd") is not None:
        lines.append(f"| `est_cost_usd` | `${meta['est_cost_usd']:.2f}` |")
    lines.append("")

    lines.append("## FoldBench summary")
    lines.append("")
    if summary_csv.exists():
        rows = summary_csv.read_text().strip().splitlines()
        if rows:
            headers = rows[0].split(",")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join("---" for _ in headers) + "|")
            for r in rows[1:]:
                lines.append("| " + " | ".join(r.split(",")) + " |")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `summary.csv` — per-category aggregate metrics")
    lines.append("- `results/{category}.csv` — per-target metrics for each of 9 FoldBench categories")
    lines.append("- `meta.json` — full run metadata (hyperparameters, git sha, cost estimate)")
    lines.append("")

    return "\n".join(lines)


def _run_hf_buckets_sync(src_dir: Path, dest: str) -> None:
    """Shell out to `hf buckets sync`. Creates the bucket if missing."""
    # Split bucket/namespace from path-within-bucket if `dest` has subpath.
    bucket_id, _, _ = dest.partition("/")
    # First three slashes: namespace/bucket/path... Buckets IDs are `{user}/{bucket}`.
    # The dest arg to `hf buckets sync` is the bucket URI including subpath.
    subprocess.run(
        ["hf", "buckets", "sync", str(src_dir), dest],
        check=True,
    )


def publish_bench_run(
    *,
    experiment: str,
    name: str,
    bucket: str = DEFAULT_BUCKET,
    issue: Optional[int] = None,
    wandb_url: Optional[str] = None,
    include_plots: bool = True,
    dry_run: bool = False,
) -> str:
    """Upload a local bench run's artifacts to a HuggingFace Bucket.

    Returns the destination URI (bucket/path) on success.
    """
    exp_dir = REPO_ROOT / "experiments" / experiment
    if not exp_dir.is_dir():
        raise SystemExit(f"No such experiment directory: {exp_dir}")
    cache_dir = exp_dir / ".cache" / "benches" / name
    if not cache_dir.is_dir():
        raise SystemExit(f"No bench cache at {cache_dir} — has the run completed?")

    meta = _load_meta(cache_dir)
    if meta.get("dry_run"):
        raise SystemExit(
            f"Refusing to publish a dry-run placeholder at {cache_dir}. "
            "Re-run without HELICO_DRY_RUN before publishing."
        )

    # If issue wasn't passed explicitly, try reading it from the notebook's
    # frontmatter so the CLI is zero-config for the common case.
    if issue is None:
        issue = _infer_issue_from_frontmatter(exp_dir / "README.md")

    summary_csv = cache_dir / "summary.csv"
    model_card = _render_model_card(
        experiment=experiment, name=name, meta=meta,
        summary_csv=summary_csv, issue=issue, wandb_url=wandb_url,
    )

    with tempfile.TemporaryDirectory() as stage:
        stage_dir = Path(stage)

        # Copy bench artifacts
        for item in cache_dir.iterdir():
            dst = stage_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        # Copy plots from the experiment dir if requested
        plots_dir = exp_dir / "plots"
        if include_plots and plots_dir.is_dir() and any(plots_dir.iterdir()):
            shutil.copytree(plots_dir, stage_dir / "plots", dirs_exist_ok=True)

        # Write model card
        (stage_dir / "README.md").write_text(model_card)

        dest = f"{bucket}/{experiment}-{name}"
        if dry_run:
            print(f"[dry-run] would sync {stage_dir} to {dest}")
            print(f"[dry-run] staged contents:")
            for p in sorted(stage_dir.rglob("*")):
                if p.is_file():
                    print(f"  {p.relative_to(stage_dir)} ({p.stat().st_size} bytes)")
            return dest

        print(f"Syncing {stage_dir} -> {dest}")
        _run_hf_buckets_sync(stage_dir, dest)

    return f"https://huggingface.co/buckets/{bucket}/{experiment}-{name}"


def _infer_issue_from_frontmatter(readme_md: Path) -> Optional[int]:
    """Read the helico_experiment.issue field from a jupytext notebook's frontmatter."""
    if not readme_md.exists():
        return None
    try:
        import yaml  # deferred — only when publish runs
    except ImportError:
        return None
    content = readme_md.read_text()
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None
    exp = fm.get("helico_experiment") or {}
    v = exp.get("issue")
    return int(v) if isinstance(v, (int, str)) and str(v).isdigit() else None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="helico-publish")
    sub = ap.add_subparsers(dest="command", required=True)

    bench_ap = sub.add_parser("bench", help="Publish a bench run to a HF Bucket")
    bench_ap.add_argument("--experiment", required=True,
                          help="Experiment slug (e.g. exp4_baseline_protenix_v1)")
    bench_ap.add_argument("--name", required=True,
                          help="Step name used in ensure_bench_run (e.g. protenix-v1-default)")
    bench_ap.add_argument("--bucket", default=DEFAULT_BUCKET,
                          help=f"HF bucket id (default: {DEFAULT_BUCKET})")
    bench_ap.add_argument("--issue", type=int, default=None,
                          help="GitHub issue number (inferred from frontmatter if omitted)")
    bench_ap.add_argument("--wandb-url", default=None,
                          help="Optional WandB run URL to embed in the card")
    bench_ap.add_argument("--no-plots", dest="include_plots", action="store_false",
                          help="Skip uploading experiment plots/ dir")
    bench_ap.add_argument("--dry-run", action="store_true",
                          help="Stage files locally but don't upload")

    args = ap.parse_args(argv)
    if args.command == "bench":
        url = publish_bench_run(
            experiment=args.experiment,
            name=args.name,
            bucket=args.bucket,
            issue=args.issue,
            wandb_url=args.wandb_url,
            include_plots=args.include_plots,
            dry_run=args.dry_run,
        )
        print(f"Published: {url}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
