#!/usr/bin/env python3
"""Upload FoldBench server-MSA cache to HuggingFace (timodonnell/helico-data).

The bench pipeline caches ColabFold paired + non-paired MSAs under
``foldbench-msas-server/{query_hash}/{mode}/out.tar.gz`` (where mode is
``p_*`` for paired and ``np_*`` for non-paired). This script uploads
only the ``out.tar.gz`` files — the extracted ``.a3m`` files are
regenerated automatically on first access.

Usage:
    # Populate the cache by running bench with --use-msa-server first, then:
    python scripts/upload_msa_cache.py \\
        ~/.cache/helico/data/benchmarks/FoldBench/foldbench-msas-server

    # Or preview what would upload:
    python scripts/upload_msa_cache.py <cache_dir> --dry-run
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

HF_REPO = "timodonnell/helico-data"
HF_PREFIX = "benchmarks/FoldBench/foldbench-msas-server"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Upload MSA cache to HuggingFace")
    parser.add_argument("cache_dir", type=Path,
                        help="Path to foldbench-msas-server cache directory")
    parser.add_argument("--repo", default=HF_REPO, help=f"HF dataset repo (default: {HF_REPO})")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    if not cache_dir.exists():
        logger.error(f"Directory not found: {cache_dir}")
        sys.exit(1)

    # Only upload the out.tar.gz files — extracted a3ms are re-derivable.
    tar_files = sorted(cache_dir.rglob("out.tar.gz"))
    if not tar_files:
        logger.error(f"No out.tar.gz files found under {cache_dir}")
        sys.exit(1)

    total_size = sum(f.stat().st_size for f in tar_files)
    logger.info(f"Found {len(tar_files)} out.tar.gz files, total {total_size / (1024**2):.1f} MB")

    if args.dry_run:
        for f in tar_files:
            rel = f.relative_to(cache_dir)
            size_kb = f.stat().st_size / 1024
            print(f"  {HF_PREFIX}/{rel}  ({size_kb:.0f} KB)")
        logger.info("Dry run — nothing uploaded.")
        return

    # Stage a clean directory containing only the tar.gz files with preserved
    # structure, then upload once. HfApi.upload_folder ignores a glob via
    # ignore_patterns, but easier to just stage.
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for src in tar_files:
            rel = src.relative_to(cache_dir)
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

        api = HfApi()
        api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)

        logger.info(f"Uploading {len(tar_files)} tars to {args.repo}:{HF_PREFIX}/ ...")
        api.upload_folder(
            repo_id=args.repo,
            folder_path=str(stage),
            path_in_repo=HF_PREFIX,
            repo_type="dataset",
            commit_message=f"Add {len(tar_files)} FoldBench server-MSA cache entries",
        )

    logger.info(f"Upload complete: https://huggingface.co/datasets/{args.repo}/tree/main/{HF_PREFIX}")


if __name__ == "__main__":
    main()
