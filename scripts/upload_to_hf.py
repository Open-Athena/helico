#!/usr/bin/env python3
"""Upload processed Helico data to HuggingFace dataset repo.

Usage:
    python scripts/upload_to_hf.py /data/tim/helico-data/processed

The processed dir is produced by:
    helico-preprocess all <raw-dir> <processed-dir>

Uploads all files in processed-dir to the 'processed/' prefix of the HF dataset.
Large files (>5GB) are automatically handled via LFS by huggingface_hub.
"""

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi

HF_REPO = "timodonnell/helico-data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Upload processed data to HuggingFace")
    parser.add_argument("processed_dir", type=Path, help="Path to processed data directory")
    parser.add_argument("--repo", default=HF_REPO, help=f"HF dataset repo (default: {HF_REPO})")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    if not processed_dir.exists():
        logger.error(f"Directory not found: {processed_dir}")
        sys.exit(1)

    # Collect files to upload
    files = sorted(processed_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    if not files:
        logger.error(f"No files found in {processed_dir}")
        sys.exit(1)

    logger.info(f"Found {len(files)} files in {processed_dir}")
    total_size = sum(f.stat().st_size for f in files)
    logger.info(f"Total size: {total_size / (1024**3):.1f} GB")

    if args.dry_run:
        for f in files:
            rel = f.relative_to(processed_dir)
            size = f.stat().st_size
            if size > 1024**3:
                print(f"  processed/{rel}  ({size / (1024**3):.1f} GB)")
            elif size > 1024**2:
                print(f"  processed/{rel}  ({size / (1024**2):.0f} MB)")
            else:
                print(f"  processed/{rel}  ({size / 1024:.0f} KB)")
        logger.info("Dry run — nothing uploaded.")
        return

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)

    # Upload the entire folder
    logger.info(f"Uploading {processed_dir} to {args.repo}:processed/ ...")
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(processed_dir),
        path_in_repo="processed",
        repo_type="dataset",
    )

    logger.info(f"Upload complete: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
