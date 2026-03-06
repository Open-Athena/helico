#!/usr/bin/env python3
"""Pre-generate MSAs for all FoldBench protein chains via ColabFold MMseqs2 server.

Saves as {sha256(seq+"\\n")}.a3m.gz — compatible with helico-bench --msa-dir.

Usage:
    uv run python generate_foldbench_msas.py /data/tim/helico-data/foldbench-msas
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-generate FoldBench MSAs")
    parser.add_argument("output_dir", type=Path, help="Output directory for .a3m.gz files")
    parser.add_argument("--host-url", default="https://api.colabfold.com",
                        help="MMseqs2 server URL")
    args = parser.parse_args()

    from helico.bench import (
        _find_gt_path,
        download_foldbench,
        load_targets,
        structure_to_chains,
    )
    from helico.data import parse_ccd, parse_mmcif
    from helico.msa_server import run_mmseqs2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download FoldBench data if needed
    foldbench_dir = download_foldbench()
    gt_dir = foldbench_dir / "examples" / "ground_truths"

    # Load CCD for parsing structures
    ccd = parse_ccd()

    # Collect all unique protein sequences across all targets
    all_targets = load_targets(foldbench_dir / "targets")
    seq_to_targets: dict[str, list[str]] = {}  # sequence -> [target names that use it]

    for category, targets in all_targets.items():
        for target in targets:
            try:
                gt_path = _find_gt_path(gt_dir, target.pdb_id)
            except FileNotFoundError:
                logger.warning(f"No ground truth for {target.pdb_id}, skipping")
                continue

            structure = parse_mmcif(gt_path, max_resolution=float("inf"))
            if structure is None:
                logger.warning(f"Failed to parse {target.pdb_id}, skipping")
                continue

            chains = structure_to_chains(structure)
            for chain in chains:
                if chain["type"] != "protein":
                    continue
                seq = chain["sequence"]
                if seq not in seq_to_targets:
                    seq_to_targets[seq] = []
                seq_to_targets[seq].append(f"{target.pdb_id}:{chain['id']}")

    logger.info(f"Found {len(seq_to_targets)} unique protein sequences across all targets")

    # Generate MSAs
    n_done = 0
    n_skipped = 0
    n_failed = 0
    for seq, target_names in sorted(seq_to_targets.items(), key=lambda x: len(x[0])):
        seq_hash = hashlib.sha256((seq + "\n").encode()).hexdigest()
        out_path = args.output_dir / f"{seq_hash}.a3m.gz"

        if out_path.exists():
            n_skipped += 1
            continue

        example_target = target_names[0]
        logger.info(
            f"[{n_done + n_skipped + 1}/{len(seq_to_targets)}] "
            f"Generating MSA for {example_target} (len={len(seq)}, "
            f"used by {len(target_names)} chain(s))"
        )

        try:
            # Use a temp cache dir for the server's internal caching
            cache_dir = args.output_dir / ".server_cache" / seq_hash
            a3m_results = run_mmseqs2(
                sequences=seq,
                result_dir=str(cache_dir),
                use_env=True,
                use_filter=True,
                host_url=args.host_url,
            )

            if a3m_results and a3m_results[0].strip():
                with gzip.open(out_path, "wt") as f:
                    f.write(a3m_results[0])
                n_done += 1
                logger.info(f"  Saved {out_path.name}")
            else:
                logger.warning(f"  Empty MSA result for {example_target}")
                n_failed += 1

        except Exception as e:
            logger.error(f"  Failed for {example_target}: {e}")
            n_failed += 1

    logger.info(
        f"Done. Generated: {n_done}, skipped (existing): {n_skipped}, failed: {n_failed}"
    )


if __name__ == "__main__":
    main()
