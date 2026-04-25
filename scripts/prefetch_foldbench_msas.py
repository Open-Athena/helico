#!/usr/bin/env python3
"""Pre-fetch paired + non-paired ColabFold MSAs for FoldBench targets.

Populates the local cache at
``<foldbench_dir>/foldbench-msas-server/<query_hash>/{p,np}_*/out.tar.gz``.
Later invocations of ``helico-bench --use-msa-server`` read these cached
tars directly instead of re-querying the server. The cache is portable —
upload with ``scripts/upload_msa_cache.py`` for Modal / CI reuse.

Usage:
    # Fetch p-protein only (default)
    python scripts/prefetch_foldbench_msas.py

    # Fetch all categories
    python scripts/prefetch_foldbench_msas.py --categories all

    # Limit to a specific cutoff / pdb list
    python scripts/prefetch_foldbench_msas.py --cutoff-date 2024-01-01 \\
        --pdb-ids 8pu1-assembly1,8wwy-assembly1
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from helico.bench import (
    _find_gt_path,
    download_foldbench,
    fetch_release_dates,
    load_targets,
    _pdb_code,
    structure_to_chains,
)
from helico.data import parse_mmcif
from helico.msa_server import fetch_paired_and_unpaired

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def protein_seqs_for_target(gt_dir: Path, pdb_id: str) -> list[str] | None:
    """Return protein chain sequences for a target, or None if parse fails."""
    try:
        gt_path = _find_gt_path(gt_dir, pdb_id)
        gt_structure = parse_mmcif(gt_path, max_resolution=float("inf"))
        if gt_structure is None:
            return None
        chains = structure_to_chains(gt_structure)
        seqs = [
            c.get("sequence", "")
            for c in chains
            if c.get("type") == "protein"
        ]
        seqs = [s for s in seqs if s]
        return seqs or None
    except Exception as e:
        logger.warning(f"{pdb_id}: failed to extract sequences ({e})")
        return None


def fetch_one(pdb_id: str, seqs: list[str], cache_dir: Path, host_url: str) -> tuple[str, str]:
    """Fetch MSAs for one target. Returns (pdb_id, status)."""
    try:
        fetch_paired_and_unpaired(seqs, cache_dir=cache_dir, host_url=host_url)
        return pdb_id, "ok"
    except Exception as e:
        return pdb_id, f"error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foldbench-dir", type=Path, default=None)
    parser.add_argument(
        "--categories", type=str, default="interface_protein_protein",
        help="Comma-separated categories or 'all' (default: interface_protein_protein)",
    )
    parser.add_argument("--cutoff-date", type=str, default="2024-01-01")
    parser.add_argument("--pdb-ids", type=str, default=None)
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel ColabFold queries (default: 6)")
    parser.add_argument("--host-url", type=str, default="https://api.colabfold.com")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    foldbench_dir = args.foldbench_dir or download_foldbench()
    cache_dir = foldbench_dir / "foldbench-msas-server"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = foldbench_dir / "examples" / "ground_truths"

    all_targets = load_targets(foldbench_dir / "targets")
    if args.categories == "all":
        cats = list(all_targets)
    else:
        cats = [c.strip() for c in args.categories.split(",")]
    all_targets = {k: v for k, v in all_targets.items() if k in cats}

    if args.cutoff_date:
        all_pdb_codes = [_pdb_code(t.pdb_id) for ts in all_targets.values() for t in ts]
        release_dates = fetch_release_dates(all_pdb_codes)
        all_targets = {
            k: [t for t in v if release_dates.get(_pdb_code(t.pdb_id), "") > args.cutoff_date]
            for k, v in all_targets.items()
        }

    if args.pdb_ids:
        wanted = set(args.pdb_ids.split(","))
        all_targets = {
            k: [t for t in v if t.pdb_id in wanted] for k, v in all_targets.items()
        }

    # Build job list: dedup by pdb_id — CSVs can have multiple rows per target
    # (one per interface pair), but we only need to fetch MSAs once.
    seen: set[str] = set()
    jobs: list[tuple[str, list[str]]] = []
    for cat, targets in all_targets.items():
        for t in targets:
            if t.pdb_id in seen:
                continue
            seen.add(t.pdb_id)
            seqs = protein_seqs_for_target(gt_dir, t.pdb_id)
            if seqs:
                jobs.append((t.pdb_id, seqs))

    logger.info(f"Prepared {len(jobs)} fetch jobs across {len(all_targets)} categories")

    if args.dry_run:
        for pid, seqs in jobs[:20]:
            logger.info(f"  {pid}: {len(seqs)} chains, lengths={[len(s) for s in seqs]}")
        logger.info("Dry run — exiting without fetching")
        return

    t0 = time.time()
    n_ok = n_err = n_skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(fetch_one, pid, seqs, cache_dir, args.host_url): pid
            for pid, seqs in jobs
        }
        for i, fut in enumerate(as_completed(futs), 1):
            pid, status = fut.result()
            if status == "ok":
                n_ok += 1
            else:
                n_err += 1
                logger.warning(f"{pid}: {status}")
            if i % 10 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (len(jobs) - i) / rate if rate > 0 else 0
                logger.info(
                    f"  [{i:>4}/{len(jobs)}] ok={n_ok} err={n_err}  "
                    f"({rate:.2f} jobs/s, ~{remaining/60:.1f} min left)"
                )

    logger.info(f"Done. ok={n_ok} err={n_err} total={len(jobs)}  "
                f"elapsed={(time.time() - t0)/60:.1f} min")
    logger.info(f"Cache at {cache_dir}")
    logger.info(f"Upload with: python scripts/upload_msa_cache.py {cache_dir}")


if __name__ == "__main__":
    main()
