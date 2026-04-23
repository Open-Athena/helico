"""Score upstream-Protenix output CIFs against FoldBench ground truths.

Run after `modal run modal/bench_upstream.py` has produced CIFs under
`experiments/exp8_ab_ag_triage/data/upstream/<pdb_id>/`. Produces:
  - upstream_per_sample_scores.csv  (seed, sample, dockq, ... per CIF)
  - upstream_oracle_vs_ranked.csv   (per-target oracle vs top-ranked)

DockQ-only for MVP — LDDT on Protenix outputs requires the same
tokenization Helico builds, which is more plumbing than we need to
answer the 8q3j/8v52 question.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump-root", type=Path,
        default=REPO_ROOT / "experiments/exp8_ab_ag_triage/data/upstream",
    )
    ap.add_argument(
        "--targets", default="8t59-assembly1,8q3j-assembly1,8v52-assembly1",
    )
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from helico.upstream_protenix import score_upstream_outputs

    gt_dir = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths"
    out_dir = args.dump_root
    per_sample_csv = out_dir / "upstream_per_sample_scores.csv"
    oracle_csv = out_dir / "upstream_oracle_vs_ranked.csv"

    all_rows = []
    oracle_rows = []

    for pdb_id in [t.strip() for t in args.targets.split(",") if t.strip()]:
        dump_dir = out_dir / pdb_id
        if not dump_dir.exists():
            print(f"[skip] {pdb_id}: no dump at {dump_dir}")
            continue
        print(f"\n=== scoring {pdb_id} ===")
        rows = score_upstream_outputs(pdb_id=pdb_id, dump_dir=dump_dir,
                                       gt_cif_path=gt_dir / f"{pdb_id}.cif.gz")
        all_rows.extend(asdict(r) for r in rows)

        ok = [r for r in rows if r.status == "ok" and r.dockq is not None]
        if ok:
            best = max(ok, key=lambda r: r.dockq or 0)
            n_success = sum(1 for r in ok if (r.dockq or 0) >= 0.23)
            print(f"  scored {len(ok)}/{len(rows)} samples "
                  f"| best DockQ {best.dockq:.3f} (seed={best.seed} sample={best.sample}) "
                  f"| success {n_success}/{len(ok)} ({100*n_success/max(len(ok),1):.0f}%)")
            # Sample with seed=42, sample=0 is approximately the ranker's
            # top pick in the first seed. For a real "ranked" number we'd
            # parse the summary_confidence JSONs; TODO if needed.
            first = next((r for r in ok if (r.seed == 42 and r.sample == 0)), ok[0])
            oracle_rows.append({
                "pdb_id": pdb_id,
                "first_seed_sample0_dockq": first.dockq,
                "oracle_dockq": best.dockq,
                "oracle_seed": best.seed,
                "oracle_sample": best.sample,
                "n_samples_scored": len(ok),
                "n_successes": n_success,
                "success_rate_pct": 100.0 * n_success / max(len(ok), 1),
            })
        else:
            print(f"  no successful scores ({len(rows)} CIFs attempted)")

    # Write CSVs
    if all_rows:
        with open(per_sample_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {per_sample_csv}")

    if oracle_rows:
        with open(oracle_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(oracle_rows[0].keys()))
            w.writeheader()
            w.writerows(oracle_rows)
        print(f"wrote {oracle_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
