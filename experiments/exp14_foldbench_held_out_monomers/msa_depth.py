"""MSA depth per target, for the low-depth cut.

The interesting question this experiment can ask of its own baselines is what
happens when there is no alignment to be had. Protenix-with-MSA and the ESM
models lean on evolutionary signal in different ways; contacts do not depend on
finding homologs at inference time at all. Splitting on depth is the closest
this set gets to a controlled test of that.

Depth is the number of sequences in the alignment Protenix was actually given
-- FoldBench's shipped a3m for the target's chain sequence, or the ColabFold
alignment `gen_missing_msas.py` fetched for the 16 it does not ship. Counting
sequences rather than computing Neff: Neff is quadratic in depth and the
question here is only whether an alignment exists at all.

Writes `data/msa_depth.csv`.

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/msa_depth.py
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

from helico.upstream_protenix import _seq_sha256  # noqa: E402

MSA_DIR = U.FOLDBENCH / "foldbench-msas"
OUT = U.DATA / "msa_depth.csv"
#: The cut this exists for.
SHALLOW = 10


def depth(path: Path) -> int:
    """Number of sequences in an a3m, the query included."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return sum(1 for line in handle if line.startswith(">"))


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    with (U.DATA / "targets.csv").open() as handle:
        targets = list(csv.DictReader(handle))

    rows, missing = [], []
    for target in targets:
        sha = _seq_sha256(target["input_seq"])
        path = MSA_DIR / f"{sha}.a3m.gz"
        if not path.exists():
            missing.append(target["target_id"])
            continue
        n = depth(path)
        rows.append({
            "target_id": target["target_id"], "eval_set": target["eval_set"],
            "n_sequences": n, "shallow": int(n <= SHALLOW),
            "seq_len": target["L_helico"], "designed": target["designed"],
        })

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    shallow = [r for r in rows if r["shallow"]]
    depths = sorted(r["n_sequences"] for r in rows)
    print(f"{len(rows)} targets with an alignment, {len(missing)} without")
    print(f"depth: min {depths[0]}, median {depths[len(depths)//2]}, "
          f"max {depths[-1]}")
    for cut in (1, 2, 5, 10, 25, 100):
        print(f"  depth <= {cut:4d}: {sum(1 for d in depths if d <= cut):3d}")
    from collections import Counter
    print(f"\nshallow (<= {SHALLOW}): {len(shallow)} -> "
          f"{dict(Counter(r['eval_set'] for r in shallow))}")
    print(f"  designed among them: {sum(int(r['designed']) for r in shallow)}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
