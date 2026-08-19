"""Step 4a -- MSAs for the handful of targets FoldBench does not already cover.

`build_protenix_input` looks an a3m up by sha256 of the chain sequence in
FoldBench's `foldbench-msas/`, and **raises when there is none regardless of
mode** -- so a missing alignment drops the target from the single-sequence arm
too, not just the +MSA one. 317 of the 333 units hit a shipped a3m under
Helico's own sequence hash; this fetches the rest from the public ColabFold
MMseqs2 server and writes them under the expected filename, so the standard
staging path picks them up with no code change.

Resumable: an existing `<sha>.a3m.gz` is skipped. Failures are counted and
reported rather than swallowed -- a silently short MSA set shows up much later
as a Protenix arm that quietly scored fewer targets than every other arm.

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/gen_missing_msas.py
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

from helico.msa_server import run_mmseqs2  # noqa: E402
# FoldBench names each a3m sha256(sequence + "\n"), NOT sha256(sequence).
# Import the hash the consumer uses instead of reimplementing it: under the
# wrong name every file is invisible to build_protenix_input, which then stages
# 0 targets and the arm scores nothing.
from helico.upstream_protenix import _seq_sha256  # noqa: E402

MSA_DIR = U.FOLDBENCH / "foldbench-msas"
CACHE = U.CACHE / "msa"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    with (U.DATA / "targets.csv").open() as handle:
        targets = [row for row in csv.DictReader(handle)
                   if row["msa_available"] == "0"]
    if args.limit:
        targets = targets[:args.limit]
    print(f"{len(targets)} targets without a FoldBench a3m")

    done = failed = skipped = 0
    for target in targets:
        sha = _seq_sha256(target["input_seq"])
        dest = MSA_DIR / f"{sha}.a3m.gz"
        if dest.exists():
            skipped += 1
            continue
        print(f"  {target['target_id']} (L={target['L_helico']}) -> {sha[:12]}...",
              flush=True)
        try:
            a3m = run_mmseqs2(target["input_seq"], str(CACHE / target["target_id"]))
        except Exception as e:  # noqa: BLE001 - the server rate-limits and stalls
            print(f"    FAILED: {type(e).__name__}: {e}")
            failed += 1
            continue
        if isinstance(a3m, (list, tuple)):
            a3m = a3m[0]
        with gzip.open(dest, "wt") as handle:
            handle.write(a3m)
        done += 1
        time.sleep(1)

    print(f"\nfetched {done}, already present {skipped}, failed {failed}")
    if failed:
        print("Re-run to retry the failures; the server is the usual cause.")
        return 1
    print("Re-run build_eval_sets.py --skip-gt to refresh the msa_available column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
