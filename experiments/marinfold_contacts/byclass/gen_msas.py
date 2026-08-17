"""Fetch MSAs for the by-class targets so the Protenix v2 +MSA arm can run.

`build_protenix_input` looks up a pre-computed a3m by sha256 of the chain
sequence, in FoldBench's `foldbench-msas/` directory. None of the 357 by-class
targets are FoldBench targets, so none have one. This queries the public
ColabFold MMseqs2 server and writes each result under the expected filename, so
the standard staging path picks them up with no code change.

Resumable: an existing `<sha>.a3m.gz` is skipped, so a killed run continues
where it stopped. Failures are counted and reported at the end rather than
swallowed -- a silently short MSA set would show up much later as a Protenix arm
that quietly scored fewer targets than every other arm.

Run:
    uv run python experiments/marinfold_contacts/byclass/gen_msas.py
"""

from __future__ import annotations

import csv
import gzip
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from helico.bench import structure_to_chains          # noqa: E402
from helico.data import parse_mmcif                    # noqa: E402
from helico.msa_server import run_mmseqs2              # noqa: E402
# FoldBench names each a3m sha256(sequence + "\\n"), NOT sha256(sequence).
# Import the hash the consumer uses instead of reimplementing it: under the
# wrong name every file is invisible to build_protenix_input, which then
# stages 0 targets and the arm scores nothing.
from helico.upstream_protenix import _seq_sha256       # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MSA_DIR = Path.home() / ".cache/helico/data/benchmarks/FoldBench/foldbench-msas"
CACHE = Path("/data/helico_contamination/msa_cache")


def main() -> None:
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with (DATA / "targets.csv").open() as f:
        targets = list(csv.DictReader(f))

    # One query per distinct sequence: the sha256 key means duplicates would
    # write the same file twice.
    wanted: dict[str, str] = {}
    for t in targets:
        st = parse_mmcif(DATA / "gt" / f"{t['target_id']}.cif.gz",
                         max_resolution=float("inf"))
        for chain in structure_to_chains(st) if st else []:
            if chain["type"] != "protein":
                continue  # ligand chains carry a CCD code, not a sequence
            wanted.setdefault(chain["sequence"], t["target_id"])

    todo = {s: t for s, t in wanted.items()
            if not (MSA_DIR / f"{_seq_sha256(s)}.a3m.gz").exists()}
    print(f"{len(wanted)} distinct sequences, {len(wanted) - len(todo)} already cached, "
          f"{len(todo)} to fetch", flush=True)

    done = failed = 0
    for i, (seq, tid) in enumerate(sorted(todo.items(), key=lambda kv: kv[1]), 1):
        sha = _seq_sha256(seq)
        try:
            a3ms = run_mmseqs2(seq, result_dir=str(CACHE / sha))
        except Exception as e:  # noqa: BLE001 - one bad query must not stop the sweep
            print(f"[{i}/{len(todo)}] {tid} FAILED: {type(e).__name__}: {e}", flush=True)
            failed += 1
            continue
        if not a3ms or not a3ms[0].strip():
            print(f"[{i}/{len(todo)}] {tid} FAILED: empty a3m", flush=True)
            failed += 1
            continue
        with gzip.open(MSA_DIR / f"{sha}.a3m.gz", "wt") as f:
            f.write(a3ms[0])
        n = sum(1 for line in a3ms[0].splitlines() if line.startswith(">"))
        done += 1
        print(f"[{i}/{len(todo)}] {tid} len={len(seq)} -> {n} sequences", flush=True)
        time.sleep(0.5)   # the ColabFold server is a shared public resource

    print(f"\ndone: {done} fetched, {failed} failed, "
          f"{len(wanted) - len(todo)} already cached")
    if failed:
        raise SystemExit(f"{failed} sequences have no MSA; the +MSA arm would be "
                         f"short by that many targets")


if __name__ == "__main__":
    main()
