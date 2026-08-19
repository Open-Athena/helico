"""Step 1 -- the target set, its ground truths, and the controls on both.

exp245 cut FoldBench's 334 monomers into eval-val (97), eval-test (218) and
eval-denovo (19), and scored 333 of them (`8uxt_A` does not fit MarinFold's
8,192-token context). This builds the same 333 units as a Helico target set:
one `targets.csv` row per unit and one ground-truth mmCIF per unit, in the
layout `modal/bench_byclass.py` mounts.

Ground truths are **not** re-fetched from RCSB. FoldBench's own
`examples/ground_truths/<pdb>-assembly1.cif.gz` is already in Helico's data
cache and is the same assembly exp245 built its records from, so copying those
bytes keeps one provenance for the structures every arm is scored against.

Three controls, all asserted rather than assumed:

* **Helico's training window.** Every unit must have been released after
  2021-09-30 or its lDDT measures memorisation. 0/333 fail; the deposit-date
  exposure exp245's baseline table uses is reported alongside, because that
  count is not zero and the two dates mean different things.
* **One protein chain per ground truth.** The contact arms address a single
  chain by id; a second polymer would silently shift every pair.
* **Set sizes** match exp245's published counts.

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/build_eval_sets.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

#: Mirrored from exp245's bucket and pinned. The results tables are pulled here
#: too: `contact_precision_all.csv` is the control the contact arms are checked
#: against, and `per_protein.csv` carries the contact-side scoreboard this
#: experiment reports its folding numbers next to.
INPUTS = (
    "eval_sets.csv",
    "eval_targets_foldbench_monomers.parquet",
    "gt_universe_scored.jsonl",
    "headline.csv",
    "per_protein.csv.gz",
    f"runs/{U.RUN_ID}/contact_precision_all.csv",
)

TARGETS = U.DATA / "targets.csv"
PINS = U.DATA / "exp245_inputs.json"
REPORT = U.DATA / "eval_set_report.json"

#: `dataset` duplicates `eval_set`: bench_byclass groups its per-run summary by
#: that column, so the three eval sets are what it reports means over.
FIELDS = ["target_id", "eval_set", "dataset", "stem", "pdb_id", "gt_chain", "L_helico",
          "L_exp245", "n_resolved", "is_viral", "designed", "kingdom",
          "exp199_stratum", "deposit_date", "initial_release_date",
          "msa_available", "input_seq"]


def seq_sha256(sequence: str) -> str:
    """FoldBench names each a3m sha256(sequence + newline), not sha256(sequence)."""
    return hashlib.sha256((sequence + "\n").encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-gt", action="store_true",
                        help="rebuild targets.csv only, leaving data/gt/ alone")
    args = parser.parse_args()

    from helico.bench import _find_gt_path, structure_to_chains
    from helico.data import parse_mmcif

    U.DATA.mkdir(parents=True, exist_ok=True)
    U.GT_DIR.mkdir(parents=True, exist_ok=True)

    mirrored = {name: U.fetch(name) for name in INPUTS}
    PINS.write_text(json.dumps(
        {"bucket": U.BUCKET, "run_id": U.RUN_ID,
         "files": {name: {"size": path.stat().st_size, "sha256": U.sha256(path)}
                   for name, path in sorted(mirrored.items())}},
        indent=2) + "\n")

    universe = U.load_gt_universe()
    with mirrored["eval_sets.csv"].open() as handle:
        sets = [row for row in csv.DictReader(handle) if row["scorable"] == "1"]

    msa_names = {path.name.split(".")[0]
                 for path in (U.FOLDBENCH / "foldbench-msas").iterdir()}

    rows, problems = [], []
    for row in sets:
        stem = row["stem"]
        pdb_id = f"{row['pdb_id']}-assembly1"
        record = universe[stem]

        structure = parse_mmcif(_find_gt_path(U.FOLDBENCH_GT, pdb_id),
                                max_resolution=float("inf"))
        if structure is None:
            problems.append({"stem": stem, "why": "ground truth did not parse"})
            continue
        chains = [c for c in structure_to_chains(structure) if c["type"] == "protein"]
        if len(chains) != 1:
            # The arms address one chain by id. Two polymers would make every
            # contact index ambiguous, and the failure would be silent.
            problems.append({"stem": stem,
                             "why": f"{len(chains)} protein chains, expected 1"})
            continue
        sequence = chains[0]["sequence"]

        if not args.skip_gt:
            shutil.copyfile(_find_gt_path(U.FOLDBENCH_GT, pdb_id),
                            U.GT_DIR / f"{stem}.cif.gz")

        rows.append({
            "target_id": stem, "eval_set": row["eval_set"],
            "dataset": row["eval_set"], "stem": stem,
            "pdb_id": pdb_id, "gt_chain": chains[0]["id"],
            "L_helico": len(sequence), "L_exp245": int(row["seq_len"]),
            "n_resolved": record["n_resolved"],
            "is_viral": row["is_viral"], "designed": row["designed"],
            "kingdom": row["kingdom"], "exp199_stratum": row["exp199_stratum"],
            "deposit_date": row["deposit_date"][:10],
            "initial_release_date": row["initial_release_date"][:10],
            "msa_available": int(seq_sha256(sequence) in msa_names),
            "input_seq": sequence,
        })

    if problems:
        raise SystemExit(f"{len(problems)} units failed the structure control: "
                         f"{problems[:5]}")

    with TARGETS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # --- controls -------------------------------------------------------
    counts = Counter(row["eval_set"] for row in rows)
    expected = {"eval-val": 97, "eval-test": 217, "eval-denovo": 19}
    if dict(counts) != expected:
        raise SystemExit(f"set sizes {dict(counts)} != exp245's {expected}")

    in_window = [r["target_id"] for r in rows
                 if r["initial_release_date"] < U.TRAIN_CUTOFF]
    if in_window:
        raise SystemExit(
            f"{len(in_window)} units predate Helico's {U.TRAIN_CUTOFF} training "
            f"cutoff and would measure memorisation: {in_window[:10]}")

    # Reported, not asserted: exp245's baseline-cutoff table keys on deposit
    # date, which is earlier than release and non-zero here. It bounds what the
    # *baselines* may have seen, not what Helico trained on.
    deposited_early = Counter(r["eval_set"] for r in rows
                              if r["deposit_date"] <= U.TRAIN_CUTOFF)

    report = {
        "n_units": len(rows),
        "sets": dict(counts),
        "released_before_helico_cutoff": 0,
        "deposited_on_or_before_cutoff": dict(deposited_early),
        "viral": dict(Counter(r["eval_set"] for r in rows if r["is_viral"] == "1")),
        "msa_available": sum(r["msa_available"] for r in rows),
        "length_agrees_with_exp245": sum(
            1 for r in rows if r["L_helico"] == r["L_exp245"]),
        "max_L_helico": max(r["L_helico"] for r in rows),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\n{len(rows)} targets -> {TARGETS}")
    if not args.skip_gt:
        print(f"{len(rows)} ground truths -> {U.GT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
