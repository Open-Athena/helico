"""Build the decontaminated by-class evaluation set from MarinFold exp211.

The by-class contact-accuracy figure (`contact_accuracy_by_dataset.py`) shows
MarinFold beating Protenix v2 single sequence on `foldbench100` and losing
everywhere else. This builds the target set needed to ask the same question in
lDDT after folding, rather than in contact accuracy.

Decontamination is the reason this is not simply exp211's 554 targets. Helico
trains on PDB entries released before 2021-09-30, so any target deposited before
that date is potentially memorised and its lDDT measures recall, not folding:

  foldbench100  98/100 kept  -- all 7z/8x depositions, and 2 have no index map
  denovo_pdb   213/396 kept  -- 183 predate the cutoff (1qys/Top7 is from 2003)
  cameo_hard    32/32  kept  -- none predate the cutoff
  casp_fm       12/26  kept  -- T11xx (CASP15, 2022); T10xx is CASP14 and its
                                structures reached the PDB across 2020-2021,
                                which straddles the cutoff

Outputs, all under this directory's `data/`:

  targets.csv     one row per kept target: dataset, stem, sequence, release date
  gt/<stem>.cif   ground truth converted from exp211's PDB with gemmi, so the
                  existing parse_mmcif / match_atoms / score_monomer path runs
                  unchanged

Run:
    uv run python experiments/marinfold_contacts/byclass/build_targets.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

EXP211 = Path("/home/bizon/git/MarinFold/.claude/worktrees/contact-consistency-exp199-9d394e"
              "/experiments/exp211_evals_contact_set_3d_self_consistency")
GT_PDB = EXP211 / "_scratch/gt/gt_structures"
UNIVERSE = EXP211 / "_scratch/gt_universe.jsonl"
TARGETS = EXP211 / "data/eval_targets.parquet"

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"

# Helico's training filter. Anything deposited before this is in-distribution.
TRAIN_CUTOFF = "2021-09-30"


def keep(dataset: str, stem: str, release: str | None) -> bool:
    """Is this target safe to score Helico on?"""
    if dataset == "casp_fm":
        # CASP15 (T11xx) postdates the cutoff; CASP14 (T10xx) straddles it and
        # the stems carry no PDB code to look a date up with.
        return stem.startswith("T11")
    if dataset == "foldbench100":
        return True  # 7z/8x depositions; verified against the manifest already
    return bool(release) and release >= TRAIN_CUTOFF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dates", required=True,
                    help="JSON mapping lowercase pdb code -> initial release date")
    args = ap.parse_args()

    import gemmi

    releases = json.loads(Path(args.release_dates).read_text())
    universe = {(r["dataset"], r["stem"]): r
                for r in map(json.loads, UNIVERSE.read_text().splitlines())}

    import pandas as pd
    seqs = {(r.dataset, r.stem): r.input_seq
            for r in pd.read_parquet(TARGETS).itertuples()}

    (OUT / "gt").mkdir(parents=True, exist_ok=True)
    rows, dropped = [], {}
    for (dataset, stem), meta in sorted(universe.items()):
        release = releases.get(stem.split("_")[0].lower(), "")
        if not keep(dataset, stem, release):
            dropped[dataset] = dropped.get(dataset, 0) + 1
            continue
        pdb = GT_PDB / dataset / f"{stem}.pdb"
        if not pdb.exists():
            dropped[dataset] = dropped.get(dataset, 0) + 1
            continue

        # gemmi's PDB -> mmCIF keeps chain ids, residue numbering and atom
        # names, which is everything parse_mmcif and match_atoms rely on.
        # assign_label_seq_id is not optional: a PDB file has no
        # entity_poly_seq, so without it every _atom_site.label_seq_id is "."
        # and helico's parser collapses the chain (190 residues -> 19).
        st = gemmi.read_structure(str(pdb))
        st.setup_entities()
        st.assign_label_seq_id(True)
        # Leave gemmi's subchain ids alone. They come out as "Axp", so helico
        # reads the chain id as "Axp" rather than "A" -- renaming them here to
        # match desynchronises the entity records and the chain loses its
        # polymer_type, which drops it from structure_to_chains entirely.
        # bench_byclass addresses contacts by the chain id it actually parses.
        cif = (OUT / "gt" / f"{dataset}__{stem}.cif.gz")
        with gzip.open(cif, "wt") as f:
            f.write(st.make_mmcif_document().as_string())

        rows.append({"dataset": dataset, "stem": stem,
                     "target_id": f"{dataset}__{stem}",
                     "release_date": release,
                     "L": meta["L"], "n_resolved": meta["n_resolved"],
                     "input_seq": seqs[(dataset, stem)]})

    with (OUT / "targets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    kept = {}
    for r in rows:
        kept[r["dataset"]] = kept.get(r["dataset"], 0) + 1
    print(f"kept {len(rows)} targets")
    for ds in sorted(set(kept) | set(dropped)):
        print(f"  {ds:14s} kept {kept.get(ds, 0):4d}  dropped {dropped.get(ds, 0):4d}")
    print(f"\nwrote {OUT / 'targets.csv'} and {len(rows)} CIFs under {OUT / 'gt'}")


if __name__ == "__main__":
    main()
