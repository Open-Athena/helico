"""Score the Protenix v2 by-class runs, into the same results/ layout as the Helico arms.

`../score_protenix_v2.py` does the same job for the FoldBench monomer arms and
hardcodes FoldBench's ground-truth directory; these targets live in this
directory's `data/gt/` instead. Everything else is shared: it imports that
module's `ranked_sample` and `lddt_against_gt`, so the sample-selection rule and
the lDDT path are identical across every Protenix arm in the project.

No prompt->residue map is needed here, unlike the exp226 top-up:
`build_protenix_input` derives its input sequence from `structure_to_chains` on
these same ground truths, so Protenix predicts exactly the resolved residues and
positional matching is correct by construction.

Run:
    uv run python experiments/marinfold_contacts/byclass/score_protenix_byclass.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/marinfold_contacts"))
sys.path.insert(0, str(ROOT / "src"))

from score_protenix_v2 import lddt_against_gt, ranked_sample  # noqa: E402

GT = HERE / "data/gt"
UPSTREAM = ROOT / "experiments/marinfold_contacts/upstream"
RESULTS = HERE / "results"
ARMS = {"byclass_v2_singleseq": "v2_singleseq", "byclass_v2_msa": "v2_msa"}


def main() -> None:
    with (HERE / "data/targets.csv").open() as f:
        meta = {r["target_id"]: r for r in csv.DictReader(f)}

    RESULTS.mkdir(parents=True, exist_ok=True)
    for dump, tag in ARMS.items():
        root = UPSTREAM / dump
        rows, seen = [], set()
        for tgt_dir in sorted(p for p in root.rglob("predictions") if p.is_dir()):
            for d in sorted(tgt_dir.iterdir()):
                tid = d.name
                if not d.is_dir() or tid in seen or tid not in meta:
                    continue
                for seed_dir in sorted(d.glob("seed_*")):
                    cif = ranked_sample(seed_dir)
                    if cif is None:
                        continue
                    v = lddt_against_gt(cif, GT / f"{tid}.cif.gz")
                    row = {"target_id": tid, "dataset": meta[tid]["dataset"],
                           "stem": meta[tid]["stem"],
                           "status": "ok" if v is not None else "no_match",
                           "lddt": round(v, 4) if v is not None else "",
                           "n_matched_atoms": "", "error": ""}
                    rows.append(row)
                    seen.add(tid)
                    break

        dest = RESULTS / f"{tag}.csv"
        with dest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["target_id"]))
        ok = [r for r in rows if r["status"] == "ok"]
        print(f"{tag}: {len(ok)}/{len(rows)} ok  "
              f"lDDT {sum(r['lddt'] for r in ok) / len(ok):.4f} -> {dest}")


if __name__ == "__main__":
    main()
