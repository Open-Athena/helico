"""Export the contacts implied by Protenix v2's single-sequence structures.

Control for helico#11: feeding these (26% precision / 26% recall) to Helico says
whether Helico is faithfully tracking contact quality or adding something of its
own on top. Same arm format as the MarinFold exports, so the bench consumes it
through the identical --contacts-arm path.

Contacts are read off the predicted structure by transplanting its coordinates
onto the GT object -- helico's parser does not classify Protenix output chains
as polymer, so the predicted file cannot be tokenised directly, and the
transplant guarantees both sides share one index space.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contacts_from_predictions import contacts_of, transplant  # noqa: E402

from helico.bench import _find_gt_path  # noqa: E402
from helico.contacts import load_rotamer_library  # noqa: E402
from helico.data import parse_ccd, parse_mmcif  # noqa: E402

GT_DIR = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths"
PRED = Path("experiments/marinfold_contacts/upstream/v2_singleseq")
ARMS = Path("experiments/marinfold_contacts/arms")


def main() -> None:
    ccd, lib = parse_ccd(), load_rotamer_library()
    scores = {r["pdb_id"]: r["cif"] for r in
              csv.DictReader((PRED.parent / "v2_singleseq_scores.csv").open())}
    arm, skipped = {}, 0
    for pdb_id, rel in sorted(scores.items()):
        cif = PRED / rel
        gt = parse_mmcif(_find_gt_path(GT_DIR, pdb_id), max_resolution=float("inf"))
        pred = parse_mmcif(cif, max_resolution=float("inf"))
        if gt is None or pred is None:
            skipped += 1
            continue
        hybrid, n_rep, n_tot = transplant(gt, pred)
        if n_rep / max(n_tot, 1) < 0.9:
            skipped += 1
            continue
        try:
            got, _ = contacts_of(hybrid, ccd, lib)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        arm[pdb_id] = [[int(i), int(j)] for i, j in sorted(got)]
    ARMS.mkdir(parents=True, exist_ok=True)
    (ARMS / "v2ss_derived.json").write_text(json.dumps(arm))
    tot = sum(len(v) for v in arm.values())
    print(f"exported {len(arm)} targets ({skipped} skipped), {tot} contacts "
          f"({tot / max(len(arm), 1):.0f}/target) -> {ARMS/'v2ss_derived.json'}")


if __name__ == "__main__":
    main()
