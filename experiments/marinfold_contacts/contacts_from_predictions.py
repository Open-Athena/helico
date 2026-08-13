"""How good are the contacts implied by Protenix v2's own single-sequence output?

The question: Helico + real MarinFold contacts scores 0.638 while Protenix v2
single-sequence scores 0.409 -- but if you take v2's predicted 3D structure and
read contacts off it, are those contacts actually much worse than MarinFold's?

If they are comparable, then MarinFold is not supplying better contact
*information*; the gain comes from somewhere else (how the two models convert
contact knowledge into coordinates), and the honest framing of the result
changes.

Contacts are computed with the identical pyconfind path used everywhere else, so
precision/recall are directly comparable to the MarinFold numbers.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from helico.bench import _find_gt_path, structure_to_chains  # noqa: E402
from helico.contacts import (  # noqa: E402
    MIN_CONTACT_DEGREE, MIN_SEQ_SEPARATION, compute_contacts, load_rotamer_library,
)
from helico.data import parse_ccd, parse_mmcif, tokenize_structure  # noqa: E402

GT_DIR = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths"
ARMS = Path(__file__).resolve().parent / "arms"


def transplant(gt, pred):
    """GT structure with predicted coordinates substituted in.

    helico's mmCIF parser does not classify Protenix's output chains as polymer
    (structure_to_chains returns nothing), so the predicted file cannot be
    tokenised directly. Its atoms are readable though, so copying coordinates
    onto the GT object -- matched by (chain order, residue order, atom name) --
    gives predicted geometry with bookkeeping that is identical to the GT by
    construction. Contacts on both sides then live in the same index space.

    Returns (structure, n_replaced, n_total).
    """
    import copy

    pidx = {}
    for ci, chain in enumerate(pred.chains):
        for ri, res in enumerate(chain.residues):
            for a in res.atoms:
                pidx[(ci, ri, a.name)] = a.coords

    out = copy.deepcopy(gt)
    n_rep = n_tot = 0
    for ci, chain in enumerate(out.chains):
        for ri, res in enumerate(chain.residues):
            for a in res.atoms:
                n_tot += 1
                c = pidx.get((ci, ri, a.name))
                if c is not None:
                    a.coords = c
                    n_rep += 1
    return out, n_rep, n_tot


def contacts_of(structure, ccd, lib):
    """{(i, j)} over resolved residues, with helico's own filters."""
    tok = tokenize_structure(structure, ccd=ccd)
    edges, _ = compute_contacts(tok.tokens, lib, MIN_CONTACT_DEGREE)
    # compute_contacts yields (ti, tj) pairs, already degree-filtered by the
    # min_degree argument -- there is no third element to unpack.
    return {(min(i, j), max(i, j)) for i, j in edges
            if abs(i - j) >= MIN_SEQ_SEPARATION}, len(tok.tokens)


def main() -> None:
    ccd, lib = parse_ccd(), load_rotamer_library()
    pred_root = Path("experiments/marinfold_contacts/upstream/v2_singleseq")
    scores = {r["pdb_id"]: Path(r["cif"])
              for r in csv.DictReader(
                  (pred_root.parent / "v2_singleseq_scores.csv").open())}

    rows = []
    for pdb_id, rel in sorted(scores.items()):
        cif = pred_root / rel
        if not cif.exists():
            continue
        gt = parse_mmcif(_find_gt_path(GT_DIR, pdb_id), max_resolution=float("inf"))
        pred = parse_mmcif(cif, max_resolution=float("inf"))
        if gt is None or pred is None:
            continue
        try:
            truth, n_gt = contacts_of(gt, ccd, lib)
            hybrid, n_rep, n_tot = transplant(gt, pred)
            got, n_pred = contacts_of(hybrid, ccd, lib)
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb_id}: {type(e).__name__}: {e}")
            continue
        # Require most atoms to have been replaced; a low rate means the two
        # files disagree about residue ordering and the comparison is invalid.
        if not truth or n_rep / max(n_tot, 1) < 0.9:
            print(f"  {pdb_id}: only {n_rep}/{n_tot} atoms transplanted; skipping")
            continue
        tp = len(got & truth)
        L = len(structure_to_chains(gt)[0]["sequence"])
        rows.append({"pdb_id": pdb_id, "L": L, "n_true": len(truth),
                     "n_pred": len(got),
                     "precision": tp / len(got) if got else 0.0,
                     "recall": tp / len(truth)})

    out = Path("experiments/marinfold_contacts/v2ss_derived_contact_accuracy.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"targets: {n}")
    print(f"  contacts implied by Protenix v2 single-sequence structures:")
    print(f"    precision {sum(r['precision'] for r in rows)/n:.3f}"
          f"   recall {sum(r['recall'] for r in rows)/n:.3f}"
          f"   n_pred/target {sum(r['n_pred'] for r in rows)/n:.0f}")
    print(f"  MarinFold exp199 top-L (measured earlier):")
    print(f"    precision 0.505   recall 0.564")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
