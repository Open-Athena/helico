"""Step 4c -- Protenix-v2 structures: contact arms, and the baseline rows.

Feeding Helico the contacts implied by another folding model's structure is the
control that says whether Helico tracks contact *quality* or adds something of
its own on top: Protenix-v2 single-sequence and +MSA bracket MarinFold's contact
accuracy from below and above, and the same structures also give the two
baseline lDDT rows the Helico arms are compared against.

Contacts are read off the prediction by transplanting its coordinates onto the
ground-truth object (`experiments/marinfold_contacts/contacts_from_predictions.py`)
-- Helico's parser does not classify Protenix output chains as polymer, so the
predicted file cannot be tokenised directly, and the transplant makes both sides
share one index space by construction.

The transplanted structure is then run through **`oracle_contact_state`**, the
same function the oracle arm uses, rather than through a separate
`tokenize_structure` path. That matters for the 30 targets carrying modified
residues (ACE, AIB, MSE, ...): those tokenize per-atom in a structure-derived
tokenization and one-per-residue in the sequence-derived one, so any indexing
that does not go residue by residue is silently off by the number of modified
residues that precede each contact.

Writes `data/arms/v2ss.json` / `data/arms/v2msa.json` (Helico token indices),
`data/protenix_v2_baseline.csv` (per-target lDDT for the structures
themselves), and `data/v2_arm_accuracy.csv` (how good those contacts are,
against exp245's ground truth, for the contact-quality-vs-lDDT plot).

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/export_v2_contacts.py \\
        --mode single_seq --mode msa
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "marinfold_contacts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402
from contacts_from_predictions import transplant  # noqa: E402
from score_protenix_v2 import lddt_against_gt, ranked_sample  # noqa: E402

PRED_ROOT = U.CACHE / "protenix_v2"
#: Modal tag -> arm name. The tags are what run_protenix_v2.py passes as
#: --out-tag, so the directory names follow from them.
MODES = {"single_seq": "v2ss", "msa": "v2msa"}

BASELINE = U.DATA / "protenix_v2_baseline.csv"
ACCURACY = U.DATA / "v2_arm_accuracy.csv"

#: Below this share of atoms replaced the prediction does not cover the ground
#: truth well enough for its contacts to describe the same molecule.
MIN_REPLACED = 0.9


def find_target_dirs(root: Path, stems: set[str]) -> dict[str, Path]:
    """stem -> the directory holding that target's Protenix output.

    Located by walking up from the sample mmCIFs rather than by assuming a
    depth. The actual tree is
    ``<root>/<tag>/<stem>/predictions/<stem>/seed_N/predictions/*.cif``: `modal
    volume get` reproduces the remote prefix under the destination, so the tag
    appears twice, and the stem appears twice as well.

    Matching against the *known stems* rather than a name pattern matters --
    `seed_42` satisfies any reasonable pattern for "four characters, underscore,
    suffix" -- and taking the innermost match lands on the directory that
    actually holds the `seed_*` level, which is what `ranked_sample` needs to
    read the per-seed confidence files.
    """
    found: dict[str, Path] = {}
    for cif in root.rglob("*_sample_*.cif"):
        for parent in cif.parents:
            if parent == root:
                break
            if parent.name in stems:
                found.setdefault(parent.name, parent)
                break
    return found


def best_prediction(target_dir: Path) -> Path | None:
    """The mmCIF this target is scored on -- Protenix's own top-ranked sample."""
    seeds = sorted(target_dir.glob("seed_*"))
    candidates = [ranked_sample(seed) for seed in seeds] if seeds else []
    candidates = [c for c in candidates if c is not None]
    if candidates:
        return candidates[0]
    direct = ranked_sample(target_dir)
    if direct is not None:
        return direct
    loose = sorted(target_dir.rglob("*_sample_*.cif"))
    return loose[0] if loose else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", action="append", choices=sorted(MODES),
                        help="repeatable; defaults to both")
    parser.add_argument("--pred-root", type=Path, default=PRED_ROOT)
    args = parser.parse_args()
    modes = args.mode or sorted(MODES)

    import torch

    from helico.bench import oracle_contact_state, structure_to_chains
    from helico.contacts import load_rotamer_library
    from helico.data import CONTACT_PRESENT, parse_ccd, parse_mmcif, tokenize_sequences

    ccd, rotamer_library = parse_ccd(), load_rotamer_library()
    universe = U.load_gt_universe()
    token_map = {stem: {int(p): t for p, t in mapping.items()}
                 for stem, mapping in
                 json.loads((U.DATA / "token_map.json").read_text()).items()}
    with (U.DATA / "targets.csv").open() as handle:
        targets = list(csv.DictReader(handle))

    U.ARMS.mkdir(parents=True, exist_ok=True)
    baseline_rows, accuracy_rows = [], []

    for mode in modes:
        arm_name = MODES[mode]
        root = args.pred_root / mode
        if not root.exists():
            raise SystemExit(f"no Protenix predictions at {root}; "
                             f"run run_protenix_v2.py --mode {mode} first")
        target_dirs = find_target_dirs(root, {t["target_id"] for t in targets})
        print(f"{arm_name}: found predictions for {len(target_dirs)} targets "
              f"under {root}")
        arm, skipped = {}, []
        for target in targets:
            stem = target["target_id"]
            target_dir = target_dirs.get(stem)
            cif = best_prediction(target_dir) if target_dir is not None else None
            if cif is None:
                skipped.append((stem, "no prediction"))
                continue

            gt_path = U.GT_DIR / f"{stem}.cif.gz"
            lddt = lddt_against_gt(cif, gt_path)
            if lddt is not None:
                baseline_rows.append({
                    "target_id": stem, "eval_set": target["eval_set"],
                    "arm": f"protenix_v2_{mode}", "lddt": round(lddt, 6),
                    "cif": str(cif.relative_to(args.pred_root)),
                })

            gt = parse_mmcif(gt_path, max_resolution=float("inf"))
            pred = parse_mmcif(cif, max_resolution=float("inf"))
            if gt is None or pred is None:
                skipped.append((stem, "structure did not parse"))
                continue
            hybrid, n_replaced, n_total = transplant(gt, pred)
            if n_replaced / max(n_total, 1) < MIN_REPLACED:
                skipped.append((stem, f"only {n_replaced}/{n_total} atoms replaced"))
                continue

            tokenized = tokenize_sequences(structure_to_chains(gt), ccd)
            state = oracle_contact_state(hybrid, tokenized, rotamer_library)
            if state is None:
                skipped.append((stem, "contacts could not be indexed"))
                continue
            pairs = torch.nonzero(torch.triu(state == CONTACT_PRESENT, diagonal=1))
            arm[stem] = [[int(a), int(b)] for a, b in pairs.tolist()]

            # How good are they? Scored against exp245's own ground truth,
            # mapped into the same token space, so this number sits beside the
            # MarinFold arm's published precision on one axis.
            mapping = token_map.get(stem)
            record = universe[stem]
            if mapping:
                length = record["L"]
                truth = {
                    (min(mapping[i], mapping[j]), max(mapping[i], mapping[j]))
                    for i, j, degree in ((int(a), int(b), c)
                                         for a, b, c in record["contacts"])
                    if degree >= U.MIN_DEG and (j - i) >= U.MIN_SEP
                    and i < j < length and i in mapping and j in mapping
                }
                got = {tuple(p) for p in arm[stem]}
                accuracy_rows.append({
                    "target_id": stem, "eval_set": target["eval_set"], "arm": arm_name,
                    "n_pred": len(got), "n_true": len(truth),
                    "precision": round(len(got & truth) / len(got), 6) if got else "",
                    "recall": round(len(got & truth) / len(truth), 6) if truth else "",
                })

        (U.ARMS / f"{arm_name}.json").write_text(json.dumps(arm))
        total = sum(len(v) for v in arm.values())
        print(f"{arm_name}: {len(arm)} targets, {len(skipped)} skipped, "
              f"{total} pairs ({total / max(len(arm), 1):.0f}/target)")
        for stem, why in skipped:
            print(f"    skipped {stem}: {why}")

    for path, rows in ((BASELINE, baseline_rows), (ACCURACY, accuracy_rows)):
        if not rows:
            continue
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
