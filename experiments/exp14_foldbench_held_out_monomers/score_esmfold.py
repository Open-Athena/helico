"""Score the ESMFold and ESMFold2 baselines on exp14's units.

Both were already run by MarinFold's exp78 and their structures sit in the
`esmfold-exp78-runs` / `esmfold2-exp78-runs` Modal volumes, covering all 333
units. Nothing is re-predicted here.

**They cannot be scored positionally.** ESMFold folds the *prompt* sequence and
numbers its residues 1-based into it, dropping anything that is not a standard
amino acid; Helico's ground truth is indexed by *resolved* residue. The two
agree on only 52 of 333 targets, so matching atoms by position would silently
compare different residues on the other 281 -- the same trap the contact arms
had, and it is defused the same way, with `data/token_map.json`.

Each mapped residue's name is checked against the ground truth, and a target
whose identity falls below `MIN_IDENTITY` is reported and dropped rather than
scored. Metrics are the ones `score_monomer` computes, over the same atoms.

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/score_esmfold.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

from helico.bench import (  # noqa: E402
    compute_gdt_ts, compute_lddt, compute_rmsd, compute_tm_score,
    structure_to_chains,
)
from helico.data import parse_mmcif  # noqa: E402

ROOT = U.CACHE / "esmfold"
OUT = U.DATA / "esmfold_baseline.csv"
MODES = {"esmfold": "ESMFold", "esmfold2": "ESMFold2"}
BACKBONE = {"CA", "C3'"}
#: Below this share of mapped residues agreeing by name, the mapping is not
#: trustworthy and the target is dropped rather than scored.
MIN_IDENTITY = 0.9


def predicted_residues(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """seqid -> {atom name: coords}, read with gemmi.

    gemmi rather than `helico.data.parse_mmcif`: helico's parser is built for
    experimental mmCIFs and recovers only 12 of ESMFold's 28 residues for
    5sbj_A. gemmi reads both predictors' output faithfully -- ESMFold numbers
    2..29 having dropped the two caps, ESMFold2 numbers 1..30 keeping them as
    UNK -- and both are 1-based into the prompt sequence, which is what the
    index map needs.
    """
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    if not len(structure) or not len(structure[0]):
        return {}
    out: dict[int, dict[str, np.ndarray]] = {}
    for residue in structure[0][0]:
        entry = {atom.name: np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                 for atom in residue}
        entry["__name__"] = residue.name
        out[int(residue.seqid.num)] = entry
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", action="append", choices=sorted(MODES))
    args = parser.parse_args()
    modes = args.mode or sorted(MODES)

    token_map = {stem: {int(p): t for p, t in mapping.items()}
                 for stem, mapping in
                 json.loads((U.DATA / "token_map.json").read_text()).items()}
    with (U.DATA / "targets.csv").open() as handle:
        targets = list(csv.DictReader(handle))

    rows, dropped = [], []
    for mode in modes:
        for target in targets:
            stem = target["target_id"]
            path = ROOT / mode / stem / "structure.cif"
            mapping = token_map.get(stem)
            if not path.exists():
                dropped.append((mode, stem, "no prediction"))
                continue
            if not mapping:
                dropped.append((mode, stem, "no verified index map"))
                continue

            gt = parse_mmcif(U.GT_DIR / f"{stem}.cif.gz", max_resolution=float("inf"))
            chains = [c for c in structure_to_chains(gt) if c["type"] == "protein"]
            if gt is None or not chains:
                dropped.append((mode, stem, "ground truth did not parse"))
                continue
            gt_residues = [r for c in gt.chains for r in c.residues][:]
            # structure_to_chains derives its sequence from the residues of the
            # first protein chain, in order, so the k-th of those is Helico
            # token k -- which is what token_map maps prompt positions onto.
            gt_chain = next(c for c in gt.chains if c.residues)
            gt_residues = list(gt_chain.residues)

            predicted = predicted_residues(path)
            if not predicted:
                dropped.append((mode, stem, "prediction did not parse"))
                continue

            # Two numbering conventions are present in exp78's outputs, and
            # which one a file uses cannot be assumed: 316 of them number
            # 1-based into the *prompt* sequence, while the rest -- every one
            # of the eight designed monomers among them -- number 1-based into
            # the *resolved* sequence, i.e. straight onto Helico's tokens.
            # Read under the wrong one the residues do not line up at all, so
            # both are tried and the residue-name check decides, rather than a
            # guess about which pipeline produced the file.
            candidates = {
                "prompt": {token: prompt_index + 1
                           for prompt_index, token in mapping.items()},
                "resolved": {token: token + 1 for token in range(len(gt_residues))},
            }
            best = None
            for convention, correspondence in candidates.items():
                coords, truths, names, matched, total = [], [], [], 0, 0
                for token, seqid in correspondence.items():
                    residue = predicted.get(seqid)
                    if residue is None or token >= len(gt_residues):
                        continue
                    truth = gt_residues[token]
                    total += 1
                    # UNK is how ESMFold2 represents a residue it did not model
                    # as a standard amino acid -- the modified residues -- so it
                    # matches anything rather than counting against the check.
                    if residue["__name__"] in (truth.name, "UNK"):
                        matched += 1
                    truth_atoms = {a.name: np.asarray(a.coords, dtype=float)
                                   for a in truth.atoms}
                    for atom_name, xyz in residue.items():
                        if atom_name == "__name__" or atom_name not in truth_atoms:
                            continue
                        coords.append(xyz)
                        truths.append(truth_atoms[atom_name])
                        names.append(atom_name)
                score = matched / total if total else 0.0
                if best is None or score > best[0]:
                    best = (score, convention, coords, truths, names, total)

            identity, convention, pred_coords, gt_coords, names, total = best
            if total == 0 or identity < MIN_IDENTITY or len(pred_coords) < 10:
                dropped.append((mode, stem,
                                f"residue identity {identity:.2f} over {total}"))
                continue

            pred = np.stack(pred_coords)
            truth = np.stack(gt_coords)
            mask = np.array([n in BACKBONE for n in names], dtype=bool)
            if mask.sum() < 3:
                dropped.append((mode, stem, "too few backbone atoms"))
                continue
            rows.append({
                "target_id": stem, "eval_set": target["eval_set"],
                "arm": mode, "numbering": convention,
                "lddt": round(compute_lddt(pred, truth), 6),
                "tm_score": round(compute_tm_score(pred[mask], truth[mask]), 6),
                "gdt_ts": round(compute_gdt_ts(pred[mask], truth[mask]), 6),
                "rmsd": round(compute_rmsd(pred[mask], truth[mask]), 6),
                "n_matched_atoms": len(pred_coords),
                "residue_identity": round(identity, 4),
            })
        done = sum(1 for r in rows if r["arm"] == mode)
        print(f"{mode}: scored {done}", flush=True)

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {OUT}")
    print(f"{len(dropped)} dropped")
    from collections import Counter
    for why, count in Counter(w.split(" over ")[0].split(" 0.")[0]
                              for _, _, w in dropped).most_common():
        print(f"  {count:4d}  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
