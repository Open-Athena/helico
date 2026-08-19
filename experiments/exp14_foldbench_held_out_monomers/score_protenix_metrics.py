"""Step 5b -- all four structural metrics for the Protenix-v2 baselines.

`export_v2_contacts.py` scores the baseline structures with lDDT only, because
that is what the contact-arm derivation needed. The Helico arms record lDDT,
TM-score, GDT-TS and RMSD, so the baselines they are compared against have to
carry the same four or the deck can only show one of them.

Nothing is re-predicted: the structures are already on disk. This re-reads the
same top-ranked mmCIF each baseline was scored on, matches predicted to
ground-truth atoms by (chain order, residue position, atom name) -- the
correspondence `helico.bench.match_atoms` uses -- and calls the same metric
functions `score_monomer` calls, with the same CA backbone.

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/score_protenix_metrics.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "marinfold_contacts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402
from export_v2_contacts import MODES, best_prediction, find_target_dirs  # noqa: E402
from score_protenix_v2 import atom_index  # noqa: E402

from helico.bench import (  # noqa: E402
    compute_gdt_ts, compute_lddt, compute_rmsd, compute_tm_score,
)
from helico.data import parse_mmcif  # noqa: E402

PRED_ROOT = U.CACHE / "protenix_v2"
BASELINE = U.DATA / "protenix_v2_baseline.csv"
#: One representative atom per residue, matching helico.bench's
#: extract_backbone_coords. Not the backbone in the N/CA/C/O sense: TM-score
#: and GDT-TS are per-residue measures and tmtools treats each coordinate as a
#: residue, so the atom set has to be one-per-residue on both sides or the
#: baselines are not comparable to the Helico arms.
BACKBONE = {"CA", "C3'"}


def metrics(pred_cif: Path, gt_cif: Path) -> dict | None:
    pred = parse_mmcif(pred_cif, max_resolution=float("inf"))
    gt = parse_mmcif(gt_cif, max_resolution=float("inf"))
    if pred is None or gt is None:
        return None
    pi, gi = atom_index(pred), atom_index(gt)
    keys = [k for k in gi if k in pi]
    if len(keys) < 10:
        return None
    pred_coords = np.stack([pi[k] for k in keys])
    gt_coords = np.stack([gi[k] for k in keys])
    mask = np.array([k[2] in BACKBONE for k in keys], dtype=bool)
    pred_bb, gt_bb = pred_coords[mask], gt_coords[mask]
    if len(pred_bb) < 3:
        return None
    return {
        "lddt": compute_lddt(pred_coords, gt_coords),
        "tm_score": compute_tm_score(pred_bb, gt_bb),
        "gdt_ts": compute_gdt_ts(pred_bb, gt_bb),
        "rmsd": compute_rmsd(pred_bb, gt_bb),
        "n_matched_atoms": len(keys),
    }


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    with (U.DATA / "targets.csv").open() as handle:
        targets = list(csv.DictReader(handle))
    stems = {t["target_id"] for t in targets}
    eval_set = {t["target_id"]: t["eval_set"] for t in targets}

    rows, missing = [], []
    for mode, _ in sorted(MODES.items()):
        root = PRED_ROOT / mode
        if not root.exists():
            continue
        found = find_target_dirs(root, stems)
        for stem in sorted(stems):
            directory = found.get(stem)
            cif = best_prediction(directory) if directory else None
            if cif is None:
                missing.append((mode, stem, "no prediction"))
                continue
            scored = metrics(cif, U.GT_DIR / f"{stem}.cif.gz")
            if scored is None:
                missing.append((mode, stem, "could not be matched"))
                continue
            rows.append({
                "target_id": stem, "eval_set": eval_set[stem],
                "arm": f"protenix_v2_{mode}",
                **{k: round(v, 6) for k, v in scored.items()},
                "cif": str(cif.relative_to(PRED_ROOT)),
            })
        print(f"{mode}: scored {sum(1 for r in rows if r['arm'].endswith(mode))}",
              flush=True)

    with BASELINE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {BASELINE}")
    for mode, stem, why in missing:
        print(f"  missing {mode}/{stem}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
