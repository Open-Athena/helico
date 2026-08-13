"""Score exp226's Protenix v2 structures for the 23 net-new targets.

exp226 already ran Protenix v2 in both modes on these and left the winning
structure on disk at `/data/exp226_gt/protenix_best/{single_seq,msa}/<stem>/
structure.cif`, so the baselines cost scoring only -- no inference is repeated.

lDDT is computed by the same path as `../score_protenix_v2.py` -- match atoms by
residue and atom name, then `helico.bench.compute_lddt` -- with one correction
that matters here. exp226 fed Protenix the **full prompt**, so its output has one
residue per prompt position, while the ground truth has only the resolved ones.
Matching those by position silently mis-registers every target with an internal
gap, which is all 23 (e.g. 8gsy_A: 144 predicted residues against 126 resolved).
So residues are paired through the same verified prompt->residue map the contact
arms use, written by `add_foldbench_rest.py`.

Run:
    uv run python experiments/marinfold_contacts/byclass/score_protenix_new23.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from helico.bench import compute_lddt  # noqa: E402
from helico.data import parse_mmcif  # noqa: E402

BEST = Path("/data/exp226_gt/protenix_best")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
ARMS = {"single_seq": "v2_singleseq", "msa": "v2_msa"}
TOKEN_MAP = "token_map_foldbench_rest.json"


def chain_atoms(structure) -> list[dict]:
    """[{atom_name: coords} per residue] for the longest chain."""
    chain = max(structure.chains, key=lambda c: len(c.residues))
    return [{a.name: np.asarray(a.coords, dtype=float) for a in res.atoms}
            for res in chain.residues]


def matched_coords(pred, gt, token_of: dict[str, int]):
    """Paired (pred, gt) coordinates over residues the map relates."""
    p_res, g_res = chain_atoms(pred), chain_atoms(gt)
    pc, gc = [], []
    for prompt_pos, gt_idx in token_of.items():
        pi, gi = int(prompt_pos), int(gt_idx)
        if pi >= len(p_res) or gi >= len(g_res):
            continue
        for name, coord in g_res[gi].items():
            if name in p_res[pi]:
                gc.append(coord)
                pc.append(p_res[pi][name])
    return pc, gc


def main() -> None:
    with (DATA / "targets.csv").open() as f:
        targets = [r for r in csv.DictReader(f) if r["dataset"] == "foldbench_rest"]
    import json
    token_maps = json.loads((DATA / TOKEN_MAP).read_text())

    RESULTS.mkdir(parents=True, exist_ok=True)
    for mode, tag in ARMS.items():
        rows = []
        for t in targets:
            pred_cif = BEST / mode / t["stem"] / "structure.cif"
            row = {"target_id": t["target_id"], "dataset": t["dataset"],
                   "stem": t["stem"], "status": "error", "lddt": "",
                   "n_matched_atoms": "", "error": ""}
            if not pred_cif.exists():
                row["error"] = f"no prediction at {pred_cif}"
                rows.append(row)
                continue
            pred = parse_mmcif(pred_cif, max_resolution=float("inf"))
            gt = parse_mmcif(DATA / "gt" / f"{t['target_id']}.cif.gz",
                             max_resolution=float("inf"))
            if pred is None or gt is None:
                row["error"] = "parse failed"
                rows.append(row)
                continue
            pc, gc = matched_coords(pred, gt, token_maps[t["target_id"]])
            if not pc:
                row["status"] = "no_match"
                rows.append(row)
                continue
            row["lddt"] = compute_lddt(np.stack(pc), np.stack(gc))
            row["n_matched_atoms"] = len(pc)
            row["status"] = "ok"
            rows.append(row)

        dest = RESULTS / f"fbrest_{tag}.csv"
        with dest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        ok = [r for r in rows if r["status"] == "ok"]
        print(f"{tag}: {len(ok)}/{len(rows)} ok  "
              f"lDDT {sum(r['lddt'] for r in ok) / len(ok):.4f} -> {dest}")
        for r in rows:
            if r["status"] != "ok":
                print(f"  FAILED {r['target_id']}: {r['status']} {r['error']}")


if __name__ == "__main__":
    main()
