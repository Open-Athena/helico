"""Score Protenix v2 predictions with the same lDDT path as every other arm.

`helico.upstream_protenix.score_upstream_outputs` is built around DockQ and
interface metrics; on these monomer targets it returns lDDT 0.000 for every
sample. Rather than debug an interface scorer for a monomer question, this
matches predicted to ground-truth atoms by (chain order, residue position within
chain, atom name) -- the same correspondence `helico.bench.match_atoms` uses --
and calls the same `compute_lddt`, so v2 numbers are directly comparable to the
Helico and Protenix-v1 arms.

Sample selection matches the other arms: Protenix's own ranking, read from the
per-seed summary confidence JSON, falling back to sample_0.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from helico.bench import compute_lddt  # noqa: E402
from helico.data import parse_mmcif  # noqa: E402


def atom_index(structure):
    """{(chain_order, residue_order, atom_name): coords} for a parsed structure."""
    out = {}
    for ci, chain in enumerate(structure.chains):
        for ri, res in enumerate(chain.residues):
            for a in res.atoms:
                out[(ci, ri, a.name)] = np.asarray(a.coords, dtype=float)
    return out


def lddt_against_gt(pred_cif: Path, gt_cif: Path) -> float | None:
    pred = parse_mmcif(pred_cif, max_resolution=float("inf"))
    gt = parse_mmcif(gt_cif, max_resolution=float("inf"))
    if pred is None or gt is None:
        return None
    pi, gi = atom_index(pred), atom_index(gt)
    keys = [k for k in gi if k in pi]
    if len(keys) < 10:
        return None
    return compute_lddt(np.stack([pi[k] for k in keys]),
                        np.stack([gi[k] for k in keys]))


def ranked_sample(seed_dir: Path) -> Path | None:
    """Protenix's own top-ranked sample for this seed."""
    cifs = sorted((seed_dir / "predictions").glob("*_sample_*.cif"))
    if not cifs:
        return None
    best, best_score = None, None
    for j in seed_dir.rglob("*summary_confidence*.json"):
        try:
            d = json.loads(j.read_text())
        except Exception:  # noqa: BLE001
            continue
        s = d.get("ranking_score")
        if s is None:
            continue
        stem = j.name.split("_summary")[0]
        cand = next((c for c in cifs if c.stem.startswith(stem)), None)
        if cand is not None and (best_score is None or s > best_score):
            best, best_score = cand, s
    if best is not None:
        return best
    return next((c for c in cifs if c.stem.endswith("_sample_0")), cifs[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    gt_dir = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths"
    rows = []
    roots = [p for p in args.dump_root.rglob("predictions") if p.is_dir()]
    seen = set()
    for root in roots:
        for tgt_dir in root.iterdir():
            if not tgt_dir.is_dir():
                continue
            pdb_id = tgt_dir.name
            if pdb_id in seen:
                continue
            for seed_dir in sorted(tgt_dir.glob("seed_*")):
                cif = ranked_sample(seed_dir)
                if cif is None:
                    continue
                v = lddt_against_gt(cif, gt_dir / f"{pdb_id}.cif.gz")
                if v is not None:
                    rows.append({"pdb_id": pdb_id, "lddt": round(v, 4),
                                 "cif": str(cif.relative_to(args.dump_root))})
                    seen.add(pdb_id)
                break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "lddt", "cif"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["pdb_id"]))
    if rows:
        m = sum(r["lddt"] for r in rows) / len(rows)
        print(f"scored {len(rows)} targets  mean lDDT = {m:.4f}  -> {args.out}")
    else:
        print("no targets scored")


if __name__ == "__main__":
    main()
