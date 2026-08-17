"""Add MarinFold exp226's 23 net-new FoldBench monomers to the by-class set.

[MarinFold #226](https://github.com/Open-Athena/MarinFold/issues/226) expanded
the contact eval set with the 234 FoldBench monomers that were never used, and
filtered the whole 776 against MarinFold's *training* corpora (AFDB +
ESM-Atlas, mmseqs, hit iff evalue <= 1e-3 and qcov >= 0.50). 23 of the net-new
survive at <40% identity -- and all 23 are natural proteins, which is the axis
the eval set is thinnest on.

They matter here because they are the one slice where MarinFold clearly beats
Protenix v2 single sequence at contact prediction after homology removal
(R-precision 0.407 vs 0.243, bootstrap CI [+0.063, +0.263]). If the folding gain
tracks contact quality, this is where it should show up; if it does not, the
foldbench100 result is weaker than it looks.

Everything needed is already on disk from exp226's run:

  /data/exp226_gt/cif/                       RCSB assembly1 mmCIFs
  /data/exp226_gt/scores/exp199_eval2_new23  vote-aggregated exp199 contacts
  /data/exp226_gt/protenix_best/{msa,single_seq}/<stem>/structure.cif

so no MarinFold or Protenix inference is repeated here.

Two of the 23 parse one residue longer than exp226's `resolved` list implies --
an internal residue Helico's parser keeps and exp226 does not. The prompt->token
map is recovered by a greedy walk and then verified position by position; a
target whose map does not verify is dropped rather than silently shifted, which
is the failure mode this whole pipeline is built to avoid.

Run (after build_targets.py):
    uv run python experiments/marinfold_contacts/byclass/add_foldbench_rest.py
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from helico.contacts import MIN_CONTACT_DEGREE, MIN_SEQ_SEPARATION

EXP226 = Path("/data/exp226_gt")
CIF_DIR = EXP226 / "cif"
VOTES = EXP226 / "scores/exp199_eval2_new23"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATASET = "foldbench_rest"


def compatible(a: str, b: str) -> bool:
    """Equal up to X, which the two pipelines use differently for modified residues."""
    return len(a) == len(b) and all(x == y or x == "X" or y == "X" for x, y in zip(a, b))


def align_map(mine: str, expect: str) -> list[int] | None:
    """Index of each `expect` position within `mine`, or None if it does not verify.

    `mine` is `expect` with zero or more residues inserted -- Helico resolves a
    residue exp226 treated as missing. A greedy left-to-right walk recovers the
    correspondence; it could in principle mis-seat on a repeat, so the result is
    verified (strictly increasing, and every mapped residue compatible) before
    it is used.
    """
    out, mi = [], 0
    for e in expect:
        while mi < len(mine) and not (mine[mi] == e or mine[mi] == "X" or e == "X"):
            mi += 1
        if mi >= len(mine):
            return None
        out.append(mi)
        mi += 1
    if any(b <= a for a, b in zip(out, out[1:])):
        return None
    if not all(mine[t] == expect[k] or mine[t] == "X" or expect[k] == "X"
               for k, t in enumerate(out)):
        return None
    return out


def main() -> None:
    from helico.bench import structure_to_chains
    from helico.data import parse_mmcif

    import gemmi

    scratch = Path(__file__).resolve().parents[3] / ".exp226"
    man_dir = scratch if scratch.exists() else Path(
        "/tmp/claude-1000/-home-bizon-git-helico--claude-worktrees-"
        "helico-residue-contacts-redesign-4cc1c4/"
        "2998fa09-44e4-48ef-ac0f-9d0289be31ec/scratchpad/exp226")
    with (man_dir / "eval2_new_predictor_manifest.csv").open() as f:
        manifest = {r["stem"]: r for r in csv.DictReader(f)}
    universe = {json.loads(l)["stem"]: json.loads(l)
                for l in (man_dir / "gt_universe_eval2_new.jsonl").read_text().splitlines()}

    votes = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(VOTES / "*.parquet")))])

    (DATA / "gt").mkdir(parents=True, exist_ok=True)
    (DATA / "arms").mkdir(parents=True, exist_ok=True)
    arms_path = DATA / "arms"
    arms = {n: json.loads((arms_path / f"marinfold_{n}.json").read_text())
            for n in ("L", "L2", "L5")}
    gt_export = json.loads((arms_path / "gt_from_exp211.json").read_text())

    # prompt position -> ground-truth residue index, persisted because the
    # Protenix baselines need it too: exp226 fed Protenix the full prompt, so
    # its output has one residue per prompt position while the ground truth has
    # only the resolved ones. Matching those by position without this map
    # mis-registers every target that has an internal gap.
    token_maps: dict[str, dict[str, int]] = {}
    rows, dropped, stats = [], [], []
    for stem, m in sorted(manifest.items()):
        tid = f"{DATASET}__{stem}"
        rec = universe[stem]
        prompt, resolved = m["input_seq"], rec["resolved"]
        expect = "".join(prompt[p] for p in resolved)

        # Keep only the target chain. The assembly can carry several, and
        # FoldBench's chain id is sometimes the label rather than the auth
        # chain -- exp226's manifest records the auth one.
        st_g = gemmi.read_structure(str(CIF_DIR / m["gt_cif"]))
        st_g.setup_entities()
        for model in st_g:
            for ch in [c.name for c in model]:
                if ch != m["gt_chain"]:
                    model.remove_chain(ch)
        st_g.assign_label_seq_id(True)
        cif = DATA / "gt" / f"{tid}.cif.gz"
        with gzip.open(cif, "wt") as f:
            f.write(st_g.make_mmcif_document().as_string())

        st = parse_mmcif(cif, max_resolution=float("inf"))
        chains = [c for c in structure_to_chains(st) if c["type"] == "protein"] if st else []
        if len(chains) != 1:
            dropped.append((tid, f"{len(chains)} protein chains"))
            cif.unlink()
            continue
        mine, cid = chains[0]["sequence"], chains[0]["id"]

        if compatible(mine, expect):
            token_of = {p: k for k, p in enumerate(resolved)}
        else:
            walk = align_map(mine, expect)
            if walk is None:
                dropped.append((tid, f"no verified map ({len(mine)} vs {len(expect)})"))
                cif.unlink()
                continue
            token_of = {p: walk[k] for k, p in enumerate(resolved)}

        L = len(mine)

        def to_tokens(pairs):
            out = []
            for a, b in pairs:
                ta, tb = token_of.get(int(a)), token_of.get(int(b))
                if ta is None or tb is None:
                    continue          # contact on a residue Helico does not see
                out.append([min(ta, tb), max(ta, tb)])
            return out

        v = votes[votes.stem == stem]
        ranked = to_tokens((int(a), int(b)) for a, b, _ in
                           sorted(zip(v.i, v.j, v.votes), key=lambda r: -r[2]))
        for label, n in (("L5", max(1, L // 5)), ("L2", max(1, L // 2)), ("L", L)):
            arms[label][tid] = ranked[:n]

        gt_export[tid] = to_tokens(
            (int(i), int(j)) for i, j, d in rec["contacts"]
            if d >= MIN_CONTACT_DEGREE and abs(int(i) - int(j)) >= MIN_SEQ_SEPARATION
        )
        token_maps[tid] = {str(p): t for p, t in token_of.items()}
        rows.append({"dataset": DATASET, "stem": stem, "target_id": tid,
                     "release_date": "", "L": rec["L"],
                     "n_resolved": rec["n_resolved"], "input_seq": prompt})
        stats.append({"target_id": tid, "chain": cid, "L": L,
                      "n_gt": len(gt_export[tid]), "n_pred_L": len(arms["L"][tid])})

    for n in ("L", "L2", "L5"):
        (arms_path / f"marinfold_{n}.json").write_text(json.dumps(arms[n]))
    (arms_path / "gt_from_exp211.json").write_text(json.dumps(gt_export))
    (DATA / "token_map_foldbench_rest.json").write_text(json.dumps(token_maps))

    tpath = DATA / "targets.csv"
    with tpath.open() as f:
        existing = list(csv.DictReader(f))
        fields = list(existing[0])
    existing = [r for r in existing if r["dataset"] != DATASET]
    with tpath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing + rows)

    print(f"added {len(rows)}, dropped {len(dropped)}")
    for tid, why in dropped:
        print(f"  dropped {tid}: {why}")

    ps, rs = [], []
    for s in stats:
        pred = {tuple(p) for p in arms["L"][s["target_id"]]}
        true = {tuple(p) for p in gt_export[s["target_id"]]}
        if pred and true:
            ps.append(len(pred & true) / len(pred))
            rs.append(len(pred & true) / len(true))
    print(f"\n{DATASET}: n={len(stats)} precision={sum(ps)/len(ps):.3f} "
          f"recall={sum(rs)/len(rs):.3f} "
          f"n_pred={sum(s['n_pred_L'] for s in stats)/len(stats):.0f} "
          f"n_true={sum(s['n_gt'] for s in stats)/len(stats):.0f}")
    print(f"targets.csv now has {len(existing) + len(rows)} rows")


if __name__ == "__main__":
    main()
