"""Export exp199 MarinFold contact predictions into Helico token indices.

MarinFold emits residue pairs indexed into the *published prompt* sequence
(full SEQRES). Helico's bench derives its sequence from the *resolved* residues
of the ground-truth structure. Those differ for most targets -- of 100 FoldBench
monomers only 15 agree outright -- so feeding MarinFold's indices straight in
would silently shift contacts on 83% of targets and look exactly like "real
contacts do not help".

The map is exp211's `resolved` list: prompt position p corresponds to Helico
token index rank(p in resolved). Targets whose sequences still disagree after
that mapping are dropped rather than fuzzily aligned.

Writes one JSON per arm: {pdb_id: [[i, j], ...]} in Helico token indices.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from helico.contacts import MIN_CONTACT_DEGREE, MIN_SEQ_SEPARATION

EXP211 = Path("/home/bizon/git/MarinFold/.claude/worktrees/contact-consistency-exp199-9d394e"
              "/experiments/exp211_evals_contact_set_3d_self_consistency")
DATASET = "foldbench100"


def compatible(a: str, b: str) -> bool:
    """Equal up to X, which the two pipelines use differently for modified residues."""
    return len(a) == len(b) and all(x == y or x == "X" or y == "X" for x, y in zip(a, b))


def ranked_pairs(df: pd.DataFrame, recipe: str) -> list[tuple[int, int]]:
    """Contact pairs in prompt indices, best first."""
    df = df[~df.duplicate]
    if recipe == "rollout":
        votes = Counter((min(int(a), int(b)), max(int(a), int(b)))
                        for a, b in zip(df.i, df.j))
        return [p for p, _ in votes.most_common()]
    if recipe == "single":
        one = df[df.rollout == 0].sort_values("order")
        return [(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(one.i, one.j)]
    raise ValueError(f"unknown recipe {recipe!r}")


def build_index_map(resolved: list[int]) -> dict[int, int]:
    """prompt position -> Helico token index."""
    return {p: k for k, p in enumerate(resolved)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/marinfold_contacts/arms")
    args = ap.parse_args()

    from helico.bench import _find_gt_path, load_targets, structure_to_chains
    from helico.data import parse_mmcif

    fb = Path.home() / ".cache/helico/data/benchmarks/FoldBench"
    gt_dir = fb / "examples" / "ground_truths"
    targets = {t.pdb_id.split("-")[0].lower(): t
               for t in load_targets(fb / "targets")["monomer_protein"]}

    uni = {}
    for line in (EXP211 / "_scratch/gt_universe.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("dataset") == DATASET:
            uni[r["stem"]] = r
    ev = pd.read_parquet(EXP211 / "data/eval_targets.parquet")
    ev = ev[ev.dataset == DATASET]
    prompt = dict(zip(ev.stem, ev.input_seq))

    arms: dict[str, dict[str, list]] = {}
    gt_export: dict[str, list] = {}
    kept = dropped = 0
    stats = []

    for stem, rec in sorted(uni.items()):
        pid = stem.split("_")[0].lower()
        if pid not in targets:
            dropped += 1
            continue
        pdb_id = targets[pid].pdb_id
        st = parse_mmcif(_find_gt_path(gt_dir, pdb_id), max_resolution=float("inf"))
        if st is None:
            dropped += 1
            continue
        mine = structure_to_chains(st)[0]["sequence"]
        p, resolved = prompt[stem], rec["resolved"]

        if compatible(mine, p):
            imap = {i: i for i in range(len(p))}
        elif compatible(mine, "".join(p[i] for i in resolved if 0 <= i < len(p))):
            imap = build_index_map([i for i in resolved if 0 <= i < len(p)])
        else:
            dropped += 1
            continue

        L = len(mine)

        def to_tokens(pairs):
            out = []
            for a, b in pairs:
                ta, tb = imap.get(a), imap.get(b)
                if ta is None or tb is None:
                    continue          # contact on an unresolved residue
                out.append([min(ta, tb), max(ta, tb)])
            return out

        cfile = EXP211 / "_scratch/rollouts/contacts" / f"{DATASET}__{stem}.parquet"
        if not cfile.exists():
            dropped += 1
            continue
        df = pd.read_parquet(cfile)

        for recipe in ("rollout", "single"):
            ranked = to_tokens(ranked_pairs(df, recipe))
            for label, n in (("L5", max(1, L // 5)), ("L2", max(1, L // 2)), ("L", L)):
                arms.setdefault(f"{recipe}_{label}", {})[pdb_id] = ranked[:n]

        # Apply Helico's own filters. exp211 stores the raw pyconfind output;
        # helico's oracle_contact_state drops low-degree and short-range pairs,
        # and comparing the two without matching filters gives 0.44 Jaccard --
        # which reads as a broken index map when the map is in fact exact.
        gt_export[pdb_id] = to_tokens(
            (min(int(i), int(j)), max(int(i), int(j)))
            for i, j, d in rec["contacts"]
            if d >= MIN_CONTACT_DEGREE and abs(int(i) - int(j)) >= MIN_SEQ_SEPARATION
        )
        stats.append({"pdb_id": pdb_id, "stem": stem, "L": L,
                      "n_gt": len(gt_export[pdb_id])})
        kept += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, mapping in arms.items():
        (out / f"{name}.json").write_text(json.dumps(mapping))
    (out / "gt_from_exp211.json").write_text(json.dumps(gt_export))
    pd.DataFrame(stats).to_csv(out / "targets.csv", index=False)

    print(f"targets kept: {kept}   dropped: {dropped}")
    for name, mapping in sorted(arms.items()):
        tot = sum(len(v) for v in mapping.values())
        print(f"  {name:14s} {len(mapping):3d} targets, {tot:6d} contacts "
              f"({tot / max(len(mapping), 1):.0f}/target)")


if __name__ == "__main__":
    main()
