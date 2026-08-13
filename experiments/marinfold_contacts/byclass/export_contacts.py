"""Export MarinFold exp199 contacts for the by-class evaluation set.

Same job as `../export_contacts.py`, which handles `foldbench100` against
FoldBench's own ground truths. This one runs over all four of exp211's target
classes against the CIFs built by `build_targets.py`, so the by-class folding
comparison and the by-class contact-accuracy figure cover the same targets.

The index map is the piece that has to be right. MarinFold indexes residues into
the published prompt; Helico indexes into the residues actually resolved in the
ground truth. `build_targets.py` verifies that Helico's parsed sequence equals
exp211's `resolved` subset of the prompt for all 357 targets, so the map is
exactly `resolved[k] -> k`, and this script re-checks it per target rather than
trusting that.

Writes to `data/arms/`:

  marinfold_L.json    top-L vote-aggregated rollout contacts, Helico token indices
  marinfold_L2.json   top-L/2
  marinfold_L5.json   top-L/5
  gt_from_exp211.json exp211's ground-truth contacts under Helico's own filters,
                      for the accuracy audit below

Run:
    uv run python experiments/marinfold_contacts/byclass/export_contacts.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from helico.contacts import MIN_CONTACT_DEGREE, MIN_SEQ_SEPARATION

EXP211 = Path("/home/bizon/git/MarinFold/.claude/worktrees/contact-consistency-exp199-9d394e"
              "/experiments/exp211_evals_contact_set_3d_self_consistency")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def compatible(a: str, b: str) -> bool:
    """Equal up to X, which the two pipelines use differently for modified residues."""
    return len(a) == len(b) and all(x == y or x == "X" or y == "X" for x, y in zip(a, b))


def ranked_pairs(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Vote-aggregated contact pairs in prompt indices, most-voted first.

    exp82 settled this recipe: 100 rollouts, per-rollout resampling, pairs
    ranked by how many rollouts proposed them. It is worth ~0.09 precision over
    reading a single rollout, and the two must never be mixed.
    """
    df = df[~df.duplicate]
    votes = Counter((min(int(a), int(b)), max(int(a), int(b)))
                    for a, b in zip(df.i, df.j))
    return [p for p, _ in votes.most_common()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DATA / "arms"))
    args = ap.parse_args()

    from helico.bench import structure_to_chains
    from helico.data import parse_mmcif

    uni = {}
    for line in (EXP211 / "_scratch/gt_universe.jsonl").read_text().splitlines():
        r = json.loads(line)
        uni[(r["dataset"], r["stem"])] = r

    with (DATA / "targets.csv").open() as f:
        targets = list(csv.DictReader(f))

    arms: dict[str, dict[str, list]] = {}
    gt_export: dict[str, list] = {}
    stats, dropped = [], []

    for t in targets:
        tid, ds, stem = t["target_id"], t["dataset"], t["stem"]
        rec = uni[(ds, stem)]
        st = parse_mmcif(DATA / "gt" / f"{tid}.cif.gz", max_resolution=float("inf"))
        chains = [c for c in structure_to_chains(st) if c["type"] == "protein"] if st else []
        if not chains:
            dropped.append((tid, "no protein chain"))
            continue
        mine = max(chains, key=lambda c: len(c["sequence"]))["sequence"]
        prompt, resolved = t["input_seq"], rec["resolved"]

        if compatible(mine, prompt):
            imap = {i: i for i in range(len(prompt))}
        elif compatible(mine, "".join(prompt[i] for i in resolved if 0 <= i < len(prompt))):
            imap = {p: k for k, p in enumerate(i for i in resolved if 0 <= i < len(prompt))}
        else:
            dropped.append((tid, f"sequence mismatch ({len(mine)} vs {len(resolved)})"))
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

        cfile = EXP211 / "_scratch/rollouts/contacts" / f"{ds}__{stem}.parquet"
        if not cfile.exists():
            dropped.append((tid, "no rollout contacts"))
            continue
        ranked = to_tokens(ranked_pairs(pd.read_parquet(cfile)))
        for label, n in (("L5", max(1, L // 5)), ("L2", max(1, L // 2)), ("L", L)):
            arms.setdefault(f"marinfold_{label}", {})[tid] = ranked[:n]

        # exp211 stores raw pyconfind output; Helico's own oracle_contact_state
        # drops low-degree and short-range pairs. Comparing the two without
        # matching filters reads as a broken index map when the map is exact.
        gt_export[tid] = to_tokens(
            (min(int(i), int(j)), max(int(i), int(j)))
            for i, j, d in rec["contacts"]
            if d >= MIN_CONTACT_DEGREE and abs(int(i) - int(j)) >= MIN_SEQ_SEPARATION
        )
        stats.append({"target_id": tid, "dataset": ds, "stem": stem, "L": L,
                      "n_gt": len(gt_export[tid]),
                      "n_pred_L": len(arms["marinfold_L"][tid])})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, mapping in arms.items():
        (out / f"{name}.json").write_text(json.dumps(mapping))
    (out / "gt_from_exp211.json").write_text(json.dumps(gt_export))
    pd.DataFrame(stats).to_csv(out / "contact_stats.csv", index=False)

    print(f"kept {len(stats)}   dropped {len(dropped)}")
    for tid, why in dropped:
        print(f"  dropped {tid}: {why}")

    # Accuracy audit: precision/recall at top-L per class. These should track
    # the R-precision figure -- if they do not, the index map is wrong.
    print(f"\n{'class':14s} {'n':>4s} {'precision':>10s} {'recall':>8s} "
          f"{'n_pred':>7s} {'n_true':>7s}")
    byds: dict[str, list] = {}
    for s in stats:
        byds.setdefault(s["dataset"], []).append(s)
    for ds, ss in sorted(byds.items()):
        ps, rs = [], []
        for s in ss:
            pred = {tuple(p) for p in arms["marinfold_L"][s["target_id"]]}
            true = {tuple(p) for p in gt_export[s["target_id"]]}
            if not pred or not true:
                continue
            hit = len(pred & true)
            ps.append(hit / len(pred))
            rs.append(hit / len(true))
        print(f"{ds:14s} {len(ss):4d} {sum(ps)/len(ps):10.3f} {sum(rs)/len(rs):8.3f} "
              f"{sum(s['n_pred_L'] for s in ss)/len(ss):7.0f} "
              f"{sum(s['n_gt'] for s in ss)/len(ss):7.0f}")


if __name__ == "__main__":
    main()
