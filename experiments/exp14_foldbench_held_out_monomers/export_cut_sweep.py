"""How many contacts should Helico be given? Arms at cuts above top-L.

[MarinFold #256](https://github.com/Open-Athena/MarinFold/issues/256). exp14
swept the cut *downwards* -- top-L/5, top-L/2, top-L -- and the curve is already
flat at the top end. Nobody has run a cut above L, and
[MarinFold #254](https://github.com/Open-Athena/MarinFold/issues/254) says
there is a lot of true contact left up there: the 100 rollouts behind one of
these predictions collectively propose **92 % of the true contacts**, and vote
rank recovers 0.52 of them at R, 0.67 at 2R and 0.79 at 5R.

Helico's `contact-list` conditioning marks unlisted pairs UNKNOWN rather than
ABSENT, so a longer list does not assert false negatives -- it only trades
precision for recall. This writes the arms that price that trade:

    mf_1p5L  mf_2L  mf_3L  mf_5L     progressively deeper cuts
    mf_union                          every pair at least one rollout emitted

**The verified cuts are re-derived and re-checked, not skipped.** L, L/2 and
L/5 go through the same assertion `export_marinfold_contacts.py` uses -- they
must reproduce exp245's published per-protein precision to floating point --
because that check is what proves the ranking and the index map are still the
ones the published numbers came from. The new cuts have no published reference,
so the old ones are the only thing standing behind them.

Unlike `export_marinfold_contacts.py` this does **not** re-mirror the dense
matrices from CoreWeave. It reads the ones exp14 already pinned and verifies
them against `data/marinfold_inputs.json`, so it needs no cloud credentials and
cannot silently score a different set of matrices.

Writes `data/arms/mf_<cut>.json` for the new cuts (Helico token indices, the
shape `modal/bench_byclass.py` consumes) and `data/cut_sweep_accuracy.csv` with
precision *and* recall per cut -- recall is the whole point of going wider and
`marinfold_arm_accuracy.csv` does not carry it.

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/export_cut_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

DENSE = U.CACHE / "marinfold_dense"
PINS = U.DATA / "marinfold_inputs.json"
ACCURACY = U.DATA / "cut_sweep_accuracy.csv"

#: Cuts with a published precision to check against. Reproducing these is the
#: gate on everything else in this file.
VERIFY_CUTS = (("L", lambda L: L), ("L2", lambda L: max(1, L // 2)),
               ("L5", lambda L: max(1, L // 5)))
CUT_LABEL = {"L": "L", "L2": "L/2", "L5": "L/5"}

#: The new cuts. ``union`` is not a multiple of L -- it is every candidate the
#: rollouts actually voted for, which is where "just give it everything" lands.
NEW_CUTS = (("1p5L", lambda L: max(1, (3 * L) // 2)),
            ("2L", lambda L: 2 * L),
            ("3L", lambda L: 3 * L),
            ("5L", lambda L: 5 * L))
UNION = "union"


def verify_pins() -> int:
    """Check the local dense matrices against exp14's recorded digests."""
    pinned = json.loads(PINS.read_text())["files"]
    missing, wrong = [], []
    for name, want in pinned.items():
        local = DENSE / name
        if not local.exists():
            missing.append(name)
            continue
        if local.stat().st_size != want["size"] or U.sha256(local) != want["sha256"]:
            wrong.append(name)
    if missing or wrong:
        raise SystemExit(
            f"dense score matrices do not match exp14's pins: {len(missing)} "
            f"missing, {len(wrong)} altered (e.g. {(missing + wrong)[:3]}). "
            f"Re-run export_marinfold_contacts.py to re-mirror them."
        )
    return len(pinned)


def published_precision() -> dict[tuple[str, str], float]:
    """exp245's `contact_precision_all.csv`, all-range rows for this checkpoint."""
    path = U.CACHE / "upstream/contact_precision_all.csv"
    out = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["model"] == U.CHECKPOINT_LABEL and row["range"] == "all":
                out[(row["stem"], row["cut"])] = float(row["precision"])
    if not out:
        raise SystemExit(f"no rows for {U.CHECKPOINT_LABEL} in {path}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--eval-set", default="eval-val",
                        help="only targets in this set get an arm entry; "
                             "#256 reads eval-val and nothing else")
    args = parser.parse_args()

    n_pinned = verify_pins()
    universe = U.load_gt_universe()
    token_map = {stem: {int(p): t for p, t in mapping.items()}
                 for stem, mapping in
                 json.loads((U.DATA / "token_map.json").read_text()).items()}
    with (U.DATA / "targets.csv").open() as handle:
        targets = [t for t in csv.DictReader(handle)
                   if t["eval_set"] == args.eval_set]
    published = published_precision()
    print(f"{n_pinned} pinned score matrices verified; "
          f"{len(targets)} targets in {args.eval_set}")

    labels = [label for label, _ in NEW_CUTS] + [UNION]
    arms: dict[str, dict[str, list]] = {label: {} for label in labels}
    rows, drift, dropped = [], [], []
    for target in targets:
        stem = target["target_id"]
        mapping = token_map.get(stem)
        if not mapping:
            dropped.append((stem, "no verified index map"))
            continue
        record = universe[stem]
        length = record["L"]
        score = np.load(DENSE / f"foldbench_monomer__{stem}.npz")["score"].astype(np.float64)
        if score.shape != (length, length):
            dropped.append((stem, f"score shape {score.shape} != L={length}"))
            continue

        ranked = U.rank_pairs(score, record["resolved"])
        truth = U.true_matrix(length, record["contacts"])
        n_true = int(sum(truth[i, j] for i, j in ranked))
        # Every pair at least one rollout voted for. `ranked` is the full
        # candidate list in score order, so the voted ones are its prefix.
        n_voted = int(sum(1 for i, j in ranked if score[i, j] > 0))

        row = {"target_id": stem, "eval_set": target["eval_set"], "L": length,
               "n_true": n_true, "n_voted": n_voted,
               "voted_over_L": round(n_voted / length, 2)}

        for label, cut in VERIFY_CUTS:
            top = ranked[:min(cut(length), len(ranked))]
            mine = float(np.mean([truth[i, j] for i, j in top])) if top else float("nan")
            want = published[(stem, CUT_LABEL[label])]
            if abs(mine - want) > args.tolerance:
                drift.append((stem, label, mine, want))

        for label in labels:
            size = n_voted if label == UNION else dict(NEW_CUTS)[label](length)
            top = ranked[:min(size, n_voted)]  # never rank into the zero-vote mass
            hits = int(sum(truth[i, j] for i, j in top))
            pairs = []
            for i, j in top:
                a, b = mapping.get(i), mapping.get(j)
                if a is None or b is None or a == b:
                    continue
                pairs.append([min(a, b), max(a, b)])
            arms[label][stem] = pairs
            row[f"n_{label}"] = len(top)
            row[f"precision_{label}"] = round(hits / len(top), 6) if top else float("nan")
            row[f"recall_{label}"] = round(hits / n_true, 6) if n_true else float("nan")
            row[f"n_pairs_{label}"] = len(pairs)
        rows.append(row)

    if drift:
        raise SystemExit(
            f"{len(drift)} (stem, cut) precisions do not reproduce exp245's "
            f"published values, e.g. {drift[:5]}. The ranking or the ground "
            f"truth differs from upstream; do not run these arms."
        )

    U.ARMS.mkdir(parents=True, exist_ok=True)
    for label, mapping in arms.items():
        (U.ARMS / f"mf_{label}.json").write_text(json.dumps(mapping))

    with ACCURACY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} targets, {len(dropped)} dropped")
    for stem, why in dropped:
        print(f"  dropped {stem}: {why}")
    print(f"L, L/2 and L/5 reproduce exp245's published precision on "
          f"{len(rows)}/{len(rows)} targets")
    print(f"  voted pairs per target: {np.mean([r['voted_over_L'] for r in rows]):.1f}x L")
    for label in labels:
        total = sum(len(v) for v in arms[label].values())
        precision = float(np.mean([r[f"precision_{label}"] for r in rows]))
        recall = float(np.mean([r[f"recall_{label}"] for r in rows]))
        print(f"  mf_{label:5s}: {total:7d} pairs "
              f"({total / max(len(rows), 1):5.0f}/target)  "
              f"precision {precision:.4f}  recall {recall:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
