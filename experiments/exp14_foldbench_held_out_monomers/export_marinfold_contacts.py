"""Step 3 -- MarinFold's predicted contacts, as Helico contact arms.

exp245 ran 100 rollouts per protein for `exp232-decontam-m2-p06-step145199` and
wrote, per protein, a dense L x L score matrix of occurrence-frequency votes.
Only its `results/` prefix was exported to the public bucket, so the matrices
are mirrored here from CoreWeave S3 and pinned by digest.

Two things this script is careful about, because getting either wrong produces
a plausible-looking arm that quietly measures something else:

**The ranking is exp245's, not a re-derivation.** Candidates are upper-triangle
pairs of *resolved* residues at separation >= 6, ordered by a stable
`argsort(-score)`, so ties fall in `np.triu_indices` order. Reproducing that
ordering reproduces the published precision at L, L/2 and L/5 on all 333
proteins to floating-point identity, and this script asserts exactly that
before writing anything. A looser rule -- ranking the sparse vote rows
directly, which drops the zero-score candidates and loses the tie order --
scores 0.572 at L against the published 0.510: close enough to look right, and
wrong enough to change the experiment.

**The cut is taken in prompt space, then mapped.** `L` is exp245's prompt
length, so `mf_L` is exactly the list whose precision is published, re-seated
onto Helico token indices by `build_index_map.py`.

Writes `data/arms/mf_{L,L2,L5}.json` -- `{target_id: [[i, j], ...]}` in Helico
token indices, the shape `modal/bench_byclass.py` consumes -- plus
`data/marinfold_arm_accuracy.csv` and the input pins.

Run:
    set -a; . ~/.config/marin/cw-rno2a.env; set +a
    uv run python experiments/exp14_foldbench_held_out_monomers/export_marinfold_contacts.py
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
ACCURACY = U.DATA / "marinfold_arm_accuracy.csv"

CUTS = (("L", lambda L: L), ("L2", lambda L: max(1, L // 2)),
        ("L5", lambda L: max(1, L // 5)))
#: exp245 publishes precision under these labels; ours are filename-safe.
CUT_LABEL = {"L": "L", "L2": "L/2", "L5": "L/5"}


def mirror_dense() -> dict[str, dict]:
    """Pull the per-protein score matrices out of CoreWeave, pinned by digest."""
    DENSE.mkdir(parents=True, exist_ok=True)
    client = U.cw_client()
    prefix = f"{U.CW_RUN_ROOT}/dense_scores/{U.CHECKPOINT_DIR}/"
    pins = {}
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=U.CW_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            local = DENSE / Path(obj["Key"]).name
            if not local.exists():
                client.download_file(U.CW_BUCKET, obj["Key"], str(local))
            pins[local.name] = {"size": local.stat().st_size,
                                "sha256": U.sha256(local)}
    if not pins:
        raise SystemExit(f"no score matrices under s3://{U.CW_BUCKET}/{prefix}")
    return pins


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
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="allowed drift from exp245's published precision")
    args = parser.parse_args()

    pins = mirror_dense()
    PINS.write_text(json.dumps(
        {"source": f"s3://{U.CW_BUCKET}/{U.CW_RUN_ROOT}/dense_scores/{U.CHECKPOINT_DIR}/",
         "checkpoint": U.CHECKPOINT_LABEL, "n_files": len(pins), "files": pins},
        indent=2) + "\n")

    universe = U.load_gt_universe()
    token_map = {stem: {int(p): t for p, t in mapping.items()}
                 for stem, mapping in
                 json.loads((U.DATA / "token_map.json").read_text()).items()}
    with (U.DATA / "targets.csv").open() as handle:
        targets = list(csv.DictReader(handle))
    published = published_precision()

    arms: dict[str, dict[str, list]] = {label: {} for label, _ in CUTS}
    rows, drift, dropped = [], [], []
    for target in targets:
        stem = target["target_id"]
        mapping = token_map.get(stem)
        if not mapping:
            dropped.append((stem, "no verified index map"))
            continue
        record = universe[stem]
        length = record["L"]
        score = np.load(DENSE / f"foldbench_monomer__{stem}.npz")["score"]
        score = score.astype(np.float64)
        if score.shape != (length, length):
            dropped.append((stem, f"score shape {score.shape} != L={length}"))
            continue

        ranked = U.rank_pairs(score, record["resolved"])
        truth = U.true_matrix(length, record["contacts"])

        row = {"target_id": stem, "eval_set": target["eval_set"], "L": length}
        for label, cut in CUTS:
            top = ranked[:min(cut(length), len(ranked))]
            mine = float(np.mean([truth[i, j] for i, j in top])) if top else float("nan")
            want = published[(stem, CUT_LABEL[label])]
            if abs(mine - want) > args.tolerance:
                drift.append((stem, label, mine, want))
            pairs = []
            for i, j in top:
                a, b = mapping.get(i), mapping.get(j)
                if a is None or b is None or a == b:
                    continue
                pairs.append([min(a, b), max(a, b)])
            arms[label][stem] = pairs
            row[f"precision_{label}"] = round(mine, 6)
            row[f"n_pairs_{label}"] = len(pairs)
            row[f"n_unmapped_{label}"] = len(top) - len(pairs)
        rows.append(row)

    if drift:
        raise SystemExit(
            f"{len(drift)} (stem, cut) precisions do not reproduce exp245's "
            f"published values, e.g. {drift[:5]}. The ranking or the ground "
            f"truth differs from upstream; do not run this arm."
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
    print(f"precision reproduces exp245 exactly on {len(rows)}/{len(rows)} targets "
          f"at every cut")
    for label, _ in CUTS:
        total = sum(len(v) for v in arms[label].values())
        unmapped = sum(r[f"n_unmapped_{label}"] for r in rows)
        mean = float(np.mean([r[f"precision_{label}"] for r in rows]))
        print(f"  mf_{label}: {total} pairs "
              f"({total / max(len(rows), 1):.0f}/target), {unmapped} unmapped, "
              f"mean precision {mean:.4f}")
    print(f"\narms -> {U.ARMS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
