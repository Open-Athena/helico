"""Measure exp199 MarinFold contact accuracy against pyconfind ground truth.

Ground truth comes from exp211's `gt_universe.jsonl`, which stores pyconfind
`[i, j, degree]` triples in the *same* 0-indexed residue space as the model's
predictions. Re-deriving it from the cached PDBs would mean re-establishing that
index mapping, which is exactly the step most likely to go wrong silently.

Helico's own filters are applied on top so the ground truth is the thing the
folding model was trained against: degree >= MIN_CONTACT_DEGREE and
|i - j| >= MIN_SEQ_SEPARATION.

Two recipes, both from the same per-rollout emissions:

  rollout   vote-aggregated over the 100 rollouts, ranked by emission frequency
  single    one rollout's own emission order

These precision/recall numbers set the operating point for the matched-synthetic
control in helico#11. Without them a low real-contact folding score cannot be
attributed to error *rate* versus error *structure*.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

D = Path("/home/bizon/git/MarinFold/.claude/worktrees/contact-consistency-exp199-9d394e"
         "/experiments/exp211_evals_contact_set_3d_self_consistency")
sys.path.insert(0, "src")
from helico.contacts import MIN_CONTACT_DEGREE, MIN_SEQ_SEPARATION  # noqa: E402

DATASET = "foldbench100"


def load_gt():
    out = {}
    for line in (D / "_scratch/gt_universe.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("dataset") != DATASET:
            continue
        out[r["stem"]] = {
            (min(int(i), int(j)), max(int(i), int(j)))
            for i, j, deg in r["contacts"]
            if deg >= MIN_CONTACT_DEGREE and abs(int(i) - int(j)) >= MIN_SEQ_SEPARATION
        }
    return out


def ranked_pairs(df, recipe):
    """Ranked contact list, best first."""
    df = df[~df.duplicate]
    if recipe == "rollout":
        votes = Counter((min(int(a), int(b)), max(int(a), int(b)))
                        for a, b in zip(df.i, df.j))
        return [p for p, _ in votes.most_common()]
    one = df[df.rollout == 0].sort_values("order")
    return [(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(one.i, one.j)]


def main():
    gt = load_gt()
    meta = {}
    for line in (D / "_scratch/gt/gt_index.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["dataset"] == DATASET:
            meta[r["stem"]] = r

    rows = []
    for stem, truth in sorted(gt.items()):
        f = D / "_scratch/rollouts/contacts" / f"{DATASET}__{stem}.parquet"
        if not f.exists() or not truth:
            continue
        df = pd.read_parquet(f)
        L = int(meta[stem]["L"])
        for recipe in ("rollout", "single"):
            ranked = ranked_pairs(df, recipe)
            for label, n in (("L/5", max(1, L // 5)), ("L/2", max(1, L // 2)), ("L", L)):
                top = set(ranked[:n])
                if not top:
                    continue
                tp = len(top & truth)
                rows.append({"stem": stem, "L": L, "recipe": recipe, "n_label": label,
                             "n_requested": n, "n_got": len(top), "n_true": len(truth),
                             "precision": tp / len(top), "recall": tp / len(truth)})
    out = pd.DataFrame(rows)
    out.to_csv("marinfold_exp199_accuracy.csv", index=False)
    print(f"targets: {out['stem'].nunique()}   (contact files present and GT non-empty)\n")
    g = (out.groupby(["recipe", "n_label"])
            .agg(precision=("precision", "mean"), recall=("recall", "mean"),
                 n_true=("n_true", "mean"), got=("n_got", "mean"))
            .round(3))
    print(g.to_string())
    print("\nHelico was trained sampling precision in [0.4, 1.0]; MarinFold's "
          "measured operating point is what the matched-synthetic control must use.")


if __name__ == "__main__":
    main()
