"""Paired, target-level analysis of the contact pilot's saved CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """Average seeds within each target; bootstrap targets, not independent seeds."""
    metrics = ["lddt", "all_recall", "all_precision", "all_bce", "all_ap", "all_brier",
               "pp_recall", "pl_recall"]
    target = results.groupby(["arm", "phase", "mode", "pdb_id"])[metrics].mean()
    comparisons = []
    for mode in ("sparse", "full", "search4", "oracle4"):
        comparisons.append((f"masked_{mode}_minus_none", ("masked", "final", mode),
                            ("masked", "final", "none")))
    comparisons.append(("masked_minus_unknown_no_contacts", ("masked", "final", "none"),
                        ("unknown", "final", "none")))
    for arm in ("masked", "unknown"):
        comparisons.append((f"{arm}_finetune_minus_initial", (arm, "final", "none"),
                            (arm, "initial", "none")))
    rows = []
    rng = np.random.default_rng(20)
    for name, treated, baseline in comparisons:
        delta = target.loc[treated] - target.loc[baseline]
        for metric in metrics:
            # Revealing labels changes the evaluated masked subset, especially
            # for positive-only search hypotheses. Only compare head metrics
            # when both conditions use the same all-masked evaluation set.
            if treated[2] != baseline[2] and metric in {"all_bce", "all_ap", "all_brier"}:
                continue
            values = delta[metric].dropna().to_numpy()
            if not len(values): continue
            if len(values) > 1:
                draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
                low, high = np.quantile(draws, [.025, .975])
            else:
                low = high = np.nan
            rows.append(dict(comparison=name, metric=metric, targets=len(values),
                             mean_delta=values.mean(), ci_low=low, ci_high=high))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    root = args.experiment
    data = root / "data"
    frames = []
    for arm in ("masked", "unknown"):
        cache = root / ".cache" / "trainings" / f"{arm}-v1"
        if (cache / "evaluation.csv").exists():
            frame = pd.read_csv(cache / "evaluation.csv")
            frames.append(frame)
    results = pd.concat(frames, ignore_index=True) if frames else pd.read_csv(data / "evaluation.csv")
    results.to_csv(data / "evaluation.csv", index=False)
    comparison = paired_comparisons(results)
    comparison.to_csv(data / "paired_comparisons.csv", index=False)
    print(comparison.to_string(index=False))
    summary = results.groupby(["arm", "phase", "mode", "kind"])[
        ["lddt", "all_recall", "all_precision", "all_bce", "all_ap",
         "pp_recall", "pl_recall", "request_precision", "request_satisfaction"]].mean()
    summary.to_csv(data / "by_category.csv")


if __name__ == "__main__":
    main()
