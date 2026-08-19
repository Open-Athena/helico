"""Step 6 -- the scoreboard, the paired deltas, and the two cuts that matter.

Every arm is scored on the same 333 units, so every comparison here is paired:
one bootstrap resample of the *proteins* is applied to every arm at once, and
intervals on differences are computed on the per-target difference rather than
on the two means. That is the convention RESULTS_contact_conditioning.md already
uses, and it is roughly half the width of an unpaired interval.

Reads the per-arm result tables produced by `ensure_byclass_run` plus
`data/protenix_v2_baseline.csv`, and writes:

``data/per_target.csv``   one row per (arm, target): lDDT, TM-score, status
``data/headline.csv``     mean per arm x eval set, with bootstrap CIs
``data/paired_deltas.csv``the comparisons the hypotheses are stated in
``data/val_vs_test.csv``  each arm's eval-val -> eval-test change
``data/strata.csv``       viral / homology-stratum / designed cuts

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/analyze.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

N_BOOT = 10_000
SEED = 20260818

#: Reporting order, and the arm -> label map used in every table and plot.
ARMS = (
    ("off", "Helico, no contacts"),
    ("mf_L5", "Helico + MarinFold, top-L/5"),
    ("mf_L2", "Helico + MarinFold, top-L/2"),
    ("mf_L", "Helico + MarinFold, top-L"),
    ("v2ss", "Helico + Protenix-v2 single-seq contacts"),
    ("v2msa", "Helico + Protenix-v2 +MSA contacts"),
    ("oracle", "Helico + oracle contacts"),
    ("protenix_v2_single_seq", "Protenix v2, single sequence"),
    ("protenix_v2_msa", "Protenix v2 + MSA"),
)

#: The comparisons the issue's hypotheses are stated in.
DELTAS = (
    # Every comparison keeps MarinFold top-L as the first term so the deck can
    # show them on one axis with a consistent sign; `oracle - mf_L` written the
    # other way round would silently drop out of that slide's filter.
    ("mf_L", "off"),
    ("mf_L", "protenix_v2_single_seq"),
    ("mf_L", "v2ss"),
    ("mf_L", "mf_L2"),
    ("mf_L", "mf_L5"),
    ("mf_L", "v2msa"),
    ("mf_L", "oracle"),
    ("mf_L", "protenix_v2_msa"),
    ("oracle", "protenix_v2_msa"),
)

SETS = ("eval-val", "eval-test", "eval-denovo")

#: Every metric score_monomer computes. `lower_is_better` matters for reading
#: the tables, not for the bootstrap: an interval on a mean is an interval
#: either way.
METRICS = (("lddt", False), ("tm_score", False), ("gdt_ts", False),
           ("rmsd", True))


def boot_indices(n: int) -> np.ndarray:
    """One resample matrix shared by every arm, so comparisons stay paired."""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def interval(values: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_arms(cache_root: Path) -> pd.DataFrame:
    """One row per (arm, target) from the byclass caches and the Protenix table."""
    frames = []
    for path in sorted(cache_root.glob("*/results.csv")):
        arm = path.parent.name
        if arm.startswith("smoke-"):
            # Smoke runs cover a handful of targets and would otherwise appear
            # as an arm with 4 proteins next to arms with 333.
            continue
        frame = pd.read_csv(path)
        frame["arm"] = arm
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no arm results under {cache_root}")
    helico = pd.concat(frames, ignore_index=True)
    helico = helico.rename(columns={"dataset": "eval_set"})

    baseline_path = U.DATA / "protenix_v2_baseline.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        baseline["status"] = "ok"
        # Carry every metric the baselines were scored on, not just lDDT:
        # selecting one column here is what previously left TM-score, GDT-TS
        # and RMSD unavailable for the two predictors the arms are compared to.
        columns = ["target_id", "eval_set", "arm", "status"]
        columns += [c for c in ("lddt", "tm_score", "gdt_ts", "rmsd")
                    if c in baseline.columns]
        helico = pd.concat([helico, baseline[columns]], ignore_index=True)
    return helico


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-root", type=Path,
                        default=U.HERE / ".cache/byclass")
    args = parser.parse_args()

    targets = pd.read_csv(U.DATA / "targets.csv")
    rows = load_arms(args.cache_root)

    # Attach the reporting cuts from targets.csv rather than trusting whatever
    # the runner wrote into its own `dataset` column.
    rows = rows.drop(columns=["eval_set"], errors="ignore").merge(
        targets[["target_id", "eval_set", "is_viral", "designed", "exp199_stratum",
                 "L_helico"]],
        on="target_id", how="left")
    rows.to_csv(U.DATA / "per_target.csv", index=False)

    ok = rows[rows.status == "ok"]
    arms = [a for a, _ in ARMS if a in set(ok.arm)]
    wide = ok.pivot_table(index="target_id", columns="arm", values="lddt")

    # Every arm must cover every target, or the means are not comparable and
    # the paired bootstrap is not paired. Report the shortfall loudly.
    complete = wide.dropna(subset=arms)
    dropped = sorted(set(wide.index) - set(complete.index))
    if dropped:
        print(f"WARNING: {len(dropped)} targets missing from at least one arm and "
              f"excluded from the paired tables: {dropped[:10]}")
    meta = targets.set_index("target_id").loc[complete.index]

    label = dict(ARMS)
    headline, deltas, val_test, strata = [], [], [], []

    for eval_set in (*SETS, "all-natural", "all"):
        if eval_set == "all":
            mask = np.ones(len(complete), bool)
        elif eval_set == "all-natural":
            mask = (meta.eval_set != "eval-denovo").to_numpy()
        else:
            mask = (meta.eval_set == eval_set).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        idx = boot_indices(n)
        for arm in arms:
            values = complete[arm].to_numpy()[mask]
            lo, hi = interval(values, idx)
            headline.append({"eval_set": eval_set, "arm": arm, "label": label[arm],
                             "n": n, "mean_lddt": values.mean(), "ci_lo": lo,
                             "ci_hi": hi})
        for a, b in DELTAS:
            if a not in arms or b not in arms:
                continue
            diff = complete[a].to_numpy()[mask] - complete[b].to_numpy()[mask]
            lo, hi = interval(diff, idx)
            deltas.append({"eval_set": eval_set, "a": a, "b": b,
                           "delta": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                           "n_better": int((diff > 0).sum()), "n": n})

    # H2: does the working set stand in for the held-out one?
    val_mask = (meta.eval_set == "eval-val").to_numpy()
    test_mask = (meta.eval_set == "eval-test").to_numpy()
    if val_mask.any() and test_mask.any():
        for arm in arms:
            val = complete[arm].to_numpy()[val_mask]
            test = complete[arm].to_numpy()[test_mask]
            val_test.append({"arm": arm, "label": label[arm],
                             "eval_val": val.mean(), "eval_test": test.mean(),
                             "change": test.mean() - val.mean()})
    else:
        print("skipping val-vs-test: one of the two sets has no scored targets")

    # The cuts exp245 found interpretable: homology to MarinFold's corpus, viral
    # status, designed vs natural.
    natural = meta.eval_set != "eval-denovo"
    cuts = {
        "viral": meta.is_viral.map({1: "viral", 0: "non-viral"}).where(natural),
        "designed": meta.designed.map({1: "designed", 0: "natural"}),
        "homology": np.where(
            meta.exp199_stratum.isin(["no_homolog", "id_20_30", "id_30_50"]),
            "under 50% id", "50% id or more"),
    }
    for cut_name, series in cuts.items():
        series = pd.Series(series, index=meta.index)
        for value, group in series.dropna().groupby(series.dropna()):
            mask = complete.index.isin(group.index)
            n = int(mask.sum())
            if n < 5:
                continue
            idx = boot_indices(n)
            for arm in arms:
                values = complete[arm].to_numpy()[mask]
                lo, hi = interval(values, idx)
                strata.append({"cut": cut_name, "value": value, "arm": arm,
                               "label": label[arm], "n": n,
                               "mean_lddt": values.mean(), "ci_lo": lo, "ci_hi": hi})

    for path, records in (
        (U.DATA / "headline.csv", headline),
        (U.DATA / "paired_deltas.csv", deltas),
        (U.DATA / "val_vs_test.csv", val_test),
        (U.DATA / "strata.csv", strata),
    ):
        pd.DataFrame(records).to_csv(path, index=False)
        print(f"-> {path}")

    # The same tables for TM-score, GDT-TS and RMSD, on the same paired
    # resamples, so the deck can show every metric with a comparable interval.
    metric_rows = []
    for metric, lower_better in METRICS:
        wide_metric = ok.pivot_table(index="target_id", columns="arm", values=metric)
        arms_here = [a for a in arms if a in wide_metric.columns]
        frame = wide_metric.dropna(subset=arms_here)
        meta_metric = targets.set_index("target_id").loc[frame.index]
        for eval_set in (*SETS, "all-natural", "all"):
            if eval_set == "all":
                mask = np.ones(len(frame), bool)
            elif eval_set == "all-natural":
                mask = (meta_metric.eval_set != "eval-denovo").to_numpy()
            else:
                mask = (meta_metric.eval_set == eval_set).to_numpy()
            n = int(mask.sum())
            if n == 0:
                continue
            idx = boot_indices(n)
            for arm in arms_here:
                values = frame[arm].to_numpy()[mask]
                lo, hi = interval(values, idx)
                metric_rows.append({
                    "metric": metric, "lower_is_better": int(lower_better),
                    "eval_set": eval_set, "arm": arm, "label": label[arm],
                    "n": n, "mean": values.mean(), "ci_lo": lo, "ci_hi": hi,
                })
    pd.DataFrame(metric_rows).to_csv(U.DATA / "headline_metrics.csv", index=False)
    print(f"-> {U.DATA / 'headline_metrics.csv'}")

    summary = {
        "n_units_scored": int(len(complete)),
        "n_units_excluded": len(dropped),
        "arms": arms,
        "n_boot": N_BOOT,
    }
    (U.DATA / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    frame = pd.DataFrame(headline)
    table = frame[frame.eval_set.isin(SETS)].pivot(
        index="label", columns="eval_set", values="mean_lddt")
    print()
    print(table.reindex([label[a] for a in arms]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
