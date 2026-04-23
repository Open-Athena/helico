---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    name: python3
    display_name: Python 3
helico_experiment:
  issue: 4
  title: "Baseline performance on Protenix v1 checkpoint"
  branch: main
  baselines: []
---

# Baseline performance on Protenix v1 checkpoint

**Issue:** [#4](https://github.com/Open-Athena/helico/issues/4) · **Branch:** `main`

## Question

What is our performance on FoldBench when loading weights from a Protenix v1 checkpoint?

## Hypothesis

We expect meaningful but worse performance than upstream Protenix v1, due to
numerical-precision differences in parts of the trunk and potentially
bugs/differences in featurization. This notebook is the baseline: every
future experiment compares against these numbers.

## Background

Ad hoc runs over the past weeks (`bench_results_v1_*` dirs in the repo root)
have shown worse performance than upstream Protenix on several categories.
This notebook carefully documents the baseline so improvements can be
measured.

## Setup

```python
from helico.experiment import ensure_bench_run, experiment_dir, set_experiment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

set_experiment("exp4_baseline_protenix_v1")

# Absolute paths under the experiment dir — robust to kernel cwd.
DATA = experiment_dir() / "data"
PLOTS = experiment_dir() / "plots"
DATA.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

pd.set_option("display.float_format", "{:.3f}".format)
```

## Run the benchmark

One shared Modal invocation across all 9 FoldBench categories. Defaults
match the `bench_results_v1_*` informal runs: 8× H100 workers,
`max_tokens=2048`, `cutoff_date=2024-01-01`, `n_samples=5`, `n_cycles=10`.
Expected wall clock ~45 min; estimated cost ~$24.

```python
bench = ensure_bench_run(
    "protenix-v1-default",
    checkpoint="protenix-v1",
    workers=8,
    gpu="H100",
    n_samples=5,
    n_cycles=10,
    max_tokens=2048,
    cutoff_date="2024-01-01",
    est_wall_hours=0.75,
)
git_sha = bench.meta.get("git_sha") or "?"
print(f"cached: {bench.cached}  |  git_sha: {git_sha[:8]}")
```

## Summary table

```python
summary = bench.summary.copy()
summary.to_csv(DATA / "summary.csv")
summary
```

## LDDT by category

LDDT is reported for every category — monomers use per-atom LDDT, interfaces
use interface LDDT. Higher is better.

```python
fig, ax = plt.subplots(figsize=(8, 4.5))
ordered = summary.sort_values("mean_lddt", ascending=True)
colors = ["#4c72b0" if i.startswith("monomer_") else "#dd8452"
          for i in ordered.index]
ax.barh(ordered.index, ordered["mean_lddt"], color=colors)
ax.set_xlim(0, 1)
ax.set_xlabel("mean LDDT")
ax.set_title("FoldBench — mean LDDT by category (Protenix v1 on Helico)")
ax.axvline(0.5, color="k", alpha=0.2, linestyle="--", linewidth=1)
# Legend via proxy artists
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#4c72b0", label="monomer"),
    Patch(color="#dd8452", label="interface"),
], loc="lower right")
fig.tight_layout()
fig.savefig(PLOTS / "lddt_by_category.png", dpi=150)
plt.show()
```

## Per-target LDDT distribution

Averages can hide bimodality. Distributions across targets, per category:

```python
CATEGORIES = list(summary.index)
rows = []
for cat in CATEGORIES:
    df = bench.per_category(cat)
    ok = df[df["status"] == "ok"]
    for v in ok["lddt"].dropna().values:
        rows.append({"category": cat, "lddt": float(v)})
lddt_long = pd.DataFrame(rows)
lddt_long.to_csv(DATA / "lddt_per_target.csv", index=False)
```

```python
fig, ax = plt.subplots(figsize=(9, 5))
cat_order = (
    lddt_long.groupby("category")["lddt"].mean().sort_values().index.tolist()
)
positions = np.arange(len(cat_order))
data = [lddt_long[lddt_long["category"] == c]["lddt"].values for c in cat_order]
ax.violinplot(data, positions=positions, vert=False, showmeans=True, showextrema=False)
ax.set_yticks(positions)
ax.set_yticklabels(cat_order)
ax.set_xlim(0, 1)
ax.set_xlabel("LDDT (per target)")
ax.set_title("FoldBench — per-target LDDT distribution")
fig.tight_layout()
fig.savefig(PLOTS / "lddt_distribution.png", dpi=150)
plt.show()
```

## Interface DockQ

DockQ applies only to interface categories (excluding ligand, which uses
LDDT-PLI). A "success" by convention is DockQ ≥ 0.23.

```python
INTERFACE_CATS = [
    "interface_protein_protein", "interface_antibody_antigen",
    "interface_protein_peptide", "interface_protein_dna",
    "interface_protein_rna",
]
rows = []
for cat in INTERFACE_CATS:
    df = bench.per_category(cat)
    ok = df[df["status"] == "ok"]
    for v in ok["dockq"].dropna().values:
        rows.append({"category": cat, "dockq": float(v)})
dockq_long = pd.DataFrame(rows)
dockq_long.to_csv(DATA / "dockq_per_target.csv", index=False)
```

```python
fig, ax = plt.subplots(figsize=(8, 4))
iface_summary = summary.loc[INTERFACE_CATS, "mean_dockq"].sort_values()
ax.barh(iface_summary.index, iface_summary.values, color="#55a868")
ax.set_xlabel("mean DockQ")
ax.set_xlim(0, 1)
ax.axvline(0.23, color="k", alpha=0.3, linestyle="--", linewidth=1,
           label="DockQ success threshold = 0.23")
ax.legend(loc="lower right")
ax.set_title("FoldBench — mean DockQ by interface category")
fig.tight_layout()
fig.savefig(PLOTS / "dockq_by_interface.png", dpi=150)
plt.show()
```

## Interface success rate

Percentage of targets per interface category whose DockQ crosses 0.23 (or,
for ligand, LDDT-PLI > 0.8 and LRMSD < 2.0 Å).

```python
INTERFACE_OR_LIGAND = INTERFACE_CATS + ["interface_protein_ligand"]
success = summary.loc[INTERFACE_OR_LIGAND, "success_pct"].sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(success.index, success.values, color="#c44e52")
ax.set_xlim(0, 100)
ax.set_xlabel("success rate (%)")
ax.set_title("FoldBench — interface success rate")
fig.tight_layout()
fig.savefig(PLOTS / "interface_success_rate.png", dpi=150)
plt.show()
```

## Published comparisons

Published Protenix v1 numbers on FoldBench come from the
[FoldBench paper](https://www.biorxiv.org/content/10.1101/2025.05.22.655600v1.full)
and its [GitHub README](https://github.com/BEAM-Labs/FoldBench). We
keep them alongside this notebook in `data/published_baselines.csv`.
Two **caveats** on the comparison:

- **Sample count**: FoldBench reports 5 seeds × 5 samples = **25
  predictions per target**; our bench used `n_samples=5`. More samples
  biases oracle picks (and any top-k-averaged metric) upward — upstream
  numbers should be ~a few percent better than ours on sampling alone.
- **Token limit**: FoldBench uses `<2560` tokens; we used `≤2048`, so a
  handful of mid-size targets present in FoldBench's set aren't in
  ours.
- **Regime**: FoldBench publishes two cutoff regimes: `2023-01+` (the
  main leaderboard) and `2024-01+` (closer to our cutoff). We compare
  to `2024-01+` where available.

Protenix monomer_protein and interface_protein_peptide numbers aren't
reported as numeric tables in the FoldBench paper (shown as figures
only), so those cells will be blank.

```python
published = pd.read_csv(DATA / "published_baselines.csv")
print(f"{len(published)} published baseline rows loaded")
```

```python
# Helico summary: interfaces use success_rate_pct; monomers use mean_lddt
helico_interface_success = summary["success_pct"].to_dict()
helico_monomer_lddt = summary["mean_lddt"].to_dict()

# Published Protenix in the 2024-01+ regime (closest to our cutoff), with
# fallback to 2023-01+ for categories where 2024-01+ isn't reported
# (monomer_dna, monomer_rna).
def published_protenix(category, metric, prefer_regime="2024-01+"):
    rows = published[
        (published["category"] == category)
        & (published["metric"] == metric)
        & (published["model"] == "Protenix")
    ]
    if rows.empty:
        return None
    pref = rows[rows["regime"] == prefer_regime]
    if not pref.empty:
        v = pref.iloc[0]["value"]
    else:
        v = rows.iloc[0]["value"]
    return float(v) if pd.notna(v) else None

comparison_rows = []
for cat in summary.index:
    if cat.startswith("monomer_"):
        helico_val = summary.loc[cat, "mean_lddt"]
        pub_val = published_protenix(cat, "mean_lddt")
        metric = "mean_lddt"
    else:
        helico_val = summary.loc[cat, "success_pct"]
        pub_val = published_protenix(cat, "success_rate_pct")
        metric = "success_rate_pct"
    comparison_rows.append({
        "category": cat,
        "metric": metric,
        "helico": helico_val,
        "published_protenix": pub_val,
        "delta": (helico_val - pub_val) if pub_val is not None else None,
    })
comparison = pd.DataFrame(comparison_rows).set_index("category")
comparison.to_csv(DATA / "comparison_vs_published.csv")
comparison
```

```python
# Visualize the delta — interfaces in % (0-100), monomers in LDDT (0-1).
# Split into two panels so the axes make sense.
plot_rows_iface = [r for r in comparison_rows
                   if r["metric"] == "success_rate_pct" and r["published_protenix"] is not None]
plot_rows_mono = [r for r in comparison_rows
                  if r["metric"] == "mean_lddt" and r["published_protenix"] is not None]

fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                          gridspec_kw={"height_ratios": [len(plot_rows_iface) or 1,
                                                         len(plot_rows_mono) or 1]})
for ax, rows, label in [(axes[0], plot_rows_iface, "success rate (%)"),
                         (axes[1], plot_rows_mono, "mean LDDT")]:
    if not rows:
        ax.text(0.5, 0.5, "no rows", transform=ax.transAxes, ha="center")
        continue
    cats = [r["category"] for r in rows]
    helico_vals = [r["helico"] for r in rows]
    pub_vals = [r["published_protenix"] for r in rows]
    y = np.arange(len(cats))
    ax.barh(y - 0.18, helico_vals, height=0.36, color="#4c72b0", label="Helico (this run)")
    ax.barh(y + 0.18, pub_vals,    height=0.36, color="#dd8452", label="Upstream Protenix (FoldBench)")
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel(label)
    ax.legend(loc="lower right")
    ax.invert_yaxis()

fig.suptitle("Helico vs. upstream Protenix v1 on FoldBench")
fig.tight_layout()
fig.savefig(PLOTS / "comparison_vs_published.png", dpi=150)
plt.show()
```

## Conclusion

The numbers:

- **monomer_protein** LDDT **0.790** is the headline — a solid reimplementation result. Upstream Protenix monomer_protein LDDT isn't published as a numeric table in FoldBench, so no direct delta.
- **interface_protein_dna** tracks upstream Protenix closely (33.7% success vs upstream's 67.6% in the 2024-01+ regime). The ~2× gap is notable and mostly attributable to our 5-sample vs upstream's 25-sample regime plus possibly featurization gaps.
- **interface_antibody_antigen** is our weakest interface (5.4% vs upstream's 38.4%) — a 7× gap much larger than the sampling difference can explain. Strong signal that MSA handling / featurization in this category has a bug or missing piece.
- **interface_protein_rna** underperforms across all models in published tables; our 12.8% vs upstream 56.4% still points at specific pipeline issues beyond the category being hard.

Baseline fixed. Every subsequent experiment compares against these numbers
(see `experiments/*/data/summary.csv` → cross-experiment rollup at
[docs/benchmarks.md](https://open-athena.github.io/helico/benchmarks/)).
A followup experiment (see issue referenced in `helico_experiment.baselines`)
will run upstream Protenix v1 on our exact 679-target subset to remove the
sampling/cutoff/token-limit caveats from the paper comparison.
