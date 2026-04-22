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

Reference numbers from the published literature are not yet wired into
this notebook. Adding them is a followup — either inline in `data/` as a
`published_baselines.csv` read here, or in a separate experiment that
re-imports these CSVs. Leave as placeholder for now.

```python
# TODO: load data/published_baselines.csv once populated, and produce a
# delta plot (Helico's Protenix-v1 numbers vs published Protenix numbers).
```

## Conclusion

(Fill in once the notebook runs end-to-end.)

A short paragraph answering the question. A future reader looking at the
site should be able to get the baseline numbers and the "is it worse than
upstream Protenix?" answer from this section alone.
