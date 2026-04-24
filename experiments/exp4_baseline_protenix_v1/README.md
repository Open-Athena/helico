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

With the TemplateEmbedder + MSA-subsample fixes now on main, we expect to
be within a small constant factor of upstream Protenix's published
FoldBench numbers. Any substantial remaining delta points at something
specific we'd want to keep hunting (MSA differences, sampling variance,
or another featurization bug).

## Background

The initial exp4 baseline in this experiment — pre-fix Helico — fell
dramatically short of Protenix on interface categories (6.8% vs 38.4%
ab-ag, 14.5% vs 64.8% p-protein, etc.). Two bugs were found and fixed
via Helico↔Protenix pipeline diffing (see commits `72f10e6` and `61e94f5`):

- **`TemplateEmbedder.forward` was stubbed to return `0`.** Protenix
  v1.0.0's weights were trained to run the embedder every recycle on
  a 4-slot dummy-template pad (`aatype=[31, 0, 0, 0]`, everything else
  zero) even when `use_template=False` — we were silently skipping
  that contribution to the pair tensor. Responsible for ~75% of the
  trunk-side divergence.
- **MSA rows weren't randomly subsampled per cycle.** Protenix draws
  a fresh `randint(1, N_msa)` subset every cycle (AF3 SI §3.5). We
  were using all rows every cycle. Mattered most on large multi-chain
  ab-ag targets.

This notebook re-runs the benchmark with both fixes on main and
compares to Protenix's published FoldBench numbers.

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
    "protenix-v1-5seeds",
    checkpoint="protenix-v1",
    workers=8,
    gpu="H100",
    n_samples=5,           # samples per seed
    n_seeds=5,             # total predictions = 25, matches published FoldBench protocol
    n_cycles=10,
    max_tokens=2048,
    cutoff_date="2024-01-01",
    est_wall_hours=4.0,    # 5× recycling per target ⇒ ~5× 0.75h
)
git_sha = bench.meta.get("git_sha") or "?"
print(f"cached: {bench.cached}  |  git_sha: {git_sha[:8]}")
```

This is the new baseline. The previous single-seed run
(`protenix-v1-default`) is kept under the old `data/summary.csv` for
reference; the n=5×5 numbers become the post-exp8-round-3 canonical
comparison point.

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

Post-fix numbers (same 679-target, 25-sample protocol, 2024-01+ cutoff):

| Category                 | Helico (this run) | Protenix (published, 2024-01+) | Helico/Protenix |
|--------------------------|-------------------|--------------------------------|-----------------|
| interface_antibody_antigen | 30.4%           | 38.4%                          | 79%             |
| interface_protein_dna      | 46.7%           | 67.6%                          | 69%             |
| interface_protein_ligand   | 33.2%           | 53.3%                          | 62%             |
| interface_protein_peptide  | 42.9%           | —                              | —               |
| interface_protein_protein  | 33.6%           | 64.8%                          | 52%             |
| interface_protein_rna      | 31.8%           | 56.4%                          | 56%             |
| monomer_dna (mean LDDT)    | 0.52            | 0.44                           | 118%            |
| monomer_rna (mean LDDT)    | 0.60            | 0.59                           | 102%            |
| monomer_protein (mean LDDT)| 0.83            | — (not published numerically)  | —               |

Headlines:

- **Monomer categories match or beat Protenix's published numbers** (monomer_dna 0.52 ≥ 0.44, monomer_rna 0.60 ≥ 0.59, monomer_protein 0.83). With a very small N for DNA/RNA these aren't statistically significant but confirm nothing is horribly wrong.
- **Interface categories are ~50–80% of Protenix's published success rates.** All of these came up substantially from the pre-fix baseline (ab-ag 6.8% → 30.4%, p-protein 14.5% → 33.6%, etc.) — the template + MSA-subsample fixes are doing the expected work.
- **interface_protein_protein remains the widest relative gap** (52% of Protenix). Worth prioritizing in followup.

The remaining gap isn't template-shaped — FoldBench doesn't ship templates
and Protenix's published numbers also use the dummy-template path. Most
likely candidates for the remaining delta:

1. **MSA differences.** Published Protenix runs with `--use_msa_server` (its own MSA server); we use the FoldBench-bundled MSAs. Different MSA depth/pairing → different predictions.
2. **bf16 numerical accumulation.** Same weights, slightly different op ordering/precision across implementations compounds over 10 recycles × 200 diffusion steps × 25 samples.

The MSA hypothesis is the most testable — see issue #TBD for the
follow-up experiment.
