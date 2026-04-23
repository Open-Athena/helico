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
  issue: 8
  title: "Triage antibody-antigen interface underperformance"
  branch: main
  baselines: [exp4_baseline_protenix_v1]
---

# Triage antibody-antigen interface underperformance

**Issue:** [#8](https://github.com/Open-Athena/helico/issues/8) · **Branch:** `main`

## Question

Why does Helico's Protenix v1 reimplementation do so much worse than upstream
Protenix on `interface_antibody_antigen` (5.4% success vs published 38.4%,
a 7× gap)?

## Hypothesis

Two candidates:
1. **Sampling** — upstream uses 25 predictions/target (5 seeds × 5 samples),
   we used 5 in exp4. Could explain much of the gap on its own.
2. **Featurization / epitope recognition** — something in our pipeline
   (CDR MSA handling, chain-pair featurization, pair-representation at
   the epitope) is broken.

If (1): Helico at n=25 should close most of the gap. If (2): Helico stays
bad and distograms show incorrect epitope localization.

## Approach

Three targets covering the exp4 DockQ range:

| target | exp4 (n=5) DockQ | exp4 (n=5) LDDT |
|---|---|---|
| `8t59-assembly1` | 0.495 (best of 74) | 0.806 |
| `8q3j-assembly1` | 0.069 (weak) | 0.560 |
| `8v52-assembly1` | 0.020 (fail) | 0.244 |

For each:
- Compare Helico n=5 (from exp4) vs Helico n=25 (this experiment)
- Compare vs upstream Protenix n=25 (blocked on #7)
- Render the best-DockQ predicted structure alongside ground truth
- Save the distogram for the best sample and overlay the true epitope

## Setup

```python
from helico.experiment import ensure_bench_run, experiment_dir, set_experiment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

set_experiment("exp8_ab_ag_triage")

DATA = experiment_dir() / "data"
PLOTS = experiment_dir() / "plots"
DATA.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

pd.set_option("display.float_format", "{:.3f}".format)

TARGETS = ["8t59-assembly1", "8q3j-assembly1", "8v52-assembly1"]
REPO_ROOT = experiment_dir().parent.parent
```

## exp4 numbers for these 3 targets (n=5)

Baseline from the exp4 bench, pulled from its committed results.

```python
exp4_per_target = pd.read_csv(
    REPO_ROOT / "experiments/exp4_baseline_protenix_v1/.cache"
    / "benches/protenix-v1-default/results/interface_antibody_antigen.csv"
)
exp4_rows = exp4_per_target[exp4_per_target["pdb_id"].isin(TARGETS)].copy()
exp4_rows = exp4_rows.drop_duplicates(subset=["pdb_id"])
exp4_rows.to_csv(DATA / "exp4_baseline_n5.csv", index=False)
exp4_rows[["pdb_id", "status", "lddt", "dockq", "irmsd", "lrmsd", "fnat"]]
```

## Run Helico at n=25 on all 74 ab-ag targets

We run the whole `interface_antibody_antigen` category rather than just the
three targets, because cost is comparable (~$15 either way given fixed
image setup) and we get the full n=5 → n=25 lift curve on a meaningful
sample, which generalizes the conclusion beyond the triage targets.

Published upstream method: 5 seeds × 5 samples = 25 predictions per target.
`modal/bench.py` currently calls the model once with `n_samples=25` (single
seed, 25 diffusion samples). Not strictly identical but the dominant source
of sample-count variance is the diffusion noise schedule, which both
approaches cover.

```python
bench25 = ensure_bench_run(
    "ab-ag-n25",
    checkpoint="protenix-v1",
    categories="interface_antibody_antigen",
    workers=8,
    gpu="H100",
    n_samples=25,
    n_cycles=10,
    max_tokens=2048,
    cutoff_date="2024-01-01",
    est_wall_hours=1.0,
)
print(f"cached: {bench25.cached} | run_name: {bench25.volume_path}")
```

## Compare n=5 vs n=25 (our 3 triage targets)

```python
n25_per_target = bench25.per_category("interface_antibody_antigen")
n25_rows = n25_per_target[n25_per_target["pdb_id"].isin(TARGETS)].copy()
n25_rows = n25_rows.drop_duplicates(subset=["pdb_id"])

triage_compare = exp4_rows[["pdb_id", "lddt", "dockq"]].rename(
    columns={"lddt": "lddt_n5", "dockq": "dockq_n5"}
).merge(
    n25_rows[["pdb_id", "lddt", "dockq"]].rename(
        columns={"lddt": "lddt_n25", "dockq": "dockq_n25"}
    ),
    on="pdb_id",
)
triage_compare["dockq_delta"] = triage_compare["dockq_n25"] - triage_compare["dockq_n5"]
triage_compare["lddt_delta"] = triage_compare["lddt_n25"] - triage_compare["lddt_n5"]
triage_compare.to_csv(DATA / "triage_n5_vs_n25.csv", index=False)
triage_compare
```

## Category-wide n=5 vs n=25 lift

Does sampling help on average across all 74 ab-ag targets?

```python
all_compare = exp4_per_target[["pdb_id", "status", "lddt", "dockq"]].rename(
    columns={"lddt": "lddt_n5", "dockq": "dockq_n5", "status": "status_n5"}
).merge(
    n25_per_target[["pdb_id", "status", "lddt", "dockq"]].rename(
        columns={"lddt": "lddt_n25", "dockq": "dockq_n25", "status": "status_n25"}
    ),
    on="pdb_id",
)
all_compare = all_compare.drop_duplicates(subset=["pdb_id"])
all_compare.to_csv(DATA / "all_n5_vs_n25.csv", index=False)

valid = all_compare[(all_compare["status_n5"] == "ok") & (all_compare["status_n25"] == "ok")]
print(f"Valid comparisons: {len(valid)} / {len(all_compare)}")
print()
print(f"Mean LDDT:  n=5 {valid['lddt_n5'].mean():.3f}  →  n=25 {valid['lddt_n25'].mean():.3f}  "
      f"(Δ {valid['lddt_n25'].mean()-valid['lddt_n5'].mean():+.3f})")
dq5 = valid["dockq_n5"].dropna()
dq25 = valid["dockq_n25"].dropna()
if len(dq5) and len(dq25):
    print(f"Mean DockQ: n=5 {dq5.mean():.3f}  →  n=25 {dq25.mean():.3f}  "
          f"(Δ {dq25.mean()-dq5.mean():+.3f})")
n5_success = (valid["dockq_n5"].fillna(0) >= 0.23).sum()
n25_success = (valid["dockq_n25"].fillna(0) >= 0.23).sum()
print(f"Success rate (DockQ≥0.23): n=5 {100*n5_success/len(valid):.1f}%  →  n=25 {100*n25_success/len(valid):.1f}%")
```

```python
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(valid["dockq_n5"], valid["dockq_n25"], alpha=0.6)
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
ax.axhline(0.23, color="g", alpha=0.3, linestyle=":", label="success threshold")
ax.axvline(0.23, color="g", alpha=0.3, linestyle=":")
ax.set_xlabel("DockQ at n=5 (exp4)")
ax.set_ylabel("DockQ at n=25 (this experiment)")
ax.set_title("interface_antibody_antigen — n=5 vs n=25 sampling")
ax.legend(loc="upper left")
ax.set_xlim(-0.02, 1); ax.set_ylim(-0.02, 1)
for _, row in valid[valid["pdb_id"].isin(TARGETS)].iterrows():
    ax.annotate(row["pdb_id"].split("-")[0], (row["dockq_n5"], row["dockq_n25"]),
                fontsize=8, alpha=0.8)
fig.tight_layout()
fig.savefig(PLOTS / "dockq_n5_vs_n25.png", dpi=150)
plt.show()
```

## Render best structures

For each triage target, extract the best-DockQ sample's predicted PDB string
from the cached predictions pickle and save it alongside the ground truth.

```python
import pickle

STRUCTURES = DATA / "structures"
STRUCTURES.mkdir(exist_ok=True)

def extract_best_prediction(bench_name: str, pdb_id: str) -> dict | None:
    """Load the cached prediction and return the `pdb_str`. We currently
    don't per-sample-rank in the bench, so this returns whichever sample
    the ranker picked (sample 0 after ranking)."""
    cache = REPO_ROOT / "experiments/exp4_baseline_protenix_v1/.cache"
    if bench_name == "ab-ag-n25":
        cache = experiment_dir() / ".cache"
    path = cache / "benches" / (
        "protenix-v1-default" if bench_name == "exp4" else bench_name
    ) / "predictions" / f"{pdb_id}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

for pdb_id in TARGETS:
    for run_name in ("exp4", "ab-ag-n25"):
        pred = extract_best_prediction(run_name, pdb_id)
        if pred is None:
            print(f"[missing] {pdb_id} / {run_name}")
            continue
        out = STRUCTURES / f"{pdb_id}_helico_{run_name}.pdb"
        out.write_text(pred["pdb_str"])
        print(f"wrote {out.relative_to(REPO_ROOT)}  ({out.stat().st_size:,} bytes)")
```

Simple Cα-trace visualization so the HTML has something showing the
predicted shape. For real structural review, open the PDB files in PyMOL.

```python
from Bio.PDB import PDBParser
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def ca_trace(pdb_path, ax, label):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    chains = {}
    for model in structure:
        for chain in model:
            pts = [r["CA"].get_coord() for r in chain if r.has_id("CA")]
            if pts:
                chains[chain.id] = np.array(pts)
    for cid, pts in chains.items():
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=1.2, label=f"chain {cid}")
    ax.set_title(label, fontsize=9)
    ax.set_axis_off()

fig = plt.figure(figsize=(12, 8))
for i, pdb_id in enumerate(TARGETS):
    for j, run_name in enumerate(("exp4", "ab-ag-n25")):
        ax = fig.add_subplot(len(TARGETS), 2, i * 2 + j + 1, projection="3d")
        pdb_path = STRUCTURES / f"{pdb_id}_helico_{run_name}.pdb"
        if not pdb_path.exists():
            ax.text(0.5, 0.5, 0.5, "missing", ha="center")
            continue
        label = f"{pdb_id.split('-')[0]}  |  Helico {run_name}"
        ca_trace(pdb_path, ax, label)
        if i == 0 and j == 0:
            ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(PLOTS / "structures_ca_trace.png", dpi=150)
plt.show()
```

## Distogram outputs — TODO

Saving distograms requires a small change to `Predictor.predict` in
`modal/bench.py`: the distogram head is on the model but isn't returned
from inference. Once exposed, we save the `(N_token, N_token, 64)` bin
probabilities for the best sample and render heatmaps with the known
antibody-antigen epitope (from ground truth) overlaid.

This cell is a placeholder — filled in once the model surgery lands.

```python
# TODO: after exposing distograms from Predictor.predict:
# - Save to data/distograms/{pdb_id}_helico_{run_name}.npy
# - For each, compute argmax distance per (i, j) pair
# - Overlay known antibody/antigen chain boundary + true interface residues
# - Heatmap per (target, model) with epitope region boxed
```

## Upstream Protenix comparison — TODO

Blocked on #7 (upstream Protenix runner on Modal). Once landed:
- Run upstream Protenix at n=25 on the same 3 targets
- Add side-by-side columns to triage_compare above
- Same structure render, same distogram saves

## Conclusion

(Filled in once the n=25 bench finishes and the TODOs land.)
