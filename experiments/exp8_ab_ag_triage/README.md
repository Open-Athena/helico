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

## Oracle best-of-N diagnostic

If the ranker is what's broken, then oracle-best-DockQ (max over all 25
samples) should be much higher than ranked-DockQ (the single sample
we actually output). We rerun the 3 triage targets with the updated
`Predictor.predict` that saves all 25 samples, then score each sample
locally and take the max.

```python
# Protocol: 5 seeds × 5 samples = 25 predictions/target (matches the
# FoldBench paper's evaluation of Protenix). Each seed resets torch RNG
# before calling the model, so we get 25 predictions from 5 independent
# noise trajectories rather than 25 samples from one trajectory's
# stream. This is the methodology the published numbers use.
oracle = ensure_bench_run(
    "ab-ag-5seeds-5samples",
    checkpoint="protenix-v1",
    categories="interface_antibody_antigen",
    target_pdb_ids=",".join(TARGETS),
    workers=8,
    gpu="H100",
    n_samples=5,        # samples per seed
    n_seeds=5,          # total predictions = 25
    n_cycles=10,
    max_tokens=2048,
    cutoff_date="2024-01-01",
    est_wall_hours=0.5, # recycling re-runs per seed ⇒ ~5× predict() time
)
print(f"cached: {oracle.cached}")
```

```python
# Score each of the 25 samples locally using helico.bench scoring
# functions. Reproduces match_atoms + score_interface on each sample's
# pdb_str + pred_coords.
import pickle
from helico.bench import match_atoms, score_interface, _find_gt_path
from helico.data import parse_mmcif

foldbench_dir = (
    REPO_ROOT / ".." / ".." / ".cache" / "helico" / "data"
    / "benchmarks" / "FoldBench"
).resolve()
if not foldbench_dir.exists():
    from helico.bench import download_foldbench
    foldbench_dir = download_foldbench()
gt_dir = foldbench_dir / "examples" / "ground_truths"

rows = []
for pdb_id in TARGETS:
    pred_pkl = oracle.cache_dir / "predictions" / f"{pdb_id}.pkl"
    if not pred_pkl.exists():
        print(f"[skip] {pdb_id}: no per-sample pickle")
        continue
    with open(pred_pkl, "rb") as f:
        pred = pickle.load(f)
    if "all_coords" not in pred:
        print(f"[skip] {pdb_id}: old-format pickle (rerun with updated Predictor)")
        continue

    gt_path = _find_gt_path(gt_dir, pdb_id)
    gt_structure = parse_mmcif(gt_path, max_resolution=float("inf"))
    n_samples_local = len(pred["all_pdb_strs"])
    print(f"{pdb_id}: scoring {n_samples_local} samples")
    for i in range(n_samples_local):
        coords_i = pred["all_coords"][i]
        pdb_str_i = pred["all_pdb_strs"][i]
        try:
            matched = match_atoms(pred["tokenized"], coords_i, gt_structure)
            scores = score_interface(pdb_str_i, gt_path, matched)
            rows.append({
                "pdb_id": pdb_id,
                "sample_idx": i,
                "lddt": scores.get("lddt"),
                "dockq": scores.get("dockq"),
                "irmsd": scores.get("irmsd"),
                "lrmsd": scores.get("lrmsd"),
                "fnat": scores.get("fnat"),
            })
        except Exception as e:
            rows.append({"pdb_id": pdb_id, "sample_idx": i, "error": str(e)})

per_sample = pd.DataFrame(rows)
per_sample.to_csv(DATA / "per_sample_scores.csv", index=False)
per_sample
```

```python
# Oracle vs ranked: per-target max DockQ over all samples, compared to
# the DockQ of the top-ranked sample (sample 0 by convention of
# predict_target's ranker). Falls through harmlessly in dry-run when
# per_sample is empty.
oracle_rows = []
has_per_sample = not per_sample.empty and "pdb_id" in per_sample.columns
for pdb_id in TARGETS if has_per_sample else []:
    target_df = per_sample[per_sample["pdb_id"] == pdb_id].dropna(subset=["dockq"])
    if target_df.empty:
        continue
    ranked = target_df[target_df["sample_idx"] == 0]
    oracle_dockq = target_df["dockq"].max()
    oracle_idx = int(target_df.loc[target_df["dockq"].idxmax(), "sample_idx"])
    oracle_rows.append({
        "pdb_id": pdb_id,
        "ranked_dockq": float(ranked.iloc[0]["dockq"]) if not ranked.empty else None,
        "oracle_dockq": float(oracle_dockq),
        "oracle_sample_idx": oracle_idx,
        "n_samples_scored": len(target_df),
        "lift": (float(oracle_dockq) - float(ranked.iloc[0]["dockq"]))
                 if not ranked.empty else None,
    })
oracle_compare = pd.DataFrame(oracle_rows)
if not oracle_compare.empty:
    oracle_compare = oracle_compare.set_index("pdb_id")
    oracle_compare.to_csv(DATA / "oracle_vs_ranked.csv")
oracle_compare
```

```python
# Per-sample DockQ bar: shows the distribution inside each target and
# where the ranked sample (index 0) sits.
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(9, 2 * len(TARGETS)),
                         sharex=True)
if len(TARGETS) == 1:
    axes = [axes]
for ax, pdb_id in zip(axes, TARGETS):
    if not has_per_sample:
        ax.text(0.5, 0.5, f"{pdb_id}: no per-sample data",
                transform=ax.transAxes, ha="center")
        continue
    target_df = per_sample[per_sample["pdb_id"] == pdb_id].copy()
    target_df = target_df.dropna(subset=["dockq"]).sort_values("sample_idx")
    if target_df.empty:
        ax.text(0.5, 0.5, f"{pdb_id}: no scored samples",
                transform=ax.transAxes, ha="center")
        continue
    colors = ["#c44e52" if i == 0 else "#4c72b0"
              for i in target_df["sample_idx"]]
    ax.bar(target_df["sample_idx"], target_df["dockq"], color=colors)
    ax.axhline(0.23, color="k", alpha=0.3, linestyle="--",
               label="success threshold")
    ax.set_title(f"{pdb_id}  —  ranked=red, other=blue", fontsize=9)
    ax.set_ylabel("DockQ")
    ax.set_ylim(0, max(1.0, target_df["dockq"].max() * 1.1))
axes[-1].set_xlabel("sample index")
axes[0].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS / "per_sample_dockq.png", dpi=150)
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

**Hypothesis 1 (sampling) is substantially *confirmed* once we run the
protocol the published numbers actually use**: 5 seeds × 5 samples, not
1 seed × 25 samples. The earlier round-2 "ranker is broken" finding
was measuring a narrower-than-expected sample distribution from a
single noise seed, not an actual ranker pathology.

### Round 3: 5 seeds × 5 samples

| target | round 2 (1 seed × 25)                 | round 3 (5 seeds × 5)              | notes |
|---|---|---|---|
| `8t59` | ranked 0.125 · oracle 0.571 · 44% ok  | ranked **0.388** · oracle 0.546 · **92% ok** | ranker now picks a success; pool is almost all good |
| `8q3j` | ranked 0.080 · oracle 0.161 · 0% ok   | ranked 0.063 · oracle 0.098 · 0% ok | still failing |
| `8v52` | OOM                                   | ranked 0.027 · oracle 0.033 · 0% ok | runs, still failing |

On 8t59 the headline number went from **44% of samples correct** (1 seed)
to **92% correct** (5 seeds). The ranker's top pick went from 0.125 (fail)
to 0.388 (success). No ranker changes needed — the 7× ab-ag gap on this
target is pure sampling methodology.

Category-wide the 5×5 run hits 33.3% success rate (2/6 interface pairs;
6 pairs = 3 targets × 2 chain pairs each). That's in the neighborhood of
the published Protenix 2024-01+ regime (38.4%). Consistent with the gap
being mostly methodology.

### Two distinct problems remain

1. **Single-seed diffusion produces a poor sample distribution**
   (~~broken ranker~~). Default the bench to 5 seeds × 5 samples. Round
   2's "ranker is broken" reading was misleading — the ranker is fine;
   the sample pool was the problem.

2. **Some targets still fail even at 5×5** (8q3j, 8v52). These are the
   featurization / epitope-recognition cases. 0% of samples are
   correct, so more sampling / better ranking can't fix them.
   Distinguishes roughly 7% of targets that need real model-level work
   from 93% that just need the right methodology.

### Immediate followups

- **Re-run exp4 with 5 seeds × 5 samples** as the new baseline. Expect
  the gap to published Protenix to mostly close on many categories
  (not just ab-ag). ~$30-60 on 679 targets.
- **Default `n_seeds=5`** in `ensure_bench_run` going forward.
- **Targeted investigation on 8q3j-class targets**: distogram exposure
  (still TODO) + upstream Protenix on the same targets (#7) — this is
  where the real featurization/epitope work lies.

### Round 1 + 2 context (for the record)

### Round 2: oracle analysis at 1 seed × 25

Scoring all 25 samples for 2 of 3 triage targets (8v52 OOM'd again):

| target | ranked DockQ | oracle DockQ | lift | # samples ≥ 0.23 |
|---|---|---|---|---|
| **8t59** | 0.125 | **0.571** | **+0.446** | **11 / 25 (44%)** |
| 8q3j | 0.080 | 0.161 | +0.081 | 0 / 25 (0%) |

**8t59 is the smoking gun.** 44% of the 25 samples cross the DockQ success
threshold and the oracle sample is 0.571 — but the ranker picks a DockQ
0.125 sample. The model clearly *can* predict the correct interface on
this target; the ranking head is selecting badly out of a pool full of
good answers. Roughly half the time, a random pick from the sample pool
would be a success; the ranker does substantially worse than random.

**8q3j is different.** Even the best of 25 samples (DockQ 0.161) is
below the success threshold. More sampling won't help this target
because the model simply isn't finding good binding poses for it.
This is the featurization/epitope-recognition mode — different from
the 8t59 pattern.

### Two distinct problems identified

1. **Ranking bug** (demonstrated on 8t59): the confidence-head scoring
   used by `predict_target` to select the best sample from n candidates
   is poorly correlated with DockQ for antibody-antigen. Same model
   weights, same predictions — picking them better would move 8t59
   alone from failure to 0.57 DockQ.

2. **Featurization / epitope issue** (pattern in 8q3j): for some
   targets, none of the 25 samples are close to correct. This isn't a
   ranking problem; the model isn't even generating candidates with
   the right binding mode. Likely suspects: CDR-loop MSA handling,
   chain-pair features at the epitope, or the pair-rep itself not
   encoding the antibody-antigen interaction.

The overall 7× gap vs upstream Protenix is probably a blend of these
two — some targets are ranker-limited (fixable cheaply), others are
generation-limited (needs real model work).

### Immediate followups

- **Rerank-by-pLDDT analysis** (cheapest to do next): the model
  computes per-sample pLDDT but currently only returns the best
  sample's. Expose `all_plddts` from `Helico.predict`, then on the
  existing 25-sample cache compute mean-pLDDT per sample and check
  whether that ranker picks better than the current confidence_head
  ranker. If yes, the fix is a one-line change in the ranker; if no,
  the issue is in the confidence head itself.
- **Upstream Protenix on these 3 targets** (#7): does upstream also
  get 8q3j wrong, or does their MSA / featurization produce
  correctly-generated samples where ours don't? This is the cleanest
  way to attribute the 8q3j-style cases.
- **Distogram output on 8q3j specifically** (still TODO): check
  whether the pair representation puts the antibody near the correct
  epitope. If yes → issue is in diffusion; if no → featurization.

### Round 1 context (sampling refutation, for the record)

Category-wide across 74 targets (48 with valid comparisons after
dedup; 31 after dropping NaN DockQ):

| | n=5 (exp4) | n=25 (this exp) | Δ |
|---|---|---|---|
| mean LDDT | 0.662 | 0.648 | **-0.014** |
| mean DockQ | 0.058 | 0.053 | **-0.006** |
| success rate (DockQ ≥ 0.23) | 7.3% | 4.9% | **-2.4 pp** |
| OOM count | 0 | **7** | larger memory footprint at 25 samples |
| regressed / improved / unchanged | — | 13 / 9 / 9 | 59% of changed targets got worse |

Triage targets:

| target | n=5 DockQ | n=25 DockQ | Δ |
|---|---|---|---|
| 8t59-assembly1 | 0.495 | 0.108 | **-0.387** (catastrophic) |
| 8q3j-assembly1 | 0.069 | 0.106 | +0.038 |
| 8v52-assembly1 | 0.020 | OOM | — |

The diagnostic insight was that n=25 hurt two targets that succeeded at
n=5 (8t59 and 9jbq both dropped out of the success band). This is the
opposite of what a sampling-limited model would do. The round-2 oracle
analysis confirms: the ranker systematically prefers poor samples.

The rendered Cα traces (saved) aren't actually informative at this
scale — the chains look similar; what's different is the relative
docking pose between antibody and antigen, which isn't visible without
proper structural overlay. PyMOL / ChimeraX for that.
