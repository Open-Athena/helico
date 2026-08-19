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
  issue: 14
  title: "exp: Helico on MarinFold exp245's held-out FoldBench monomer sets (eval-val / eval-test / eval-denovo)"
  branch: main
  baselines: []
---

# exp: Helico on MarinFold exp245's held-out FoldBench monomer sets (eval-val / eval-test / eval-denovo)

**Issue:** [#14](https://github.com/Open-Athena/helico/issues/14) · **Branch:** `main`

## Question

How accurately does Helico fold MarinFold [exp245](https://github.com/Open-Athena/MarinFold/tree/main/experiments/exp245_evals_foldbench_held_out_monomers)'s three held-out FoldBench monomer sets — **eval-val (97)**, **eval-test (217)**, **eval-denovo (19)** — when conditioned on contacts from the best *decontaminated* MarinFold checkpoint, versus no contacts, oracle contacts, and contacts read off Protenix-v2 structures?

## Hypothesis

- **H1.** The published contact-conditioning ordering survives on a 9x larger, held-out set: `oracle` > `mf_L` > `protenix_v2_singleseq` > `off`, and `oracle` ≈ `protenix_v2_msa`.
- **H2.** eval-val and eval-test agree within noise for every arm. exp245 found this for contact R-precision (every predictor within 0.03, difference-in-differences −0.006 for the contaminated model); folding lDDT should behave the same way, because 0/333 units fall inside Helico's training window.
- **H3.** The gain from MarinFold contacts is smaller here than the published +0.119 over Protenix-v2 single sequence. Two reasons push the same way: the contacts now come from a decontaminated checkpoint (0.510 precision@L on these 333 vs the contaminated cooldown's numbers), and the set is not homology-filtered, so the baselines are stronger.

## Background

- Helico's published contact-conditioning result ([`RESULTS_contact_conditioning.md`](https://github.com/Open-Athena/helico/blob/main/RESULTS_contact_conditioning.md), #10, #13) is **38 homology-filtered FoldBench monomers**, conditioned on the *contaminated* exp199 cooldown checkpoint. Its headline interval is ±0.03 on the key delta, and the contact source trained on a corpus never filtered against FoldBench.
- MarinFold #225 published corpora decontaminated against all of FoldBench; #232 trained on them; exp245 scored the two best checkpoints on all 334 monomers, and found the historical FoldBench-100 was **not** flattering anyone (every predictor scores the same or slightly better on the 217 never-touched proteins).
- exp245 publishes its eval sets, ground truth, and per-protein contact metrics openly at `hf://buckets/open-athena/MarinFold/data/contacts-v1-foldbench-monomers-exp245/`.

Two facts verified while scoping this:

- **0 of the 333 scored units were released before 2021-09-30**, Helico's training cutoff. Unlike the 38-target headline set, no homology filter is needed for Helico's own contamination.
- All 334 ground truths are already in Helico's FoldBench cache, each parsing to exactly one protein chain; 317/333 already have a precomputed a3m under Helico's own sequence hash.

## Setup

The eval sets, their ground truths, the index map and the contact arms are all
built by scripts in this directory rather than in the notebook: they are slow,
they have their own controls, and rebuilding them is not part of reading the
result. Each prints what it checked. Run them in this order:

```
uv run python build_eval_sets.py           # targets.csv + gt/, decontamination controls
uv run python build_index_map.py           # token_map.json + the oracle-agreement control
uv run python export_marinfold_contacts.py # arms/mf_{L,L2,L5}.json  (needs CW_KEY_ID/SECRET)
uv run python gen_missing_msas.py          # the 16 alignments FoldBench does not ship
uv run python run_protenix_v2.py           # both Protenix-v2 modes on all 333 units
uv run python export_v2_contacts.py        # arms/v2ss.json, arms/v2msa.json + baseline lDDT
```

```python
import json

import matplotlib.pyplot as plt
import pandas as pd

from helico.experiment import ensure_byclass_run, experiment_dir, set_experiment

set_experiment("exp14_foldbench_held_out_monomers")

PLOTS = experiment_dir() / "plots"
DATA = experiment_dir() / "data"
PLOTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

CHECKPOINT = "/ckpts/contacts-msafree-01/final.pt"
```

## The target set, and why it needs no homology filter

exp245's 333 scored units, joined to FoldBench's own ground truths. The
published contact-conditioning numbers are homology-filtered twice over
(MarinFold's `eval2` at 40% identity, then Helico's release-date window) and
that is what cut them to 38 targets. Here only the second filter applies, and it
removes nothing: every unit was released after 2021-09-30.

```python
targets = pd.read_csv(DATA / "targets.csv")
report = json.loads((DATA / "eval_set_report.json").read_text())
report
```

## The index map, and the control on it

MarinFold indexes contacts into its published prompt sequence; Helico indexes
the resolved residues of the ground truth. The two agree outright on 52 of 333
targets. `build_index_map.py` re-seats one onto the other and then checks the
result the only way that can catch an error: exp245's own ground-truth contacts,
pushed through the map, against Helico's `oracle_contact_state` on the same
structure. Both sides are pyconfind at the same thresholds, so a correct map
agrees essentially perfectly.

```python
index_map = pd.read_csv(DATA / "index_map_report.csv")
index_map.groupby("rule").jaccard.describe()[["count", "50%", "min"]]
```

## The contact arms

`mf_L` is exactly the contact list whose precision exp245 published: the same
dense score matrix, the same resolved-pair candidates, the same stable
tie-break. `export_marinfold_contacts.py` refuses to write the arm unless it
reproduces `contact_precision_all.csv` to floating point on every target, so the
arm's contact quality is the published number by construction.

```python
mf = pd.read_csv(DATA / "marinfold_arm_accuracy.csv")
v2 = pd.read_csv(DATA / "v2_arm_accuracy.csv")
pd.concat([
    mf.groupby("eval_set")[["precision_L", "precision_L2", "precision_L5"]].mean(),
    v2.pivot_table(index="eval_set", columns="arm", values="precision"),
], axis=1).round(3)
```

## Run the arms

Seven arms over the same 333 targets, MSA-free throughout, one
`ensure_byclass_run` per arm so each caches and re-runs independently. The arm
is an explicit argument rather than ambient environment state — a bench that
silently runs with no contacts looks exactly like "predicted contacts do not
help", and this repo has shipped that mistake twice.

Cost: 7 arms x 333 targets on H100 workers, about $120 total, plus the
Protenix-v2 predictions run outside the notebook (~$220). Check with
`HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py` before launching.

```python
ARMS = [
    ("off", {}),
    ("oracle", {"oracle_contacts": True}),
    ("mf_L", {"contacts_arm": "mf_L"}),
    ("mf_L2", {"contacts_arm": "mf_L2"}),
    ("mf_L5", {"contacts_arm": "mf_L5"}),
    ("v2ss", {"contacts_arm": "v2ss"}),
    ("v2msa", {"contacts_arm": "v2msa"}),
]

runs = {
    name: ensure_byclass_run(
        name,
        targets_dir=DATA,
        checkpoint=CHECKPOINT,
        workers=8,
        gpu="H100",
        n_samples=3,
        n_cycles=6,
        est_wall_hours=1.0,
        **kwargs,
    )
    for name, kwargs in ARMS
}
{name: (run.cached, run.arm) for name, run in runs.items()}
```

## Analyze

`analyze.py` writes every table this section reads. One bootstrap resample of
the proteins is shared across all arms, so the deltas below are paired.

```python
!uv run python analyze.py
```

```python
headline = pd.read_csv(DATA / "headline.csv")
headline[headline.eval_set.isin(["eval-val", "eval-test", "eval-denovo"])].pivot(
    index="label", columns="eval_set", values="mean_lddt").round(3)
```

```python
deltas = pd.read_csv(DATA / "paired_deltas.csv")
deltas[deltas.eval_set == "eval-test"].round(3)
```

```python
val_vs_test = pd.read_csv(DATA / "val_vs_test.csv")
val_vs_test.round(3)
```

## Results

<!-- Filled in once the runs land: the scoreboard, the paired deltas, the
     val-vs-test answer to H2, and the strata. -->

## Conclusion

<!-- Answer the question. A future reader should get the answer from this
     section alone. -->
