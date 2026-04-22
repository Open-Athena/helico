# Experiments

Every research question we run is documented as a GitHub issue tagged
`experiment`. The experiment's notebook lives at
`experiments/exp<N>_<slug>/README.md` in the repo and is a self-contained
record: prose, Modal invocations (training, benchmarks), analysis, plots.
Raw result CSVs are committed alongside the notebook under `data/` so
every plot on this site is re-plottable without rerunning anything.

!!! note
    This index is currently maintained manually. A `scripts/pm/itemize_experiments.py`
    generator will populate it automatically from
    `gh issue list --label experiment` in Wave 2 of the experiment system rollout.

## How to read an experiment page

1. **Question** and **Hypothesis** are the preregistration.
2. **Setup** / **Run** cells show exactly what was dispatched to Modal.
3. **Summary**, **LDDT by category**, **DockQ**, **Success rate** are the headline numbers.
4. **Conclusion** answers the Question for the reader in hindsight.

Every plot links to its source CSV so you can re-plot without rerunning
Modal.

## Open experiments

<!-- Generator will fill this; for now maintain by hand. -->

- [#4 · Baseline performance on Protenix v1 checkpoint](https://github.com/Open-Athena/helico/blob/main/experiments/exp4_baseline_protenix_v1/README.md) — branch `main`

## Closed experiments

_(None yet.)_
