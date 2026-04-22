<!--
Experiment notebook template. Copy this file to
experiments/exp<N>_<slug>/README.md and edit.

Format: jupytext plain markdown. Triple-backtick ```python fences are
executable cells; other content is prose. Outputs are NOT stored in this
file — they live in the paired .ipynb (under .cache/, gitignored) and in
committed plots under plots/.

Run locally:
    HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py .  # dry
    uv run python scripts/pm/run_experiment.py .                     # real
-->

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
  issue: 0
  title: "TEMPLATE: replace me"
  branch: main
  baselines: []
---

# TEMPLATE: replace this line with the experiment title

**Issue:** #0 · **Branch:** `main`

## Question

One sentence — what do we want to learn?

## Hypothesis

What we expect to see, and why.

## Setup

```python
from helico.experiment import (
    ensure_bench_run,
    estimate_cost,
    experiment_dir,
    set_experiment,
)

import matplotlib.pyplot as plt

# Set the experiment slug explicitly — avoids cwd-autodetect surprises.
# set_experiment("exp0_template")

# Absolute paths under the experiment dir — robust to kernel cwd.
PLOTS = experiment_dir() / "plots"
DATA = experiment_dir() / "data"
PLOTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
```

## Run the bench

Prose describing what this step does and why.

```python
bench = ensure_bench_run(
    "example-run",
    checkpoint="protenix-v1",
    workers=8,
    gpu="H100",
)
bench.cached, bench.meta.get("est_cost_usd")
```

## Analyze

```python
summary = bench.summary
summary
```

```python
# Save summary for re-plotting without rerunning Modal
summary.to_csv(DATA / "example_run_summary.csv")
```

```python
fig, ax = plt.subplots(figsize=(8, 4))
summary["mean_lddt"].plot.barh(ax=ax)
ax.set_xlabel("mean LDDT")
ax.set_title("Example run — LDDT by category")
fig.tight_layout()
fig.savefig(PLOTS / "example_run_lddt.png", dpi=150)
plt.show()
```

## Conclusion

Answer the question. Two or three paragraphs. A future reader should be
able to get the answer from this section alone; they can scroll up for
methods.
