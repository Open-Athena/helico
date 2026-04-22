# experiments/

One directory per experiment, keyed by GitHub issue number. Each directory's
`README.md` is a **jupytext-markdown notebook**: prose, code, and analysis in
one file that GitHub renders natively and `nbconvert` executes.

Canonical design: `.agents/project/20260422_experiment_system_design.md`.
Agent rules: `AGENTS.md` in this directory.

## Filing an experiment

1. Open an issue using the **Experiment** template.
2. Create a branch from `main`: `exp/<issue-number>-<slug>`.
3. Copy `experiments/TEMPLATE.md` to `experiments/exp<N>_<slug>/README.md`.
4. Fill in the frontmatter (issue, title, branch) and the body.
5. Run the notebook locally to iterate.
6. Push. (In Wave 2+, the site will auto-rebuild on merge to main.)

## Running a notebook

```bash
# Dry run: prints per-step cost estimates + total. No Modal dispatch.
HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py \
    experiments/exp1_protenix_baseline/

# Real run: executes the notebook, dispatching Modal where needed.
# Cached steps return instantly; new/forced steps block until done.
uv run python scripts/pm/run_experiment.py experiments/exp1_protenix_baseline/

# Interactive (open in Jupyter / VS Code):
uv run jupytext --sync experiments/exp1_protenix_baseline/README.md
# Edit the paired .ipynb; jupytext syncs changes back to README.md on save.
```

The runner converts `README.md` → `.ipynb` → `README.html`. The HTML is
what the website publishes (Wave 2+).

## What's cached

`ensure_bench_run(name=...)` and `ensure_training_run(name=...)` cache by
name alone. Repeat calls with the same name return instantly without
re-dispatching Modal. To force a rerun:

- `force=True` on a single call (use when debugging one step), or
- bump the name (`"baseline"` → `"baseline-v2"`) for a persistent re-run.

Cache locations (in order of precedence):
- Local `experiments/<slug>/.cache/` — gitignored, fast re-reads.
- Modal volumes (`helico-checkpoints`, `helico-experiments`) — authoritative.

Code or data changes do **not** invalidate cache. If the meaning of a run
has changed, rename the step or `force=True`.

## Directory layout per experiment

```
experiments/exp1_protenix_baseline/
  README.md          # jupytext notebook; the source of truth
  data/              # small CSVs committed; site links here
  plots/             # PNGs committed; also embedded in README.html
  .cache/            # gitignored; downloaded artifacts, paired .ipynb
  README.html        # gitignored; built by CI and published to site
```
