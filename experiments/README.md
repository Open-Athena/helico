# experiments/

One directory per experiment, keyed by GitHub issue number. Each directory's
`README.md` is a **jupytext-markdown notebook**: prose, code, and analysis in
one file that GitHub renders natively and `nbconvert` executes.

Canonical design: `.agents/project/20260422_experiment_system_design.md`.
Agent playbook rules: `AGENTS.md` in this directory.

## Current flow (manual, Claude-in-the-loop)

1. **File an issue** using the **Experiment** template. Label it
   `experiment`. Describe the question, hypothesis, approach, and
   compute estimate.

2. **Scaffold the notebook** (from a local Claude session or by hand):
   ```bash
   uv run python scripts/pm/scaffold_experiment.py --issue <N>
   ```
   Creates `experiments/exp<N>_<slug>/README.md` from
   `experiments/TEMPLATE.md`, pre-filling the frontmatter and copying
   the issue's Question/Hypothesis/Background prose into the body.
   Use `--branch exp/<N>-<slug>` if the experiment needs speculative
   code changes (default is `main`).

3. **Edit the notebook**: fill in the `ensure_bench_run` /
   `ensure_training_run` calls, success criteria, baselines, and
   analysis cells.

4. **Cost check**:
   ```bash
   HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py \
       experiments/exp<N>_<slug>/
   ```
   Sum the `~$X` estimates from each `ensure_*` call and compare with
   `.github/experiments.yaml` → `cost_gate_usd` before launching.

5. **Run**:
   ```bash
   uv run python scripts/pm/run_experiment.py experiments/exp<N>_<slug>/
   ```
   This blocks until Modal completes, then builds `README.html`. Cached
   (already-run) steps return instantly; new or `force=True` steps
   dispatch to Modal.

6. **Publish**:
   ```bash
   uv run helico-publish bench --experiment exp<N>_<slug> --name <step-name>
   ```
   Uploads bench artifacts + a generated model card to
   `buckets/timodonnell/helico-experiments/`. Set `publish=True` on
   `ensure_bench_run` to auto-publish at notebook runtime instead.

7. **Report results**: post a summary comment on the issue via
   `gh issue comment <N> --body ...` with headline numbers, HF link,
   and (when you close the issue) a short conclusion.

## On main vs. a branch

An experiment can live directly on `main` (baselines, characterizations,
bug analyses — anything whose results belong in the permanent record
regardless of outcome) or on a branch `exp/<N>-<slug>` (when it requires
speculative model or data-pipeline changes that may not ship). Set
`branch:` in the notebook frontmatter accordingly.

## What's cached

`ensure_bench_run(name=...)` and `ensure_training_run(name=...)` cache
by name alone. Repeat calls with the same name return instantly without
re-dispatching Modal. To force a rerun:

- `force=True` on a single call (use when debugging one step), or
- bump the name (`"baseline"` → `"baseline-v2"`) for a persistent re-run.

Cache locations (in order of precedence):
- Local `experiments/<slug>/.cache/` — gitignored, fast re-reads
- Modal volumes (`helico-checkpoints`, `helico-experiments`) — authoritative

Code or data changes do **not** invalidate cache. If the meaning of a
run has changed, rename the step or pass `force=True`.

## Directory layout per experiment

```
experiments/exp<N>_<slug>/
  README.md          # jupytext notebook; the source of truth
  data/              # small CSVs committed; site links here
  plots/             # PNGs committed; also embedded in README.html
  .cache/            # gitignored; downloaded artifacts, paired .ipynb
  README.html        # gitignored; built by the runner and served via mkdocs
```
