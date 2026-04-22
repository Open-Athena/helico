---
name: run-experiment
description: Execute an experiment's notebook end-to-end and report results back to its GitHub issue. Use when the user points you at an experiment issue ("run #7", "run exp4", etc.).
---

# run-experiment

Execute an experiment's notebook on Modal, then post results back to the
issue. Runs in whatever environment Claude is currently in (local laptop,
CI runner, etc.) — no assumption of being in GitHub Actions.

## Prerequisites

- The issue is labeled `experiment` and has an associated
  `experiments/exp<N>_<slug>/README.md` notebook. If not, use
  `scripts/pm/scaffold_experiment.py --issue <N>` to create the skeleton,
  edit the body from the issue text, then run.
- Modal auth is set up on this machine (`modal token new` has been run).
- HF auth: `HF_TOKEN` env var (or `hf auth login`) for `helico-publish`.
- `.github/experiments.yaml` defines the cost gate.

## Workflow

1. **Identify the notebook.** From the issue number, locate
   `experiments/exp<N>_<slug>/`. Read its frontmatter for the branch.
   If the branch in frontmatter is not `main`, `git checkout <branch>`.

2. **Dry run + cost gate.**
   ```bash
   HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py \
       experiments/exp<N>_<slug>/
   ```
   Read the `[helico.experiment] ensure_* — dry-run (~$X)` lines. Sum
   them; compare against `cost_gate_usd` in `.github/experiments.yaml`.

3. **Gate decision.**
   - Under gate → tell the user the estimate and proceed.
   - Over gate → stop and ask the user for explicit approval in chat
     before spending. Don't proceed without a clear "yes".

4. **Run.**
   ```bash
   uv run python scripts/pm/run_experiment.py experiments/exp<N>_<slug>/
   ```
   This blocks until Modal completes. If wall clock exceeds what the
   user is willing to wait, use `run_in_background` on the tool call
   and report when done.

5. **Publish.** After a successful run (unless the notebook already used
   `ensure_bench_run(..., publish=True)`):
   ```bash
   uv run helico-publish bench --experiment exp<N>_<slug> --name <step-name>
   ```

6. **Commit committed artifacts** (CSVs under `data/`, PNGs under `plots/`)
   to the experiment's branch. Open a PR back to `main` if the branch is
   not `main`.

7. **Post results to the issue.** Use `gh issue comment <N>` with a
   headline-numbers + links body. Prefix with 🤖 so the convention is
   consistent. Example:
   ```
   🤖 Bench complete (~$actual_cost).

   Headline:
   - monomer_protein.mean_lddt = 0.734
   - interface_protein_protein.mean_lddt = 0.487
   (full table in the notebook)

   HF: <bucket url>
   Notebook: experiments/exp<N>_<slug>/README.md
   ```

## Rules

- **Never bypass the cost gate** by splitting a run into smaller named
  steps. The gate exists to catch runaway spend; route around it only
  with explicit user approval.
- **Never destroy prior artifacts**. If `.cache/benches/<name>/` exists,
  the library returns cached results — don't delete to force a rerun.
  If the researcher wants a rerun, either add `force=True` to the
  specific call or bump the step name.
- **One agent comment per run**. If you posted a status while dispatching,
  edit it with `gh issue comment --edit-last <N>`. Don't spam multiple
  updates.
- **Respect the branch**. If the frontmatter says `branch: exp/N-slug`,
  operate on that branch. Don't commit to `main` implicitly.

## Failure modes

- Modal auth missing → surface clearly, ask the user to run `modal token new`.
- `cuDNN Frontend error` during inference → usually a cuequivariance
  version drift. Check pyproject.toml pins and modal/bench.py's image
  spec. See commits 4d23f5e, ec58b03 for history.
- Numpy `ModuleNotFoundError: No module named 'numpy._core.numeric'` →
  numpy version skew between local and Modal; both must be >=2.0. See
  commits 6f8e152, 0d521d3.
- Notebook execution fails mid-way → the `ensure_*` cache still has
  partial artifacts. Inspect `.cache/benches/<name>/` and `meta.json` to
  decide whether to rerun with `force=True` or bump the step name.
