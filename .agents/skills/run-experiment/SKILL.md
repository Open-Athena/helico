---
name: run-experiment
description: Dispatch an experiment's notebook end-to-end. Use when the user comments `@claude run` or similar in an experiment-labeled issue.
---

# run-experiment

Run an experiment's notebook on Modal, post results back to the issue.

## Prerequisites

- The issue is labeled `experiment` and has an associated
  `experiments/exp<N>_<slug>/README.md` notebook on the branch in its
  frontmatter (`helico_experiment.branch`). If not, ask the researcher
  to create the notebook first (pointing them at `experiments/TEMPLATE.md`).
- `.github/experiments.yaml` exists and contains `cost_gate_usd`.
- Modal auth: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` env vars available.
- HF auth: `HF_TOKEN` env var for `helico-publish` to work (optional in
  dry-run).

## Workflow

1. **Identify the notebook.** From the issue, derive `experiments/exp<N>_<slug>/`.
   If the branch in frontmatter is not `main`, `git checkout <branch>`.

2. **Post a single 🤖 status comment** using `gh issue comment --edit-last`
   pattern. Initial content:
   ```
   🤖 Starting experiment run. Parsing spec and estimating cost.
   ```

3. **Dry run + cost gate.**
   ```bash
   HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py \
       experiments/exp<N>_<slug>/
   ```
   Read the `[helico.experiment] ensure_*` lines in the output — they print
   per-step cost estimates. Sum them or call
   `helico.experiment.dry_run_total_usd()` from a small script to get the
   total. Compare to `.github/experiments.yaml` → `cost_gate_usd`.

4. **Gate decision.**
   - Under gate: update the status comment with the estimate and launch.
   - Over gate: update the status comment:
     ```
     🤖 Cost estimate $X exceeds gate $Y. Reply `@claude approve` to proceed.
     ```
     Stop. Do not run.

5. **Approval path.** If the user replies `@claude approve`, set
   `HELICO_COST_APPROVED=1` in the environment and launch. Treat this
   variable as the approval token — it's how we tell the next invocation
   the gate has been lifted.

6. **Run.**
   ```bash
   uv run python scripts/pm/run_experiment.py experiments/exp<N>_<slug>/
   ```
   This blocks until Modal completes. Stream logs if possible.

7. **Publish.** After a successful run:
   - Read `experiments/exp<N>_<slug>/.cache/benches/<name>/summary.csv`
     and `.meta.json` to get headline numbers.
   - If auto-publish wasn't already done in the notebook
     (`publish=True`), run:
     ```bash
     uv run helico-publish bench --experiment exp<N>_<slug> --name <name>
     ```

8. **Commit committed artifacts.** The notebook writes small CSVs under
   `experiments/exp<N>_<slug>/data/` and plots under `plots/`. Commit
   those on the experiment's branch (create a PR back to `main` if the
   notebook's branch is not `main`).

9. **Post results comment.** Update the status comment with headline
   numbers, HF link, and a one-paragraph interpretation:
   ```
   🤖 Bench complete ($actual_cost actual). Headline:
   - monomer_protein.mean_lddt = 0.734
   - interface_protein_protein.mean_lddt = 0.487
   - ...
   See [HF bucket](<url>) for full artifacts.
   ```

## Rules

- **Never bypass the cost gate.** Splitting a run into smaller named
  steps to avoid the gate is explicitly disallowed — the gate exists to
  catch runaway spend.
- **Never destroy prior artifacts.** If `.cache/benches/<name>/` exists
  already, the library's idempotency layer returns cached results — do
  not delete to force a rerun. If the researcher wants a rerun, either
  add `force=True` to the specific call or bump the step name.
- **One 🤖 comment per run.** Use `gh issue comment --edit-last` for
  progress updates. Don't post a fresh comment per step.
- **Respect the branch.** If the notebook's frontmatter says `branch:
  exp/N-slug`, operate on that branch. Don't commit to `main` implicitly.

## Failure modes

- Modal auth missing → surface clearly, ask user to set secrets.
- Cost estimate fails → don't run blind; post the error.
- GH Actions 6h limit exceeded → post a comment suggesting either
  `max_targets=100` for a subset run, or that the researcher execute
  locally where the time limit doesn't apply.
