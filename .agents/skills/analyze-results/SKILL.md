---
name: analyze-results
description: Post a results-analysis comment to an experiment issue after a completed run.
---

# analyze-results

After a bench or training run finishes, produce a short analysis comment
on the issue: headline numbers, comparison against baselines, and a
one-paragraph interpretation.

## Inputs

- `experiments/exp<N>_<slug>/.cache/benches/<name>/summary.csv` (or
  `.cache/trainings/<name>/meta.json`)
- `experiments/exp<N>_<slug>/README.md` frontmatter — `helico_experiment.baselines`
  names prior runs to compare against
- The issue body's "Success criteria" section

## Workflow

1. Read `summary.csv` and pull the metrics named in the issue's Success
   criteria section.
2. For each baseline listed in the frontmatter, try to load
   `experiments/exp<N_baseline>_<slug>/data/summary.csv` (these are the
   committed summaries, not the `.cache/` ones). Compute deltas per
   category.
3. Render a comment along these lines:

   ```
   🤖 Results posted.

   **Headline**
   - monomer_protein.mean_lddt: 0.734 (was 0.700, +0.034)
   - interface_protein_protein.mean_lddt: 0.487 (was 0.374, +0.113)

   **Success criteria**
   - [x] interface_protein_protein.mean_lddt improves by ≥0.02 — met (+0.113)
   - [x] monomer_protein.mean_lddt does not regress — met (+0.034)

   **Artifacts**
   - HF bucket: <url>
   - Notebook: experiments/exp<N>_<slug>/README.md
   - WandB: <url>  (if available)

   See the notebook for plots and discussion.
   ```

4. Post via `gh issue comment <N> --body "..."` (or
   `--edit-last` if a 🤖 comment already exists for this run).

## Rules

- **One comment per run.** Edit the existing 🤖 comment rather than
  posting a new one.
- **No speculation.** Report the numbers and whether criteria were met.
  Interpretation beyond "criteria met / not met" belongs in the
  notebook's Conclusion section, which the researcher writes.
- **No re-dispatching.** This skill only reads existing artifacts. If a
  rerun is needed, invoke `run-experiment` separately.

## When baseline is missing

If a named baseline's `data/summary.csv` can't be loaded, note it and
skip the delta for that baseline. Don't fail — still post the absolute
numbers.
