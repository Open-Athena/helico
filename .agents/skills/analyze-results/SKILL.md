---
name: analyze-results
description: Post a results-analysis comment to an experiment issue after a completed run.
---

# analyze-results

After a bench or training run finishes, produce a short analysis comment
on the issue: headline numbers, comparison against the stated baselines
in the notebook's frontmatter, and a one-paragraph interpretation.

## Inputs

- `experiments/exp<N>_<slug>/.cache/benches/<name>/summary.csv` (or
  `.cache/trainings/<name>/meta.json`)
- `experiments/exp<N>_<slug>/README.md` frontmatter (`helico_experiment.baselines`)
- The issue body's "Success criteria" section

## Workflow

1. Read `summary.csv` and pull the numbers named in the issue's Success
   criteria.
2. For each baseline listed in the frontmatter, try to load
   `experiments/exp<N_baseline>_<slug_baseline>/data/summary.csv` (or
   `.cache/`). Compute deltas per category.
3. Render a comment:

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
   - WandB: <url>  ← if available

   See the notebook for plots and discussion.
   ```

4. If any success criteria fail, flag them with `- [ ]` prefix and
   mention in the interpretation paragraph.

## Rules

- **One comment.** Edit the existing 🤖 comment rather than posting a new
  one (use `gh issue comment --edit-last`).
- **No speculation.** Report the numbers and whether criteria were met.
  Interpretation beyond "criteria met/not met" belongs in the notebook's
  Conclusion section, which the researcher writes.
- **No re-dispatching.** This skill only reads existing artifacts. If
  you need to rerun something, invoke the `run-experiment` skill
  separately.

## When baseline is missing

If a named baseline can't be loaded (no summary.csv in its experiment
dir), note this and skip the delta for that baseline. Don't fail the
analysis — still post the absolute numbers.
