# experiments/AGENTS.md

Rules for agents working under `experiments/`. Layered on top of the root
`AGENTS.md`. Canonical design:
`.agents/project/20260422_experiment_system_design.md`. If this file
conflicts with the design doc, the design doc wins — update this file in
the same PR.

## Scope

Each experiment is a jupytext-markdown notebook at
`experiments/exp<N>_<slug>/README.md` that encodes its own question,
procedure, and analysis. `<N>` is the GitHub issue number; `<slug>` is a
2-5 word kebab-case descriptor.

Code changes that implement new functionality (new model variants, new
features) belong in `src/helico/`. An experiment's notebook *uses* that
code via normal imports.

## Main vs. branch

Experiments can live on `main` or on a dedicated branch
`exp/<N>-<slug>`. Default to main for baselines, characterizations, and
bug analyses — any experiment whose results belong in the permanent
record regardless of outcome. Use a branch when the experiment depends
on speculative code that may not ship. The `branch:` field in the
notebook's frontmatter records the choice and should be kept accurate.

## Hard rules

1. **Never rerun by mutating an existing step name.** To iterate, pass
   `force=True` for a single call (during debugging), or bump the name
   (`"baseline"` → `"baseline-v2"`) for a persistent re-run. The cache is
   keyed by name alone; silently overwriting corrupts the record.

2. **Don't commit `.cache/`, `README.html`, or `*.ipynb` files.** They are
   gitignored. Commit small CSVs under `data/` and plots under `plots/`.

3. **Use `ensure_bench_run` / `ensure_training_run`, not subprocess calls
   to `modal run modal/bench.py` directly.** The library handles idempotent
   caching, dry-run mode, and cost estimation. Direct `modal run` calls
   bypass all of that.

4. **Declare intent in prose before each code cell.** The notebook will be
   read by future-you and the agent. A two-sentence intro above each
   code block is the minimum.

5. **Save key plots to `plots/` AND display inline.** `plt.savefig(...)`
   followed by `plt.show()`. The `data/` dir holds the CSVs that feed the
   plots, so readers can re-plot without rerunning Modal.

6. **Cost hygiene.** Before any sequence of `ensure_*` calls, put a
   comment noting total expected cost. The library will print estimates
   on each call; reading them is part of your review.

7. **Cost gate.** For agent-driven runs, the agent must first execute with
   `HELICO_DRY_RUN=1` and gate on total estimated cost. Do not route
   around this by splitting one expensive run into smaller named steps.

8. **Agent comment hygiene.** Agent comments start with 🤖. Use
   `gh issue comment --edit-last` for progress updates, not a fresh
   comment per update. PRs/issues opened by the agent carry the
   `agent-generated` label.

## When a researcher replies with a variant

- If the change is small (tweak an `ensure_*` kwarg or add a new named
  step): edit the notebook on the branch, push, the next run picks up
  new work without re-running cached steps.
- If the change is a different hypothesis: suggest opening a new issue
  that links back, rather than extending this one.

## Closing out

There is no separate SUMMARY.md. The notebook itself, rendered to HTML,
is the permanent record. Before the issue is closed:

- Make sure the notebook's final section (`## Conclusion` or similar)
  gives a future reader the answer without requiring them to scroll
  through the full thread.
- Ensure every plot on the page has its source CSV in `data/`.
- Confirm the issue body's title and hypothesis still match what the
  notebook actually did (update the issue body if not).

## Notebook format quick reference

- Jupytext flavor: **plain markdown** (not MyST, not percent). Code cells
  are triple-backtick `python` fences.
- Required frontmatter keys: `jupytext.text_representation`,
  `kernelspec`, and `helico_experiment.{issue, title, branch}`.
- Run locally:
  `uv run python scripts/pm/run_experiment.py experiments/exp<N>_<slug>/`.
- Dry-run:
  `HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py ...`
  (prints per-step cost estimates + total; doesn't touch Modal).
