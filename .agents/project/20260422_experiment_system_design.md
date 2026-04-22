# Helico Experiments System — Design

Date: 2026-04-22 (pivoted same day)
Status: Wave 1b scaffold in progress.

This doc is the reference for how experiments are organized, executed, and
published in Helico. Agents working in this repo should read it before
modifying anything under `experiments/`, `src/helico/experiment.py`,
`scripts/pm/`, or the experiment-related GitHub Actions. If you change the
design, update this doc in the same PR.

## Goal

A **GitHub issue** is the research record — the question, thread of
discussion, and close-out. An **experiment directory** on a branch holds a
jupytext-markdown `README.md` that **is** the notebook: prose, code, and
analysis interleaved. Running the notebook produces plots and HTML; the
HTML is published to a GitHub Pages site so results are shareable with
links back to the issue, the branch, and the underlying CSVs.

## Why a notebook, not a config file

A config file assumes each experiment follows the same template. Real
research doesn't: one experiment trains three variants and compares them;
another sweeps a hyperparameter; another benches an existing checkpoint
with a data-pipeline tweak and no training at all. The notebook absorbs
that variance — it's ordinary Python plus helpers, and the written
analysis lives next to the code that produced it.

Design non-goal: a Marin-style content-addressed executor DAG. Too much
machinery for our scale.

## Lifecycle

```
1. Researcher opens issue with the Experiment template.
   Labels: `experiment`, priority.

2. Experiment is scaffolded at experiments/exp<N>_<slug>/README.md,
   either on `main` (baselines, characterizations, anything whose results
   we want to reference going forward regardless of outcome) or on a
   branch `exp/<N>-<slug>` (when the experiment requires speculative
   code changes that may not ship). The `branch:` frontmatter field
   tracks this choice.

   Wave 1b: researcher creates the notebook by hand using
   experiments/TEMPLATE.md. Wave 4: agent drafts from issue body.

3. Researcher (or later, agent) runs the notebook. Each ensure_* call
   dispatches to Modal (blocking) or returns cached results. Plots are
   saved as PNGs next to the notebook.

4. (Wave 2) A CI workflow executes the notebook on push and publishes the
   resulting HTML to the GH Pages site under /experiments/exp<N>_<slug>/.

5. Discussion happens in the issue. Iterations = edits to the notebook
   on the branch; results update in place thanks to idempotent caching.

6. Researcher closes the issue. The notebook's rendered HTML is the
   permanent record on the site. No separate SUMMARY.md step.
```

## Directory layout

```
experiments/
  AGENTS.md                          Rules for agents working on experiments
  README.md                          User-facing intro + conventions
  TEMPLATE.md                        Empty-but-working jupytext notebook
  exp<N>_<slug>/
    README.md                        The experiment notebook (jupytext md)
    data/                            Small CSVs committed; site links here
    plots/                           PNGs committed; also embedded in HTML
    .cache/                          Local pull-through cache; gitignored
    README.html                      Built by CI; not committed

src/helico/
  experiment.py                      Library: ensure_*, estimate_cost, dataclasses

scripts/pm/
  run_experiment.py                  Executes a notebook + exports HTML
  modal_prices.yaml                  GPU $/hr table
  itemize_experiments.py             (Wave 2) Scrape issues -> site index
  publish_checkpoint.py              (Wave 2) HF upload wrapper

docs/                                (Wave 2) MkDocs source; GH Pages target
  index.md
  experiments/
    index.md                         Auto-generated from experiments/*

.github/
  experiments.yaml                   Cost gate + HF repo names
  ISSUE_TEMPLATE/experiment.md       Preregistration form
  workflows/
    docs-build.yml                   (Wave 2) Execute notebooks + publish
    experiment-agent.yml             (Wave 3/4) Triggered by @claude mentions
```

## The experiment notebook

Format: **jupytext/md** (plain markdown). GitHub renders it natively.
Triple-backtick code fences with `python` are executable cells; other
fences are just prose. Outputs are NOT stored in the .md — they live in
the executed .ipynb (gitignored, under `.cache/`) and in committed plots.

Frontmatter:

```markdown
---
jupytext:
  text_representation: {extension: .md, format_name: markdown}
kernelspec: {name: python3}
helico_experiment:
  issue: 42
  title: "Deeper diffusion: 24 → 32 token blocks"
  branch: exp/42-deeper-diffusion
  baselines: [exp7-run-3]
---
```

The `helico_experiment` block is our metadata; the `jupytext` and
`kernelspec` blocks are what jupytext needs to pair the .md with an
.ipynb on execution. See `experiments/TEMPLATE.md` for a working example.

## The helico.experiment library

`src/helico/experiment.py` exports the primitives the notebook calls:

```python
ensure_training_run(name, spec, *, gpu, force=False) -> TrainingRun
ensure_bench_run(name, *, checkpoint, force=False, **bench_kwargs) -> BenchRun
estimate_cost(*, gpu, hours, workers=1) -> float
set_experiment(slug_or_path)    # auto-detects from cwd if not called
```

**Idempotency — cache by name alone.**
- Full key: `{experiment_slug}/{step_name}`.
- `ensure_bench_run` checks `.cache/benches/{name}/` then the Modal
  `helico-experiments` volume at `/experiments/{slug}/{name}/bench/`.
  If found: load and return. Otherwise: launch `modal run modal/bench.py`,
  sync results to volume on success.
- No content hashing. Code or data changes do NOT invalidate cache. To
  rerun, pass `force=True` or bump the name (`"baseline"` → `"baseline-v2"`).
- The library emits one line per call: `ensure_bench_run("foo") — cached`
  or `ensure_bench_run("foo") — launching (~$15.80)`.

**Dry-run mode (HELICO_DRY_RUN=1).**
- Every `ensure_*` short-circuits: records its cost estimate, returns a
  sentinel object, doesn't touch Modal. Used by the agent in Wave 3 to
  gate on total cost before executing for real.
- `estimate_cost()` is usable standalone at any time.

**Blocking dispatch.**
- `ensure_*` blocks until the Modal run completes (streaming logs).
  Matches researcher mental model; avoids a parallel dependency tracker.
- For Wave 3 agent-in-CI runs, the agent wraps the notebook in the Modal
  detach-and-callback pattern (see "Agent runtime" below).

## Cost gate

Stored in `.github/experiments.yaml` (`cost_gate_usd`, default $100).

Gate enforcement happens at two points:
- **Researcher in notebook**: soft — `ensure_*` prints the estimate and
  cumulative; no block (it's their Modal account).
- **Agent in CI (Wave 3+)**: hard — agent first executes notebook with
  `HELICO_DRY_RUN=1`, sums costs, aborts and posts for approval if above
  threshold. On approval, reruns without dry-run.

## Agent runtime (Wave 3)

Primary: **GitHub Actions + `anthropics/claude-code-action`**, ACL-gated
to OWNER/MEMBER/COLLABORATOR. Agent comments start with 🤖; agent-opened
PRs and issues carry the `agent-generated` label; progress updates via
`gh issue comment --edit-last`.

**Workflow**: `.github/workflows/experiment-agent.yml` fires on
`@claude` mentions in experiment-labeled issues (and on `issues.labeled`
when `experiment` is added). Elevated perms (contents/issues/PRs write)
versus the read-only `claude.yml`. Loads skills from `.agents/skills/`:

- `run-experiment` — dispatch notebook → cost gate → Modal → publish
- `estimate-cost` — HELICO_DRY_RUN + gate comparison
- `analyze-results` — post headline numbers + baseline deltas

**Required secrets** (repo Settings → Secrets and variables → Actions):

- `CLAUDE_CODE_OAUTH_TOKEN` — already set, used by existing `claude.yml`
- `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` — `modal token new` locally and copy
- `HF_TOKEN` — huggingface.co/settings/tokens; needed for helico-publish

Plus one **Modal-side secret**, `helico-github-pat`, containing
`GITHUB_TOKEN` set to a fine-grained GH PAT with contents/issues/PRs
write on this repo. Used by Modal-detach + callback pattern (below).

**Long-running constraint.** GH Actions jobs cap at 6h; full FoldBench
bench is 3-7h and training can be 12h+. Two strategies coexist:

- **In-Action blocking** (current MVP): `experiment-agent.yml` runs the
  notebook end-to-end in the Action runner, blocking on Modal. Works
  for anything under 6h (e.g. bench with `max_targets=100`, short
  training runs). Simple, no additional infra.
- **Modal-detach + callback** (for longer runs, TBD): Action parses
  spec, gates on cost, dispatches a Modal function that runs the
  notebook inside Modal and uses `helico-github-pat` to post the
  results comment when done. Action exits immediately. A short
  polling cron workflow is the safety net for abandoned Modal jobs.

## Checkpoint publishing

Two HF repos (`.github/experiments.yaml`):
- `buckets/timodonnell/helico-experiments` — auto-published per
  experiment run, path `exp<N>-<name>/`.
- `timodonnell/helico` — releases only, manual tag trigger.

Implemented via `src/helico/hf_publish.py` + `helico-publish` CLI
(Wave 2). Each upload includes the checkpoint, the resolved config,
a model card with issue/wandb links, and the bench summary CSV.

## Website (Wave 2)

MkDocs Material, published via GH Pages on every merge to main.

- `docs/experiments/index.md` generated by `scripts/pm/itemize_experiments.py`
  from `gh issue list --label experiment --state all` + frontmatter of each
  notebook.
- Per-experiment pages are the `README.html` rendered by a `docs-build.yml`
  workflow that executes each notebook with `HELICO_DRY_RUN=0` (cache hits
  make this cheap). CSVs under `data/` are linked from the page with
  "Source data" download links so readers can re-plot.

## What Wave 1b delivers

- Notebook-based experiment format locked in
- `src/helico/experiment.py` with `ensure_bench_run` + `estimate_cost`
  (enough for the first intended experiment: "run FoldBench on Protenix v1")
- `ensure_training_run` stub with a clear NotImplementedError (we'll flesh
  out when the first training experiment is filed)
- Working `experiments/TEMPLATE.md`
- `scripts/pm/run_experiment.py` runner (jupytext md → ipynb → html)
- Updated AGENTS.md, issue template, memory pointer

Waves 2-4 are unchanged from the pre-pivot plan.

## Marin patterns still adopted

- Issue-as-experiment with preregistration
- `experiments/exp<N>_<slug>/` directory convention keyed by issue number
- `.agents/skills/<name>/SKILL.md` playbooks for repeatable chores
- Layered AGENTS.md
- 🤖 comment prefix + `gh issue comment --edit-last` progress pattern
- `agent-generated` label
- Dated design docs under `.agents/project/YYYYMMDD_*.md`
- ACL-gated agent workflows

## Marin patterns **not** adopted

- Executor DAG + content-addressed hashing (overkill; name-based cache is
  enough for our scale)
- Python-DSL spec files (our "spec" is the notebook itself)
- GCS as artifact store (HF + Modal volumes)
- Cross-region transfer budgets (Modal is flat)
