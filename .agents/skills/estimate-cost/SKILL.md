---
name: estimate-cost
description: Compute $ cost estimate for an experiment notebook without dispatching to Modal.
---

# estimate-cost

Return the total Modal $ cost an experiment would spend if run, without
actually dispatching anything. Used by `run-experiment` to gate on
`cost_gate_usd` in `.github/experiments.yaml`, and by humans who want a
sanity check before kicking off a long run.

## How

Dry-run mode short-circuits every `ensure_*` call in the notebook:
instead of launching Modal, the library records the estimated cost and
returns a sentinel object with placeholder data so analysis cells can
still execute.

```bash
HELICO_DRY_RUN=1 uv run python scripts/pm/run_experiment.py \
    experiments/exp<N>_<slug>/
```

Output lines to look for:
```
[helico.experiment] ensure_bench_run('protenix-v1-default') — dry-run (~$23.70)
[helico.experiment] ensure_training_run('baseline') — dry-run (~$379.20)
```

Sum the `~$X` values manually or parse them out of the output; that's
the total.

## Gate rules

- Threshold in `.github/experiments.yaml` as `cost_gate_usd` (default $100).
- Total estimate < gate → go.
- Total estimate ≥ gate → stop and ask the user for explicit approval
  before proceeding.

## What's NOT counted

- Local notebook execution time
- Image-build / cold-start time on Modal (first use per image version
  adds ~5-10 min; subsequent runs reuse cached images)
- `helico-publish` HF upload time (negligible)
