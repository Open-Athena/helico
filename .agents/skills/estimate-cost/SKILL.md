---
name: estimate-cost
description: Compute $ cost estimate for an experiment notebook without dispatching to Modal.
---

# estimate-cost

Return the total Modal $ cost an experiment would spend if run, without
actually dispatching anything. Used by `run-experiment` to gate on the
threshold in `.github/experiments.yaml`, and by humans who want a sanity
check before kicking off a long run.

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

## Programmatic

From Python (e.g. in a SKILL-driven agent action):

```python
import os, subprocess, sys
from helico.experiment import dry_run_records, dry_run_total_usd

os.environ["HELICO_DRY_RUN"] = "1"
subprocess.run(
    ["uv", "run", "python", "scripts/pm/run_experiment.py",
     f"experiments/{slug}/"],
    check=True,
)
# dry_run_total_usd is process-local — need to read it in the same process.
# If run_experiment.py was a subprocess, parse its stdout instead.
```

Cleaner: import the library directly and call ensure_* functions in a
sandboxed script. But run_experiment.py with stdout-parsing is fine for
the Action use case.

## Gate rules

- Threshold lives in `.github/experiments.yaml` as `cost_gate_usd` (default $100).
- If total estimate < gate: go.
- If total estimate >= gate: post the estimate, wait for `@claude approve`.
- Calibration runs and bench runs under the flat floor ($5) are always
  allowed — they're too cheap to meaningfully gate.

## What's NOT counted

- Local notebook execution time
- Image-build / cold-start time on Modal (first use per image version
  adds ~5-10 min; subsequent runs reuse cached images)
- `helico-publish` HF upload time (tiny, negligible)
