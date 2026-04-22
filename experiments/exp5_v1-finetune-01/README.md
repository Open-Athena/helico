<!--
Retrospective experiment wrapping an already-running fine-tune (v1-finetune-01).
The notebook pulls train + val metrics from W&B and plots their trajectory.
FoldBench comparison vs. exp4 baseline is deferred until both (a) training
finishes and (b) ensure_bench_run supports custom checkpoints (see TODO at
the end of the notebook + followup issue).

Run locally:
    uv run python scripts/pm/run_experiment.py experiments/exp5_v1-finetune-01/
-->

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    name: python3
    display_name: Python 3
helico_experiment:
  issue: 5
  title: "Fine tuning improves accuracy"
  branch: main
  baselines: [exp4_baseline_protenix_v1]
---

# Fine tuning improves accuracy

**Issue:** [#5](https://github.com/Open-Athena/helico/issues/5) · **Branch:** `main`

## Question

Starting from Protenix v1 weights and fine-tuning on our pre-2021-09-30
PDB train split, does FoldBench accuracy improve over the exp4 baseline?
Secondarily: is our training pipeline sound end-to-end (no divergence, no
NaN, val metrics tracking train)?

## Hypothesis

Val LDDT and train loss should both trend in the right direction. We have
early evidence (val/lddt_hard 0.627 at step 500 → 0.738 at step 1000) that
the model is learning. Hypothesis: the final fine-tuned checkpoint beats
the Protenix v1 baseline from #4 on most FoldBench categories.

## Background

This issue wraps an already-running fine-tune (`v1-finetune-01`, H100:8,
10K steps, launched 2026-04-22, W&B run `usvcruzi`). The run itself was
kicked off as the Phase 4 milestone of the broader roadmap — the
experiment-system wrapper here is retrospective, so the notebook pulls
live metrics rather than launching new work.

## Setup

```python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helico.experiment import experiment_dir, set_experiment

set_experiment("exp5_v1-finetune-01")

WANDB_ENTITY = "timodonnell"
WANDB_PROJECT = "helico"
WANDB_RUN_ID = "usvcruzi"  # v1-finetune-01

DATA = experiment_dir() / "data"
PLOTS = experiment_dir() / "plots"
DATA.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

pd.set_option("display.float_format", "{:.3f}".format)
```

## Pull metrics from W&B

We use the W&B public API to download the logged metrics as a DataFrame.
The run is on my personal account (`timodonnell/helico`) and `WANDB_API_KEY`
needs to be set in the environment for the pull to work.

```python
import wandb

api = wandb.Api()
run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN_ID}")
# history() streams all step-indexed scalars into a DataFrame.
# Pass samples=None (deprecated but stable default) or a large number to
# get every row rather than a downsampled subset.
history = run.history(samples=50_000)
history = history.sort_values("_step").reset_index(drop=True)
history.to_csv(DATA / "wandb_history.csv", index=False)
print(f"pulled {len(history)} rows, last step = {int(history['_step'].max())}")
history.head()
```

## Training loss and its components

`loss` is the total (`diffusion + 0.1 * distogram`). Components are also
logged individually so we can see whether one is dominating.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
ax = axes[0]
if "loss" in history:
    ax.plot(history["_step"], history["loss"], label="loss", color="#222", lw=1.5)
if "loss/diffusion" in history:
    ax.plot(history["_step"], history["loss/diffusion"], label="diffusion", alpha=0.8)
if "loss/distogram" in history:
    ax.plot(history["_step"], history["loss/distogram"], label="distogram", alpha=0.8)
ax.set_xlabel("step")
ax.set_ylabel("loss (train)")
ax.set_yscale("log")
ax.legend(loc="upper right")
ax.set_title("Train loss — window average (log scale)")

ax = axes[1]
if "grad_norm" in history:
    ax.plot(history["_step"], history["grad_norm"], color="#c44e52", lw=1.0)
ax.set_xlabel("step")
ax.set_ylabel("grad norm (pre-clip)")
ax.set_yscale("log")
ax.set_title("Gradient norm")

fig.tight_layout()
fig.savefig(PLOTS / "train_loss_and_grad.png", dpi=150)
plt.show()
```

## Learning rate and throughput

Sanity check: warmup then cosine decay, and tokens/sec roughly constant.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
ax = axes[0]
if "lr" in history:
    ax.plot(history["_step"], history["lr"], color="#4c72b0")
ax.set_xlabel("step")
ax.set_ylabel("learning rate")
ax.set_title("LR schedule")

ax = axes[1]
if "tokens_per_sec" in history:
    ax.plot(history["_step"], history["tokens_per_sec"], color="#55a868", alpha=0.6, lw=0.8)
    # Smooth with rolling mean for readability
    smoothed = history["tokens_per_sec"].rolling(20, min_periods=1).mean()
    ax.plot(history["_step"], smoothed, color="#55a868", lw=2)
ax.set_xlabel("step")
ax.set_ylabel("tokens/sec (per GPU)")
ax.set_title("Throughput (20-step rolling mean)")

fig.tight_layout()
fig.savefig(PLOTS / "lr_and_throughput.png", dpi=150)
plt.show()
```

## Validation metrics over time

This is the core of the experiment. If fine-tuning is working we expect
`val/lddt_hard` and `val/gdt_ts` to trend up, and `val/rmsd` and
`val/*_loss` to trend down.

```python
val_cols = [c for c in history.columns if c.startswith("val/")]
# Drop the step-indexing bookkeeping columns (val/n_attempted, val/n_skipped)
plot_cols = [c for c in val_cols if not c.endswith(("n_attempted", "n_skipped"))]
print("val metrics found:", plot_cols)
val_rows = history[["_step"] + val_cols].dropna(subset=val_cols, how="all")
val_rows.to_csv(DATA / "val_metrics.csv", index=False)
val_rows.tail(5)
```

```python
# Pair each metric with the direction that counts as "good" (up or down).
HIGHER_IS_BETTER = {"val/lddt", "val/lddt_hard", "val/gdt_ts", "val/plddt"}

# 2 rows × 4 cols fits all eight val metrics cleanly.
to_plot = [c for c in [
    "val/total_loss", "val/diffusion_loss", "val/distogram_loss", "val/lddt",
    "val/lddt_hard", "val/rmsd", "val/gdt_ts", "val/plddt",
] if c in val_rows]

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, col in zip(axes.flat, to_plot):
    y = val_rows[col].values
    x = val_rows["_step"].values
    good = col in HIGHER_IS_BETTER
    ax.plot(x, y, marker="o", markersize=4, lw=1.5, color="#2b8a3e" if good else "#c44e52")
    ax.set_title(col)
    ax.set_xlabel("step")
    ax.grid(alpha=0.2)
for ax in axes.flat[len(to_plot):]:
    ax.axis("off")
fig.suptitle("Validation metrics (green = higher-is-better, red = lower-is-better)")
fig.tight_layout()
fig.savefig(PLOTS / "val_metrics.png", dpi=150)
plt.show()
```

## Num val samples attempted vs skipped

Sanity check that gh#3 (the cuDNN crash that used to torpedo val sweeps)
stays fixed. Post-torch-pin we expect `val/n_skipped == 0` every sweep.

```python
if {"val/n_attempted", "val/n_skipped"}.issubset(history.columns):
    vsamp = history[["_step", "val/n_attempted", "val/n_skipped"]].dropna()
    print(vsamp.tail(10))
    print(f"total skipped across all val sweeps: "
          f"{int(vsamp['val/n_skipped'].sum())}")
```

## Final-checkpoint FoldBench vs. exp4 baseline

Bench the final fine-tuned checkpoint (`/ckpts/v1-finetune-01/final.pt`
on the `helico-checkpoints` Volume) and diff the per-category
mean LDDT / mean DockQ against the exp4 Protenix-v1 baseline.

`ensure_bench_run` is idempotent — once cached locally or on the
`helico-experiments` Volume, re-runs are free.

> **Note**: the run was stopped at step 3,250 (out of a planned 10,000)
> after val metrics had clearly converged on the cuDNN-fixed val sweep.
> `/ckpts/v1-finetune-01/final.pt` was set to `step_3250.pt` so the
> usual `final.pt` reference resolves; raw `step_*.pt` checkpoints are
> still on the volume for sweeps.

```python
FINAL_CKPT = "/ckpts/v1-finetune-01/final.pt"

from helico.experiment import ensure_bench_run
bench_ft = ensure_bench_run(
    "v1-finetune-01-final",
    checkpoint=FINAL_CKPT,
    workers=8, gpu="H100", n_samples=5, n_cycles=10,
    max_tokens=2048, cutoff_date="2024-01-01",
    est_wall_hours=0.75,
)
bench_ft.summary.to_csv(DATA / "bench_ft_summary.csv")
print(f"bench cached: {bench_ft.cached}")
bench_ft.summary
```

```python
from helico.experiment import _experiment_dir
baseline_csv = _experiment_dir("exp4_baseline_protenix_v1") / "data" / "summary.csv"
if baseline_csv.exists():
    baseline = pd.read_csv(baseline_csv, index_col=0)
    # Align by category; subtract fine-tune − protenix-v1.
    cols = ["mean_lddt", "mean_dockq"]
    delta = (bench_ft.summary[cols].astype(float)
             - baseline[cols].astype(float))
    delta.to_csv(DATA / "delta_vs_exp4.csv")
    print("Δ (fine-tune − protenix-v1):")
    print(delta)
else:
    print(f"exp4 baseline summary not found at {baseline_csv}; run exp4 first.")
    delta = None
```

```python
if delta is not None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = delta["mean_lddt"].sort_values().index
    ax.barh(order, delta.loc[order, "mean_lddt"],
            color=["#c44e52" if v < 0 else "#2b8a3e"
                   for v in delta.loc[order, "mean_lddt"]])
    ax.axvline(0, color="k", alpha=0.5, lw=1)
    ax.set_xlabel("Δ mean LDDT (fine-tune − Protenix v1)")
    ax.set_title("FoldBench — fine-tuning delta vs. exp4 baseline")
    fig.tight_layout()
    fig.savefig(PLOTS / "delta_vs_exp4_lddt.png", dpi=150)
    plt.show()
```

## Conclusion

(Fill in after training finishes and the FoldBench follow-on runs.)

A short paragraph: did training converge? Did val metrics improve? Does
the final checkpoint beat the Protenix-v1 baseline on FoldBench, and by
how much per category?
