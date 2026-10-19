---
jupytext:
  text_representation:
    extension: .md
    format_name: markdown
    format_version: '1.3'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
helico_experiment:
  issue: 24
  title: Focused continuation for sparse contact control
  branch: codex/clean-contact-pilot
---

# Focused continuation for sparse contact control

Continue the full 368.5M-parameter exp22 model from step 20,000 to test whether genuinely sparse conditioning plus geometric compliance produces structural steering. Exp23 found little response to four supplied contacts; its mask-support audit showed that the previous sparsest nonempty time had about 72 expected revealed pairs in a fully observed 384-residue crop.

The training data remain `timodonnell/helico-protenix-v1-pdb@164e41cf98d7fdf4d437192a9947138ca2464011`, with lock SHA256 `2855c34acc97c8990eb49268f0f17a35f014f76fcd1fac9195dbdd9aaa699114`. The Protenix-v1 train/validation split and FoldBench exclusions remain unchanged. No nonbinding, Flock, pMHC or TCR examples are added in this experiment.

## Recipe

Restore the exact raw model, optimizer and EMA from `exp22-contact-scale-v11/final.pt`, SHA256 `355788e40f610c8cc662e4c10f8225204668dfdcaf92ee1f04d744f8d5ea8b9a`. Parent source is `46cffd6939c9ea17d2103d5b8f5d79ba9e35444d`; the released model is `timodonnell/helico-masked-contact-exp22@01a6025408d078aa8305f302d1d664a0291c45ee`. Explicit continuation starts a new 5,000-update schedule and crop draw stream (seed 2401); recovery within that recipe preserves its step, draw suffix and scientific configuration. Checkpoints record both continuation and cumulative steps.

Use eight H100s, batch 32 (four accumulation steps), 384-token crops, four denoising samples, one trunk cycle, MSA depth 512 and independent 20% MSA dropout. Backbone/contact-module peak learning rates are 1e-5/1e-4, with 100 warmup steps and cosine decay to 10%; AdamW state and EMA decay 0.999 are retained. Confidence and dummy-template parameters remain frozen.

Mask-time mixture: 20% all unknown, 10% fully known, 50% sparse (equally likely expected counts 1/4/16/64), and 20% uniform continuous time. For M valid unordered pairs and expected count k, reveal each pair independently with probability min(1,k/M). Counts are binomial, not exactly k. The same label-independent absorbing kernel applies in all regimes; a sparse draw can be empty. Time selection depends on observation validity and crop size, never contact values.

Retain `4*(aligned MSE + smooth lDDT loss) + .03*distogram + .1*masked BCE + .01*visible BCE`. Add `0.2*geometric_compliance`, using the exact minimum predicted heavy-atom distance for each supplied valid pair. Positive violations are `relu(d-4.9 Å)`; negative violations are `relu(5.1 Å-d)`. Apply smooth-L1 with beta=1 Å, average over denoising samples and pairs within each present class, then average the present classes. Revelation is unbiased; the loss is class-balanced. Empty conditioning contributes differentiable zero. Only observed heavy atoms participate: negatives require complete tokens and positives require an observed close-pair witness. Closest-atom selection is chunked without a graph, followed by a differentiable norm giving the minimum's exact subgradient away from ties. This avoids retaining quadratic atom-pair graphs.

This is an empirical weighted training objective, not an unbiased diffusion ELBO. Binary margins are deliberately only 0.1 Å. Native structural labels and the existing identity-mapped coordinate losses still define the training target; the objective does not turn marginal contact probabilities into simultaneous hard constraints.

## Held-out checks and continuation gates

`data/development_panel.csv` fixes the existing 24 intra-protein/protein–protein/protein–ligand development examples plus eight protein–DNA and four protein–RNA retention examples, chosen by metadata/hash before predictions. Validation dataset: `timodonnell/helico-contact-eval-v1@ec4de019de89f79f03da31c5e026ddf7d858e77e`. FoldBench is not used for these decisions. This development panel is for recipe selection; it is not an independent final test.

Run parent EMA inference once, with two seeds, one cycle and 200 diffusion steps. Freeze 1/4/16 native contacts missed by that parent's structure (nested selections) and up to four spurious contacts to negate; reuse the exact matrices for every continuation checkpoint. Evaluate raw and EMA weights at steps 500, 1,500 and 5,000. Each has unconditioned baseline and same-seed unconditioned repeat, because exp23 measured numerical variation; steering is conditioned minus repeated-unconditioned satisfaction. Native labels select oracle diagnostic constraints but are removed from every model input. No control model is trained.

At 500, stop on missing outputs/nonfinite metrics or gross structure regressions: mean unconditioned lDDT delta below -0.02, mean interface lDDT delta below -0.05, or inter-chain clash fraction increase above 0.05, relative to parent EMA. Apply these to raw and EMA, and report DNA/RNA retention separately. At 1,500 additionally require either raw or EMA mean sparse steering of at least +0.05 and at least +0.03 over the parent, with a nonnegative 95% target-bootstrap lower bound and no sparse subgroup worse than -0.05. Otherwise stop at 1,500 and retain all artifacts. These are predeclared operational gates, not claims of definitive statistical significance. Final interpretation includes paired target intervals and per-group outcomes.

## Resources and dispatch

Use the existing prepaid cw-rno2a H100 pool at interactive priority, without changing fleet capacity. The entire sequence is bounded by **192 H100-hours**, including staging, evaluation and bounded recovery, and **$24** additional first-month storage (at most 400 GiB at [$0.06/GiB-month](https://www.coreweave.com/pricing), checked 2026-10-16). Existing immutable training shards are reused. Each stage has a fixed deadline; recovery consumes its remaining time. Storage persists after completion. GPU consumption is recorded separately from incremental spend.

The following cell declares every planned stage so the dry run covers the entire study. For actual dispatch set `HELICO_FOCUSED_STAGE` to one stage; the controller advances only after the declared gates, and existing receipts are adopted without duplicating jobs.

```python
import json, os
from pathlib import Path
from helico.experiment import set_experiment, ensure_training_run, ensure_bench_run, is_dry_run, REPO_ROOT
set_experiment("exp24_focused-continuation")
plan = json.loads((REPO_ROOT / "experiments/exp24_focused-continuation/plan.json").read_text())
# Whole-study estimate: $24 incremental first-month storage; <=192 prepaid H100-hours.
requested = os.environ.get("HELICO_FOCUSED_STAGE", "train-500")
for name, spec in plan["stages"].items():
    if not is_dry_run() and name != requested:
        continue
    if spec["mode"] == "train":
        run = ensure_training_run(name, gpu="H100:8", max_steps=5000,
            crop_size=384, lr=1e-5, est_wall_hours=spec["timeout_seconds"]/3600, coreweave=spec)
    else:
        run = ensure_bench_run(name, gpu="H100", workers=8, est_wall_hours=spec["timeout_seconds"]/3600, coreweave=spec)
    print(run.meta)
```

## Status

The first continuation stage completed **500 additional updates / 20,500 cumulative updates** on October 16, processing 16,000 weighted crops in 4,180 training seconds on eight H100s. Model, optimizer and EMA are durable in `exp24-focused-500-v1/final.pt`, SHA256 `ee97fdd0e0a5aeb538641fcf95cc1a67470aa31d2f3b0268b1449f4a6d66170a`. Parent evaluation completed all 336 predictions. See `data/checkpoint-500.json` and `data/result-train-500.json`.

The candidate evaluation then failed at 17:30 UTC on October 16: CoreWeave object storage rejected a 1,629-byte JSON metadata upload for raw-weight 7n8v with `IncompleteBody`. The tensor-upload path already retried, but metadata did not. One rank failed after completing its other targets, and torchrun terminated another rank, leaving **654/672 prediction records**. The controller stopped for inspection; no 1,500- or 5,000-update training stage ran during the pause. This was an infrastructure failure, not a failed scientific gate. The original traceback is retained in `data/evaluation-upload-failure.json`.

On October 19, source `d51f467296e10280b2b0963fedba4e605c6b31a3` adds bounded metadata-upload retries and propagates target errors into controller-visible logs. All 29 targeted publication/watchdog/gate tests pass. Recovery `/bizon/helico-exp24-focused-eval-500-v1-recovery-1` reuses the identical resolved evaluation configuration and completed predictions. No model, loss, input, threshold or scientific gate changed. The persistent controller is restarted and may advance only after the complete original gate passes.

Resource accounting explicitly excludes the job-free weekend pause while charging 1,200 seconds (rounded up) for the failed evaluation attempt; recovery is limited to the remaining 4,200 seconds of that stage's original 5,400-second allocation. The full-study 192 H100-hour and $24 first-month storage caps are unchanged. See `data/recovery-20261019.json`.

Validation contact BCE was 0.021023 at update 250 and 0.020844 at 500. Logged training BCE and compliance fluctuate without a clear downward trend over this short continuation (`data/training-through-500.csv`). Incomplete evaluation results are not used to advance the gate or establish improved structural steering.

The 58 focused tests passed, including sparse mask support/label independence, loss gradients, a CUDA contact-projection/diffusion gradient test, restoration guards, artifact publication, evaluation gates and notebook checks. A separate full 368,537,859-parameter real-complex preflight on 7qeh (144 tokens) produced finite gradients throughout, nonzero contact-projection gradient norm 0.1713, and compliance loss 0.002658; no optimizer update was performed. Peak local GPU allocation was 7.81 GB. See `data/full_model_gradient_preflight.json`.

The enabled user service `helico-exp24-controller.service` keeps a durable state at `scratch/exp24-controller-state.json`, survives user-service restarts, and automatically sequences stage evaluation and approved continuations. Each stage's original deadline bounds retries; scientific/data/OOM/nonfinite failures stop automatic continuation. Only this experiment's jobs can be cancelled. Gate decisions are written to `data/gate-<step>.json` and the single experiment issue comment. No scientific result is inferred from submission or training loss alone.

[Training W&B](https://wandb.ai/timodonnell/helico/runs/exp24-focused-500-v1) · [Parent inference W&B](https://wandb.ai/timodonnell/helico/runs/exp24-focused-eval-parent-v1) · [Issue status](https://github.com/Open-Athena/helico/issues/24#issuecomment-6219362619).

The initial full CI attempt passed 401 tests and skipped three optional fixtures, but repeated the pre-existing `TestTrunkSnapshot.test_z_init` BF16 failure (max absolute difference 0.5), matching the exp23 evidence. The unchanged rerun passed **402 tests, with three optional fixture skips** ([CI](https://github.com/Open-Athena/helico/actions/runs/38927942551/job/117067176075)); snapshots, tolerances and model code were not modified to make this check pass. The startup step-50 checkpoint is also durable, with remote size verified, and W&B independently records finite metrics through step 50 (`data/wandb_initial_metrics.csv`).
