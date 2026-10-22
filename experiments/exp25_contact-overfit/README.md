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
  issue: 25
  title: Fixed-training-set overfit diagnostic for sparse contact control
  branch: codex/clean-contact-pilot
---

# Fixed-training-set overfit diagnostic

Exp24 reached 21,500 cumulative updates but failed its 1,500-update sparse-steering gate. This diagnostic asks whether sustained repetition on a small training set can teach useful contact compliance, and separates ordinary memorization from an effect of supplying contacts. It does not test generalization or replace a broad-data continuation.

## Data and recipe

The six frozen crops are 2fsx/5ahj (intra-protein), 3bnj/4qnp (protein–protein), and 3dn4/3sii (protein–ligand). Group labels specify evaluated pairs, not exclusive molecule composition. They were selected from previously prepared training crops before predictions from this checkpoint. All six IDs were checked against the original training membership, validation pool and FoldBench exclusions. Native heavy-atom contacts and validity masks were recomputed and exactly match the saved features. Validation and test membership are unchanged.

Dataset: `timodonnell/helico-contact-overfit-v1@ed0f7372cf033e18168236fb4dadd8a12cb211e7`, manifest SHA256 `1733da59edae03d98a132d12a0c3e2b1f69f6eda5cdc6743431a527ba44e3791`. The full source dataset and lock remain `timodonnell/helico-protenix-v1-pdb@164e41cf98d7fdf4d437192a9947138ca2464011`, lock `2855c34acc97c8990eb49268f0f17a35f014f76fcd1fac9195dbdd9aaa699114`. See `data/fixed-crops-manifest.json`. Samples stay fixed; masks, MSA sampling/dropout, rigid augmentation and diffusion noise change on every draw.

Restore the complete raw model, AdamW optimizer and EMA from `exp24-focused-1500-v1/final.pt`, SHA256 `3167b567a602c2c4a7e9b01def1cddbc5c9462d2ab5aed5b65ab98b4221d501f`. Retain the full 368.5M architecture and exp24 loss/masking recipe: 20% unknown, 10% full, 50% sparse expected 1/4/16/64 pairs, 20% uniform continuous time, with value-independent Bernoulli revelation and 20% independent MSA dropout. Structure/distogram/contact losses and geometric-compliance weight 0.2 are unchanged. No positive-biased revelation is introduced.

Use eight H100s, global batch eight (one crop per GPU), four denoising samples, MSA depth 512, and uniformly sample the six fixed crops. Peak backbone/contact rates are 1e-5/1e-4, warmup 100, cosine decay over **5,000 updates**, seed 2501. This is about 6,667 draws per crop on average. The smaller global batch and repeated fixed crops are explicit diagnostic changes. There is no steering-based stop at 1,500. Nonfinite/data failures or the resource ceiling still stop execution.

## Measurements and interpretation

Evaluate raw weights at updates 0/100/500/1,500/3,000/5,000, and EMA at the final update. Use two fixed seeds, MSA on, one cycle, 200 diffusion steps and one sample per call. Freeze native contacts missed by the initial raw-model structure, and spurious contacts to negate; reuse those exact matrices for unknown/repeat/rescue1/rescue4/rescue16/negative4 at all checkpoints. Parent constraints are read from raw step zero. Native labels are removed before model inference.

Report absolute requested-contact satisfaction, satisfaction with the contacts masked, their difference, contact-head response, native lDDT and per-target records. Average seeds/modes within targets, then targets equally. A target with no missed contacts contributes no sparse-steering measurement; retain it and report the measured-target count. Success at memorizing a native structure without supplied contacts is not counted as successful conditioning. The six examples are training data and must never be presented as held-out performance.

Track training BCE, geometry losses, positive/negative exposure, sparse draws that actually contain positive constraints, diffusion noise and contact-projection gradient norm. The projection norm includes all loss terms; a separate compliance-only backward preflight checks the structural gradient path. Improving loss without improved full-sampling satisfaction is a distinct outcome from ordinary memorization or conditional steering. A failure under this recipe does not by itself establish an architectural defect or rule out further broad-data training.

The mask schedule also limits exposure to any particular pair. Over the full run, an individual pair is expected to be revealed roughly 1–9 times in the sparse component, depending on crop size; it is revealed much more often in the full/uniform components. Conditional on a sparse draw, the probability of at least one positive pair is 14–39% across these six crops. `data/expected-exposure.csv` records these expectations separately from observed training counters. Repetition of structures is therefore not equivalent to repetition of each exact sparse conditioning task.

## Resources and dispatch

Whole diagnostic cap: **64 prepaid H100-hours** (one eight-H100 job, eight hours including staging, inference and bounded recovery), **$8** first-month additional storage (at most 133 GiB within the existing $0.06/GiB-month allowance). No fleet capacity changes. Training's internal seven-hour limit leaves time for final inference and checkpointing; recovery shares the original eight-hour deadline. W&B routes to `timodonnell/helico`.

This cell declares the whole study for the cost gate, and dispatches it through the standard idempotent training interface. The job performs its paired diagnostics internally and retains checkpoints and per-prediction artifacts in durable object storage.

```python
import json
from helico.experiment import set_experiment, ensure_training_run, REPO_ROOT
set_experiment("exp25_contact-overfit")
plan = json.loads((REPO_ROOT / "experiments/exp25_contact-overfit/plan.json").read_text())
# Whole diagnostic: <=64 prepaid H100-hours; <=$8 new first-month storage.
run = ensure_training_run("fixed-six-v1", gpu="H100:8", max_steps=5000,
    crop_size=384, lr=1e-5, est_wall_hours=8, coreweave=plan["spec"])
print(run.meta)
```

## Results as of October 22

The diagnostic stopped at update **4,500 of 5,000** (26,000 cumulative), while publishing its next checkpoint. CoreWeave object storage returned `MethodNotAllowed` / `PermissionError`: “Account is write suspended for this availability zone. Please check capacity quota and contact support.” Rank zero exited at this storage operation; the distributed launcher then terminated the other ranks. The full traceback was recovered from the separate Iris output archive and its SHA256 verified; see `data/failure-excerpt.txt`. This was not the scientific steering gate. The last durable model/optimizer/EMA checkpoint is update **4,000** (25,500 cumulative), 5,795,898,755 bytes, SHA256 `5342f7c3d106d0f5149e0f1451ae16d79757a3a53bacd55f1460eb3f0175a51a`.

Both Iris and W&B report failure. The monitor stopped without restarting because its available error tail did not identify a retryable transient failure. Storage failures also left its diagnostic progress at 4,214; W&B and the recovered full log establish that optimizer updates reached 4,500. The original eight-hour recovery deadline has elapsed. The task ran for 2h 53m 22s, approximately 23.12 H100-hours. It is no longer using GPUs. A fresh October 22 multipart-upload creation probe succeeded and was immediately aborted; that specific storage operation is available again, but a complete large upload has not been retested. No recovery job was launched during this status review.

Raw paired evaluation completed at 0/100/500/1,500/3,000: **72 records at every checkpoint**, all six targets measured, two seeds per target. Final raw and EMA evaluations are missing; no `result.json` was published. The frozen requests are native contacts missed by the initial raw checkpoint. Satisfaction is averaged over rescue1/4/16 and seeds within each target, then over targets equally.

| Additional updates | Supplied satisfaction | Masked satisfaction | Supplied minus masked | Unconditioned lDDT |
|---:|---:|---:|---:|---:|
| 0 | 1.83% | 2.01% | −0.19 pp | 0.9165 |
| 100 | 9.69% | 5.15% | +4.54 pp | 0.9236 |
| 500 | 48.86% | 50.19% | −1.34 pp | 0.9613 |
| 1,500 | 65.74% | 65.43% | +0.31 pp | 0.9820 |
| 3,000 | 67.64% | 67.46% | +0.17 pp | 0.9881 |

At update 3,000, the intra-protein group has mean steering +0.52 pp; protein–protein and protein–ligand each have 0.00 pp on their two targets. These are tiny descriptive groups, not generalization estimates. Contact-head response at requested pairs also shows no sustained positive trajectory: mean −0.00220 at 3,000. Per-target and aggregate values are in `data/eval-summary-step-*-raw.json` and `data/trajectory.csv`.

Training masked-contact BCE fell from **0.01911** over updates 1–500 to **0.01441** over 4,001–4,500, approximately **25%**. These are means of 50 equally spaced logged updates per window, not every training batch. Corresponding loss/exposure windows are in `data/training-windows.csv`. Pre-clipping gradient norms grew substantially late in training: the median rose from about 492 in the first window to 19,319 in the last. Global clipping is enabled at norm 1. These finite but large norms deserve investigation before treating further updates as equivalent effective learning; they were not the recorded crash cause.

## Interpretation and remaining work

Repeated training clearly improved native structure fit and satisfaction of the frozen initially missed contacts. Almost the same improvement occurs with the conditioning matrix entirely masked. The transient +4.54 pp result at update 100 was dominated by one target and was not sustained. This diagnostic therefore supports memorization under the current recipe, without demonstrating useful sparse contact control. It neither proves that the architecture is incapable of conditioning nor rules out substantially longer training on the broad dataset. The limited exposure of any particular pair in the sparse regime, and the absence of alternate native states for the same sequence in this six-example set, limit the inference.

The unfinished work is to resume the verified update-4,000 checkpoint through 5,000 and obtain final raw/EMA comparisons, after explicitly accounting for the recovery allocation and verifying durable writes. Before a large broad-data extension, inspect why supplied contacts contribute so little, including sparse task exposure and the increasing pre-clipping norms. Do not count the overfit checkpoint as a candidate production model or any of these six examples as held-out evidence.

## Validation and provenance

The cost dry run estimated $8 additional first-month storage, within the whole-study cap. All 57 focused tests passed. Full GPU CI for the submitted source passed **412 tests**, with three optional fixture skips ([CI](https://github.com/Open-Athena/helico/actions/runs/39615141453/job/119274053808)). A full 368,537,859-parameter preflight restored the parent on real training crop 2fsx: compliance-only loss 0.005082, contact-input projection gradient norm 1.0405, finite nonzero diffusion-module gradients, and 7.85 GB peak allocation. Full 200-step inference passed without an optimizer update. The preflight caught missing inference-only atom-count metadata in the old training cache; deriving it from input tokenization fixed execution without adding native geometry or labels.

Job `/bizon/helico-exp25-contact-overfit-v1` launched October 21 from source `4278cbead7e02ac225c39f1c7c7de7309ce036b4`. See `data/submission.json`, `data/partial-run-status.json`, and `data/storage-probe.json`. The monitor's historical state is `scratch/exp25-monitor-state.json`; its service is inactive after marking the failure as needing attention.

[Training and diagnostic W&B](https://wandb.ai/timodonnell/helico/runs/exp25-contact-overfit-v1) · [Experiment issue](https://github.com/Open-Athena/helico/issues/25).
