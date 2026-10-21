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

## Status

The cost dry run estimates $8 additional first-month storage, within the whole-study cap. The 57 focused tests pass, including final raw/EMA diagnostic completion checks. A full 368,537,859-parameter preflight restored the exact parent on real training crop 2fsx (148 tokens): compliance-only loss 0.005082, contact-input projection gradient norm 1.0405, finite nonzero diffusion-module gradients, and 7.85 GB peak GPU allocation. No optimizer update was performed. Full 200-step structure sampling also completed with finite coordinates; see `data/full-model-preflight.json`.

The preflight caught missing inference-only atom-count metadata in the older training cache. `FixedCrops` now derives `atoms_per_token` from input tokenization, matching the standard evaluation preparation; it adds no geometry or labels to inference. The second full-model preflight passed. This verifies execution and a nonzero structural gradient path, not useful conditional behavior.
