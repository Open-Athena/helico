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
  issue: 22
  title: Full-data masked-contact fine-tuning on CoreWeave
  branch: codex/clean-contact-pilot
---

# Full-data masked-contact fine-tuning on CoreWeave

Train one full-size Helico model initialized from Protenix base default v1.0.0 on the pinned Protenix-v1 PDB split. This is a sustained fine-tune with 167,997 training entries and 1,839,712 weighted chain/interface records; no fine-tuning control arm is launched.

The upstream Protenix feature pipeline preserves atom tokenization, CCD conformers, unresolved-atom masks, paired/unpaired protein MSAs and RNA MSAs. Its source is pinned to `85767b811c40ed46e73a9b39519cf6bfca8701ba`. Real template retrieval is explicitly disabled because Helico currently implements the checkpoint's dummy-template path. Confidence and dummy-template weights remain frozen; input embedding, MSA, Pairformer, coordinate diffusion, distogram and contact parameters train.

Contacts mean minimum observed heavy-atom distance strictly below 5 Å. An observed close atom pair establishes a positive; negatives require complete heavy-atom coordinates in both tokens. Supervision includes protein-involving pairs, excluding within-chain protein pairs with residue-index separation below six and the diagonal. Missing observations never become negative interactions. This release has no curated experimental nonbinding examples.

The absorbing-mask process and endpoint/sparse time mixture are unchanged from exp20, with no positive-biased revelation. MSA input, including profile/deletion information, is dropped independently for 20% of crops. Contact input is added to pair initialization and a symmetric contact head predicts clean labels. Train with upstream aligned molecule-weighted MSE and sparse smooth lDDT, using `4 × (MSE + smooth_lDDT) + 0.03 × distogram + 0.1 × masked_contact_BCE + 0.01 × visible_contact_BCE`. The time mixture defines a denoising surrogate rather than an unbiased diffusion ELBO. Coordinate labels retain their supplied chain/atom identity; permutation-aware confidence retraining is not included in this first run.

Use eight H100s, 384-token crops, one crop per GPU, four gradient-accumulation steps (global batch 32), four diffusion samples per crop, one trunk recycle and at most 512 MSA rows. AdamW peaks at 2e-5 for pretrained parameters and 2e-4 for the new contact projection/head after 500 warmup updates and decays to 10% of that rate. Target 20,000 updates (640,000 weighted crop draws), capped at 96 hours of training; the job allows six additional hours for staging. Save optimizer/model/EMA snapshots at step 1 and every 250 updates, with EMA decay 0.999. Resume preserves the data lock and deterministic draw/crop seeds. Online validation uses 32 fixed crops from the unchanged validation membership, reporting contact BCE and denoising losses; these are not FoldBench structure-generation scores. FoldBench remains held out for subsequent structure/search evaluation.

## Resources and provenance

The inspected `cw-rno2a` cluster configuration identifies its H100 fleet as prepaid and fully warm. This run uses that reservation at batch priority, without provisioning nodes or altering the cluster. GPU allocation is bounded to 8 × 102 = 816 reserved GPU-hours including staging; incremental GPU rental is zero under that reservation, with the resource usage recorded separately from spend.

The 50 immutable source shards total 1,057,335,306,240 bytes. A CPU-only job caches their verified bytes in object storage, while the training node extracts them on local scratch. Dataset identity remains the Hugging Face revision/checksum, not the cache location. Source cache plus retained checkpoints are estimated below 1,500 GiB; at the [published hot-object-storage price](https://coreweave.com/pricing) of $0.06/GiB/month (checked 2026-10-08), a conservative full-month incremental storage allowance is **$90**. This is the cost-gate estimate for the whole run, including staging and checkpoints. Ongoing storage persists after training; warm/cold tiering may reduce its cost. The repository's Modal reference rate would imply $3,223.20 for 816 H100-hours, but that is not incremental CoreWeave reservation spending.

The launcher records a durable Iris job receipt and returns while training runs. Re-executing this cell adopts its existing receipt and never describes a submitted job as a completed model. Set `HELICO_IRIS_PYTHON` to an environment containing Iris/Fray and `HELICO_IRIS_CONFIG` to the installed CoreWeave cluster config before a real launch.

```python
from helico.experiment import set_experiment, ensure_training_run
set_experiment("exp22_contact-scale")
# Whole-run estimate: $90 first-month incremental storage, prepaid GPU allocation.
spec = dict(
    job_name="helico-exp22-contact-scale-v1", mode="train", gpus=8,
    cpu=96, memory="1000g", disk="3000g",
    image="pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime",
    timeout_seconds=367200, config="configs/train/contact-scale-v1.json",
    output_uri="s3://marin-us-east-02a/helico/runs/exp22-contact-scale-v1",
    estimated_incremental_cost_usd=90,
    cost_accounting="Prepaid cw-rno2a reservation; <=1500 GiB storage at $0.06/GiB for one month",
)
run = ensure_training_run("full-data-v1", gpu="H100:8", max_steps=20000,
    crop_size=384, lr=2e-5, est_wall_hours=102, coreweave=spec)
print(run.meta)
```

## Status

Preparation underway. CPU feature checks on real antibody/protein and protein/heme complexes preserve paired/unpaired MSAs and unresolved atoms. Contact validity, padding, independent diffusion-sample weights and resumed distributed draw order have regression tests. Training results and job receipts will be recorded here after dispatch and validation.
