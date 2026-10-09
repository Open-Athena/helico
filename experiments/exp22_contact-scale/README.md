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

The inspected `cw-rno2a` cluster configuration identifies its H100 fleet as prepaid and fully warm. This run uses that reservation at batch priority, without provisioning nodes or altering the cluster. The replacement job is bounded to 8 × 102 = 816 reserved GPU-hours including staging; allowing for the failed starts and gradient diagnostic gives a conservative overall ceiling of 832 reserved GPU-hours; incremental GPU rental is zero under that reservation, with the resource usage recorded separately from spend.

The 50 immutable source shards total 1,057,335,306,240 bytes. A CPU-only job caches their verified bytes in object storage, while the training node extracts them on local scratch. Dataset identity remains the Hugging Face revision/checksum, not the cache location. Source cache plus retained checkpoints are estimated below 1,500 GiB; at the [published hot-object-storage price](https://coreweave.com/pricing) of $0.06/GiB/month (checked 2026-10-08), a conservative full-month incremental storage allowance is **$90**. This is the cost-gate estimate for the whole run, including staging and checkpoints. Ongoing storage persists after training; warm/cold tiering may reduce its cost. The repository's Modal reference rate would imply $3,286.40 for 832 H100-hours, but that is not incremental CoreWeave reservation spending.

The launcher records a durable Iris job receipt and returns while training runs. Re-executing this cell adopts its existing receipt and never describes a submitted job as a completed model. Set `HELICO_IRIS_PYTHON` to an environment containing Iris/Fray and `HELICO_IRIS_CONFIG` to the installed CoreWeave cluster config before a real launch.

```python
from helico.experiment import set_experiment, ensure_training_run
set_experiment("exp22_contact-scale")
# Whole-run estimate: $90 first-month incremental storage, prepaid GPU allocation.
spec = dict(
    job_name="helico-exp22-contact-scale-v3", mode="train", gpus=8,
    cpu=96, memory="1000g", disk="3000g",
    image="pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca",
    timeout_seconds=367200, config="configs/train/contact-scale-v1.json",
    output_uri="s3://marin-us-east-02a/helico/runs/exp22-contact-scale-v3",
    estimated_incremental_cost_usd=90,
    cost_accounting="Prepaid cw-rno2a reservation; <=1500 GiB storage at $0.06/GiB for one month",
)
run = ensure_training_run("full-data-v3", gpu="H100:8", max_steps=20000,
    crop_size=384, lr=2e-5, est_wall_hours=102, coreweave=spec)
print(run.meta)
```

## Status

The replacement job is `/bizon/helico-exp22-contact-scale-v3`. Its predecessors failed before any optimizer updates: the first during dependency installation, and v2 (`f311dbe0e70cc63b3c75974adfef10be9fdcef23`) during the first full-data backward pass. The latter exposed a padding bug: coordinate diffusion attended to tokens without atoms, allowing gradients through zero-variance padded pair products. Diffusion now excludes those tokens as keys, and triangle updates mask their padded outputs. Model weights and valid-token architecture are unchanged.

All 32 exact first-batch crops pass individual full-model backward checks after the fix; the formerly failing 2d2h crop has finite gradients. Results are in `data/first_batch_gradient_replay.csv`. Four padding regression checks and the existing data/model/notebook checks pass (74 total). Source staging now uses eight download workers and four checksum/extraction workers, rejects duplicate archive paths, and retains the content-addressed data cache on the worker's designated cache volume. The pinned source inventory contains 931,266 distinct file paths. All 50 source shards are also present in the regional object cache; two stalled uploads were recovered from verified local copies and the CPU-only staging job was stopped.

The real-complex distributed GPU preflight completed two optimizer updates on all eight H100s, with dynamic MSA branches and four accumulated crops per device. Peak allocated GPU memory was 50.27 GB; the warmed-up update took 7.31 seconds on the largest rank. This is a setup/gradient check on one 384-token crop, not evidence of generalization or a full-data throughput measurement. Per-rank measurements are in `data/gpu_preflight.csv`.

A second distributed check used a 384-token RNA/protein crop from 4V9D (5,483 atoms, including 3,930 RNA atoms). All eight ranks completed both updates with finite gradients, at 50.70 GB peak allocation and 7.53 seconds per warmed-up update. Measurements are in `data/gpu_rna_preflight.csv`. The preflight updates are discarded; only the full-data trainer produces the model checkpoints.
