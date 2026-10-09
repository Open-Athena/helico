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

Use four H100s, 384-token crops, one crop per GPU, eight gradient-accumulation steps (global batch 32), four diffusion samples per crop, one trunk recycle and at most 512 MSA rows. AdamW peaks at 2e-5 for pretrained parameters and 2e-4 for the new contact projection/head after 500 warmup updates and decays to 10% of that rate. Target 20,000 updates (640,000 weighted crop draws), capped at 96 hours of training; the job allows six additional hours for staging. Save optimizer/model/EMA snapshots at step 1, every 50 updates through warmup, and every 250 updates thereafter, with EMA decay 0.999. Resume preserves the data lock and deterministic draw/crop seeds. Online validation uses 32 fixed crops from the unchanged validation membership, reporting contact BCE and denoising losses; these are not FoldBench structure-generation scores. FoldBench remains held out for subsequent structure/search evaluation.

## Resources and provenance

The inspected `cw-rno2a` cluster configuration identifies its H100 fleet as prepaid and fully warm. This run uses that reservation at normal research (`interactive`) priority, without provisioning nodes or altering the cluster. The current four-GPU run is bounded to 4 × 102 = 408 reserved GPU-hours including staging and recovery, retaining its original deadline; the original conservative whole-experiment ceiling of 832 reserved GPU-hours remains an upper bound after allowing for the earlier attempts and diagnostics. Incremental GPU rental is zero under that reservation, with resource usage recorded separately from spend.

The 50 immutable source shards total 1,057,335,306,240 bytes. A CPU-only job cached their verified bytes in object storage, while the training node extracts them on local scratch. Dataset identity remains the Hugging Face revision/checksum, not the cache location. Source cache, retained checkpoints and diagnostics are estimated below 1,600 GiB; at the [published hot-object-storage price](https://coreweave.com/pricing) of $0.06/GiB/month (checked 2026-10-08), a conservative full-month incremental storage allowance is **$96**. This is the cost-gate estimate for the whole run, including staging and checkpoints. Ongoing storage persists after training; warm/cold tiering may reduce its cost. The repository's Modal reference rate would imply $3,286.40 for 832 H100-hours, but that is not incremental CoreWeave reservation spending.

The launcher records a durable Iris job receipt and returns while training runs. Re-executing this cell adopts its existing receipt and never describes a submitted job as a completed model. Set `HELICO_IRIS_PYTHON` to an environment containing Iris/Fray and `HELICO_IRIS_CONFIG` to the installed CoreWeave cluster config before a real launch.

```python
from helico.experiment import set_experiment, ensure_training_run
set_experiment("exp22_contact-scale")
# Whole-run estimate: $96 first-month incremental storage, prepaid GPU allocation.
spec = dict(
    job_name="helico-exp22-contact-scale-v7", mode="train", gpus=4,
    cpu=64, memory="750g", disk="3000g",
    image="pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca",
    timeout_seconds=362400, config="configs/train/contact-scale-v7.json",
    output_uri="s3://marin-us-east-02a/helico/runs/exp22-contact-scale-v7",
    priority_band="interactive",
    estimated_incremental_cost_usd=96,
    cost_accounting="Prepaid cw-rno2a reservation; <=1600 GiB storage at $0.06/GiB for one month",
)
run = ensure_training_run("full-data-v7", gpu="H100:4", max_steps=20000,
    crop_size=384, lr=2e-5, est_wall_hours=362400 / 3600, coreweave=spec)
print(run.meta)
```

## Status

The v3 run failed with rank-0 `SIGABRT` at 2026-10-09 01:32 UTC. Its last logged update was 150 (4,800 crops); only step 1 was durably saved. No validation or FoldBench result was produced. The normal Iris/Finelog endpoints have no retained training log for this attempt, so the underlying abort is not yet established. W&B history is retained in `data/v3_training_metrics.csv`.

The v4 request was capacity-gated before allocation and was cancelled. The v5 recovery uses four H100s with eight accumulation steps, preserving the effective batch of 32 and global draw stream. It restores v3's step-1 model, EMA and optimizer with identical scientific settings and deterministic data draws. It adds per-rank and data-worker progress files, slow-operation stack dumps, a five-minute data-loader timeout, and a supervisor that writes native stdout/stderr and progress files to durable storage every minute. Startup checkpoints are saved every 50 updates through step 500 and regular checkpoints are saved before validation. There are 89 scheduled snapshots plus the final snapshot; the combined source cache, prior snapshot, diagnostics and checkpoints fit a conservative 1,600 GiB / $96 first-month storage allowance. GPUs remain on the existing prepaid reservation. This restart is an instrumented recovery; it is not yet evidence that the original failure is fixed.

The recovery job `/bizon/helico-exp22-contact-scale-v5` was allocated on four H100s from source `287d7d9733679588a30518e4fad1f646dd826140`. It downloaded all 50 source shards, but Kueue preempted it during extraction at 2026-10-09 20:22:43 UTC to admit higher-priority work. Iris automatically rescheduled it; that retry was then cancelled to correct the launcher's hard-coded opportunistic `batch` priority. The v6 replacement uses Iris's documented normal research `interactive` band, with the same scientific configuration and the remaining original deadline. A non-spot resource request does not prevent scheduler priority preemption. The original v3 native abort and this v5 scheduler interruption are distinct events.

A separate debugging pass inside the v5 allocation replayed all 640 crops corresponding to updates 141–160, with full protein/RNA MSAs: CPU preprocessing succeeded for every crop (maximum 29.3 seconds), and all four GPU ranks completed 20 global updates with finite gradients. Peak GPU allocation was 52.30 GB and the median update was 14.64 seconds. These diagnostic updates were discarded; the sustained trainer restores v3's step-1 checkpoint. See `data/failure_window_full_preprocessing.csv` and `data/failure_window_gpu_replay.csv`. The original abort has not reproduced, so a definite underlying cause cannot be assigned from the surviving evidence.

The v6 attempt completed staging and restored step 1, but all 16 forked data workers blocked on their first crop. A live `py-spy` stack identified `SeededCrops.progress` at `faulthandler.dump_traceback_later`: the new diagnostics had started a watchdog thread in the parent before forking. The replacement uses explicit `spawn` workers, so CUDA/NCCL and watchdog thread locks are not inherited. A regression test exercises a real worker with the parent watchdog active. A full-dataset worker check on the actual node then successfully prepared draws 32 (1WBZ) and 36 (7JK3); see `data/spawn_loader_preflight.csv` and `data/v6_loader_deadlock.txt`. This fixes a newly introduced diagnostic deadlock; it is not an explanation of the original v3 abort. The v7 launch from `1a4d655237601b275a50abc7f2e7316c747b6a13` retains the scientific configuration and remaining original deadline.

A persistent local user service, `helico-exp22-watch.service`, checks Iris task state, actual per-rank progress, checkpoints and final artifacts every 570 seconds. Its single-owner record is `scratch/20261009_helico_monitoring_state.json`, with actionable events in `scratch/20261009_helico_monitor.log`. It can resume recognized infrastructure/collective-timeout failures at most three times, requiring checkpoint progress between recoveries and retaining the original time budget. Nonfinite gradients, OOM, data failures, illegal memory access and unexplained aborts stop unattended retries and record an event. Desktop notification delivery is best effort; the notification-server check timed out on this host. Completion requires successful Iris status, the final checkpoint/result and a finished W&B run; reaching the configured time cap is reported separately from reaching the step target. Routine checks stay quiet. The monitor resumes after a local user-service restart; cluster training itself does not depend on this workstation staying connected.

The original v3 job was `/bizon/helico-exp22-contact-scale-v3`, launched from `3fe2242c11c54101b01ec242219dbab15ca0cd4d`. Its predecessors failed before any optimizer updates: the first during dependency installation, and v2 (`f311dbe0e70cc63b3c75974adfef10be9fdcef23`) during the first full-data backward pass. The latter exposed a padding bug: coordinate diffusion attended to tokens without atoms, allowing gradients through zero-variance padded pair products. Diffusion now excludes those tokens as keys, and triangle updates mask their padded outputs. Model weights and valid-token architecture are unchanged.

All 32 exact first-batch crops pass individual full-model backward checks after the fix; the formerly failing 2d2h crop has finite gradients. Results are in `data/first_batch_gradient_replay.csv`. Four padding regression checks and the existing data/model/notebook checks pass (76 total, including the inference-default checks). Source staging now uses eight download workers and four checksum/extraction workers, rejects duplicate archive paths, and retains the content-addressed data cache on the worker's designated cache volume. The pinned source inventory contains 931,266 distinct file paths. All 50 source shards are also present in the regional object cache; two stalled uploads were recovered from verified local copies and the CPU-only staging job was stopped.

The real-complex distributed GPU preflight completed two optimizer updates on all eight H100s, with dynamic MSA branches and four accumulated crops per device. Peak allocated GPU memory was 50.27 GB; the warmed-up update took 7.31 seconds on the largest rank. This is a setup/gradient check on one 384-token crop, not evidence of generalization or a full-data throughput measurement. Per-rank measurements are in `data/gpu_preflight.csv`.

A second distributed check used a 384-token RNA/protein crop from 4V9D (5,483 atoms, including 3,930 RNA atoms). All eight ranks completed both updates with finite gradients, at 50.70 GB peak allocation and 7.53 seconds per warmed-up update. Measurements are in `data/gpu_rna_preflight.csv`. The preflight updates are discarded; only the full-data trainer produces the model checkpoints.

The corrected exact-first-batch distributed check completed two updates on all eight ranks with finite gradients, 50.78 GB peak allocation and 7.37 seconds per warmed-up update. Measurements are in `data/gpu_first_batch_preflight.csv`; these diagnostic updates are also discarded.

For new contact-prediction checkpoints, inference with omitted contact conditioning now supplies the learned all-unknown embedding, matching the training endpoint. This inference default does not change the active trainer, whose batches always provide explicit contact states. Abandoned multipart uploads from the initial cache job were removed after complete replacement objects were verified.
