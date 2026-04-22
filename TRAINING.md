# Training

## Training Data

Processed data is hosted on HuggingFace at [`timodonnell/helico-data`](https://huggingface.co/datasets/timodonnell/helico-data) and auto-downloads to `~/.cache/helico/data/` on first use.

```bash
# Download all processed data
helico-download

# Download just the CCD cache (needed for inference)
helico-download --subset ccd-only

# Download to a custom location
helico-download --data-dir /data/helico
```

Override the default data location with the `HELICO_DATA_DIR` env var.

Processed data in `<data-dir>/processed/` (hosted on HuggingFace):

| File | Size | Description |
|------|------|-------------|
| `ccd_cache.pkl` | 112 MB | Pickled CCD (parsed from components.cif) |
| `structures/` | ~236K `.pkl` files | Pickled TokenizedStructures across ~1,000 subdirectories |
| `manifest.json` | 1.9 GB | Metadata for all 236,326 processed structures |
| `rcsb_raw_msa_index.pkl` | 15 MB | Tar index for rcsb_raw_msa.tar (151,403 entries) |
| `openfold_raw_msa_index.pkl` | 11 MB | Tar index for openfold_raw_msa.tar (268,778 entries) |

> **Note**: `token_bonds` in each structure pickle is stored as a sparse edge
> list (`list[tuple[int,int]]`) and densified to `(N_tok, N_tok)` at training
> time in `TokenizedStructure.to_features()`. The previous dense-tensor format
> blew up preprocess workers to 200+ GB RSS on ribosomes/capsids. The loader
> still accepts legacy dense pickles for backward compat.

Raw data (not hosted on HuggingFace — download from PDB/OpenFold directly, see `LOG.md`):

| File | Size | Description |
|------|------|-------------|
| `components.cif` | 473 MB | PDB Chemical Component Dictionary (atom/bond definitions for all ligands) |
| `pdb_seqres.txt.gz` | 60 MB | Sequences for ~254K PDB entries |
| `rcsb_raw_msa.tar` | 131 GB | Pre-computed MSA files (.a3m.gz) from RCSB |
| `openfold_raw_msa.tar` | 88 GB | Pre-computed MSA files from OpenFold |
| `mmCIF/` | 81 GB | PDB structure archive (248,942 `.cif.gz` files across 1,089 subdirectories) |


## Quick Start (Synthetic Data)

Verify the installation with a quick training run on synthetic data:

```bash
# Minimal smoke test (~30 seconds)
helico-train --synthetic --n-blocks 2 --n-diffusion-token-blocks 2 --max-steps 100

# Longer test with full model dimensions
helico-train --synthetic --max-steps 500 --log-every 50
```

## Data Preparation

The easiest path is `helico-download` which fetches everything from HuggingFace. For preprocessing from scratch:

```bash
RAW=/data/helico/raw          # contains components.cif, mmCIF/, MSA tars
PROCESSED=/data/helico/processed   # output directory

# Parse CCD only (generates ccd_cache.pkl from components.cif)
helico-preprocess ccd $RAW $PROCESSED

# Process mmCIF structures into pickles + manifest (also generates ccd_cache.pkl if missing)
helico-preprocess structures $RAW $PROCESSED

# Build tar indices for O(1) MSA lookup
helico-preprocess msa-index --tar-path $RAW/rcsb_raw_msa.tar --output $PROCESSED/rcsb_raw_msa_index.pkl
helico-preprocess msa-index --tar-path $RAW/openfold_raw_msa.tar --output $PROCESSED/openfold_raw_msa_index.pkl

# Or run everything in sequence (ccd + structures + msa indices):
helico-preprocess all $RAW $PROCESSED
```

The preprocessing pipeline:
- Parses ~252K mmCIF files, filters by resolution (≤9 Å), removes water/hydrogen
- Tokenizes: 1 token/residue (protein), 1 token/nucleotide, 1 token/heavy atom (ligand)
- Saves each structure as a pickle file for fast loading during training
- Builds a manifest with metadata for all passing structures (236,326 from the most recent run)
- Indexes MSA tar archives for O(1) random access during training
- Supports resumption — re-run safely without reprocessing existing structures

Default worker count is capped at 16 with `maxtasksperchild=25` — each worker can briefly spike to several GB during large structures (ribosomes, capsids), so 16 workers on 64-CPU boxes keeps a safety margin.

To upload processed data to HuggingFace after preprocessing:

```bash
bash scripts/upload_to_hf.sh $PROCESSED
```

### Preprocess on Modal (bypass local upstream)

If your local upstream is slow or you don't want to keep ~300 GB of raw data around, `modal/preprocess_on_modal.py` downloads raw data directly onto a Modal Volume (S3 for the MSA tars, rsync for the mmCIF mirror) and runs preprocess there. Processed data lands on the `helico-train-data` Volume at `/processed/` and the training script picks it up without any upload step.

```bash
modal run modal/preprocess_on_modal.py

# Re-run with different filters / worker count
HELICO_MAX_RES=9.0 HELICO_N_WORKERS=32 modal run modal/preprocess_on_modal.py

# Download only (skip preprocess)
HELICO_SKIP_PREPROCESS=1 modal run modal/preprocess_on_modal.py

# Preprocess only (skip download, re-use existing /raw)
HELICO_SKIP_DOWNLOAD=1 modal run modal/preprocess_on_modal.py
```

Expected runtime end-to-end: ~3 h (S3/rsync at tens-of-MB/s + ~85 min preprocess on a 32-core container).

## Single-GPU Training

```bash
# Stage 1 only
helico-train --crop-size 384 --lr 1e-3 --max-steps 50000

# With custom checkpoint directory and logging frequency
helico-train \
    --crop-size 384 \
    --lr 1e-3 \
    --max-steps 100000 \
    --warmup-steps 1000 \
    --save-every 5000 \
    --log-every 50 \
    --checkpoint-dir /data/checkpoints/helico

# Resume from checkpoint
helico-train --resume checkpoints/step_50000.pt --max-steps 100000
```

## Single-Node Multi-GPU (1 node, 8 H100s)

```bash
torchrun \
    --standalone \
    --nproc_per_node=8 \
    -m helico.train \
    --distributed \
    --batch-size 1 \
    --crop-size 384 \
    --lr 1e-3 \
    --grad-accum-steps 1 \
    --max-steps 100000 \
    --checkpoint-dir checkpoints/
```

## Multi-Node Training (128 H100s)

For large-scale training across 16 nodes with 8 GPUs each (128 H100s total):

### With `torchrun` (manual launch)

On each node, run the same command with matching `--rdzv-id`:

```bash
torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_id=helico_run \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m helico.train \
    --distributed \
    --batch-size 1 \
    --crop-size 384 \
    --lr 1e-3 \
    --grad-accum-steps 1 \
    --warmup-steps 1000 \
    --max-steps 100000 \
    --save-every 2000 \
    --log-every 10 \
    --num-workers 8 \
    --checkpoint-dir /shared/checkpoints/helico
```

Set `MASTER_ADDR` to the hostname/IP of node 0 and `MASTER_PORT` to a free port (e.g. 29500).

### With SLURM

Example SLURM job script for 128 H100s (16 nodes × 8 GPUs):

```bash
#!/bin/bash
#SBATCH --job-name=helico
#SBATCH --partition=h100
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/helico_%j.out
#SBATCH --error=logs/helico_%j.err
#SBATCH --exclusive

# Environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# NCCL tuning for multi-node H100
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=1

# Data path (use shared filesystem)
export HELICO_DATA_DIR=/shared/data/helico-data

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m helico.train \
    --distributed \
    --batch-size 1 \
    --crop-size 384 \
    --lr 1e-3 \
    --grad-accum-steps 1 \
    --warmup-steps 1000 \
    --max-steps 100000 \
    --save-every 2000 \
    --log-every 10 \
    --num-workers 8 \
    --checkpoint-dir /shared/checkpoints/helico
```

Submit with:

```bash
mkdir -p logs
sbatch train_helico.sh
```

## Multi-Stage Training Schedule

Training uses a 3-stage schedule with increasing crop sizes. The schedule transitions automatically based on the global step count:

| Stage | Crop Size | Learning Rate | Steps | Cumulative |
|-------|-----------|---------------|-------|------------|
| 1 | 384 | 1e-3 | 50,000 | 50,000 |
| 2 | 640 | 5e-4 | 30,000 | 80,000 |
| 3 | 768 | 1e-4 | 20,000 | 100,000 |

The crop size and learning rate switch automatically when the global step crosses stage boundaries. No manual restart is needed — a single `--max-steps 100000` run covers all three stages.

When resuming from a checkpoint, the training loop picks up at the correct stage based on the saved step count.

## Effective Batch Size

The effective batch size is:

```
effective_batch_size = batch_size_per_gpu × num_gpus × grad_accum_steps
```

| Setup | GPUs | `--batch-size` | `--grad-accum-steps` | Effective Batch Size |
|-------|------|----------------|---------------------|---------------------|
| Single GPU | 1 | 1 | 128 | 128 |
| Single node | 8 | 1 | 16 | 128 |
| 4 nodes | 32 | 1 | 4 | 128 |
| 16 nodes | 128 | 1 | 1 | 128 |

With 128 H100s and `--batch-size 1 --grad-accum-steps 1`, the effective batch size is 128 structures per optimizer step. For crop size 384 at bfloat16, each GPU uses ~40 GB of the 80 GB available, leaving headroom for activations.

## Training CLI Reference

```
helico-train [OPTIONS]

Model:
  --n-blocks N              Pairformer blocks (default: 48)
  --n-diffusion-token-blocks N    Diffusion transformer blocks (default: 24)

Optimizer:
  --lr LR                   Base learning rate (default: 1e-3)
  --weight-decay WD         AdamW weight decay (default: 0.01)
  --warmup-steps N          Linear LR warmup steps (default: 1000)
  --grad-clip NORM          Max gradient norm (default: 1.0)
  --grad-accum-steps N      Gradient accumulation steps (default: 1)
  --ema-decay DECAY         EMA decay rate (default: 0.999)

Data:
  --crop-size N             Initial crop size in tokens (default: 384)
  --batch-size N            Per-GPU batch size (default: 1)
  --num-workers N           DataLoader workers per GPU (default: 4)
  --synthetic               Use synthetic data for testing
  --manifest PATH           Path to manifest.json (default: <data-dir>/processed/manifest.json)
  --processed-dir PATH      Path to processed data directory
  --msa-dir PATH            Path to extracted MSA directory

Train/val split (defaults match AF3 / Protenix v1 / OF3-preview2 for direct comparison):
  --train-cutoff DATE       Train on release_date < this date (default: 2021-09-30)
  --val-cutoff-start DATE   Val window lower bound, inclusive (default: 2022-05-01)
  --val-cutoff-end DATE     Val window upper bound, inclusive (default: 2023-01-12)

Training:
  --max-steps N             Total training steps (default: 100000)
  --save-every N            Checkpoint interval (default: 1000)
  --log-every N             Logging interval (default: 10)
  --val-every N             Run validation every N steps (0 disables; default: 0)
  --val-samples N           Val batches per validation pass (default: 32)
  --checkpoint-dir PATH     Checkpoint directory (default: checkpoints/)
  --resume PATH             Resume from checkpoint
  --distributed             Enable DDP (required for multi-GPU)
```

The default dates match the canonical AF3 "Low-Homology Recent PDB" split that
Protenix and OpenFold3-preview2 both reuse. Structures released in the
`2021-09-30 → 2022-05-01` gap are in neither split (AF3's leakage-prevention
design). Using the same cutoffs makes our metrics directly comparable to
their published numbers.

## Training on Modal

`modal/train.py` runs DDP training inside a Modal container with the
`helico-train-data` Volume mounted and the `helico-checkpoints` Volume for
durable checkpoint storage. All settings are exposed via env vars (no CLI
flags on the Modal side):

```bash
# Proof run: 1×H100, 500 steps, crop 256
HELICO_TRAIN_GPU=H100:1 \
HELICO_TRAIN_MAX_STEPS=500 \
HELICO_TRAIN_CROP=256 \
HELICO_TRAIN_RUN_NAME=proof-v1 \
  modal run modal/train.py

# Full fine-tune from Protenix v1 weights: 8×H100
HELICO_TRAIN_GPU=H100:8 \
HELICO_TRAIN_MAX_STEPS=10000 \
HELICO_TRAIN_CROP=384 \
HELICO_TRAIN_RUN_NAME=v1-finetune-01 \
HELICO_TRAIN_VAL_EVERY=500 \
  modal run modal/train.py
```

Env vars (and defaults):

| Variable | Default | Notes |
|----------|---------|-------|
| `HELICO_TRAIN_GPU` | `H100:8` | Modal GPU string `TYPE:COUNT` |
| `HELICO_TRAIN_RUN_NAME` | `helico-run` | W&B run name + checkpoint subdir |
| `HELICO_TRAIN_MAX_STEPS` | `10000` | Hard step cap |
| `HELICO_TRAIN_CROP` | `384` | Token crop size |
| `HELICO_TRAIN_BATCH` | `1` | Per-GPU batch size |
| `HELICO_TRAIN_LR` | `5e-5` | Fine-tune-friendly default |
| `HELICO_TRAIN_WARMUP` | `200` | Linear warmup steps |
| `HELICO_TRAIN_SAVE_EVERY` | `250` | Checkpoint cadence |
| `HELICO_TRAIN_LOG_EVERY` | `10` | wandb log cadence |
| `HELICO_TRAIN_VAL_EVERY` | `0` | 0 disables; e.g. `500` enables val sweeps |
| `HELICO_TRAIN_VAL_SAMPLES` | `32` | Val batches per sweep |
| `HELICO_TRAIN_RESUME` | — | `/ckpts/<run>/step_<N>.pt` to resume |
| `HELICO_TRAIN_PROTENIX_INIT` | `1` | Warm-start from Protenix v1 weights |
| `HELICO_TRAIN_CUTOFF` | `2021-09-30` | Train date upper bound |
| `HELICO_VAL_START` | `2022-05-01` | Val window lower bound |
| `HELICO_VAL_END` | `2023-01-12` | Val window upper bound |
| `HELICO_TRAIN_WANDB_PROJECT` | `helico` | W&B project |

Checkpoints are written to `/ckpts/<run_name>/` on the `helico-checkpoints`
Volume; `protenix_v1_seed.pt` is created on first run when warm-starting.

## W&B Logging

W&B is enabled automatically on Modal runs (the `wandb-credentials` secret
provides `WANDB_API_KEY`). Local runs opt in with `HELICO_WANDB_ENABLE=1`
and the same `WANDB_PROJECT` / `WANDB_RUN_NAME` env vars.

Metrics logged every `log_every` steps (rank 0 only):

| Key | Description |
|-----|-------------|
| `loss` | Total loss — `diffusion_loss + 0.1 * distogram_loss` |
| `loss/diffusion` | Diffusion MSE loss |
| `loss/distogram` | Distogram cross-entropy (when confidence head is active) |
| `train/lddt` | Smooth-LDDT on denoised vs ground-truth atom coords (running mean over the log window) |
| `train/lddt_hard` | Hard LDDT (exact, AF3 definition) — snapshot on most recent batch |
| `train/rmsd` | RMSD (Å) after Kabsch superposition — snapshot |
| `train/gdt_ts` | GDT-TS (1/2/4/8 Å thresholds, Kabsch-aligned) — snapshot |
| `train/plddt` | Mean per-atom pLDDT from confidence head (0-100) — snapshot |
| `grad_norm` | Pre-clip total gradient norm |
| `gpu/peak_mem_gb` | Peak GPU memory in the log window |
| `lr`, `tokens_per_sec`, `stage` | Schedule / throughput |

Validation metrics (when `val_every > 0`, logged at that cadence — averaged over `val_samples` batches):

| Key | Description |
|-----|-------------|
| `val/diffusion_loss` | Diffusion loss |
| `val/distogram_loss` | Distogram loss |
| `val/total_loss` | Weighted total matching the training loss |
| `val/lddt` | Smooth-LDDT on the val split |
| `val/lddt_hard` | Hard LDDT (AF3 definition) — directly comparable to AF3/Protenix/OF3 reports |
| `val/rmsd` | RMSD (Å) after Kabsch |
| `val/gdt_ts` | GDT-TS (Kabsch-aligned, 1/2/4/8 Å) |
| `val/plddt` | Mean per-atom pLDDT (0-100) |

The hard-quality metrics (`lddt_hard`, `rmsd`, `gdt_ts`, `plddt`) live in
`src/helico/eval_metrics.py` as torch-batched `(B, N, 3) → (B,)` ops. They
match the offline numpy implementations in `bench.py` (verified by
regression tests in `tests/test_eval_metrics.py`).

Per-mol-type splits, intra/inter-chain LDDT, DockQ, and TM-score are
**not** logged inline — those are run offline via `helico-bench` /
`modal/bench.py` on saved predictions, and require per-atom
chain/entity-type metadata that's not currently in the training batch.

Val is rank-0 only to avoid DDP gymnastics; other ranks skip the sweep.

## Re-benchmarking After Fine-Tuning

The inline W&B metrics give a fast signal during training, but for
publishable scores against AF3 / Protenix / OF3 we run the offline
FoldBench scorer. We've been iterating against a small **~100-target
subset** for fast turnaround, defined by the `helico-bench` defaults:

- `--cutoff-date 2024-01-01` — keeps targets released after this date
- `--max-tokens 2048` — drops anything larger
- All 9 FoldBench categories included

Net: ~11 targets per category (12 for `interface_antibody_antigen`),
~100 total. Each historical run lives in a `bench_results_*` directory
in the repo root (untracked); examples: `bench_results_v1_modres/`,
`bench_results_v1_ptxmsa/`, `bench_results_v1_rna_frame/`. Each has
`results/<category>.csv` (per-target metrics) and `summary.csv`
(aggregate per-category mean LDDT / mean DockQ / success%).

To rerun after a fine-tune so the numbers are directly comparable:

```bash
# Local (slower; ~30-60 min on a single H100)
helico-bench \
    --checkpoint /ckpts/<run>/final.pt \
    --output-dir bench_results_<run> \
    --foldbench-dir <FOLDBENCH_DIR>

# Modal (fast; fans out across N workers)
HELICO_BENCH_WORKERS=8 modal run modal/bench.py \
    --checkpoint /ckpts/<run>/final.pt \
    --output-dir bench_results_<run>
```

To compare two checkpoints, diff their `summary.csv` row-by-row or load
both `results/<cat>.csv` and pair on `pdb_id`. The full FoldBench leaderboard
(1,522 targets) is reachable by overriding `--cutoff-date 1900-01-01` —
expect ~10× the runtime.
