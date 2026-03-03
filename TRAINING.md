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
| `structures/` | ~233K `.pkl` files | Pickled TokenizedStructures across 1,085 subdirectories |
| `manifest.json` | 1.5 GB | Metadata for all 233,215 processed structures |
| `rcsb_raw_msa_index.pkl` | 15 MB | Tar index for rcsb_raw_msa.tar (151,403 entries) |
| `openfold_raw_msa_index.pkl` | 11 MB | Tar index for openfold_raw_msa.tar (268,778 entries) |

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
- Parses 248,942 mmCIF files, filters by resolution (≤9 Å), removes water/hydrogen
- Tokenizes: 1 token/residue (protein), 1 token/nucleotide, 1 token/heavy atom (ligand)
- Saves each structure as a pickle file for fast loading during training
- Builds a manifest with metadata for all 233,215 passing structures
- Indexes MSA tar archives for O(1) random access during training
- Supports resumption — re-run safely without reprocessing existing structures

To upload processed data to HuggingFace after preprocessing:

```bash
bash scripts/upload_to_hf.sh $PROCESSED
```

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
  --val-date-cutoff DATE    Date cutoff for train/val split (default: 2022-01-01)
  --msa-dir PATH            Path to extracted MSA directory

Training:
  --max-steps N             Total training steps (default: 100000)
  --save-every N            Checkpoint interval (default: 1000)
  --log-every N             Logging interval (default: 10)
  --checkpoint-dir PATH     Checkpoint directory (default: checkpoints/)
  --resume PATH             Resume from checkpoint
  --distributed             Enable DDP (required for multi-GPU)
```
