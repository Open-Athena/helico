# Helico
Our goal is to enable robust experimentation around modeling and data improvements for AlphaFold3-like architectures.

## Key ideas

Open source [AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w) clones have so far kept the AF3 architecture remarkably intact with mostly small tweaks and limited ablations.

If we are to match or eventually exceed more recent proprietary models like [IsoDDE](https://www.isomorphiclabs.com/articles/the-isomorphic-labs-drug-design-engine-unlocks-a-new-frontier), we need organized efforts to find better architectures. Helico is an opinionated take on how to do this. We prioritize:

**Open development**. We want to capture and share with the community not only the final best-performing model, but also the incremental and failed experiments that got us there, in real time.

**Automated workflows**. We want to configure compute environments (e.g. Lambda Labs, or AWS) with code that lives in this codebase. It should be possible for anyone to kick off training and evals on any supported compute environment that they have access to.

**Everything lives on github, wandb, or huggingface**. The source of truth on datasets, code, checkpoints, an so on is on public services not private filesystems (or in people's brains). For example, it should be possible for anyone to tell exactly what dataset and code was used to train a given checkpoint.

**Agentic coding**.
We aim for a low-abstraction codebase that is easy for agents to work with. Tests are prioritized over code. It should be possible for agents to autonomously run experiments and analyze the results. We try to document in this repo everything an agent has done wrong so it doesn't do it in the future. We also need to have good guardrails in place to monitor compute usage and data transfer costs.


## Project status

We are just getting started. Our initial implementation closely follows [Protenix](https://github.com/bytedance/Protenix) and our model can load protenix weights. Before we do expensive training runs from scratch we are planning to iterate on modeling improvements starting from these weights.


## Architecture

Helico implements the full AlphaFold3 architecture:

- **Pairformer trunk** (48 blocks) — iteratively refines single-representation and pair-representation tensors using triangle multiplicative updates, triangle attention, and single attention with pair bias.
- **Diffusion module** (24 blocks) — predicts 3D atom coordinates via iterative denoising, conditioned on the Pairformer output.
- **Confidence head** — predicts per-residue pLDDT, pTM, and predicted aligned error (PAE).
- **Affinity module** (Boltz2 extension) — architecture for binding affinity prediction exists in the model but has no training data pipeline or loss wiring yet.

All triangle operations and attention-with-pair-bias use [NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance) fused CUDA/Triton kernels directly. There are no PyTorch-only fallback code paths. Target GPUs are H100 and B200.

## Project Structure

```
src/helico/
    __init__.py               Package entry (exports Helico, HelicoConfig)
    model.py                  All neural network modules in a single file
    data.py                   Data pipeline (CCD, mmCIF, tokenizer, MSA, cropping)
    train.py                  Training loop, DDP, checkpointing, inference
  tests/
    test_data.py              Integration tests for the data pipeline
    test_model.py             Integration tests for all model components
```

### Source Modules

**`model.py`** contains the entire model:

| Component | Class | Description |
|-----------|-------|-------------|
| Config | `HelicoConfig` | All model hyperparameters (dims, block counts, heads) |
| Input embedder | `InputFeatureEmbedder` | Atom-level feature encoding aggregated to token representations |
| Triangle ops | `TriangleMultiplicativeUpdate` | Fused cuEquivariance kernel (outgoing/incoming) |
| | `TriangleAttention` | Fused cuEquivariance kernel (starting/ending) |
| | `SingleAttentionWithPairBias` | Fused cuEquivariance `attention_pair_bias` kernel |
| Pairformer | `PairformerBlock`, `Pairformer` | 48-block trunk with gradient checkpointing |
| MSA | `MSAModule` | MSA processing: outer product mean, pair-weighted averaging, pair stack |
| Diffusion | `DiffusionConditioning` | Noise conditioning with Fourier embeddings and RelPE |
| | `AtomAttentionEncoder` / `AtomAttentionDecoder` | Windowed atom attention with atom-to-token aggregation |
| | `DiffusionTransformerBlock`, `DiffusionModule` | 24-block denoising transformer with pair bias |
| Losses | `diffusion_loss`, `smooth_lddt_loss`, `distogram_loss`, `violation_loss` | All training losses |
| Confidence | `ConfidenceHead` | Predicts pLDDT, pTM, PAE, distogram |
| Affinity | `AffinityModule` | Binding affinity prediction architecture (not yet wired for training) |
| Full model | `Helico` | Combines all components; `forward()` for training, `predict()` for inference |

**`data.py`** contains the full data pipeline:

| Component | Function/Class | Description |
|-----------|---------------|-------------|
| CCD parser | `parse_ccd()` | Parses the 473 MB Chemical Component Dictionary into a lookup table |
| mmCIF parser | `parse_mmcif()` | Parses PDB structures (including `.cif.gz`) with filtering (resolution, water, hydrogen) |
| Tokenizer | `tokenize_structure()` | Proteins: 1 token/residue, nucleotides: 1 token/nt, ligands: 1 token/heavy atom |
| Sequence tokenizer | `tokenize_sequences()` | Tokenizes from sequence strings (no 3D coords needed), uses CCD ideal coords for ref_coords |
| Input parsing | `parse_sequences_arg()`, `parse_input_yaml()` | Parse `--sequences` CLI arg or Boltz2-style YAML input into chain dicts |
| MSA | `parse_a3m()`, `compute_msa_features()`, `load_msa_for_chain()` | A3M parsing, MSA profile computation, sequence clustering, tar-based MSA lookup |
| Tar indexing | `TarIndex`, `build_tar_index()` | O(1) random access into MSA tar archives via offset index |
| Cropping | `spatial_crop()`, `contiguous_crop()` | Interface-biased spatial and contiguous cropping |
| Preprocessing | `preprocess_structures()`, `build_manifest()` | Multiprocessing mmCIF-to-pickle pipeline with resumability |
| DataLoader | `HelicoDataset`, `LazyHelicoDataset`, `collate_fn` | In-memory and lazy-loading PyTorch Datasets with padding-aware collation |
| Synthetic data | `make_synthetic_structure()`, `make_synthetic_batch()` | Test data generation |
| CLI | `preprocess_main()` | `helico-preprocess` entry point (structures, msa-index, all) |

**`train.py`** contains:

| Component | Description |
|-----------|-------------|
| `TrainConfig` | Training hyperparameters (LR, schedule, crop size, DDP settings) |
| `StageConfig` | Multi-stage schedule (384 -> 640 -> 768 crop sizes) |
| `EMAModel` | Exponential moving average of model weights |
| `train()` | Main training loop with DDP, gradient accumulation, mixed precision; accepts synthetic or real datasets |
| `run_inference()` | Multi-sample inference ranked by pLDDT |
| `coords_to_pdb()` | Convert predicted coordinates to PDB format |

## Setup

Requires Python 3.10+ and an H100 or B200 GPU.

```bash
# Clone
git clone https://github.com/Open-Athena/helico.git
cd helico

# Create environment and install
uv venv --python 3.10
uv pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch>=2.7` | PyTorch with CUDA support |
| `cuequivariance-torch>=0.8` | Fused CUDA kernels for triangle ops and attention |
| `cuequivariance-ops-torch-cu12>=0.8` | cuEquivariance CUDA/Triton operator backends |
| `biopython>=1.80` | mmCIF parsing |
| `numpy`, `scipy` | Numerical operations |
| `pyyaml>=6.0` | YAML input parsing for inference |
| `pytest>=7` | Testing (dev dependency) |

## Training Data

Data paths are configured via environment variables (required):

```bash
export HELICO_RAW_DIR=/path/to/raw
export HELICO_PROCESSED_DIR=/path/to/processed
```

Raw data in `$HELICO_RAW_DIR`:

| File | Size | Description |
|------|------|-------------|
| `components.cif` | 473 MB | PDB Chemical Component Dictionary (atom/bond definitions for all ligands) |
| `pdb_seqres.txt.gz` | 60 MB | Sequences for ~254K PDB entries |
| `rcsb_raw_msa.tar` | 131 GB | Pre-computed MSA files (.a3m.gz) from RCSB |
| `openfold_raw_msa.tar` | 88 GB | Pre-computed MSA files from OpenFold |
| `mmCIF/` | 81 GB | PDB structure archive (248,942 `.cif.gz` files across 1,089 subdirectories) |

Processed data in `$HELICO_PROCESSED_DIR`:

| File | Size | Description |
|------|------|-------------|
| `ccd_cache.pkl` | 112 MB | Pickled CCD (parsed from components.cif) |
| `structures/` | ~233K `.pkl` files | Pickled TokenizedStructures across 1,085 subdirectories |
| `manifest.json` | 1.5 GB | Metadata for all 233,215 processed structures |
| `rcsb_raw_msa_index.pkl` | 15 MB | Tar index for rcsb_raw_msa.tar (151,403 entries) |
| `openfold_raw_msa_index.pkl` | 11 MB | Tar index for openfold_raw_msa.tar (268,778 entries) |

See `LOG.md` for download commands and preprocessing details.

## Training

### Quick Start (Synthetic Data)

Verify the installation with a quick training run on synthetic data:

```bash
# Minimal smoke test (~30 seconds)
helico-train --synthetic --n-blocks 2 --n-diffusion-token-blocks 2 --max-steps 100

# Longer test with full model dimensions
helico-train --synthetic --max-steps 500 --log-every 50
```

### Data Preparation

Before training on real data, download and preprocess the PDB structures. See `LOG.md` for download commands.

After downloading, run the preprocessing pipeline:

```bash
export HELICO_RAW_DIR=/path/to/raw
export HELICO_PROCESSED_DIR=/path/to/processed

# Process all mmCIF files into pickled TokenizedStructures + manifest
helico-preprocess structures

# Build tar indices for O(1) MSA lookup
helico-preprocess msa-index --tar-path $HELICO_RAW_DIR/rcsb_raw_msa.tar --output $HELICO_PROCESSED_DIR/rcsb_raw_msa_index.pkl
helico-preprocess msa-index --tar-path $HELICO_RAW_DIR/openfold_raw_msa.tar --output $HELICO_PROCESSED_DIR/openfold_raw_msa_index.pkl

# Or run everything in sequence:
helico-preprocess all
```

The preprocessing pipeline:
- Parses 248,942 mmCIF files, filters by resolution (≤9 Å), removes water/hydrogen
- Tokenizes: 1 token/residue (protein), 1 token/nucleotide, 1 token/heavy atom (ligand)
- Saves each structure as a pickle file for fast loading during training
- Builds a manifest with metadata for all 233,215 passing structures
- Indexes MSA tar archives for O(1) random access during training
- Supports resumption — re-run safely without reprocessing existing structures

### Single-GPU Training

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

### Single-Node Multi-GPU (1 node, 8 H100s)

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

### Multi-Node Training (128 H100s)

For large-scale training across 16 nodes with 8 GPUs each (128 H100s total):

#### With `torchrun` (manual launch)

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

#### With SLURM

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

# Data paths (use shared filesystem)
export HELICO_RAW_DIR=/shared/data/helico-data/raw
export HELICO_PROCESSED_DIR=/shared/data/helico-data/processed

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

### Multi-Stage Training Schedule

Training uses a 3-stage schedule with increasing crop sizes. The schedule transitions automatically based on the global step count:

| Stage | Crop Size | Learning Rate | Steps | Cumulative |
|-------|-----------|---------------|-------|------------|
| 1 | 384 | 1e-3 | 50,000 | 50,000 |
| 2 | 640 | 5e-4 | 30,000 | 80,000 |
| 3 | 768 | 1e-4 | 20,000 | 100,000 |

The crop size and learning rate switch automatically when the global step crosses stage boundaries. No manual restart is needed — a single `--max-steps 100000` run covers all three stages.

When resuming from a checkpoint, the training loop picks up at the correct stage based on the saved step count.

### Effective Batch Size

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

### Training CLI Reference

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
  --manifest PATH           Path to manifest.json (default: $HELICO_PROCESSED_DIR/manifest.json)
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

### Inference

Helico supports three input modes for inference: protein sequences, YAML input files, and mmCIF structures.

#### From Protein Sequences

Predict a structure directly from one-letter amino acid sequences:

```bash
# Single chain
helico-infer --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --sequences "A:MKFLILFNIFTG" --output pred.pdb

# Multi-chain complex (homodimer)
helico-infer --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --sequences "A:MKFLILFNIFTG,B:MKFLILFNIFTG" --output pred.pdb

# With explicit CCD cache path
helico-infer --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --sequences "A:MKFLILFNIFTG" --output pred.pdb \
    --ccd /path/to/ccd_cache.pkl
```

#### From YAML Input (Protein, RNA, DNA, Ligands)

For inputs with mixed chain types (RNA, DNA, ligands), use a Boltz2-style YAML file:

```yaml
# input.yaml
sequences:
  - protein: {id: A, sequence: MKFLILFNIFTG}
  - rna: {id: B, sequence: AUGCCU}
  - ligand: {id: C, ccd: ATP}
```

```bash
helico-infer --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --yaml input.yaml --output pred.pdb
```

#### From mmCIF Structure

Re-predict coordinates for an existing structure (e.g., for benchmarking):

```bash
helico-infer --checkpoint checkpoints/final.pt \
    --input structure.cif --output pred.pdb --n-samples 5
```

When using `--input`, CCD is loaded automatically so that reference coordinates (ref_coords) are populated from ideal coordinates.

#### Inference CLI Reference

```
helico-infer [OPTIONS]

Input (at least one required):
  --sequences STR         Comma-separated chain:seq pairs, e.g. "A:MKFLILF,B:ACDEF"
  --yaml PATH             Path to YAML input file (Boltz2-style, supports protein/RNA/DNA/ligand)
  --input PATH            Path to input mmCIF file

Model (one required):
  --checkpoint PATH       Path to Helico checkpoint
  --protenix PATH         Path to Protenix checkpoint (.pt)

Options:
  --output PATH           Output PDB file (default: output.pdb)
  --n-samples N           Number of diffusion samples, best by pLDDT is kept (default: 5)
  --ccd PATH              Path to CCD cache pickle (default: uses $HELICO_PROCESSED_DIR/ccd_cache.pkl)
```

Generates N structure samples and selects the one with the highest mean pLDDT. Outputs per-atom pLDDT scores in the B-factor column of the PDB file.

### Python API

```python
from helico import Helico, HelicoConfig
from helico.data import make_synthetic_batch

# Create model with custom config
config = HelicoConfig(
    n_pairformer_blocks=4,
    n_diffusion_token_blocks=4,
    d_single=384,
    d_pair=128,
)
model = Helico(config).cuda()

# Training forward pass
batch = make_synthetic_batch(n_tokens=64, device="cuda")
outputs = model(batch)
loss = outputs["diffusion_loss"]

# Inference
results = model.predict(batch, n_samples=5)
coords = results["coords"]      # (B, N_atoms, 3)
plddt = results["plddt"]        # (B, N_tokens) in [0, 1]
ptm = results["ptm"]            # (B,) in [0, 1]
```

## Tests

All tests are full integration tests (no mocks or stubs) that run on GPU with bfloat16 precision.

```bash
# Run all tests
uv run pytest

# Run fast tests only (skip CCD parsing and seqres loading)
uv run pytest -k "not CCD and not Seqres"

# Run model tests only
uv run pytest tests/test_model.py -v

# Run data pipeline tests only
uv run pytest tests/test_data.py -v
```

### Test Coverage

**Data pipeline (42 tests):**
- CCD parser: parses ALA, GLY, ATP, HEM; validates atom counts, bonds, ideal coordinates
- Tokenizer: protein, multi-chain, ligand tokenization; feature tensor shapes
- Sequence tokenizer: protein/ligand from sequences, multi-chain entity IDs, to_features round-trip, input parsing (CLI args and YAML)
- MSA: A3M parsing, matrix conversion, deletion counting, profile computation
- Cropping: spatial and contiguous cropping; atom-token consistency
- Dataset/DataLoader: batching, collation, padding
- Preprocessing: mmCIF parsing (.cif.gz), structure discovery, process+pickle round-trip, manifest I/O, lazy dataset loading, tar index build/save/load

**Model (68 tests):**
- Input feature embedder: output shapes, gradient flow
- Triangle ops: `triangle_multiplicative_update` (outgoing/incoming), `triangle_attention` (starting/ending); shape and gradient checks
- Single attention with pair bias: shape verification via `attention_pair_bias` kernel
- Pairformer: block and stack shapes, residual connection scale stability, pair-only mode
- MSA module: outer product mean, pair-weighted averaging, block structure, gradient flow
- Diffusion primitives: adaptive layer norm, Fourier embedding, conditioned transition, attention pair bias (global, cross, windowed modes)
- Atom attention: encoder output shapes (with and without coords), decoder output shape
- Diffusion conditioning and module: training forward, loss scalar, inference shapes
- Losses: zero loss for perfect predictions, positive loss for random, distogram, violations
- Template embedder, distogram head: output shapes, symmetry
- Confidence head: pLDDT/pTM/PAE/ipTM shapes, ranges, perfect-prediction checks, ranking score
- Affinity module: classification/regression output shapes (architecture only, no training data)
- Full model: forward pass, gradient flow
- End-to-end: training loss decreases over 50 steps; synthetic structure through full pipeline; predict returns all confidence scores
- Protenix weight transfer: transfer stats, weight verification, GPU forward pass with loaded weights

## cuEquivariance Kernels

Helico uses three cuEquivariance fused kernels:

| Kernel | Used In | What It Fuses |
|--------|---------|---------------|
| `cuet.triangle_multiplicative_update()` | `TriangleMultiplicativeUpdate` | LayerNorm + projection + gating + triangle einsum + LayerNorm + projection + gating |
| `cuet.triangle_attention()` | `TriangleAttention` | Batched attention over triangle edges with bias and masking |
| `cuet.attention_pair_bias()` | `SingleAttentionWithPairBias` | LayerNorm on pair repr + pair bias projection + scaled dot-product attention + sigmoid gating + output projection |

Constraints for these kernels:
- Pair representation dimension must be a multiple of 32 (for `triangle_multiplicative_update`)
- Attention head dimension must be divisible by 8 for bf16, by 4 for fp32 (for `triangle_attention`)
- B200 (Blackwell) kernels require sequence length to be a multiple of 8

## References

- **AlphaFold3**: [paper](https://www.nature.com/articles/s41586-024-07487-w) / [code](https://github.com/google-deepmind/alphafold3)
- **Boltz2**: [paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full) / [code](https://github.com/jwohlwend/boltz)
- **OpenFold3**: [whitepaper](https://github.com/aqlaboratory/openfold-3/blob/main/assets/whitepaper.pdf) / [code](https://github.com/aqlaboratory/openfold-3) / [docs](https://openfold-3.readthedocs.io/en/latest/)
- **cuEquivariance**: [code](https://github.com/NVIDIA/cuEquivariance) / [docs](https://docs.nvidia.com/cuda/cuequivariance/)
- **Protenix**: [code](https://github.com/bytedance/Protenix) / [paper](https://github.com/bytedance/Protenix/blob/main/docs/PTX_V1_Technical_Report_202602042356.pdf)

## License

Apache 2.0
