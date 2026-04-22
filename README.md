# Helico

[![GPU Tests](https://github.com/Open-Athena/helico/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Open-Athena/helico/actions/workflows/test.yml)
[![W&B](https://img.shields.io/badge/W%26B-timodonnell%2Fhelico-FFBE00?logo=weightsandbiases&logoColor=white)](https://wandb.ai/timodonnell/helico)
[![HF Dataset](https://img.shields.io/badge/🤗%20dataset-helico--data-yellow)](https://huggingface.co/datasets/timodonnell/helico-data)

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

Current state:
- Model, data pipeline, FoldBench benchmarking, and training loop all working
- Training data preprocessed from the 2026-04 PDB snapshot (236,326 structures)
- Modal infrastructure for preprocess (`modal/preprocess_on_modal.py`) and multi-GPU DDP training (`modal/train.py`)
- Proof-of-pipeline run on 1×H100 succeeded end-to-end — see `TRAINING.md` for full training usage


## Setup

Requires Python 3.10+ and a GPU.

```bash
# Clone
git clone https://github.com/Open-Athena/helico.git
cd helico

# Create environment and install
uv venv --python 3.10
uv pip install -e ".[dev]"
```

To run inference, download the pretrained Protenix checkpoint into `checkpoints/`:

```bash
mkdir -p checkpoints
wget -P checkpoints/ https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt
```

This is the recommended 368M-parameter base model (v1.0.0). Other available checkpoints:

| Model | Params | URL |
|-------|--------|-----|
| `protenix_base_20250630_v1.0.0` | 368M | [download](https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_20250630_v1.0.0.pt) |

## Tests

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

## Training

See [`TRAINING.md`](TRAINING.md) for the full guide: data preparation (local
or on Modal), single / multi-GPU / multi-node recipes, the default AF3-shared
train/val date cutoffs, and the W&B metrics that are logged.

Quick smoke test (synthetic data, ~30s):

```bash
helico-train --synthetic --n-blocks 2 --n-diffusion-token-blocks 2 --max-steps 100
```

Proof run on Modal (1×H100, 500 steps, warm-start from Protenix v1):

```bash
HELICO_TRAIN_GPU=H100:1 HELICO_TRAIN_MAX_STEPS=500 HELICO_TRAIN_CROP=256 \
    HELICO_TRAIN_RUN_NAME=proof-v1 modal run modal/train.py
```

## Inference

Helico supports three input modes for inference: protein sequences, YAML input files, and mmCIF structures.

### From Protein Sequences

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

### From YAML Input (Protein, RNA, DNA, Ligands)

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

### From mmCIF Structure

Re-predict coordinates for an existing structure (e.g., for benchmarking):

```bash
helico-infer --checkpoint checkpoints/final.pt \
    --input structure.cif --output pred.pdb --n-samples 5
```

When using `--input`, CCD is loaded automatically so that reference coordinates (ref_coords) are populated from ideal coordinates.

#### With MSA (recommended)

For significantly better predictions, use the `--use-msa-server` flag to query the public [ColabFold MMseqs2 server](https://api.colabfold.com) for evolutionary information:

```bash
helico-infer --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --sequences "A:MKFLILFNIFTG" --output pred.pdb \
    --use-msa-server

# With a custom MSA server URL
helico-infer --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --sequences "A:MKFLILFNIFTG" --output pred.pdb \
    --use-msa-server --msa-server-url https://your-server.com

# MSA results are cached automatically; re-runs skip the server query
```

This requires the `requests` package (`pip install requests`). MSA results are cached in `<output>.msa_cache/` by default.

### Inference CLI Reference

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
  --ccd PATH              Path to CCD cache pickle (auto-downloads from HuggingFace if not found)
  --use-msa-server        Generate MSA using the public ColabFold MMseqs2 server
  --msa-server-url URL    MMseqs2 server URL (default: https://api.colabfold.com)
  --msa-cache-dir PATH    Directory to cache MSA results (default: <output>.msa_cache)
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

## Benchmarking (FoldBench)

`helico-bench` evaluates prediction accuracy against ground truth structures from the [FoldBench](https://github.com/BEAM-Labs/FoldBench) benchmark (1,522 biological assemblies across 9 categories). Results are directly comparable to the FoldBench leaderboard (AlphaFold3, Boltz-2, Protenix, etc.).

### Install Benchmark Dependencies

```bash
uv pip install -e ".[bench]"
```

This adds `tmtools` (TM-score), `DockQ` (interface scoring), and `tqdm`.

### FoldBench Data

FoldBench data (target CSVs, ground truth structures, AF3 inputs, and pre-computed MSAs) is hosted on HuggingFace at [`timodonnell/helico-data`](https://huggingface.co/datasets/timodonnell/helico-data/tree/main/benchmarks/FoldBench) and **auto-downloads** on first run. No manual setup is needed — MSAs are used automatically.

Data is cached at `~/.cache/helico/data/benchmarks/FoldBench/` (or `$HELICO_DATA_DIR/benchmarks/FoldBench/`).

### Running the Benchmark

```bash
# Run on all categories with Protenix weights (data auto-downloads)
helico-bench \
    --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --output-dir bench_results/

# Run a single category
helico-bench \
    --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --output-dir bench_results/ \
    --categories monomer_protein

# Run multiple specific categories
helico-bench \
    --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --output-dir bench_results/ \
    --categories monomer_protein,interface_protein_ligand

# With a Helico checkpoint instead of Protenix
helico-bench \
    --checkpoint checkpoints/final.pt \
    --output-dir bench_results/

# Resume a partially completed run (reuses cached predictions)
helico-bench \
    --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --output-dir bench_results/ \
    --resume

# Use a local FoldBench directory instead of auto-download
helico-bench \
    --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --foldbench-dir /path/to/FoldBench \
    --output-dir bench_results/
```

### Input Sources

For each target, `helico-bench` tries two input sources in order:

1. **AF3-style JSON** (`examples/alphafold3_inputs.json`) — used if the target name matches an entry in the JSON. Supports protein, DNA, RNA, ligands, and modifications.
2. **Ground truth CIF fallback** — extracts chain sequences directly from the ground truth CIF file. This works for all targets but only captures sequences (no ligand CCD codes or modifications).

### Metrics

| Metric | Method | Categories |
|--------|--------|------------|
| **LDDT** | Hard thresholds at 0.5/1/2/4 Å, 15 Å distance cutoff | All |
| **TM-score** | `tmtools` (C-alpha atoms, wraps TMalign) | Monomers |
| **GDT-TS** | Fraction within 1/2/4/8 Å after Kabsch superposition | Monomers |
| **RMSD** | Kabsch superposition via `scipy.spatial.transform.Rotation` | Monomers |
| **DockQ** | `DockQ` package on predicted PDB vs native CIF | Interfaces |
| **LDDT-PLI** | LDDT restricted to protein-ligand cross-boundary pairs | Protein-ligand |
| **LRMSD** | Ligand RMSD after superposing on receptor atoms | Protein-ligand |

Success criteria: DockQ >= 0.23 (interfaces), LRMSD < 2 Å AND LDDT-PLI > 0.8 (protein-ligand).

### Output

Results are saved to `--output-dir`:

```
bench_results/
  results/
    monomer_protein.csv           # Per-target metrics for each category
    interface_protein_ligand.csv
    ...
  predictions/                    # Cached prediction pickles (for --resume)
    5sbj-assembly1.pkl
    ...
  summary.csv                     # Aggregate metrics across all categories
```

A summary table is also printed to stdout:

```
==========================================================================================
Category                            |    N | Predicted | Success% | Mean LDDT | Mean DockQ
------------------------------------------------------------------------------------------
monomer_protein                     |  334 |       330 |        - |      0.85 |          -
interface_protein_protein           |  279 |       275 |    45.1% |      0.72 |       0.38
interface_protein_ligand            |  558 |       550 |    12.3% |      0.65 |          -
...
==========================================================================================
```

### Parallel Benchmark on Modal

For faster runs, `modal/bench.py` fans out predictions across multiple GPU workers on [Modal](https://modal.com). Scoring is done locally after all predictions complete.

```bash
# Default: 4 H100 workers
modal run modal/bench.py

# Specific categories
modal run modal/bench.py --categories monomer_dna

# Resume an interrupted run
modal run modal/bench.py --resume --output-dir bench_results

# Override worker count or GPU type via environment variables
HELICO_BENCH_WORKERS=8 HELICO_BENCH_GPU=H100 modal run modal/bench.py
```

Prediction caches (`predictions/*.pkl`) are compatible between `helico-bench` and `modal/bench.py`, so `--resume` works across both.


## References

- **AlphaFold3**: [paper](https://www.nature.com/articles/s41586-024-07487-w) / [code](https://github.com/google-deepmind/alphafold3)
- **Boltz2**: [paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full) / [code](https://github.com/jwohlwend/boltz)
- **OpenFold3**: [whitepaper](https://github.com/aqlaboratory/openfold-3/blob/main/assets/whitepaper.pdf) / [code](https://github.com/aqlaboratory/openfold-3) / [docs](https://openfold-3.readthedocs.io/en/latest/)
- **cuEquivariance**: [code](https://github.com/NVIDIA/cuEquivariance) / [docs](https://docs.nvidia.com/cuda/cuequivariance/)
- **Protenix**: [code](https://github.com/bytedance/Protenix) / [paper](https://github.com/bytedance/Protenix/blob/main/docs/PTX_V1_Technical_Report_202602042356.pdf)

## License

Apache 2.0
