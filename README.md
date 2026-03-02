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
  --ccd PATH              Path to CCD cache pickle (default: uses $HELICO_PROCESSED_DIR/ccd_cache.pkl)
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

### Download FoldBench Data

```bash
# Clone the FoldBench repository
git clone https://github.com/BEAM-Labs/FoldBench.git
cd FoldBench

# Download ground truth structures (1.06 GB tar from Google Drive)
pip install gdown
gdown 17KdWDXKATaeHF6inPxhPHIRuIzeqiJxS -O ground_truths.tar
tar xf ground_truths.tar
```

After extraction, the directory layout should be:

```
FoldBench/
  targets/                              # 9 CSV files (one per category)
    interface_protein_protein.csv
    interface_antibody_antigen.csv
    interface_protein_peptide.csv
    interface_protein_ligand.csv
    interface_protein_dna.csv
    interface_protein_rna.csv
    monomer_protein.csv
    monomer_rna.csv
    monomer_dna.csv
  examples/
    alphafold3_inputs.json              # AF3-style input JSON (4 example targets)
  ground_truth_20250520/                # Ground truth CIF files
    *.cif                               # One CIF per target assembly
```

### Download CCD (if not already available)

The Chemical Component Dictionary is needed for tokenization. If you already have it from training setup, point `--ccd` at your existing cache. Otherwise:

```bash
wget https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
gunzip components.cif.gz

# Place it where helico expects it
export HELICO_RAW_DIR=/path/to/dir/containing/components.cif
export HELICO_PROCESSED_DIR=/path/to/processed
```

The CCD cache (`ccd_cache.pkl`) will be built automatically on first run and reused subsequently.

### Running the Benchmark

```bash
# Set environment variables (required for CCD loading)
export HELICO_RAW_DIR=/path/to/raw       # Directory containing components.cif
export HELICO_PROCESSED_DIR=/path/to/processed  # Directory for ccd_cache.pkl

# Run on all categories with Protenix weights
helico-bench \
    --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --foldbench-dir /path/to/FoldBench \
    --output-dir bench_results/

# Run a single category
helico-bench \
    --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --foldbench-dir /path/to/FoldBench \
    --output-dir bench_results/ \
    --categories monomer_protein

# Run multiple specific categories
helico-bench \
    --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --foldbench-dir /path/to/FoldBench \
    --output-dir bench_results/ \
    --categories monomer_protein,interface_protein_ligand

# With a Helico checkpoint instead of Protenix
helico-bench \
    --checkpoint checkpoints/final.pt \
    --foldbench-dir /path/to/FoldBench \
    --output-dir bench_results/

# Resume a partially completed run (reuses cached predictions)
helico-bench \
    --protenix checkpoints/protenix_base_default_v1.0.0.pt \
    --foldbench-dir /path/to/FoldBench \
    --output-dir bench_results/ \
    --resume
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


## References

- **AlphaFold3**: [paper](https://www.nature.com/articles/s41586-024-07487-w) / [code](https://github.com/google-deepmind/alphafold3)
- **Boltz2**: [paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full) / [code](https://github.com/jwohlwend/boltz)
- **OpenFold3**: [whitepaper](https://github.com/aqlaboratory/openfold-3/blob/main/assets/whitepaper.pdf) / [code](https://github.com/aqlaboratory/openfold-3) / [docs](https://openfold-3.readthedocs.io/en/latest/)
- **cuEquivariance**: [code](https://github.com/NVIDIA/cuEquivariance) / [docs](https://docs.nvidia.com/cuda/cuequivariance/)
- **Protenix**: [code](https://github.com/bytedance/Protenix) / [paper](https://github.com/bytedance/Protenix/blob/main/docs/PTX_V1_Technical_Report_202602042356.pdf)

## License

Apache 2.0
