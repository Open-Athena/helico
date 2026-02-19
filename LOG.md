# LOG.md

Operations log for Helico data processing and training.

## Environment Setup

All data commands require these env vars:

```bash
export HELICO_RAW_DIR=/home/ubuntu/tim1/helico-data/raw
export HELICO_PROCESSED_DIR=/home/ubuntu/tim1/helico-data/processed
```

## Raw Data Inventory

Location: `/home/ubuntu/tim1/helico-data/raw/`

| File | Size | Description |
|------|------|-------------|
| `components.cif` | 473 MB | Chemical Component Dictionary (CCD) — all ligand/residue definitions |
| `mmCIF/` | 81 GB | 248,942 `.cif.gz` files across 1,089 subdirectories — all PDB structures |
| `rcsb_raw_msa.tar` | 131 GB | RCSB MSA alignments as `{sha256}.a3m.gz` (from Boltz S3) |
| `openfold_raw_msa.tar` | 88 GB | OpenFold MSA alignments (from Boltz S3) |
| `pdb_seqres.txt.gz` | 60 MB | PDB sequence database |

Download commands (already run):

```bash
cd /home/ubuntu/tim1/helico-data/raw
wget https://files.wwpdb.org/pub/pdb/data/monomers/components.cif
wget https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
wget https://boltz1.s3.us-east-2.amazonaws.com/rcsb_raw_msa.tar
wget https://boltz1.s3.us-east-2.amazonaws.com/openfold_raw_msa.tar
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ mmCIF/
```

## Processed Data

Location: `/home/ubuntu/tim1/helico-data/processed/`

| File | Size | Description |
|------|------|-------------|
| `ccd_cache.pkl` | 112 MB | Pickled CCD (parsed from components.cif) |
| `structures/` | ~233K `.pkl` files | Pickled TokenizedStructures across 1,085 subdirectories |
| `manifest.json` | 1.5 GB | Metadata for all 233,215 processed structures |
| `rcsb_raw_msa_index.pkl` | 15 MB | Tar index for rcsb_raw_msa.tar (151,403 entries) |
| `openfold_raw_msa_index.pkl` | 11 MB | Tar index for openfold_raw_msa.tar (268,778 entries) |

## Preprocessing

### Initial run (completed)

Processed 248,942 mmCIF files -> 233,215 structures passed filters (resolution <= 9.0, at least one polymer chain). MSA tar indices built for both RCSB and OpenFold archives.

### Re-run required: add chain sequences to manifest

The initial manifest lacks `chain_sequences`, which are needed for MSA lookup.
RCSB MSA files are named by `sha256(chain_sequence + "\n")`, so the sequence
must be stored in the manifest. Re-run with `--no-skip-existing` to regenerate
pickles and manifest with sequences:

```bash
helico-preprocess structures --no-skip-existing
```

Expected time: ~2-4 hours with 32+ cores. The MSA tar indices do not need to be rebuilt.

## Preprocessing Statistics

- 233,215 structures processed
- Token count: min=2, median=553, max=587,457
- Atom count: min=22, median=3,962, max=4,527,365
- Resolution: min=0.00, median=2.10, max=9.00
- Date range: 1976-05-19 to 2026-02-11
- Methods: X-RAY 201,849 / EM 30,934 / E-CRYST 267 / NEUTRON 117 / FIBER 29
- Entity types: protein 229,792 / ligand 194,967 / nucleotide 18,732
- Train (release < 2022-01-01): 170,927
- Val (release >= 2022-01-01): 62,288
- MSA coverage: RCSB 151,403 chains / OpenFold 268,778 chains

## Training Commands

(To be added when we start training runs.)
