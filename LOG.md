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

Location: `/home/ubuntu/tim1/helico-data/processed/` (local path from earlier runs).
Modal equivalent: `/cache/helico-data/processed/` on the `helico-train-data` Volume.

| File | Size | Description |
|------|------|-------------|
| `ccd_cache.pkl` | 112 MB | Pickled CCD (parsed from components.cif) |
| `structures/` | ~236K `.pkl` files | Pickled TokenizedStructures across ~1,000 subdirectories |
| `manifest.json` | 1.9 GB | Metadata for all 236,326 processed structures |
| `rcsb_raw_msa_index.pkl` | 15 MB | Tar index for rcsb_raw_msa.tar (151,403 entries) |
| `openfold_raw_msa_index.pkl` | 11 MB | Tar index for openfold_raw_msa.tar (268,778 entries) |

## Preprocessing

### 2026-04 Modal run (current)

Switched to running preprocess on Modal via `modal/preprocess_on_modal.py` after
a local attempt at ~1.5 MB/s upstream would have taken 11 h to upload the
results. Downloaded raw data directly on Modal (S3 + rsync at ~45 MB/s), ran
preprocess with the new sparse-`token_bonds` format, committed the Volume.
End-to-end ~3 h.

- 252,091 mmCIF files processed → 236,326 structures passed filters (resolution <= 9.0, at least one polymer chain)
- ~3,000 structures more than the earlier 233,215 run due to a fresher rsync snapshot
- MSA tar indices rebuilt at the same time

### Earlier local run (obsolete)

Crashed the machine via OOM: workers accumulated 200+ GB RSS each because
`token_bonds` was stored as a dense `(N_tok, N_tok)` tensor — 400 MB for
ribosomes, multi-GB for capsids. Fix committed as `16c904d`, preprocess now
stores the sparse edge list and workers stay under ~6 GB RSS.

## Preprocessing Statistics

From the 2026-04 Modal run:

- 236,326 structures processed from 252,091 mmCIF files
- Train split (`release_date < 2021-09-30`, matches AF3/Protenix/OF3): 170,926 structures
- Val split (`2022-05-01 ≤ release_date ≤ 2023-01-12`, AF3 Recent PDB window): counts logged at train start
- MSA coverage: RCSB 151,403 chains / OpenFold 268,778 chains

## Training Commands

### Proof run (1×H100, 500 steps, warm-start from Protenix v1)

```bash
HELICO_TRAIN_GPU=H100:1 HELICO_TRAIN_MAX_STEPS=500 HELICO_TRAIN_CROP=256 \
    HELICO_TRAIN_RUN_NAME=proof-v1 modal run modal/train.py
```

Completed 2026-04-21 in ~22 min on H100:1 (commit `733a439`). Loss stayed
bounded 0.4-0.9 with occasional spikes. Checkpoints written to
`/ckpts/proof-v1/{step_250.pt, final.pt}` on `helico-checkpoints` Volume.
W&B run: [proof-v1](https://wandb.ai/PrinceOA/helico/runs/l2k08dqo).

### Full fine-tune (8×H100, validation every 500 steps)

```bash
HELICO_TRAIN_GPU=H100:8 HELICO_TRAIN_MAX_STEPS=10000 HELICO_TRAIN_CROP=384 \
    HELICO_TRAIN_VAL_EVERY=500 HELICO_TRAIN_RUN_NAME=v1-finetune-01 \
    modal run modal/train.py
```

See `TRAINING.md` for the full env-var / CLI reference.
