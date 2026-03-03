#!/usr/bin/env bash
# Upload Helico training data to HuggingFace dataset repo.
#
# Run this on the machine that has the raw/processed data.
# Requires: huggingface-cli (pip install huggingface_hub)
#
# Usage:
#   export HELICO_RAW_DIR=/data/helico/raw
#   export HELICO_PROCESSED_DIR=/data/helico/processed
#   bash scripts/upload_to_hf.sh
#
# The script:
#   1. Creates a staging directory with the HF repo layout
#   2. Tars directories with many files (mmCIF/, structures/)
#   3. Splits large tars to <50GB parts (HF per-file limit)
#   4. Compresses manifest.json
#   5. Copies small files directly
#   6. Uploads via huggingface-cli

set -euo pipefail

HF_REPO="timodonnell/helico-data"
SPLIT_SIZE="20G"
STAGING_DIR="${STAGING_DIR:-/tmp/helico-hf-staging}"

RAW_DIR="${HELICO_RAW_DIR:?Set HELICO_RAW_DIR to the raw data directory}"
PROCESSED_DIR="${HELICO_PROCESSED_DIR:?Set HELICO_PROCESSED_DIR to the processed data directory}"

echo "=== Helico HuggingFace Upload ==="
echo "Raw dir:       $RAW_DIR"
echo "Processed dir: $PROCESSED_DIR"
echo "Staging dir:   $STAGING_DIR"
echo "HF repo:       $HF_REPO"
echo ""

mkdir -p "$STAGING_DIR/raw" "$STAGING_DIR/processed"

# --- Raw files ---

# Small files: copy directly
for f in components.cif pdb_seqres.txt.gz; do
    if [ -f "$RAW_DIR/$f" ]; then
        echo "Copying raw/$f..."
        cp "$RAW_DIR/$f" "$STAGING_DIR/raw/$f"
    else
        echo "WARNING: $RAW_DIR/$f not found, skipping"
    fi
done

# mmCIF directory -> tar and split
if [ -d "$RAW_DIR/mmCIF" ]; then
    echo "Tarring mmCIF/ (~81 GB, this will take a while)..."
    tar cf - -C "$RAW_DIR" mmCIF | split -b "$SPLIT_SIZE" -d --additional-suffix="" - "$STAGING_DIR/raw/mmcif.tar."
    echo "mmCIF split into parts:"
    ls -lh "$STAGING_DIR/raw/mmcif.tar."*
else
    echo "WARNING: $RAW_DIR/mmCIF not found, skipping"
fi

# MSA tars: split if >50GB
for tar_name in rcsb_raw_msa.tar openfold_raw_msa.tar; do
    if [ -f "$RAW_DIR/$tar_name" ]; then
        size=$(stat -c%s "$RAW_DIR/$tar_name" 2>/dev/null || stat -f%z "$RAW_DIR/$tar_name")
        limit=$((50 * 1024 * 1024 * 1024))  # 50 GB
        if [ "$size" -gt "$limit" ]; then
            echo "Splitting $tar_name ($size bytes)..."
            split -b "$SPLIT_SIZE" -d --additional-suffix="" "$RAW_DIR/$tar_name" "$STAGING_DIR/raw/${tar_name}."
            ls -lh "$STAGING_DIR/raw/${tar_name}."*
        else
            echo "Copying raw/$tar_name (under 50 GB, no split needed)..."
            cp "$RAW_DIR/$tar_name" "$STAGING_DIR/raw/$tar_name"
        fi
    else
        echo "WARNING: $RAW_DIR/$tar_name not found, skipping"
    fi
done

# --- Processed files ---

# Small files: copy directly
for f in ccd_cache.pkl rcsb_raw_msa_index.pkl openfold_raw_msa_index.pkl; do
    if [ -f "$PROCESSED_DIR/$f" ]; then
        echo "Copying processed/$f..."
        cp "$PROCESSED_DIR/$f" "$STAGING_DIR/processed/$f"
    else
        echo "WARNING: $PROCESSED_DIR/$f not found, skipping"
    fi
done

# Compress manifest.json (1.5 GB JSON -> ~small gzip, since HF doesn't LFS-track .json)
if [ -f "$PROCESSED_DIR/manifest.json" ]; then
    echo "Compressing manifest.json..."
    gzip -c "$PROCESSED_DIR/manifest.json" > "$STAGING_DIR/processed/manifest.json.gz"
    echo "manifest.json.gz: $(ls -lh "$STAGING_DIR/processed/manifest.json.gz" | awk '{print $5}')"
else
    echo "WARNING: $PROCESSED_DIR/manifest.json not found, skipping"
fi

# structures/ directory -> tar and split
if [ -d "$PROCESSED_DIR/structures" ]; then
    echo "Tarring structures/ (~233K pkl files, this will take a while)..."
    tar cf - -C "$PROCESSED_DIR" structures | split -b "$SPLIT_SIZE" -d --additional-suffix="" - "$STAGING_DIR/processed/structures.tar."
    echo "structures split into parts:"
    ls -lh "$STAGING_DIR/processed/structures.tar."*
else
    echo "WARNING: $PROCESSED_DIR/structures not found, skipping"
fi

echo ""
echo "=== Staging complete ==="
echo "Contents of staging dir:"
find "$STAGING_DIR" -type f -exec ls -lh {} \; | awk '{print $5, $NF}'
echo ""

# Upload
echo "=== Uploading to HuggingFace ==="
echo "Running: huggingface-cli upload-large-folder $HF_REPO $STAGING_DIR --repo-type dataset"
huggingface-cli upload-large-folder "$HF_REPO" "$STAGING_DIR" --repo-type dataset

echo ""
echo "=== Upload complete ==="
echo "Dataset: https://huggingface.co/datasets/$HF_REPO"
