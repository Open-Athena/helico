#!/usr/bin/env bash
# Upload processed Helico data to HuggingFace dataset repo.
#
# Requires: huggingface-cli (pip install huggingface_hub)
#
# Usage:
#   bash scripts/upload_to_hf.sh <processed-dir>
#   bash scripts/upload_to_hf.sh /data/tim/helico-data/processed
#
# The processed dir is produced by:
#   helico-preprocess all <raw-dir> <processed-dir>
#
# The script:
#   1. Creates a staging directory
#   2. Copies small files (ccd_cache.pkl, index pickles)
#   3. Compresses manifest.json to manifest.json.gz
#   4. Tars and splits structures/ to <50GB parts
#   5. Uploads via huggingface-cli

set -euo pipefail

HF_REPO="timodonnell/helico-data"
SPLIT_SIZE="20G"
STAGING_DIR="${STAGING_DIR:-/tmp/helico-hf-staging}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <processed-dir>"
    echo "Example: $0 /data/tim/helico-data/processed"
    exit 1
fi

PROCESSED_DIR="$1"

echo "=== Helico HuggingFace Upload ==="
echo "Processed dir: $PROCESSED_DIR"
echo "Staging dir:   $STAGING_DIR"
echo "HF repo:       $HF_REPO"
echo ""

mkdir -p "$STAGING_DIR/processed"

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
