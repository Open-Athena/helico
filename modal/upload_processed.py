"""Upload locally-preprocessed Helico data into the helico-train-data Modal Volume.

Run AFTER `helico-preprocess all` finishes on your local machine. Mirrors the
local processed/ tree into /processed/ on the Modal Volume — modal/train.py
expects exactly that layout (manifest.json at /cache/helico-data/processed/).

Usage:
    HELICO_UPLOAD_SRC=/data/tim/helico-data/processed \
        modal run modal/upload_processed.py
"""

import os
from pathlib import Path

import modal

SRC = Path(os.environ.get("HELICO_UPLOAD_SRC", "/data/tim/helico-data/processed")).resolve()
DEST_SUBDIR = os.environ.get("HELICO_UPLOAD_DEST", "processed")

app = modal.App("helico-upload-processed")
data_volume = modal.Volume.from_name("helico-train-data", create_if_missing=True)


@app.local_entrypoint()
def main():
    assert SRC.exists() and SRC.is_dir(), f"source not a directory: {SRC}"
    print(f"Uploading {SRC} → volume 'helico-train-data' under /{DEST_SUBDIR}/")

    # batch_upload streams many files in parallel; much faster than
    # file-at-a-time puts for a tree of pickles.
    with data_volume.batch_upload(force=True) as batch:
        for path in sorted(SRC.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(SRC)
            remote_path = f"/{DEST_SUBDIR}/{rel.as_posix()}"
            batch.put_file(str(path), remote_path)
    print("Upload complete; volume committed.")
