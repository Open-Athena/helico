"""Upload a date-stamped Helico data snapshot from the Modal Volume to HuggingFace.

Designed to scale: HF becomes the authoritative source, Modal Volume is the cache.

Usage:
    # Audit what would be uploaded — no actual transfer
    modal run modal/upload_to_hf.py --dry-run

    # Upload current Volume contents as snapshot pdb-2026-04-22 and update latest
    HF_TOKEN=... modal run modal/upload_to_hf.py

    # Custom snapshot id
    HELICO_SNAPSHOT_ID=pdb-2026-04-22-rerun modal run modal/upload_to_hf.py

    # Use a different source-type label (for OF3 distillation in the future)
    HELICO_SOURCE_TYPE=of3-distillation HELICO_SNAPSHOT_ID=of3-distillation-2026-05-15 \
        modal run modal/upload_to_hf.py

Env vars:
    HF_REPO          (default timodonnell/helico-data)
    HELICO_SOURCE_TYPE (default pdb) — top-level taxonomy in latest.json
    HELICO_SNAPSHOT_ID (default <source_type>-YYYY-MM-DD)
    HELICO_TAR_CHUNK_GB (default 25)  — split-tar chunk size in GB
    HELICO_SKIP_LATEST_UPDATE=1 — upload snapshot but don't repoint latest.json
    HELICO_DRY_RUN=1 — alias for --dry-run flag

Requires Modal secret `helico-hf-modal` providing `HF_TOKEN` (write scope on
the dataset).

Layout produced on HF:
    processed/
        ccd_cache.pkl                  (shared across snapshots)
        latest.json                    ({"pdb": "pdb-2026-04-22", ...})
        <snapshot_id>/
            SOURCE.json
            manifest.json.gz
            structures.tar.00, .01, ... (split tar)
            rcsb_raw_msa_index.pkl
            openfold_raw_msa_index.pkl
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tar", "gzip", "coreutils")
    .pip_install("huggingface_hub>=0.20")
)

app = modal.App("helico-upload-to-hf", image=image)
data_volume = modal.Volume.from_name("helico-train-data", create_if_missing=True)
DATA_ROOT = "/cache/helico-data"

DEFAULT_REPO = "timodonnell/helico-data"
HF_PROCESSED_PREFIX = "processed"

# Files copied verbatim from a snapshot (small, no special handling).
_SNAPSHOT_VERBATIM = [
    "rcsb_raw_msa_index.pkl",
    "openfold_raw_msa_index.pkl",
]

# Shared across snapshots — uploaded once at processed/ root, not per-snapshot.
_SHARED_FILES = ["ccd_cache.pkl"]


def _today() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd="/root/helico", text=True,
        ).strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    """Hash up to the first max_bytes — full hash on multi-GB files would be slow.
    For provenance/dedup, this is sufficient as a fingerprint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def _build_source_json(
    snapshot_id: str,
    source_type: str,
    processed_dir: Path,
    structures_dir: Path,
    n_structures: int,
    git_sha: str,
) -> dict:
    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "source_type": source_type,
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": git_sha,
        "n_structures": n_structures,
        "raw_data_sources": {
            "mmcif": "rsync://rsync.rcsb.org::ftp_data/structures/divided/mmCIF/",
            "ccd": "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif",
            "rcsb_msa": "https://boltz1.s3.us-east-2.amazonaws.com/rcsb_raw_msa.tar",
            "openfold_msa": "https://boltz1.s3.us-east-2.amazonaws.com/openfold_raw_msa.tar",
        } if source_type == "pdb" else {},
        "preprocess": {
            "command": "helico-preprocess all <raw> <processed>",
            "max_resolution": 9.0,
            "token_bonds_format": "sparse",  # sparse since 16c904d
        },
        "fingerprints": {
            "ccd_cache.pkl": _file_sha256(processed_dir / "ccd_cache.pkl"),
            "manifest.json": _file_sha256(processed_dir / "manifest.json"),
        },
    }


@app.function(
    cpu=8.0,
    memory=64 * 1024,
    timeout=6 * 3600,
    volumes={DATA_ROOT: data_volume},
    secrets=[modal.Secret.from_name("helico-hf-modal")],
    ephemeral_disk=600 * 1024,
)
def upload_remote(
    snapshot_id: str,
    source_type: str,
    repo: str,
    tar_chunk_gb: int,
    dry_run: bool,
    skip_latest_update: bool,
    git_sha: str = "unknown",
) -> dict:
    """Pack the processed/ tree into snapshot files and upload to HF."""
    from huggingface_hub import HfApi

    processed_dir = Path(DATA_ROOT) / "processed"
    structures_dir = processed_dir / "structures"
    if not (processed_dir / "manifest.json").exists():
        raise SystemExit(f"manifest.json missing under {processed_dir} — preprocess first")

    workspace = Path("/tmp/helico-upload")
    workspace.mkdir(parents=True, exist_ok=True)
    snap_workspace = workspace / snapshot_id
    if snap_workspace.exists():
        shutil.rmtree(snap_workspace)
    snap_workspace.mkdir(parents=True)

    print(f"=== Snapshot {snapshot_id} (source_type={source_type}) ===", flush=True)
    print(f"Reading from {processed_dir}", flush=True)

    # 1. Gzip manifest
    manifest_src = processed_dir / "manifest.json"
    manifest_gz = snap_workspace / "manifest.json.gz"
    print(f"[1/4] Gzipping manifest.json ({manifest_src.stat().st_size / 1e9:.2f} GB)...", flush=True)
    with open(manifest_src, "rb") as src, gzip.open(manifest_gz, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
    print(f"  → {manifest_gz.name} ({manifest_gz.stat().st_size / 1e9:.2f} GB)", flush=True)

    # Determine n_structures from manifest (cheap streaming count)
    with open(manifest_src, "rb") as f:
        # Manifest is `{"id1": {...}, "id2": {...}, ...}` — count top-level keys.
        # For 1.9GB JSON we don't want to parse fully; load just to count keys.
        data = json.load(f)
        n_structures = len(data)
    print(f"  manifest has {n_structures} structures", flush=True)

    # 2. Verbatim files
    for fname in _SNAPSHOT_VERBATIM:
        src = processed_dir / fname
        if src.exists():
            shutil.copy2(src, snap_workspace / fname)
            print(f"  copied {fname} ({src.stat().st_size / 1e6:.1f} MB)", flush=True)

    # 3. SOURCE.json (git_sha captured at the local entrypoint and passed in,
    # since the Modal container has no .git dir of its own).
    source_json = _build_source_json(
        snapshot_id, source_type, processed_dir, structures_dir, n_structures, git_sha,
    )
    with open(snap_workspace / "SOURCE.json", "w") as f:
        json.dump(source_json, f, indent=2)
    print(f"  wrote SOURCE.json ({n_structures} structures, git_sha={source_json['git_sha'][:8]})",
          flush=True)

    # 4. Tar + split structures (large)
    chunk_bytes = tar_chunk_gb * 1024 * 1024 * 1024
    print(f"[2/4] tar | split structures/ → structures.tar.NN (chunk={tar_chunk_gb} GB)...",
          flush=True)
    n_pkls = sum(1 for _ in structures_dir.rglob("*.pkl"))
    print(f"  packing {n_pkls} pickle files...", flush=True)
    # Use shell pipe to avoid materializing tar in memory: tar -C ... -cf - structures | split -b 25G
    cmd = (
        f"tar -C '{processed_dir}' -cf - structures "
        f"| split -b {chunk_bytes} -d -a 2 - '{snap_workspace}/structures.tar.'"
    )
    if dry_run:
        print(f"  [dry-run] would run: {cmd}", flush=True)
    else:
        subprocess.run(cmd, shell=True, check=True)
        parts = sorted(snap_workspace.glob("structures.tar.*"))
        total = sum(p.stat().st_size for p in parts)
        print(f"  → {len(parts)} parts, total {total / 1e9:.2f} GB", flush=True)

    # 5. Audit
    print(f"[3/4] Audit:", flush=True)
    upload_plan = []
    if not dry_run:
        for p in sorted(snap_workspace.iterdir()):
            sz = p.stat().st_size
            upload_plan.append((p, f"{HF_PROCESSED_PREFIX}/{snapshot_id}/{p.name}", sz))
            print(f"  {sz/1e9:7.2f} GB  →  {HF_PROCESSED_PREFIX}/{snapshot_id}/{p.name}", flush=True)
        # Shared (uploaded only if missing on HF)
        for fname in _SHARED_FILES:
            src = processed_dir / fname
            if src.exists():
                upload_plan.append((src, f"{HF_PROCESSED_PREFIX}/{fname}", src.stat().st_size))
                print(f"  {src.stat().st_size/1e9:7.2f} GB  →  {HF_PROCESSED_PREFIX}/{fname} (shared)",
                      flush=True)
        total = sum(sz for _, _, sz in upload_plan)
        print(f"  Total upload: {total / 1e9:.2f} GB across {len(upload_plan)} files", flush=True)

    if dry_run:
        print("[4/4] DRY RUN — nothing uploaded.", flush=True)
        return {"status": "dry-run", "snapshot_id": snapshot_id, "n_structures": n_structures}

    # 6. Upload
    print(f"[4/4] Uploading to https://huggingface.co/datasets/{repo} ...", flush=True)
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    for src, repo_path, sz in upload_plan:
        print(f"  ↑ {repo_path} ({sz/1e9:.2f} GB)", flush=True)
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=repo_path,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Upload {repo_path}",
        )

    # 7. Update latest.json (atomic via overwrite)
    if not skip_latest_update:
        print(f"  ↑ {HF_PROCESSED_PREFIX}/latest.json (pointing {source_type} → {snapshot_id})",
              flush=True)
        # Read existing latest.json if present, else start fresh
        try:
            latest_path = api.hf_hub_download(
                repo_id=repo, repo_type="dataset",
                filename=f"{HF_PROCESSED_PREFIX}/latest.json",
            )
            with open(latest_path) as f:
                latest = json.load(f)
        except Exception:
            latest = {"schema_version": 1, "sources": {}}
        latest.setdefault("sources", {})[source_type] = snapshot_id
        latest["updated_at_utc"] = dt.datetime.utcnow().isoformat() + "Z"
        latest_local = workspace / "latest.json"
        with open(latest_local, "w") as f:
            json.dump(latest, f, indent=2)
        api.upload_file(
            path_or_fileobj=str(latest_local),
            path_in_repo=f"{HF_PROCESSED_PREFIX}/latest.json",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"latest: {source_type} → {snapshot_id}",
        )

    return {
        "status": "ok",
        "snapshot_id": snapshot_id,
        "source_type": source_type,
        "repo": repo,
        "n_structures": n_structures,
        "url": f"https://huggingface.co/datasets/{repo}/tree/main/{HF_PROCESSED_PREFIX}/{snapshot_id}",
    }


@app.local_entrypoint()
def main(dry_run: bool = False):
    source_type = os.environ.get("HELICO_SOURCE_TYPE", "pdb")
    snapshot_id = os.environ.get("HELICO_SNAPSHOT_ID", f"{source_type}-{_today()}")
    repo = os.environ.get("HF_REPO", DEFAULT_REPO)
    tar_chunk_gb = int(os.environ.get("HELICO_TAR_CHUNK_GB", "25"))
    skip_latest = os.environ.get("HELICO_SKIP_LATEST_UPDATE") == "1"
    dry_run = dry_run or os.environ.get("HELICO_DRY_RUN") == "1"
    # Capture git sha locally — the Modal container has no .git dir.
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        git_sha = "unknown"

    print(f"Snapshot id: {snapshot_id}")
    print(f"Source type: {source_type}")
    print(f"HF repo:     {repo}")
    print(f"Tar chunk:   {tar_chunk_gb} GB")
    print(f"Dry run:     {dry_run}")
    print(f"Skip latest: {skip_latest}")
    print(f"Git sha:     {git_sha}")
    result = upload_remote.remote(
        snapshot_id, source_type, repo, tar_chunk_gb, dry_run, skip_latest, git_sha,
    )
    print(result)
