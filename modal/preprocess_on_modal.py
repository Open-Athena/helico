"""Download raw data + run helico-preprocess on Modal.

Avoids the slow local upstream (~1.5 MB/s here would take 11h to upload the
processed dataset). Downloads from S3/RCSB on Modal at ~hundreds of MB/s,
preprocesses on a beefy container, writes directly to the helico-train-data
Volume. Total runtime target: ~1-1.5h vs 11h local→Modal upload.

Usage:
    modal run --detach modal/preprocess_on_modal.py

**Use --detach.** A plain `modal run` creates an *ephemeral* app tied to the
client process: if the terminal (or agent session) that launched it exits, the
app is torn down and the remaining work is lost. This is a multi-hour job, so
it needs to outlive its launcher. A 2026-08-06 run was killed at ~95% this way.

Volume writes are flushed as they happen rather than only at the explicit
`data_volume.commit()`, so an interrupted run leaves the pickles it already
wrote intact — but `build_manifest` never runs, so `manifest.json` keeps its
previous (still valid) contents and the un-reached structures keep their old
pickles.

Env vars:
    HELICO_SKIP_DOWNLOAD=1   # skip the raw-data download step (reuse existing /raw)
    HELICO_SKIP_PREPROCESS=1 # skip the preprocess step (download-only)
    HELICO_MAX_RES=9.0       # resolution cutoff passed to preprocess
    HELICO_N_WORKERS=32      # override preprocess worker count
    HELICO_NO_SKIP_EXISTING=1  # reprocess structures that already have a pickle.
                               # REQUIRED after a schema change (e.g. adding
                               # residue/residue contacts) — otherwise an
                               # already-populated structures/ dir makes the
                               # whole step a silent no-op.
    HELICO_REQUIRE_CONTACTS=1  # skip a structure only if its pickle already has
                               # contacts. Makes a contacts migration resumable:
                               # re-running processes only what is still missing.
    HELICO_STEP=all          # "all" or "structures". Use "structures" to redo
                             # only the tokenize+pickle+manifest step, reusing
                             # the existing CCD cache and MSA tar indices.

Re-preprocess after a schema change, reusing raw data already on the Volume:
    HELICO_SKIP_DOWNLOAD=1 HELICO_STEP=structures HELICO_NO_SKIP_EXISTING=1 \
        modal run modal/preprocess_on_modal.py
"""

import os
import shlex
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl", "rsync", "git")
    .pip_install(
        "biopython>=1.80",
        "numpy",
        "scipy",
        "pyyaml>=6.0",
        "torch>=2.10,<2.11",  # see gh#3
        "huggingface_hub>=0.20",
        "tqdm",
        "pyconfind>=0.6",  # residue/residue contacts
    )
    # Warm the rotamer-library cache at build time. Otherwise the first worker
    # to need it downloads from GitHub mid-run, which is both a failure mode
    # hours into a long job and a race across concurrent workers.
    .run_commands(
        "python -c 'from pyconfind import cached_rotamer_library;"
        " print(cached_rotamer_library())'"
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/helico/src")
    .add_local_file(str(ROOT / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
    .add_local_file(str(ROOT / "README.md"), remote_path="/root/helico/README.md")
)

app = modal.App("helico-preprocess-on-modal", image=image)
data_volume = modal.Volume.from_name("helico-train-data", create_if_missing=True)

DATA_ROOT = "/cache/helico-data"
RAW_DIR = f"{DATA_ROOT}/raw"
PROCESSED_DIR = f"{DATA_ROOT}/processed"

RAW_DOWNLOADS = [
    # (url, dest relative to RAW_DIR, size GB for logging)
    ("https://files.wwpdb.org/pub/pdb/data/monomers/components.cif", "components.cif", 0.47),
    ("https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz", "pdb_seqres.txt.gz", 0.06),
    ("https://boltz1.s3.us-east-2.amazonaws.com/rcsb_raw_msa.tar", "rcsb_raw_msa.tar", 131.0),
    ("https://boltz1.s3.us-east-2.amazonaws.com/openfold_raw_msa.tar", "openfold_raw_msa.tar", 88.0),
]

RSYNC_SRC = "rsync.rcsb.org::ftp_data/structures/divided/mmCIF/"
RSYNC_DST_SUBDIR = "mmCIF"


def _run(cmd: list[str] | str, cwd: str | None = None, env: dict | None = None) -> None:
    if isinstance(cmd, list):
        printable = " ".join(shlex.quote(a) for a in cmd)
    else:
        printable = cmd
    print(f"\n+ {printable}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=cwd, env=env, check=True, shell=isinstance(cmd, str))
    print(f"  done in {time.time() - t0:.1f}s", flush=True)


@app.function(
    # CPU-only; preprocess is CPU-bound and no GPU is needed. Give enough cores
    # and RAM to hold the worst-case parsed mmCIFs comfortably.
    cpu=32.0,
    memory=128 * 1024,  # 128 GB RAM
    timeout=6 * 3600,
    volumes={DATA_ROOT: data_volume},
    ephemeral_disk=600 * 1024,  # Modal min is 512 GiB
)
def run_remote(
    skip_download: bool,
    skip_preprocess: bool,
    max_res: float,
    n_workers: int,
    no_skip_existing: bool = False,
    step: str = "all",
    require_contacts: bool = False,
) -> dict:
    """Download raw data into the Volume and run helico-preprocess all."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not skip_download:
        for url, rel, size_gb in RAW_DOWNLOADS:
            dest = Path(RAW_DIR) / rel
            if dest.exists() and dest.stat().st_size > 0:
                print(f"[download] {rel} already present ({dest.stat().st_size / 1e9:.2f} GB), skip")
                continue
            print(f"[download] {rel} (~{size_gb} GB) from {url}")
            _run([
                "curl", "-fL", "--retry", "5", "--retry-delay", "10",
                "--connect-timeout", "30", "-o", str(dest), url,
            ])

        mmcif_dest = Path(RAW_DIR) / RSYNC_DST_SUBDIR
        mmcif_dest.mkdir(exist_ok=True)
        # --delete would prune local-only files; skip on first sync
        _run([
            "rsync", "-rlpt", "-v", "-z", "--port=33444",
            RSYNC_SRC, str(mmcif_dest) + "/",
        ])
        data_volume.commit()
        print("[download] committed volume")

    if skip_preprocess:
        print("Skipping preprocess (HELICO_SKIP_PREPROCESS=1)")
        return {"status": "download-only"}

    # Install helico into the container's Python so the CLI is available.
    _run(["pip", "install", "-e", "/root/helico"])

    # Run preprocess. The data.py fix from commit 16c904d stores token_bonds
    # sparse, so worker RSS stays bounded.
    # "structures" reruns only the tokenize+pickle+manifest step, reusing the
    # existing ccd_cache.pkl and MSA tar indices. That is what a schema change
    # like adding contacts needs — "all" would additionally re-parse the 473 MB
    # CCD and re-index 219 GB of MSA tars for no benefit.
    cmd = [
        "helico-preprocess", step, RAW_DIR, PROCESSED_DIR,
        "--max-resolution", str(max_res),
        "--n-workers", str(n_workers),
    ]
    if no_skip_existing:
        cmd.append("--no-skip-existing")
    if require_contacts:
        cmd.append("--require-contacts")
    _run(cmd)
    data_volume.commit()
    print("[preprocess] committed volume")

    # Quick sanity
    manifest = Path(PROCESSED_DIR) / "manifest.json"
    assert manifest.exists(), f"manifest missing: {manifest}"
    size_gb = manifest.stat().st_size / 1e9
    return {
        "status": "ok",
        "manifest_size_gb": round(size_gb, 3),
        "raw_dir": RAW_DIR,
        "processed_dir": PROCESSED_DIR,
    }


@app.local_entrypoint()
def main():
    skip_download = os.environ.get("HELICO_SKIP_DOWNLOAD") == "1"
    skip_preprocess = os.environ.get("HELICO_SKIP_PREPROCESS") == "1"
    max_res = float(os.environ.get("HELICO_MAX_RES", "9.0"))
    n_workers = int(os.environ.get("HELICO_N_WORKERS", "32"))
    no_skip_existing = os.environ.get("HELICO_NO_SKIP_EXISTING") == "1"
    step = os.environ.get("HELICO_STEP", "all")
    require_contacts = os.environ.get("HELICO_REQUIRE_CONTACTS") == "1"
    result = run_remote.remote(
        skip_download, skip_preprocess, max_res, n_workers, no_skip_existing,
        step, require_contacts,
    )
    print(result)
