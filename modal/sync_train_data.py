"""Download Protenix v1 training data into a Modal Volume.

Downloads the tars published at https://protenix.tos-cn-beijing.volces.com/
for v2024.05.22 (what Protenix v1.0.0 was trained on). Each tar is streamed +
extracted in-place inside the Volume, then the tar is deleted to save space.

Usage:
    HELICO_DATA_COMPONENTS=common,mmcif_bioassembly,... modal run modal/sync_train_data.py

Default components: everything needed for training + validation.
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
BASE_URL = "https://protenix.tos-cn-beijing.volces.com"

# Data version (matches Protenix v1.0.0). See Protenix's
# scripts/database/download_protenix_data.sh.
DATA_VERSION = "2024.05.22"

# Components to download. Sizes are compressed tar.gz sizes.
#   common.tar.gz (0.4 GB) — shared constants, CCD, PDB seqres, cluster files
#   indices.tar.gz (30 MB) — sample index CSVs for WeightedPDB / Distillation
#   mmcif_bioassembly.tar.gz (31.6 GB) — pre-processed bioassemblies (pkl.gz)
#   mmcif_msa_template.tar.gz (175.9 GB) — MSA (a3m.gz) + hhsearch templates
#   recentPDB_bioassembly.tar.gz (1.3 GB) — val set (post-cutoff)
#   rna_msa.tar.gz — RNA-specific MSAs, needed if training on RNA
# Skip: mmcif.tar.gz (82 GB raw mmCIFs — we have these locally and they're
# redundant with mmcif_bioassembly); posebusters_{mmcif,bioassembly} (test set,
# add later when running posebusters eval).
_DEFAULT_COMPONENTS = [
    "common",
    "indices",
    "recentPDB_bioassembly",  # val set, small — download early
    "mmcif_bioassembly",
    "mmcif_msa_template",     # biggest, save for last
]

_ENV_COMPONENTS = os.environ.get("HELICO_DATA_COMPONENTS")
COMPONENTS = (
    _ENV_COMPONENTS.split(",") if _ENV_COMPONENTS else _DEFAULT_COMPONENTS
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl", "pv")
)

app = modal.App("helico-train-data-sync", image=image)

data_volume = modal.Volume.from_name("helico-train-data", create_if_missing=True)
DATA_ROOT = "/cache/protenix-v1-data"


@app.function(
    volumes={DATA_ROOT: data_volume},
    timeout=86400,
    cpu=4,
    memory=8192,
    ephemeral_disk=600 * 1024,  # 600 GB scratch (Modal min 512 GB)
)
def download_component(component: str) -> dict:
    """Download one component tar.gz, extract to Volume, remove tar.gz."""
    import os
    import subprocess
    import time

    fname = f"{component}.tar.gz"
    url = f"{BASE_URL}/{fname}"
    marker = Path(DATA_ROOT) / f".{component}.done"
    if marker.exists():
        print(f"[{component}] already downloaded (marker exists), skipping")
        return {"component": component, "status": "cached"}

    os.makedirs(DATA_ROOT, exist_ok=True)
    tar_path = Path(DATA_ROOT) / fname

    # Download (resume-capable via wget -c).
    t0 = time.time()
    print(f"[{component}] downloading {url} → {tar_path}")
    subprocess.run(
        ["wget", "-c", "-q", "--show-progress", "-O", str(tar_path), url],
        check=True,
    )
    dl_time = time.time() - t0
    size_gb = tar_path.stat().st_size / (1024**3)
    print(f"[{component}] download complete: {size_gb:.2f} GB in {dl_time:.0f}s")

    # Extract.
    t0 = time.time()
    print(f"[{component}] extracting...")
    subprocess.run(
        ["tar", "-xzf", str(tar_path), "-C", DATA_ROOT],
        check=True,
    )
    ex_time = time.time() - t0
    print(f"[{component}] extract complete in {ex_time:.0f}s")

    # Remove the tar and commit.
    tar_path.unlink()
    marker.touch()
    data_volume.commit()
    print(f"[{component}] done; Volume committed")
    return {
        "component": component,
        "status": "ok",
        "size_gb": size_gb,
        "dl_seconds": dl_time,
        "extract_seconds": ex_time,
    }


@app.local_entrypoint()
def main():
    """Download all selected components (sequentially, for disk/bandwidth stability)."""
    print(f"Syncing Protenix training data v{DATA_VERSION}")
    print(f"Components: {COMPONENTS}")
    results = []
    for comp in COMPONENTS:
        print(f"\n=== {comp} ===")
        r = download_component.remote(comp)
        results.append(r)
        print(f"  result: {r}")

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r.get('component')}: {r.get('status')}")

    # Final layout check
    list_layout.remote()


@app.function(volumes={DATA_ROOT: data_volume})
def list_layout():
    import subprocess
    print(f"\nVolume layout under {DATA_ROOT}:")
    subprocess.run(["du", "-sh", f"{DATA_ROOT}/*"], check=False)
    print("\nTop-level tree (depth 2):")
    subprocess.run(["bash", "-c", f"ls -la {DATA_ROOT} && find {DATA_ROOT} -maxdepth 2 -type d | head -30"], check=False)
