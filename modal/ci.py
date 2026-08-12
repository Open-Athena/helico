from pathlib import Path
import modal

ROOT = Path(__file__).parent.parent
PROTENIX_URL = "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget")
    .pip_install(
        "torch>=2.7",
        "cuequivariance-torch>=0.8",
        "cuequivariance-ops-torch-cu12>=0.8",
        "biopython>=1.80",
        "numpy",
        "scipy",
        "pyyaml>=6.0",
        "huggingface_hub>=0.20",
        "pytest>=7",
        "requests",
        "tmtools",
        "DockQ",
        "tqdm",
        # Needed to warm the rotamer cache below.
        "pyconfind>=0.6",
    )
    # Warm the pyconfind rotamer library into ~/.cache at build time.
    # preprocess_structures fetches it on first use with no retry, so a
    # transient network blip fails the whole suite (seen: RemoteDisconnected).
    # The venv the tests run in shares the same on-disk cache.
    .run_commands(
        "python -c 'from pyconfind import cached_rotamer_library;"
        " print(cached_rotamer_library())'"
    )
    # Protenix checkpoint baked into image (1.4 GB, cached by Modal)
    .run_commands(
        f"mkdir -p /root/helico/checkpoints && wget -q -O /root/helico/checkpoints/protenix_base_default_v1.0.0.pt {PROTENIX_URL}"
    )
    # Huggingface data (ccd_cache only; matches default HELICO_DATA_DIR)
    .run_commands(
        "mkdir -p /root/.cache/helico/data && "
        "hf download timodonnell/helico-data processed/ccd_cache.pkl --repo-type dataset --local-dir /root/.cache/helico/data"
    )
    # Project code last (changes most frequently)
    .add_local_dir(str(ROOT / "src"), remote_path="/root/helico/src")
    .add_local_dir(str(ROOT / "tests"), remote_path="/root/helico/tests")
    .add_local_file(str(ROOT / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
    .add_local_file(str(ROOT / "README.md"), remote_path="/root/helico/README.md")
)

app = modal.App("helico-ci", image=image)


@app.function(gpu=["A10", "L40S", "A100", "H100"], timeout=600,
              env={"HELICO_CI_PYTEST_ARGS": __import__("os").environ.get("HELICO_CI_PYTEST_ARGS", "")})
def run_tests():
    import subprocess
    subprocess.run("cd /root/helico && uv venv --python 3.11", check=True, shell=True)
    # Pre-seed the fresh venv with DockQ's no-isolation build deps. The
    # setup.py imports numpy at build time (numpy.get_include() for the
    # cython extension), and uv builds DockQ before installing the rest of
    # the project's deps under [bench]. Order matters: numpy first so the
    # cython extension links against numpy 2 ABI.
    subprocess.run(
        "cd /root/helico && uv pip install 'setuptools>=68' 'numpy>=2.0' cython",
        check=True, shell=True,
    )
    subprocess.run("cd /root/helico && uv pip install -e '.[dev,bench]'", check=True, shell=True)
    import os
    extra = os.environ.get("HELICO_CI_PYTEST_ARGS", "").split()
    subprocess.run(["uv", "run", "pytest", "-v", "--tb=short", *extra],
                   check=True, cwd="/root/helico")
