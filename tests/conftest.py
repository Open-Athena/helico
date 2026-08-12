"""Fixtures shared across test modules.

`ccd` and `rotamer_library` are expensive to build and are needed by both the
contacts tests and the inference tests, so they live here rather than being
duplicated. `pdb_path` downloads and caches reference structures.
"""

import os
import pickle
import urllib.request
from pathlib import Path

import pytest

_env_dir = os.environ.get("HELICO_TEST_PDB_DIR")
# Note: Path("") is Path("."), which is truthy — so check the string, not the Path.
_CACHE = Path(_env_dir) if _env_dir else Path(__file__).parent / ".pdb_cache"


@pytest.fixture(scope="session")
def ccd():
    from helico.data import _processed_dir

    path = _processed_dir() / "ccd_cache.pkl"
    if not path.exists():
        pytest.skip("CCD cache not available (run helico-download --subset ccd-only)")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def rotamer_library():
    from helico.contacts import load_rotamer_library

    try:
        return load_rotamer_library()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"rotamer library unavailable: {e}")


@pytest.fixture(scope="session")
def pdb_path():
    """Return a callable that fetches and caches a reference mmCIF by PDB id."""

    def _get(pdb_id: str) -> Path:
        _CACHE.mkdir(parents=True, exist_ok=True)
        path = _CACHE / f"{pdb_id}.cif"
        if not path.exists():
            url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"could not download {pdb_id}: {e}")
        return path

    return _get
