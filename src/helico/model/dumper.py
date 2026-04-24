"""Intermediate-tensor dumper for pipeline-diff analysis.

Used by ``Helico.predict(..., dump_intermediates_to=...)`` to write
stage-tagged ``.npz`` files for offline comparison against other
implementations (e.g. Protenix). See ``scripts/pm/diff_dumps.py`` and
``scripts/pm/diff_activations.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _maybe_build_dumper(dump_dir: str | None):
    """Return a callable ``dump(stage_name, tensors_dict)`` or None.

    Each call writes ``<dump_dir>/<stage_name>.npz`` with all given
    tensors cast to float32 numpy arrays. Returns None when
    ``dump_dir`` is None, so callers can use ``if _dump is not None``
    without a no-op branch.
    """
    if dump_dir is None:
        return None

    out = Path(dump_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _dump(stage: str, tensors: dict):
        arrs = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                arrs[k] = v.detach().to(dtype=torch.float32, device="cpu").numpy()
            elif isinstance(v, (int, float, bool)):
                arrs[k] = np.array(v)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        arrs[f"{k}.{sk}"] = (
                            sv.detach().to(dtype=torch.float32, device="cpu").numpy()
                        )
        np.savez_compressed(out / f"{stage}.npz", **arrs)

    _dump.dump_dir = out  # type: ignore[attr-defined]
    return _dump
