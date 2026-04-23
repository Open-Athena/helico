"""Diff intermediate activations between Helico and Protenix dumps.

Helico dumps 4 stages (00_batch, 01_pre_recycle, 02_post_recycle,
03_post_diffusion) via `model.predict(..., dump_intermediates_to=...)`.
Protenix dumps the same stages via bench_upstream.UpstreamPredictor.predict_and_dump.

Since Helico sees the full structure (with ions) and Protenix sees
protein-only, we align via a token mask derived from Helico's
chain_indices.

Usage:
    uv run python scripts/pm/diff_activations.py \\
        experiments/exp8_ab_ag_triage/data/diff_dumps/helico/8t59-assembly1 \\
        experiments/exp8_ab_ag_triage/data/upstream_dump/8t59-assembly1/dump_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _norm_stats(label: str, arr: np.ndarray) -> str:
    flat = arr.reshape(-1)
    mu = flat.mean()
    std = flat.std()
    return (f"{label}: shape={tuple(arr.shape)} dtype={arr.dtype} "
            f"mean={mu:+.4g} std={std:+.4g} "
            f"min={flat.min():+.4g} max={flat.max():+.4g}")


def _align_tok_axis(arr: np.ndarray, tok_mask: np.ndarray, tok_dim: int) -> np.ndarray:
    """Slice Helico's array along token dim to match Protenix's size."""
    idx = np.where(tok_mask)[0]
    return np.take(arr, idx, axis=tok_dim)


def _align_atom_axis(arr: np.ndarray, atom_mask: np.ndarray, atom_dim: int) -> np.ndarray:
    idx = np.where(atom_mask)[0]
    return np.take(arr, idx, axis=atom_dim)


def _compare_arrays(ha: np.ndarray, pa: np.ndarray, label: str) -> None:
    if ha.shape != pa.shape:
        print(f"  {label} SHAPE_MISMATCH  helico={tuple(ha.shape)} protenix={tuple(pa.shape)}")
        return
    diff = ha.astype(np.float64) - pa.astype(np.float64)
    abs_diff = np.abs(diff)
    print(f"  {label}")
    print(f"    helico   : mean={ha.mean():+.4g} std={ha.std():+.4g} "
          f"range=[{ha.min():+.4g},{ha.max():+.4g}]")
    print(f"    protenix : mean={pa.mean():+.4g} std={pa.std():+.4g} "
          f"range=[{pa.min():+.4g},{pa.max():+.4g}]")
    print(f"    |h-p|    : mean={abs_diff.mean():.4g} "
          f"max={abs_diff.max():.4g} "
          f"rel_L2={np.linalg.norm(diff) / (np.linalg.norm(pa) + 1e-8):.4g}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("helico_dir", type=Path)
    ap.add_argument("protenix_dir", type=Path)
    args = ap.parse_args()

    # Load Helico's batch to derive protein-token mask
    h_batch = _load(args.helico_dir / "00_batch.npz")
    chain_indices = h_batch["chain_indices"][0]  # (447,)
    protein_tok_mask = chain_indices < 3
    # Atom-level mask: keep atoms whose token is in protein_tok_mask
    atom_to_token = h_batch["atom_to_token"][0]  # (3387,)
    protein_atom_mask = np.isin(atom_to_token, np.where(protein_tok_mask)[0])

    print(f"Alignment: helico protein tokens={protein_tok_mask.sum()} "
          f"atoms={protein_atom_mask.sum()}")

    # --- 01_pre_recycle ---
    print("\n" + "=" * 80)
    print("[01_pre_recycle]")
    print("=" * 80)
    h = _load(args.helico_dir / "01_pre_recycle.npz")
    p_path = args.protenix_dir / "01_pre_recycle.npz"
    if not p_path.exists():
        print(f"  (missing {p_path})")
    else:
        p = _load(p_path)
        common = sorted(set(h) & set(p))
        only_h = sorted(set(h) - set(p))
        only_p = sorted(set(p) - set(h))
        print(f"  common: {common}")
        print(f"  only_helico: {only_h}")
        print(f"  only_protenix: {only_p}")
        for k in ("s_inputs", "s_init"):
            if k in h and k in p:
                ha = h[k][0] if h[k].ndim == p[k].ndim + 1 else h[k]
                # Align token axis (first)
                ha_aligned = _align_tok_axis(ha, protein_tok_mask, 0)
                _compare_arrays(ha_aligned, p[k], k)
        if "z_init" in h and "z_init" in p:
            ha = h["z_init"][0]  # (447, 447, c_z)
            # Slice both token axes
            ha_aligned = _align_tok_axis(_align_tok_axis(ha, protein_tok_mask, 0),
                                          protein_tok_mask, 1)
            _compare_arrays(ha_aligned, p["z_init"], "z_init")
        if "relpe_pair" in p:
            print(f"  {_norm_stats('protenix relpe_pair', p['relpe_pair'])}")

    # --- 02_post_recycle ---
    print("\n" + "=" * 80)
    print("[02_post_recycle]")
    print("=" * 80)
    h = _load(args.helico_dir / "02_post_recycle.npz")
    p_path = args.protenix_dir / "02_post_recycle.npz"
    if not p_path.exists():
        print(f"  (missing {p_path})")
    else:
        p = _load(p_path)
        for k in ("s", "z", "s_inputs"):
            if k in h and k in p:
                ha = h[k][0]
                if k == "z":
                    ha_aligned = _align_tok_axis(_align_tok_axis(ha, protein_tok_mask, 0),
                                                  protein_tok_mask, 1)
                else:
                    ha_aligned = _align_tok_axis(ha, protein_tok_mask, 0)
                _compare_arrays(ha_aligned, p[k], k)

    # --- 03_post_diffusion ---
    print("\n" + "=" * 80)
    print("[03_post_diffusion]")
    print("=" * 80)
    h = _load(args.helico_dir / "03_post_diffusion.npz")
    p_path = args.protenix_dir / "03_post_diffusion.npz"
    if not p_path.exists():
        print(f"  (missing {p_path})")
    else:
        p = _load(p_path)
        print(f"  helico keys: {sorted(h.keys())}")
        print(f"  protenix keys: {sorted(p.keys())}")
        # Try to match common keys
        matches = [
            ("all_coords", "coordinate"),   # helico_key, protenix_key
            ("pae", "pae"),
            ("plddt_flat", "plddt"),
            ("ptm", "ptm"),
            ("iptm", "iptm"),
        ]
        for hk, pk in matches:
            if hk in h and pk in p:
                print(f"  {hk} vs {pk}")
                print(f"    helico:   {_norm_stats('', h[hk])}")
                print(f"    protenix: {_norm_stats('', p[pk])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
