"""Diff Helico's feature batch against Protenix's input_feature_dict.

Loads two `00_batch.npz` files (Helico's from `helico.predict` dumps,
Protenix's from `bench_upstream.UpstreamPredictor.predict_and_dump`) and
reports:

  1. Keys present only on one side.
  2. For common keys: shape, dtype, finite-fraction, simple stats.
  3. For semantically equivalent but differently-named fields, a
     side-by-side comparison under an `alias` map.

The goal is to find the first divergence in featurization — not to
assert bit-equality.

Usage:
    uv run python scripts/pm/diff_dumps.py \\
        experiments/exp8_ab_ag_triage/data/helico_dump/8t59-assembly1 \\
        experiments/exp8_ab_ag_triage/data/upstream_dump/8t59-assembly1/dump
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Helico key -> Protenix key. Used for side-by-side on semantically
# equivalent fields. The lists below are best-effort; the tool still
# prints everything, so new pairings can be spotted in the raw output.
ALIAS: dict[str, str] = {
    "chain_indices": "asym_id",
    "sym_id": "sym_id",
    "entity_id": "entity_id",
    "msa": "msa",
    "ref_coords": "ref_pos",
    "res_indices": "residue_index",
    "token_index": "token_index",
    "has_frame": "has_frame",
    "atom_to_token": "atom_to_token_idx",
    "restype": "restype",
}


def _stat(arr: np.ndarray) -> str:
    if arr.size == 0:
        return f"shape={tuple(arr.shape)} empty"
    flat = arr.ravel()
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(flat)
        if finite.any():
            f = flat[finite]
            return (f"shape={tuple(arr.shape)} dtype={arr.dtype} "
                    f"range=[{f.min():+.4g},{f.max():+.4g}] "
                    f"mean={f.mean():+.4g} frac_finite={finite.mean():.3f}")
        return f"shape={tuple(arr.shape)} dtype={arr.dtype} all-nonfinite"
    # Integer-ish
    uniq, counts = np.unique(flat, return_counts=True)
    if uniq.size <= 12:
        dist = " ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq, counts))
    else:
        dist = f"n_unique={uniq.size} min={int(uniq.min())} max={int(uniq.max())}"
    return f"shape={tuple(arr.shape)} dtype={arr.dtype} {dist}"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _fmt_keylist(keys):
    keys = sorted(keys)
    if not keys:
        return "  (none)"
    w = max(len(k) for k in keys) + 2
    lines = []
    row = []
    per_row = max(1, 80 // w)
    for i, k in enumerate(keys):
        row.append(k.ljust(w))
        if (i + 1) % per_row == 0:
            lines.append("  " + "".join(row))
            row = []
    if row:
        lines.append("  " + "".join(row))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("helico_dir", type=Path,
                    help="Helico dump dir containing 00_batch.npz")
    ap.add_argument("protenix_dir", type=Path,
                    help="Protenix dump dir containing 00_batch.npz")
    args = ap.parse_args()

    h_path = args.helico_dir / "00_batch.npz"
    p_path = args.protenix_dir / "00_batch.npz"
    if not h_path.exists():
        raise SystemExit(f"missing: {h_path}")
    if not p_path.exists():
        raise SystemExit(f"missing: {p_path}")

    h = _load(h_path)
    p = _load(p_path)

    h_keys = set(h.keys())
    p_keys = set(p.keys())

    print("=" * 80)
    print(f"HELICO  : {h_path}  ({len(h_keys)} keys)")
    print(f"PROTENIX: {p_path}  ({len(p_keys)} keys)")
    print("=" * 80)

    # Token-count sanity
    for label, arr_dict, cand_keys in [
        ("helico", h, ["token_index", "chain_indices", "restype"]),
        ("protenix", p, ["token_index", "asym_id", "restype"]),
    ]:
        for k in cand_keys:
            if k in arr_dict:
                print(f"  {label} N_token ({k}): {arr_dict[k].shape}")
                break

    print()
    print("[keys only in Helico]")
    print(_fmt_keylist(h_keys - p_keys))
    print()
    print("[keys only in Protenix]")
    print(_fmt_keylist(p_keys - h_keys))

    print()
    print("[common keys]")
    for k in sorted(h_keys & p_keys):
        ha, pa = h[k], p[k]
        match = "shape_match" if ha.shape == pa.shape else "SHAPE_MISMATCH"
        print(f"  {k:35s} [{match}]")
        print(f"    helico   : {_stat(ha)}")
        print(f"    protenix : {_stat(pa)}")
        if ha.shape == pa.shape and np.issubdtype(ha.dtype, np.number) and np.issubdtype(pa.dtype, np.number):
            # Quick similarity check
            try:
                if ha.dtype == pa.dtype or (np.issubdtype(ha.dtype, np.integer) and np.issubdtype(pa.dtype, np.integer)):
                    eq = float((ha == pa).mean())
                    print(f"    exact_eq_frac: {eq:.4f}")
                else:
                    diff = np.abs(ha.astype(np.float64) - pa.astype(np.float64))
                    print(f"    |h-p|: mean={diff.mean():.4g}  max={diff.max():.4g}")
            except Exception as e:
                print(f"    (compare skipped: {e})")

    print()
    print("[aliased fields] Helico -> Protenix")
    for hk, pk in ALIAS.items():
        if hk in h_keys and pk in p_keys and hk != pk:
            ha, pa = h[hk], p[pk]
            match = "shape_match" if ha.shape == pa.shape else "SHAPE_MISMATCH"
            print(f"  {hk:20s} -> {pk:20s} [{match}]")
            print(f"    helico   : {_stat(ha)}")
            print(f"    protenix : {_stat(pa)}")

    # Meta
    meta = args.protenix_dir / "sample_meta.json"
    if meta.exists():
        print()
        print("[protenix sample meta]")
        with open(meta) as f:
            m = json.load(f)
        for k in ("sample_name", "N_asym", "N_token", "N_atom", "N_msa"):
            print(f"  {k}: {m.get(k)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
