"""Inspect Helico pipeline-diff dumps for a target.

Prints a per-stage summary: tensor shapes, dtype, finite-fraction,
min/mean/max, and a few specific features (MSA depth, asym_id
distribution, ref_coords validity) that are common places for
featurization bugs.

Usage:
    uv run python scripts/pm/inspect_dumps.py path/to/dump/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_stage(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _summary(name: str, arr: np.ndarray) -> str:
    if arr.size == 0:
        return f"{name:40s}  shape={tuple(arr.shape)!s:<20s}  empty"
    flat = arr.ravel()
    finite_mask = np.isfinite(flat)
    frac_finite = finite_mask.mean()
    suffix = ""
    if frac_finite == 1.0:
        mn, mx, mu = flat.min(), flat.max(), flat.mean()
        suffix = f"range=[{mn:+.4g}, {mx:+.4g}]  mean={mu:+.4g}"
    elif frac_finite > 0:
        f = flat[finite_mask]
        suffix = f"range=[{f.min():+.4g}, {f.max():+.4g}]  mean={f.mean():+.4g}  frac_finite={frac_finite:.2f}"
    else:
        suffix = "all-nonfinite"
    return f"{name:40s}  shape={tuple(arr.shape)!s:<20s}  dtype={str(arr.dtype):<10s}  {suffix}"


def _inspect_batch(arrs: dict[str, np.ndarray]) -> list[str]:
    lines = ["[00_batch] input features to Helico.predict"]
    for k in sorted(arrs):
        lines.append("  " + _summary(k, arrs[k]))

    # Spot-checks
    lines.append("")
    lines.append("  --- spot checks ---")
    if "msa" in arrs:
        msa = arrs["msa"]
        # Expected shape: (B, N_seqs, N_tok, ...) or similar
        lines.append(f"  MSA shape: {tuple(msa.shape)}  "
                     f"# unique vals in row 0 = "
                     f"{int(np.unique(msa.reshape(-1, msa.shape[-1])[0]).size) if msa.ndim >= 2 else 'n/a'}")
    if "token_types" in arrs:
        tt = arrs["token_types"]
        uniq, counts = np.unique(tt, return_counts=True)
        top = sorted(zip(counts, uniq), reverse=True)[:6]
        lines.append(f"  token_types top-6: {[(int(u), int(c)) for c, u in top]}")
    if "asym_id" in arrs:
        aid = arrs["asym_id"]
        uniq, counts = np.unique(aid, return_counts=True)
        lines.append(f"  asym_id distribution: {list(zip(uniq.tolist(), counts.tolist()))}")
    if "chain_indices" in arrs:
        ci = arrs["chain_indices"]
        uniq, counts = np.unique(ci, return_counts=True)
        lines.append(f"  chain_indices distribution: {list(zip(uniq.tolist(), counts.tolist()))}")
    if "ref_coords" in arrs:
        rc = arrs["ref_coords"]
        zero_frac = (np.linalg.norm(rc.reshape(-1, 3), axis=-1) == 0).mean()
        lines.append(f"  ref_coords zero-position fraction: {zero_frac:.3f}")
    return lines


def _inspect_pre_recycle(arrs: dict[str, np.ndarray]) -> list[str]:
    lines = ["[01_pre_recycle] trunk init + MSA raw"]
    for k in sorted(arrs):
        lines.append("  " + _summary(k, arrs[k]))
    # Check MSA
    if "msa_raw" in arrs:
        m = arrs["msa_raw"]
        lines.append(f"  msa_raw sparsity: nonzero_frac={np.mean(m != 0):.3f}")
    if "msa_mask" in arrs:
        mm = arrs["msa_mask"]
        lines.append(f"  msa_mask mean: {mm.mean():.3f}  sum: {mm.sum():.0f}")
    return lines


def _inspect_post_recycle(arrs: dict[str, np.ndarray]) -> list[str]:
    lines = ["[02_post_recycle] final s, z after recycling"]
    for k in sorted(arrs):
        lines.append("  " + _summary(k, arrs[k]))
    return lines


def _inspect_post_diffusion(arrs: dict[str, np.ndarray]) -> list[str]:
    lines = ["[03_post_diffusion] coords + confidence"]
    for k in sorted(arrs):
        lines.append("  " + _summary(k, arrs[k]))
    if "ranking_score_per_sample" in arrs:
        rs = arrs["ranking_score_per_sample"].ravel()
        lines.append(f"  ranking_score: {rs.tolist()}")
    if "ptm" in arrs:
        lines.append(f"  pTM: {arrs['ptm'].tolist()}  iPTM: {arrs['iptm'].tolist()}")
    if "all_coords" in arrs:
        ac = arrs["all_coords"]
        lines.append(f"  all_coords: {tuple(ac.shape)}  "
                     f"(B={ac.shape[0]}, n_samples={ac.shape[1]}, N_atoms={ac.shape[2]})")
    return lines


INSPECTORS = {
    "00_batch": _inspect_batch,
    "01_pre_recycle": _inspect_pre_recycle,
    "02_post_recycle": _inspect_post_recycle,
    "03_post_diffusion": _inspect_post_diffusion,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path, help="Directory containing *.npz dumps")
    args = ap.parse_args(argv)

    if not args.dump_dir.is_dir():
        raise SystemExit(f"not a directory: {args.dump_dir}")

    for stage, inspect in INSPECTORS.items():
        path = args.dump_dir / f"{stage}.npz"
        if not path.exists():
            print(f"[MISSING] {path}", file=sys.stderr)
            continue
        arrs = _load_stage(path)
        for line in inspect(arrs):
            print(line)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
