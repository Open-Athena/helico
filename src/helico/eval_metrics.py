"""GPU-batched structural-quality metrics for inline training validation.

These mirror the offline numpy versions in `bench.py` but operate on
batched (B, N, 3) tensors with optional (B, N) masks, so they can be
called from the training/validation loop without round-tripping through
CPU. They match the conventions used by AF3 / Protenix / OpenFold3 so
the resulting numbers are directly comparable.

Differences from `model.smooth_lddt_loss`:
  - LDDT here uses HARD step thresholds (0.5/1/2/4 Å), matching the
    published metric, not a sigmoid-soft approximation. Not differentiable
    and only used for evaluation.
"""

from __future__ import annotations

import torch


_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
_GDT_TS_THRESHOLDS = (1.0, 2.0, 4.0, 8.0)


def hard_lddt(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    cutoff: float = 15.0,
    thresholds: tuple[float, ...] = _LDDT_THRESHOLDS,
) -> torch.Tensor:
    """Hard LDDT — exact (non-smooth) Local Distance Difference Test.

    Args:
        pred_coords: (B, N, 3) predicted coordinates.
        gt_coords:   (B, N, 3) ground truth coordinates.
        atom_mask:   (B, N) bool/0-1 mask. If None, all atoms count.
        cutoff:      Inclusion radius in Å (pairs with gt distance < cutoff).
        thresholds:  Per-pair tolerances; score is mean fraction of pairs
                     under each threshold.

    Returns:
        (B,) per-batch LDDT scores in [0, 1].
    """
    if pred_coords.dim() != 3 or gt_coords.dim() != 3:
        raise ValueError("pred_coords and gt_coords must be (B, N, 3)")
    pred = pred_coords.float()
    gt = gt_coords.float()
    B, N, _ = pred.shape

    pred_d = torch.cdist(pred, pred)  # (B, N, N)
    gt_d = torch.cdist(gt, gt)

    pair_in = (gt_d < cutoff) & (gt_d > 0.01)
    if atom_mask is not None:
        am = atom_mask.bool()
        pm = am.unsqueeze(-1) & am.unsqueeze(-2)
        pair_in = pair_in & pm

    diff = (pred_d - gt_d).abs()

    # Per-batch sums; avoid 0/0 with .clamp.
    pair_in_f = pair_in.float()
    denom = pair_in_f.sum(dim=(1, 2)).clamp(min=1.0)
    score_sum = torch.zeros(B, device=pred.device, dtype=pred.dtype)
    for t in thresholds:
        score_sum = score_sum + ((diff < t).float() * pair_in_f).sum(dim=(1, 2)) / denom
    out = score_sum / len(thresholds)
    # If a sample had zero valid pairs, denom was clamped to 1 — actual
    # score is undefined; report 0 so it shows up rather than NaN.
    has_pairs = pair_in_f.sum(dim=(1, 2)) > 0
    return torch.where(has_pairs, out, torch.zeros_like(out))


def kabsch_align(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optimally rotate+translate pred onto gt (Kabsch). Per-batch.

    Args:
        pred_coords: (B, N, 3)
        gt_coords:   (B, N, 3)
        atom_mask:   (B, N), only masked atoms are used to fit the rotation;
                     all atoms are then transformed.

    Returns:
        (B, N, 3) — pred superposed onto gt's reference frame.
    """
    if pred_coords.dim() != 3 or gt_coords.dim() != 3:
        raise ValueError("pred_coords and gt_coords must be (B, N, 3)")
    pred = pred_coords.float()
    gt = gt_coords.float()
    B = pred.shape[0]

    if atom_mask is None:
        am = torch.ones(pred.shape[:2], device=pred.device)
    else:
        am = atom_mask.float()
    weight = am.unsqueeze(-1)

    # Weighted centroids
    n_eff = am.sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    pred_c = (pred * weight).sum(dim=1, keepdim=True) / n_eff
    gt_c = (gt * weight).sum(dim=1, keepdim=True) / n_eff
    P = (pred - pred_c) * weight
    G = (gt - gt_c) * weight

    # Cross-covariance per-batch (fine on GPU).
    H = torch.bmm(P.transpose(1, 2), G)  # (B, 3, 3)
    # Run all the linear algebra on CPU. H is only (B, 3, 3) so the round
    # trip cost is trivial, and torch.linalg.{svd,det} on GPU triggers
    # PyTorch's CUDA kernel JIT, which has been failing with NVRTC version
    # mismatch errors on Modal images (libnvrtc-builtins.so.13.0 missing).
    H_cpu = H.detach().cpu()
    U_cpu, _, Vt_cpu = torch.linalg.svd(H_cpu)
    det_cpu = torch.linalg.det(torch.bmm(Vt_cpu.transpose(1, 2), U_cpu.transpose(1, 2)))
    sign_cpu = torch.ones(B, 3, dtype=H_cpu.dtype)
    sign_cpu[:, -1] = det_cpu.sign()
    R_cpu = torch.bmm(Vt_cpu.transpose(1, 2) * sign_cpu.unsqueeze(1), U_cpu.transpose(1, 2))
    R = R_cpu.to(pred.device, dtype=pred.dtype)

    aligned = torch.bmm(pred - pred_c, R.transpose(1, 2)) + gt_c
    return aligned


def rmsd_after_kabsch(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSD after optimal Kabsch superposition. (B,) Å."""
    aligned = kabsch_align(pred_coords, gt_coords, atom_mask)
    sq = (aligned - gt_coords.float()).pow(2).sum(dim=-1)  # (B, N)
    if atom_mask is None:
        return sq.mean(dim=-1).sqrt()
    am = atom_mask.float()
    n = am.sum(dim=-1).clamp(min=1.0)
    msd = (sq * am).sum(dim=-1) / n
    return msd.sqrt()


def gdt_ts(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    thresholds: tuple[float, ...] = _GDT_TS_THRESHOLDS,
) -> torch.Tensor:
    """GDT-TS after Kabsch — fraction of atoms within 1/2/4/8 Å. (B,) in [0, 1]."""
    aligned = kabsch_align(pred_coords, gt_coords, atom_mask)
    dist = (aligned - gt_coords.float()).pow(2).sum(dim=-1).sqrt()  # (B, N)
    if atom_mask is None:
        am = torch.ones_like(dist)
    else:
        am = atom_mask.float()
    n = am.sum(dim=-1).clamp(min=1.0)
    score = torch.zeros(dist.shape[0], device=dist.device, dtype=dist.dtype)
    for t in thresholds:
        score = score + ((dist < t).float() * am).sum(dim=-1) / n
    return score / len(thresholds)


def mean_plddt(
    plddt_per_atom: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-atom pLDDT, masked. (B,) on the same scale as input (e.g. 0-100)."""
    if plddt_per_atom.dim() != 2:
        raise ValueError("plddt_per_atom must be (B, N_atoms)")
    if atom_mask is None:
        return plddt_per_atom.float().mean(dim=-1)
    am = atom_mask.float()
    n = am.sum(dim=-1).clamp(min=1.0)
    return (plddt_per_atom.float() * am).sum(dim=-1) / n
