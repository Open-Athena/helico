"""Loss functions for training.

- ``diffusion_loss``: EDM weighted MSE on denoised coords (AF3 SI §3.7.1 Eq. 3)
- ``smooth_lddt_loss``: differentiable lDDT (AF3 SI Algorithm 27)
- ``distogram_loss``: cross-entropy on binned pairwise distances (§4.4)
- ``violation_loss``: soft steric-clash penalty
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_loss(
    x_denoised: torch.Tensor,
    gt_coords: torch.Tensor,
    sigma: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """EDM diffusion loss: weighted MSE on denoised coordinates.

    AF3 SI Eq. 3: L_MSE = mean_l(w_l · ‖x_l - x_l_GT_aligned‖²) with per-atom
    weights w_l (upweighted for DNA/RNA/ligand atoms, Eq. 4). The EDM
    weighting factor 1/σ² normalizes across sampled noise levels so the
    gradient doesn't blow up at small σ.
    """
    while sigma.dim() < gt_coords.dim():
        sigma = sigma.unsqueeze(-1)
    weight = 1.0 / sigma.pow(2).clamp(min=1e-6)

    loss = weight * (x_denoised - gt_coords).pow(2).sum(dim=-1)

    if atom_mask is not None:
        loss = loss * atom_mask
        return loss.sum() / atom_mask.sum().clamp(min=1)
    return loss.mean()


def smooth_lddt_loss(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    cutoff: float = 15.0,
) -> torch.Tensor:
    """Differentiable lDDT — AF3 SI Algorithm 27.

    lDDT is the fraction of ground-truth pairwise distances within 15 Å
    that are preserved in the prediction to within {0.5, 1, 2, 4} Å.
    The smooth variant replaces the hard thresholds with sigmoids so the
    metric is differentiable:
      ε_lm = 1/4 [sigmoid(½ - |Δ|) + sigmoid(1 - |Δ|) + sigmoid(2 - |Δ|) + sigmoid(4 - |Δ|)]
    where Δ = |d_lm - d_lm_GT|.

    Helico's implementation uses a single sigmoid at each threshold with
    sharpness 5 rather than the exact set of four thresholds — simpler
    but still a close approximation.
    """
    pred_dists = torch.cdist(pred_coords, pred_coords)
    gt_dists = torch.cdist(gt_coords, gt_coords)

    close_mask = (gt_dists < cutoff) & (gt_dists > 0.01)

    if atom_mask is not None:
        pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
        close_mask = close_mask & pair_mask

    diff = torch.abs(pred_dists - gt_dists)

    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=pred_coords.device)
    # Smooth (sigmoid) thresholding instead of step function
    scores = torch.sigmoid(5.0 * (thresholds.view(1, 1, 1, -1) - diff.unsqueeze(-1)))
    score = scores.mean(dim=-1)  # average the 4 thresholds

    if close_mask.any():
        lddt = (score * close_mask).sum() / close_mask.sum().clamp(min=1)
    else:
        lddt = torch.tensor(1.0, device=pred_coords.device)

    return 1.0 - lddt


def distogram_loss(
    pred_logits: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    n_bins: int = 64,
) -> torch.Tensor:
    """Cross-entropy on binned pairwise distances — AF3 SI §4.4.

    The DistogramHead predicts ``n_bins`` logits per pair; we bucketize
    the ground-truth pairwise distance into those bins and compute CE.
    ``pred_logits`` is expected to be symmetrized by the DistogramHead.
    """
    gt_dists = torch.cdist(gt_coords, gt_coords)

    boundaries = torch.linspace(min_dist, max_dist, n_bins - 1, device=gt_coords.device)
    gt_bins = torch.bucketize(gt_dists, boundaries)

    loss = F.cross_entropy(
        pred_logits.reshape(-1, n_bins),
        gt_bins.reshape(-1),
        reduction="none",
    ).reshape(gt_bins.shape)

    if atom_mask is not None:
        pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
        loss = loss * pair_mask
        return loss.sum() / pair_mask.sum().clamp(min=1)
    return loss.mean()


def violation_loss(
    pred_coords: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    clash_threshold: float = 1.2,
) -> torch.Tensor:
    """Steric clash penalty.

    Soft ReLU on (threshold - pairwise distance). Not part of AF3 SI as
    a standalone loss — AF3 uses bonded-ligand length loss (Eq. 5); this
    is a simpler per-atom clash term that's useful for training from
    scratch.
    """
    dists = torch.cdist(pred_coords, pred_coords)

    # Exclude self-distances by adding a large value on the diagonal
    eye = torch.eye(dists.shape[1], device=dists.device).unsqueeze(0)
    dists = dists + eye * 1e6

    clash = F.relu(clash_threshold - dists)

    if atom_mask is not None:
        pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
        clash = clash * pair_mask
        return clash.sum() / pair_mask.sum().clamp(min=1)
    return clash.mean()
