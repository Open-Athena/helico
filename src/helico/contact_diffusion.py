"""Clean absorbing-mask contact labels, independent of the legacy pyconfind path.

Geometry and observation validity are separate: missing atoms never establish
an absence. The absorbing chain masks unordered pairs together, independently
of their values. Oversampling particular times does not change that kernel.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F


def heavy_atom_contacts(coords, atom_to_token, n_tokens: int, threshold: float = 5.0,
                        atom_mask=None) -> torch.Tensor:
    """Return symmetric min-heavy-atom-distance < threshold labels (CPU).

Callers must exclude hydrogens/deuterium in atom_mask or upstream tokenization.
The diagonal is always false; eligibility for supervision is supplied separately.
"""
    xyz = torch.as_tensor(coords).detach().float().cpu().numpy()
    indices = torch.as_tensor(atom_to_token).detach().cpu().numpy()
    keep = np.isfinite(xyz).all(axis=-1)
    if atom_mask is not None:
        keep &= torch.as_tensor(atom_mask).detach().cpu().numpy().astype(bool)
    xyz, indices = xyz[keep], indices[keep]
    pairs = cKDTree(xyz).query_pairs(threshold, output_type="ndarray")
    labels = np.zeros((n_tokens, n_tokens), dtype=bool)
    if len(pairs):
        # query_pairs includes the boundary; our definition is strictly <.
        pairs = pairs[np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=-1) < threshold]
        i, j = indices[pairs[:, 0]], indices[pairs[:, 1]]
        labels[i, j] = labels[j, i] = True
    np.fill_diagonal(labels, False)
    return torch.from_numpy(labels)


def absorbing_mask(target: torch.Tensor, valid: torch.Tensor, t: int, *,
                   steps: int = 1000, generator=None) -> torch.Tensor:
    """q(x_t|x_0): retain each unordered valid pair with probability 1-t/T.

States follow Helico's convention: unknown=0, absent=1, present=2.
There is no label corruption or class-dependent revelation.
"""
    if not 0 <= t <= steps or steps <= 0:
        raise ValueError("Require 0 <= t <= steps and steps > 0")
    if target.ndim != 2 or target.shape != valid.shape or target.shape[0] != target.shape[1]:
        raise ValueError("Expected square target and validity matrices")
    if not torch.equal(target, target.T) or not torch.equal(valid, valid.T):
        raise ValueError("Contact labels and validity must be symmetric")
    keep = torch.rand(target.shape, generator=generator, device=target.device) < (1 - t / steps)
    keep = torch.triu(keep & valid, diagonal=1)
    keep = keep | keep.T
    return torch.where(keep, target.to(torch.uint8) + 1, 0)


def contact_bce(logits, target, valid, conditioning):
    """Unweighted BCE on masked and observed pairs, counted once per example.

Empty subsets give differentiable zero. Keeping the two losses separate avoids
mistaking trivial reconstruction of visible labels for denoising performance.
"""
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    eligible = torch.triu(valid.bool(), diagonal=1)
    def mean(mask):
        count = mask.sum((-1, -2))
        per_example = (loss * mask).sum((-1, -2)) / count.clamp_min(1)
        return per_example.mean()
    return {"contact_loss": mean(eligible & (conditioning == 0)),
            "contact_observed_loss": mean(eligible & (conditioning != 0))}
