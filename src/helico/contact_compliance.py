"""Exact heavy-atom constraint distances with memory-bounded gradients."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def prepare_constraints(features, state, device):
    """Prepare observed heavy atoms and clean, valid supplied pairs on CPU.

    Positive labels need only an observed witness. Restricting their loss to
    observed atoms prevents unresolved coordinates satisfying a constraint.
    Negative labels already require complete tokens in contact_valid.
    """
    pairs = torch.triu(features["contact_valid"] & (state != 0), 1).nonzero()
    atoms = [[] for _ in range(features["n_tokens"])]
    keep = features["coordinate_mask"] & (features["atom_element_idx"] != 0)
    for i in keep.nonzero().flatten().tolist():
        atoms[int(features["atom_to_token"][i])].append(i)
    width = max(1, max(map(len, atoms), default=0))
    indices = torch.full((len(atoms), width), -1, dtype=torch.long)
    for token, members in enumerate(atoms):
        indices[token, :len(members)] = torch.tensor(members, dtype=torch.long)
    if len(pairs) and (indices[pairs][:, :, 0] < 0).any():
        raise ValueError("Valid contact constraint has no observed heavy atoms")
    values = state[pairs[:, 0], pairs[:, 1]]
    if len(pairs) and not torch.equal(values, features["contact_target"][pairs[:, 0], pairs[:, 1]].to(values.dtype) + 1):
        raise ValueError("Training constraints must be clean native labels")
    return {"constraint_atoms": indices.to(device), "constraint_pairs": pairs.to(device),
            "constraint_values": values.to(device)}


def constraint_distances(prediction, atoms, pairs, chunk_size=256):
    """Minimum distance for each sample/pair; exact subgradient off ties.

    Select the closest atom pair without a graph, then differentiate its norm.
    This is the gradient of a minimum, without retaining all atom-pair distance
    graphs for a fully revealed contact matrix. Memory scales with chunk size.
    """
    xyz = prediction.float().reshape(-1, prediction.shape[-2], 3)
    width = atoms.shape[1]
    distances = []
    for chunk in pairs.split(chunk_size):
        left, right = atoms[chunk[:, 0]], atoms[chunk[:, 1]]
        with torch.no_grad(), torch.autocast(device_type=xyz.device.type, enabled=False):
            d = torch.cdist(xyz[:, left.clamp_min(0)], xyz[:, right.clamp_min(0)],
                            compute_mode="donot_use_mm_for_euclid_dist")
            d.masked_fill_((left < 0)[None, :, :, None] | (right < 0)[None, :, None, :], float("inf"))
            nearest = d.flatten(-2).argmin(-1)
            a = left.unsqueeze(0).expand(len(xyz), -1, -1).gather(-1, (nearest // width).unsqueeze(-1)).squeeze(-1)
            b = right.unsqueeze(0).expand(len(xyz), -1, -1).gather(-1, (nearest % width).unsqueeze(-1)).squeeze(-1)
        sample = torch.arange(len(xyz), device=xyz.device)[:, None]
        distances.append(torch.linalg.vector_norm(xyz[sample, a] - xyz[sample, b], dim=-1))
    return torch.cat(distances, dim=-1) if distances else xyz[:, :0, 0]


def compliance_loss(prediction, batch, *, threshold=5., margin=.1, chunk_size=256):
    """Class-balanced smooth-L1 hinge on supplied positive/negative geometry.

    Average each present class over its pairs and all denoising samples, then
    average present classes. This reweights the loss, never the revelation.
    Empty conditioning has differentiable zero loss. Units are Angstroms.
    """
    values = batch["constraint_values"]
    distances = constraint_distances(prediction, batch["constraint_atoms"],
                                     batch["constraint_pairs"], chunk_size)
    zero = prediction.float().sum() * 0
    losses, metrics = [], {}
    for value, name in ((2, "positive"), (1, "negative")):
        mask = values == value
        count = int(mask.sum())
        if count:
            d = distances[:, mask]
            violation = F.relu(d - (threshold - margin) if value == 2 else (threshold + margin) - d)
            loss = F.smooth_l1_loss(violation, torch.zeros_like(violation))
            satisfied = ((d < threshold) if value == 2 else (d >= threshold)).float().mean()
            losses.append(loss)
        else:
            loss, satisfied = zero, zero.detach()
        metrics[name + "_loss"] = loss
        metrics[name + "_satisfaction"] = satisfied.detach()
        metrics[name + "_count"] = count
    return (torch.stack(losses).mean() if losses else zero), metrics
