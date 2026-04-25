"""Confidence-score computation — AF3 SI §4.3, §5.9, Algorithm 27-style.

Post-processing functions that turn ConfidenceHead logits into
interpretable scores:

- ``compute_plddt``    — per-atom pLDDT in [0, 100] (§4.3.1)
- ``compute_pae``      — PAE matrix in Å (§4.3.2)
- ``compute_ptm``      — predicted TM-score (SI §5.9.1)
- ``compute_iptm``     — interface predicted TM-score (§5.9.1)
- ``compute_clash``    — inter-chain atom-clash flag (§5.9.2)
- ``compute_ranking_score``   — AF3's 0.8·iptm + 0.2·ptm − 100·clash (§5.9.3)
- ``_flatten_plddt``   — (B, N_tok, max_atoms_per_token) → (B, N_atoms)

All are pure functions operating on tensors; no learned parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_plddt(plddt_logits: torch.Tensor) -> torch.Tensor:
    """Per-atom pLDDT in [0, 100].

    ``plddt_logits`` is (B, N_tok, max_atoms, n_plddt_bins=50). Bins cover
    [0, 1] so the expected value multiplied by 100 gives the usual
    AlphaFold-style 0-100 score.
    """
    n_bins = plddt_logits.shape[-1]
    bin_centers = torch.linspace(
        1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins,
        device=plddt_logits.device, dtype=plddt_logits.dtype,
    )
    probs = F.softmax(plddt_logits, dim=-1)
    plddt = (probs * bin_centers).sum(dim=-1)
    return plddt * 100.0


def compute_pae(pae_logits: torch.Tensor) -> torch.Tensor:
    """PAE matrix in Å — AF3 SI §4.3.2.

    64 bins of width 0.5 Å covering 0-32 Å (bin centers 0.25, 0.75, ..., 31.75).
    """
    n_bins = pae_logits.shape[-1]
    bin_centers = torch.linspace(
        0.25, 31.75, n_bins,
        device=pae_logits.device, dtype=pae_logits.dtype,
    )
    probs = F.softmax(pae_logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


def _compute_tm_term(pae_logits: torch.Tensor, d0: torch.Tensor) -> torch.Tensor:
    """Expected TM-score contribution per pair, given PAE logits and d0.

    TM per bin = 1 / (1 + (d/d0)²). Expectation over the softmax gives
    the contribution pTM / ipTM sum over each (i, j).
    """
    n_bins = pae_logits.shape[-1]
    bin_centers = torch.linspace(
        0.25, 31.75, n_bins,
        device=pae_logits.device, dtype=pae_logits.dtype,
    )
    probs = F.softmax(pae_logits, dim=-1)
    tm_per_bin = 1.0 / (1.0 + (bin_centers / d0.unsqueeze(-1)) ** 2)
    return (probs * tm_per_bin).sum(dim=-1)


def compute_ptm(
    pae_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    has_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predicted TM-score — AF3 SI §5.9.1.

    pTM = max_i mean_j TM_ij where TM_ij uses the Zhang-Skolnick d0(N_res)
    scaling: d0 = 1.24·(max(N_res,34) - 15)^(1/3) - 1.8. ``has_frame``
    masks tokens that don't contribute (ligand atoms without a local
    frame can still receive mean TM from others but aren't the anchor i).
    """
    B, N = pae_logits.shape[:2]
    device = pae_logits.device

    if mask is None:
        mask = torch.ones(B, N, device=device, dtype=pae_logits.dtype)
    else:
        mask = mask.to(dtype=pae_logits.dtype)

    n_res = mask.sum(dim=-1).clamp(min=19)
    d0 = 1.24 * (n_res.clamp(min=19 + 15) - 15).pow(1.0 / 3.0) - 1.8
    d0 = d0.reshape(B, 1, 1)

    tm_pair = _compute_tm_term(pae_logits, d0)
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    tm_pair = tm_pair * pair_mask

    n_scored = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    tm_per_aligned = tm_pair.sum(dim=-1) / n_scored

    # pTM = max over alignment dimension, filtered by has_frame
    frame_mask = mask.clone()
    if has_frame is not None:
        frame_mask = frame_mask * has_frame.to(dtype=frame_mask.dtype)
    tm_per_aligned = tm_per_aligned.masked_fill(frame_mask == 0, 0.0)
    return tm_per_aligned.max(dim=-1).values


def compute_iptm(
    pae_logits: torch.Tensor,
    chain_indices: torch.Tensor,
    mask: torch.Tensor | None = None,
    has_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    """Interface predicted TM-score — AF3 SI §5.9.1.

    Same as pTM but averages only over inter-chain pairs (chain_i != chain_j).
    Returns 0 if there are no inter-chain pairs.
    """
    B, N = pae_logits.shape[:2]
    device = pae_logits.device

    if mask is None:
        mask = torch.ones(B, N, device=device, dtype=pae_logits.dtype)
    else:
        mask = mask.to(dtype=pae_logits.dtype)

    inter_mask = (chain_indices.unsqueeze(-1) != chain_indices.unsqueeze(-2)).float()
    pair_mask = inter_mask * mask.unsqueeze(-1) * mask.unsqueeze(-2)

    n_res = mask.sum(dim=-1).clamp(min=19)
    d0 = 1.24 * (n_res.clamp(min=19 + 15) - 15).pow(1.0 / 3.0) - 1.8
    d0 = d0.reshape(B, 1, 1)

    tm_pair = _compute_tm_term(pae_logits, d0)
    tm_pair = tm_pair * pair_mask

    n_inter = pair_mask.sum(dim=-1).clamp(min=1)
    tm_per_aligned = tm_pair.sum(dim=-1) / n_inter

    has_inter = (pair_mask.sum(dim=-1) > 0).float()
    frame_mask = has_inter.clone()
    if has_frame is not None:
        frame_mask = frame_mask * has_frame.to(dtype=frame_mask.dtype)
    tm_per_aligned = tm_per_aligned * frame_mask
    any_inter = has_inter.sum(dim=-1) > 0
    iptm = tm_per_aligned.max(dim=-1).values
    return iptm * any_inter.float()


def compute_clash(
    pred_coords: torch.Tensor,
    chain_indices: torch.Tensor,
    atom_to_token: torch.Tensor,
    atom_mask: torch.Tensor,
    threshold: float = 1.1,
) -> torch.Tensor:
    """Inter-chain atom-clash flag — AF3 SI §5.9.2 / Protenix metrics/clash.py.

    Per Protenix's `Clash.get_chain_pair_violations` (the source the
    Protenix v1 ranking_score was trained against), a chain pair is
    flagged as clashing iff EITHER:

      * total clashing atom pairs > 100, OR
      * (clashing atom pairs / min(N_atoms_chain_i, N_atoms_chain_j)) > 0.5

    A sample is flagged as clashing iff ANY chain pair is. Earlier
    Helico logic flagged "any single inter-chain atom-pair < 1.1Å"
    which is far more sensitive — legitimate close interface contacts
    trip the flag, giving good predictions the −100 ranking_score
    penalty and demoting them below worse predictions.

    Atoms are subsampled per chain (cap 5000 each) to bound memory on
    very large structures; counts are scaled accordingly so the >100 /
    >0.5 thresholds remain meaningful.
    """
    B = pred_coords.shape[0]
    device = pred_coords.device
    has_clash = torch.zeros(B, device=device)

    for b in range(B):
        mask = atom_mask[b].bool()
        coords = pred_coords[b][mask]
        tok_ids = atom_to_token[b][mask]
        chain_ids = chain_indices[b][tok_ids]

        # Iterate over unique chain ids; for each pair compute the
        # number of < threshold inter-atom pairs and apply the AF3 dual
        # criterion (>100 OR >50%).
        uniq = torch.unique(chain_ids)
        if uniq.numel() < 2:
            continue
        sample_clash = False
        for i_idx in range(uniq.numel()):
            if sample_clash:
                break
            ci = uniq[i_idx]
            mask_i = chain_ids == ci
            coords_i = coords[mask_i].float()
            n_i = coords_i.shape[0]
            if n_i > 5000:
                sub = torch.randperm(n_i, device=device)[:5000]
                coords_i = coords_i[sub]
                n_i = coords_i.shape[0]
            for j_idx in range(i_idx + 1, uniq.numel()):
                cj = uniq[j_idx]
                mask_j = chain_ids == cj
                coords_j = coords[mask_j].float()
                n_j = coords_j.shape[0]
                if n_j > 5000:
                    sub = torch.randperm(n_j, device=device)[:5000]
                    coords_j = coords_j[sub]
                    n_j = coords_j.shape[0]
                d = torch.cdist(coords_i, coords_j)
                clash_count = int((d < threshold).sum().item())
                rel = clash_count / max(min(n_i, n_j), 1)
                if clash_count > 100 or rel > 0.5:
                    sample_clash = True
                    break
        has_clash[b] = 1.0 if sample_clash else 0.0

    return has_clash


def compute_ranking_score(
    ptm: torch.Tensor,
    iptm: torch.Tensor,
    has_interface: torch.Tensor,
    has_clash: torch.Tensor | None = None,
) -> torch.Tensor:
    """AF3 SI §5.9.3 — sample ranking score.

    AF3 uses ``0.8·iptm + 0.2·ptm − 100·has_clash`` for multi-chain
    targets, falling back to pure pTM for single-chain. The heavy clash
    penalty (−100) effectively disqualifies any sample with an
    inter-chain clash.
    """
    multi = has_interface.float()
    score = multi * (0.8 * iptm + 0.2 * ptm) + (1.0 - multi) * ptm
    if has_clash is not None:
        score = score - 100.0 * has_clash
    return score


def _flatten_plddt(
    plddt: torch.Tensor,
    atom_to_token: torch.Tensor,
    atoms_per_token: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """(B, N_tok, max_atoms_per_token) → (B, N_atoms).

    The confidence head emits pLDDT in a "jagged token" layout — for
    each token it predicts up to ``max_atoms_per_token`` atoms. To match
    the actual atom axis, we gather each atom's pLDDT from
    ``plddt[token_id, within_token_idx]`` and mask invalid atoms to 0.
    """
    B, N_atoms = atom_to_token.shape
    device = plddt.device

    tok_ids = atom_to_token

    # Token start offsets: cumsum of atoms_per_token
    tok_starts = torch.zeros_like(atoms_per_token)
    tok_starts[:, 1:] = atoms_per_token[:, :-1].cumsum(dim=-1)

    atom_indices = torch.arange(N_atoms, device=device).unsqueeze(0).expand(B, -1)
    token_start_per_atom = tok_starts.gather(1, tok_ids)
    within_idx = atom_indices - token_start_per_atom
    within_idx = within_idx.clamp(min=0, max=plddt.shape[-1] - 1)

    # plddt[b, tok_ids[b, a], within_idx[b, a]]
    flat_plddt = plddt.gather(
        1, tok_ids.unsqueeze(-1).expand(-1, -1, plddt.shape[-1]),
    )
    result = flat_plddt.gather(2, within_idx.unsqueeze(-1)).squeeze(-1)
    return result * atom_mask.float()
