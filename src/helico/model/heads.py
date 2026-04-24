"""Auxiliary prediction heads — AF3 SI §4.

- ``DistogramHead``    (AF3 SI §4.4) — pair binned-distance logits
- ``ConfidenceHead``   (AF3 SI §4.3) — PAE/PDE/pLDDT/resolved
- ``AffinityModule``   (Boltz2 extension, not in AF3) — binding affinity
  regression + binder/non-binder classification over a pocket mask

All heads read from the trunk's detached (s_trunk, z_trunk) and the
input embedding s_inputs; the ConfidenceHead also receives predicted
coordinates from the diffusion rollout.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import LayerNorm, linear_no_bias
from .pairformer import Pairformer


class DistogramHead(nn.Module):
    """AF3 SI §4.4 — DistogramHead.

    Linear layer from c_z → n_distogram_bins, symmetrized by adding the
    transpose. Predicts a distribution over Cβ-Cβ distance bins for each
    pair of tokens. Trained with cross-entropy against ground-truth
    bucketed distances (see ``losses.distogram_loss``).
    """

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.d_pair, config.n_distogram_bins)  # WITH bias

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        return logits + logits.transpose(-2, -3)


class ConfidenceHead(nn.Module):
    """AF3 SI §4.3 — ConfidenceHead.

    Takes detached trunk (s, z) + the input embedding s_inputs + predicted
    coords, processes them through a small pairformer (``n_confidence_blocks``),
    and predicts four outputs:

    - ``pae_logits``       (B, N, N, n_pae_bins)    — predicted aligned error (§4.3.2)
    - ``pde_logits``       (B, N, N, n_pae_bins)    — predicted distance error (§4.3.3)
    - ``plddt_logits``     (B, N, max_atoms, n_plddt_bins) — per-atom pLDDT (§4.3.1)
    - ``resolved_logits``  (B, N, max_atoms, 2)     — experimentally-resolved (§4.3.4)

    Confidence z_init is built as z_trunk + outer(s_inputs) + distance-pair
    embedding of predicted coord distances between each token's
    "distogram rep atom" (CB for proteins, C4 for purines, etc.). This
    distance embedding is what lets the head "see" the predicted structure
    without diffusing into it — the pairformer then propagates that
    signal.
    """

    def __init__(self, config):
        super().__init__()
        c = config

        self.input_s_norm = LayerNorm(c.d_single)
        self.linear_s1 = linear_no_bias(c.c_s_inputs, c.d_pair)
        self.linear_s2 = linear_no_bias(c.c_s_inputs, c.d_pair)

        # Distance pair embeddings (39 bins over 3.25-52.0 Å)
        n_dist_bins = c.n_distance_bins
        lower = torch.linspace(3.25, 50.75, n_dist_bins)
        upper = torch.cat([torch.linspace(4.50, 52.0, n_dist_bins - 1), torch.tensor([1e6])])
        self.register_buffer("lower_bins", lower)
        self.register_buffer("upper_bins", upper)
        self.linear_d = linear_no_bias(n_dist_bins, c.d_pair)
        self.linear_d_raw = linear_no_bias(1, c.d_pair)

        # 4-block PairformerStack (same dims as the trunk)
        # Using a fresh small config so the HelicoConfig doesn't need to
        # expose n_blocks=4 as a trunk-level flag.
        from .config import HelicoConfig as _Config
        conf_config = _Config(
            d_single=c.d_single,
            d_pair=c.d_pair,
            n_pairformer_blocks=c.n_confidence_blocks,
            n_heads_pair=c.n_heads_pair,
            n_heads_single=c.n_heads_single,
            pair_head_dim=c.pair_head_dim,
            single_head_dim=c.single_head_dim,
            gradient_checkpointing=c.gradient_checkpointing,
            dropout=c.dropout,
        )
        self.pairformer_stack = Pairformer(conf_config)

        # Output heads
        self.pae_norm = LayerNorm(c.d_pair)
        self.linear_pae = linear_no_bias(c.d_pair, c.n_pae_bins)
        self.pde_norm = LayerNorm(c.d_pair)
        self.linear_pde = linear_no_bias(c.d_pair, c.n_pae_bins)
        self.plddt_norm = LayerNorm(c.d_single)
        self.plddt_weight = nn.Parameter(
            torch.zeros(c.max_atoms_per_token, c.d_single, c.n_plddt_bins)
        )
        self.resolved_norm = LayerNorm(c.d_single)
        self.resolved_weight = nn.Parameter(
            torch.zeros(c.max_atoms_per_token, c.d_single, 2)
        )

    def forward(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        pred_coords: torch.Tensor,
        atom_to_token: torch.Tensor,
        atom_mask: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        rep_atom_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        s = self.input_s_norm(torch.clamp(s_trunk.detach(), min=-512, max=512))
        s_inp = s_inputs.detach()

        # z_init = z_trunk + outer(s_inputs)
        z = z_trunk.detach() + self.linear_s1(s_inp).unsqueeze(2) + self.linear_s2(s_inp).unsqueeze(1)

        # Distance pair embeddings from representative atom coords (Cβ/Cα/C4/C2)
        B, N_tok = s.shape[:2]
        if rep_atom_idx is not None:
            idx3 = rep_atom_idx.unsqueeze(-1).expand(-1, -1, 3)
            token_centers = torch.gather(pred_coords, 1, idx3)
        else:
            token_centers = self._get_token_centers(pred_coords, atom_to_token, atom_mask, N_tok)

        # Compute pairwise distances in float32 for numerical stability
        with torch.amp.autocast("cuda", enabled=False):
            dists = torch.cdist(token_centers.float(), token_centers.float())
        d_unsq = dists.unsqueeze(-1)
        one_hot = ((d_unsq > self.lower_bins) & (d_unsq < self.upper_bins)).to(z.dtype)
        z = z + self.linear_d(one_hot)
        z = z + self.linear_d_raw(d_unsq.to(z.dtype))

        # Confidence PairformerStack
        s, z = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

        # Output heads in float32 for stability
        z = z.float()
        s = s.float()

        pae_logits = self.linear_pae(self.pae_norm(z))
        pde_logits = self.linear_pde(self.pde_norm(z + z.transpose(-2, -3)))
        plddt_logits = torch.einsum(
            "...tc,acb->...tab", self.plddt_norm(s), self.plddt_weight,
        )
        resolved_logits = torch.einsum(
            "...tc,acb->...tab", self.resolved_norm(s), self.resolved_weight,
        )

        return {
            "pae_logits": pae_logits,          # (B, N, N, n_pae_bins)
            "pde_logits": pde_logits,          # (B, N, N, n_pae_bins)
            "plddt_logits": plddt_logits,      # (B, N, max_atoms, n_plddt_bins)
            "resolved_logits": resolved_logits,  # (B, N, max_atoms, 2)
        }

    def _get_token_centers(self, coords: torch.Tensor, atom_to_token: torch.Tensor,
                           atom_mask: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Fallback: mean of all atoms per token when no rep_atom_idx is given."""
        B = coords.shape[0]
        device = coords.device
        dt = coords.dtype
        centers = torch.zeros(B, n_tokens, 3, device=device, dtype=dt)
        counts = torch.zeros(B, n_tokens, 1, device=device, dtype=dt)
        masked_coords = coords * atom_mask.unsqueeze(-1).to(dt)
        idx3 = atom_to_token.unsqueeze(-1).expand(-1, -1, 3)
        centers.scatter_add_(1, idx3, masked_coords)
        counts.scatter_add_(1, atom_to_token.unsqueeze(-1),
                            atom_mask.unsqueeze(-1).to(dt))
        return centers / counts.clamp(min=1)


class AffinityModule(nn.Module):
    """Binding affinity prediction — Boltz2 extension (not in AF3 SI).

    Small PairFormer over pocket tokens, then:
      - binary binder/non-binder logit (classifier)
      - continuous log₁₀(IC50/Ki/Kd) regression

    Currently unused by the default inference path; kept here because it
    shares the trunk's Pairformer primitive.
    """

    def __init__(self, config):
        super().__init__()
        c = config

        self.single_proj = nn.Linear(c.d_single, c.d_affinity)
        self.pair_proj = nn.Linear(c.d_pair, c.d_affinity)

        from .config import HelicoConfig as _Config
        pocket_config = _Config(
            d_single=c.d_affinity,
            d_pair=c.d_affinity,
            n_pairformer_blocks=c.n_affinity_pairformer_blocks,
            n_heads_pair=max(1, c.d_affinity // 32),
            n_heads_single=max(1, c.d_affinity // 16),
            pair_head_dim=32,
            single_head_dim=min(16, c.d_affinity),
            gradient_checkpointing=False,
            dropout=c.dropout,
        )
        self.pocket_pairformer = Pairformer(pocket_config)

        self.classifier = nn.Sequential(
            LayerNorm(c.d_affinity),
            nn.Linear(c.d_affinity, 1),
        )
        self.regressor = nn.Sequential(
            LayerNorm(c.d_affinity),
            nn.Linear(c.d_affinity, c.d_affinity),
            nn.ReLU(),
            nn.Linear(c.d_affinity, 1),
        )

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        pocket_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """single: (B, N, d_single), pair: (B, N, N, d_pair), pocket_mask: (B, N)."""
        s = self.single_proj(single)
        z = self.pair_proj(pair)

        s = s * pocket_mask.unsqueeze(-1)
        z = (z * pocket_mask.unsqueeze(-1).unsqueeze(-2)
                 * pocket_mask.unsqueeze(-2).unsqueeze(-1))

        pair_mask = pocket_mask.unsqueeze(-1) & pocket_mask.unsqueeze(-2)
        s, z = self.pocket_pairformer(s, z, mask=pocket_mask, pair_mask=pair_mask)

        pooled = (
            (s * pocket_mask.unsqueeze(-1)).sum(dim=1)
            / pocket_mask.sum(dim=1, keepdim=True).clamp(min=1)
        )

        return {
            "bind_logits": self.classifier(pooled),
            "affinity": self.regressor(pooled),
        }
