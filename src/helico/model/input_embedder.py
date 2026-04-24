"""Input feature embedder — AF3 SI §3.1.1 Algorithm 2.

Produces the per-token single representation ``s_inputs`` that feeds the
trunk. Runs an ``AtomAttentionEncoder(has_coords=False)`` over the
reference conformer (no noisy coords, no trunk injection), aggregates
atoms to tokens, then concatenates per-token restype + MSA profile +
deletion_mean:

    s_inputs = concat(a_token, restype(32), profile(32), deletion_mean(1))

resulting in a ``c_s_inputs = c_s + 65 = 449``-dim vector per token.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .diffusion import AtomAttentionEncoder


class InputFeatureEmbedder(nn.Module):
    """AF3 SI Algorithm 2 — InputFeatureEmbedder.

    Wraps AtomAttentionEncoder(has_coords=False) and concatenates the
    per-token scalar features. Output dim: ``c_s + 65`` (``c_s = 384``).
    """

    def __init__(self, config):
        super().__init__()
        self.atom_attention_encoder = AtomAttentionEncoder(
            config, has_coords=False, c_token_override=config.d_single,
        )

    def forward(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_features: torch.Tensor,
        atom_to_token: torch.Tensor,
        atom_mask: torch.Tensor,
        n_tokens: int,
        restype: torch.Tensor,
        profile: torch.Tensor,
        deletion_mean: torch.Tensor,
        ref_space_uid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns (B, N_tok, c_s_inputs) — 384 (a_token) + 32 + 32 + 1 = 449."""
        a_token, _, _, _, _ = self.atom_attention_encoder(
            ref_pos, ref_charge, ref_features,
            atom_to_token, atom_mask, n_tokens,
            ref_space_uid=ref_space_uid,
        )
        return torch.cat([a_token, restype, profile, deletion_mean], dim=-1)
