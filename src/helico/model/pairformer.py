"""Pairformer trunk (AF3 SI §3.6) + Relative Position Encoding (§3.1.2).

The Pairformer is AF3's main trunk module, replacing AF2's Evoformer: it
jointly refines the single representation ``s`` and pair representation
``z`` without an MSA axis (the MSA is handled by a separate
``MSAModule`` that updates ``z`` only). Each ``PairformerBlock`` does:

  1. Triangle multiplicative update (outgoing)    on z
  2. Triangle multiplicative update (incoming)    on z
  3. Triangle attention (starting)                on z
  4. Triangle attention (ending)                  on z
  5. Pair transition (SwiGLU MLP)                 on z
  6. Single attention with pair bias              on s (if has_single)
  7. Single transition (SwiGLU MLP)               on s (if has_single)

All steps are residual with optional dropout.

Relative Position Encoding (Algorithm 3) produces a ``(B, N, N, c_z)``
tensor added to ``z_init`` that encodes per-pair structural relations:
same chain, same residue, same entity, and relative chain index.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import cuequivariance_torch as cuet
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .blocks import LayerNorm, Transition, linear_no_bias
from .triangle import TriangleMultiplicativeUpdate, TriangleAttention


# ---------------------------------------------------------------------------
# Single attention with pair bias
# ---------------------------------------------------------------------------

class SingleAttentionWithPairBias(nn.Module):
    """Self-attention on the single representation, biased by the pair tensor.

    AF3 SI Algorithm 24 (diffusion variant without AdaLN is Algorithm 24;
    the Pairformer version differs only in that it doesn't use AdaLN
    conditioning — the attention is just self-attention over the sequence
    axis, with bias from ``z_ij``).

    Uses the fused ``cuet.attention_pair_bias`` kernel which fuses the
    ``LayerNorm(z) → linear(n_heads) → add as bias`` chain into the
    attention softmax.
    """

    def __init__(self, d_single: int, d_pair: int, n_heads: int):
        super().__init__()
        self.d_single = d_single
        self.d_pair = d_pair
        self.n_heads = n_heads
        self.head_dim = d_single // n_heads
        assert self.head_dim * n_heads == d_single

        self.norm_s = LayerNorm(d_single)
        self.q_proj = nn.Linear(d_single, d_single, bias=True)
        self.k_proj = nn.Linear(d_single, d_single, bias=False)
        self.v_proj = nn.Linear(d_single, d_single, bias=False)

        # Pair-bias projection: d_pair -> n_heads
        self.norm_z = LayerNorm(d_pair)
        self.z_proj = nn.Linear(d_pair, n_heads, bias=False)

        # Output
        self.gate = nn.Linear(d_single, d_single, bias=False)
        self.out_proj = nn.Linear(d_single, d_single, bias=False)

        nn.init.zeros_(self.gate.weight)

    @torch.compiler.disable
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """s: (B, N, d_single). z: (B, N, N, d_pair). mask: (B, N) or None."""
        B, N, D = s.shape
        H = self.n_heads
        dh = self.head_dim

        s_norm = self.norm_s(s)
        q = self.q_proj(s_norm).reshape(B, N, H, dh).permute(0, 2, 1, 3)
        k = self.k_proj(s_norm).reshape(B, N, H, dh).permute(0, 2, 1, 3)
        v = self.v_proj(s_norm).reshape(B, N, H, dh).permute(0, 2, 1, 3)

        if mask is None:
            mask_input = torch.ones(B, N, device=s.device, dtype=s.dtype)
        else:
            mask_input = mask.float()

        out, _ = cuet.attention_pair_bias(
            s=s_norm, q=q, k=k, v=v, z=z, mask=mask_input,
            num_heads=H,
            w_proj_z=self.z_proj.weight,
            w_proj_g=self.gate.weight,
            w_proj_o=self.out_proj.weight,
            w_ln_z=self.norm_z.weight, b_ln_z=self.norm_z.bias,
            b_proj_g=None,
        )
        return out


# ---------------------------------------------------------------------------
# Pairformer block + stack
# ---------------------------------------------------------------------------

class PairformerBlock(nn.Module):
    """One Pairformer block — AF3 SI Algorithm 17 (PairformerStack body)."""

    def __init__(self, config, has_single: bool = True):
        super().__init__()
        c = config
        self.has_single = has_single

        # Triangle ops on z
        self.tri_mul_out = TriangleMultiplicativeUpdate(c.d_pair, direction="outgoing")
        self.tri_mul_in = TriangleMultiplicativeUpdate(c.d_pair, direction="incoming")
        self.tri_att_start = TriangleAttention(c.d_pair, c.n_heads_pair, mode="starting")
        self.tri_att_end = TriangleAttention(c.d_pair, c.n_heads_pair, mode="ending")
        self.pair_transition = Transition(c.d_pair)

        if has_single:
            self.single_attention = SingleAttentionWithPairBias(
                c.d_single, c.d_pair, c.n_heads_single,
            )
            self.single_transition = Transition(c.d_single)

        self.dropout = nn.Dropout(c.dropout) if c.dropout > 0 else nn.Identity()

    def forward(
        self,
        single: torch.Tensor | None,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # Pair path
        pair = pair + self.dropout(self.tri_mul_out(pair, mask=pair_mask))
        pair = pair + self.dropout(self.tri_mul_in(pair, mask=pair_mask))
        pair = pair + self.dropout(self.tri_att_start(pair, mask=pair_mask))
        pair = pair + self.dropout(self.tri_att_end(pair, mask=pair_mask))
        pair = pair + self.dropout(self.pair_transition(pair))

        # Single path (skipped by MSA-module sub-blocks with has_single=False)
        if self.has_single and single is not None:
            single = single + self.dropout(self.single_attention(single, pair, mask=mask))
            single = single + self.dropout(self.single_transition(single))

        return single, pair


class Pairformer(nn.Module):
    """Stack of Pairformer blocks (AF3 SI §3.6).

    Default n_blocks=48 for AF3. Each cycle of the main inference loop
    runs the full stack once (after the Template + MSA modules). Supports
    gradient checkpointing for memory-constrained training.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            PairformerBlock(config) for _ in range(config.n_pairformer_blocks)
        ])

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                single, pair = grad_checkpoint(
                    block, single, pair, mask, pair_mask,
                    use_reentrant=False,
                )
            else:
                single, pair = block(single, pair, mask, pair_mask)
        return single, pair


# ---------------------------------------------------------------------------
# Relative Position Encoding
# ---------------------------------------------------------------------------

class RelativePositionEncoding(nn.Module):
    """AF3 SI Algorithm 3 — Relative position encoding.

    Concatenates four pairwise one-hot features and projects to c_z:

      a_rel_pos   (66): relative residue index within same chain, else sentinel
      a_rel_token (66): relative token index within same residue+chain, else sentinel
      b_same_entity (1): bool, same sequence-entity (e.g. homodimer chains)
      a_rel_chain (6):   relative sym_id within same entity (for symmetric copies)

    Total: 66 + 66 + 1 + 6 = 139 features → LinearNoBias → c_z.

    Compared with AF2's RelPE, the `a_rel_token` feature is new in AF3:
    it encodes relative position *within a residue*, needed for
    per-atom-tokenized ligands/modified residues where a single residue
    produces multiple tokens.
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        num_features = 2 * (2 * (r_max + 1)) + 1 + 2 * (s_max + 1)  # 139
        self.linear_no_bias = linear_no_bias(num_features, c_z)

    def forward(
        self,
        residue_index: torch.Tensor,
        token_index: torch.Tensor,
        asym_id: torch.Tensor,
        entity_id: torch.Tensor,
        sym_id: torch.Tensor,
    ) -> torch.Tensor:
        """(B, N) integer features → (B, N, N, c_z)."""
        r_max = self.r_max
        s_max = self.s_max
        dtype = self.linear_no_bias.weight.dtype

        same_chain = (asym_id.unsqueeze(2) == asym_id.unsqueeze(1))

        # a_rel_pos (66): clipped residue-index difference, sentinel if cross-chain
        d_res = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
        d_res_clipped = (d_res + r_max).clamp(0, 2 * r_max)
        sentinel = 2 * r_max + 1
        d_res_final = torch.where(
            same_chain, d_res_clipped,
            torch.full_like(d_res_clipped, sentinel),
        )
        a_rel_pos = F.one_hot(d_res_final.long(), 2 * (r_max + 1)).to(dtype)

        # a_rel_token (66): relative token-within-residue; sentinel outside same residue+chain
        d_tok = token_index.unsqueeze(2) - token_index.unsqueeze(1)
        d_tok_clipped = (d_tok + r_max).clamp(0, 2 * r_max)
        same_res = (residue_index.unsqueeze(2) == residue_index.unsqueeze(1))
        d_tok_final = torch.where(
            same_chain & same_res, d_tok_clipped,
            torch.full_like(d_tok_clipped, sentinel),
        )
        a_rel_token = F.one_hot(d_tok_final.long(), 2 * (r_max + 1)).to(dtype)

        # b_same_entity (1)
        same_entity = (entity_id.unsqueeze(2) == entity_id.unsqueeze(1))
        b_same_entity = same_entity.unsqueeze(-1).to(dtype)

        # a_rel_chain (6): relative sym_id for chains of the same entity
        d_sym = sym_id.unsqueeze(2) - sym_id.unsqueeze(1)
        d_sym_clipped = (d_sym + s_max).clamp(0, 2 * s_max)
        sym_sentinel = 2 * s_max + 1
        d_sym_final = torch.where(
            same_entity, d_sym_clipped,
            torch.full_like(d_sym_clipped, sym_sentinel),
        )
        a_rel_chain = F.one_hot(d_sym_final.long(), 2 * (s_max + 1)).to(dtype)

        features = torch.cat([a_rel_pos, a_rel_token, b_same_entity, a_rel_chain], dim=-1)
        return self.linear_no_bias(features)
