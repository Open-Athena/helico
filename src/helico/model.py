"""Helico: AlphaFold3 model implementation in a single file.

All neural network modules: input embeddings, Pairformer, diffusion module,
confidence head, affinity module, and loss functions.

Uses cuEquivariance kernels for triangle attention, triangle multiplicative update,
and attention with pair bias. Targets H100/B200 GPUs with bfloat16 precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import cuequivariance_torch as cuet

from helico.data import NUM_TOKEN_TYPES, UNK_ELEM_IDX

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HelicoConfig:
    """Model configuration with AlphaFold3 defaults."""
    # Representation dimensions
    d_single: int = 384
    d_pair: int = 128
    d_msa: int = 64
    n_msa_blocks: int = 4
    c_msa_opm_hidden: int = 32
    n_msa_pw_heads: int = 8
    msa_pw_head_dim: int = 8
    # Pairformer
    n_pairformer_blocks: int = 48
    n_heads_pair: int = 4         # d_pair / 32 = 4
    n_heads_single: int = 16     # d_single / 24 = 16
    pair_head_dim: int = 32
    single_head_dim: int = 24

    # Diffusion module
    c_token: int = 768              # token-level diffusion transformer dim
    c_atom: int = 128               # atom embedding dim
    c_atompair: int = 16            # atom-pair feature dim
    c_noise_embedding: int = 256    # Fourier noise embedding dim
    sigma_data: float = 16.0        # EDM preconditioning constant
    n_diffusion_token_blocks: int = 24
    n_heads_diffusion_token: int = 16
    diffusion_token_head_dim: int = 48  # 768/16
    n_atom_encoder_blocks: int = 3
    n_atom_decoder_blocks: int = 3
    n_heads_atom: int = 4
    atom_head_dim: int = 32            # 128/4
    noise_log_mean: float = -1.2       # EDM log-normal noise sampling
    noise_log_std: float = 1.5
    n_diffusion_steps: int = 200  # inference sampling steps
    n_atom_queries: int = 32   # query window size for atom attention
    n_atom_keys: int = 128     # key window size for atom attention

    # Atom features
    n_elements: int = UNK_ELEM_IDX + 1  # 24
    n_token_types: int = NUM_TOKEN_TYPES

    # Template embedder
    n_template_blocks: int = 2        # PairformerBlocks in template embedder
    d_template: int = 64              # Template pair dim (NOT same as d_pair)

    # Confidence head
    n_plddt_bins: int = 50
    n_pae_bins: int = 64
    n_distogram_bins: int = 64
    n_confidence_blocks: int = 4      # PairformerBlocks in confidence head
    n_distance_bins: int = 39         # Distogram bins for confidence (3.25-52.0 Å)

    # Recycling
    n_cycles: int = 1                 # Number of recycling cycles

    # Affinity module (Boltz2)
    n_affinity_pairformer_blocks: int = 4
    d_affinity: int = 64

    # Training
    max_atoms_per_token: int = 24
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    @property
    def d_atom(self) -> int:
        return self.c_atom

    @property
    def d_pair_head(self) -> int:
        return self.pair_head_dim

    @property
    def d_single_head(self) -> int:
        return self.single_head_dim

    @property
    def c_s_inputs(self) -> int:
        """Input feature dim: d_single (from atom encoder) + 32 restype + 32 profile + 1 deletion_mean."""
        return self.d_single + 65


# ============================================================================
# Building Blocks
# ============================================================================

class LayerNorm(nn.Module):
    """LayerNorm that handles mixed precision (casts weights to input dtype)."""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)


class Transition(nn.Module):
    """SwiGLU transition block: LayerNorm -> gated MLP (matches Protenix)."""
    def __init__(self, d: int, factor: int = 4):
        super().__init__()
        self.norm = LayerNorm(d)
        self.linear_a = nn.Linear(d, d * factor, bias=False)
        self.linear_b = nn.Linear(d, d * factor, bias=False)
        self.linear_out = nn.Linear(d * factor, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.linear_out(F.silu(self.linear_a(h)) * self.linear_b(h))


# (Old InputEmbedder removed — replaced by InputFeatureEmbedder + trunk init in Helico)


# ============================================================================
# Triangle Operations (Phase 2b) — cuEquivariance kernels
# ============================================================================

class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle multiplicative update using cuEquivariance kernel.

    Fused operation: LayerNorm -> projection + gating -> triangle einsum -> LayerNorm -> projection + gating.
    """

    def __init__(self, d: int, direction: str = "outgoing"):
        super().__init__()
        assert direction in ("outgoing", "incoming")
        assert d % 32 == 0, f"d={d} must be multiple of 32 for cuEquivariance kernel"
        self.direction = direction
        self.d = d

        # Input normalization and projection
        self.layer_norm_in = LayerNorm(d)
        self.linear_p = nn.Linear(d, 2 * d, bias=False)  # projection
        self.linear_g = nn.Linear(d, 2 * d, bias=False)  # gate

        # Output normalization and projection
        self.layer_norm_out = LayerNorm(d)
        self.output_projection = nn.Linear(d, d, bias=False)
        self.output_gate = nn.Linear(d, d, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Gate biases initialized to favor open gates
        nn.init.zeros_(self.output_gate.weight)

    @torch.compiler.disable
    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: (B, N, N, D) pair representation
            mask: (B, N, N) optional mask
        Returns:
            updated z: (B, N, N, D)
        """
        dt = z.dtype
        return cuet.triangle_multiplicative_update(
            x=z,
            direction=self.direction,
            mask=mask.to(dt) if mask is not None else None,
            norm_in_weight=self.layer_norm_in.weight.to(dt),
            norm_in_bias=self.layer_norm_in.bias.to(dt),
            p_in_weight=self.linear_p.weight.to(dt),
            g_in_weight=self.linear_g.weight.to(dt),
            norm_out_weight=self.layer_norm_out.weight.to(dt),
            norm_out_bias=self.layer_norm_out.bias.to(dt),
            p_out_weight=self.output_projection.weight.to(dt),
            g_out_weight=self.output_gate.weight.to(dt),
            eps=1e-5,
        )


class TriangleAttention(nn.Module):
    """Triangle attention using cuEquivariance kernel.

    Attention over one edge of the triangle (starting or ending node).
    """

    def __init__(self, d: int, n_heads: int, mode: str = "starting"):
        super().__init__()
        assert mode in ("starting", "ending")
        self.mode = mode
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        assert self.head_dim * n_heads == d
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm = LayerNorm(d)
        self.qkv_proj = nn.Linear(d, 3 * d, bias=False)
        self.bias_proj = nn.Linear(d, n_heads, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d, bias=False)

        nn.init.zeros_(self.gate.weight)

    @torch.compiler.disable
    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: (B, N, N, D) pair representation
            mask: (B, N, N) optional mask
        Returns:
            output: (B, N, N, D)
        """
        B, N, _, D = z.shape
        H = self.n_heads
        dh = self.head_dim

        # For "ending" mode: transpose, apply starting-mode attention, transpose back
        z_input = z if self.mode == "starting" else z.transpose(1, 2).contiguous()
        mask_input = mask if self.mode == "starting" else (mask.transpose(1, 2).contiguous() if mask is not None else None)

        z_norm = self.norm(z_input)
        qkv = self.qkv_proj(z_norm)  # (B, N, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # q,k,v shape for cuet: (B, N, H, N, dh)
        q = q.reshape(B, N, N, H, dh).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(B, N, N, H, dh).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(B, N, N, H, dh).permute(0, 1, 3, 2, 4).contiguous()

        # Bias from pair representation: (B, 1, H, N, N)
        bias = self.bias_proj(z_norm).permute(0, 3, 1, 2).unsqueeze(1).contiguous()

        # Mask: (B, N, 1, 1, N)
        tri_mask = None
        if mask_input is not None:
            tri_mask = mask_input.unsqueeze(2).unsqueeze(3)

        out = cuet.triangle_attention(q, k, v, bias, mask=tri_mask, scale=self.scale)
        # out: (B, N, H, N, dh)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, N, N, D)

        # Gate before output projection (matching Protenix: out_proj(gate * attn_out))
        gate = torch.sigmoid(self.gate(z_norm))
        out = self.out_proj(gate * out)

        if self.mode == "ending":
            out = out.transpose(1, 2)

        return out


class SingleAttentionWithPairBias(nn.Module):
    """Self-attention on single representation with pair bias using cuEquivariance."""

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

        # Pair bias projection: d_pair -> n_heads
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
        """
        Args:
            s: (B, N, d_single) single representation
            z: (B, N, N, d_pair) pair representation
            mask: (B, N) token mask
        Returns:
            output: (B, N, d_single)
        """
        B, N, D = s.shape
        H = self.n_heads
        dh = self.head_dim

        s_norm = self.norm_s(s)
        q = self.q_proj(s_norm).reshape(B, N, H, dh).permute(0, 2, 1, 3)  # (B, H, N, dh)
        k = self.k_proj(s_norm).reshape(B, N, H, dh).permute(0, 2, 1, 3)
        v = self.v_proj(s_norm).reshape(B, N, H, dh).permute(0, 2, 1, 3)

        # Use cuEquivariance attention_pair_bias
        # Expected shapes: s=(B*M, S, D), q/k/v=(B*M, H, S, DH), z=(B, S, S, z_dim), mask=(B, S)
        s_input = s_norm  # (B, N, D) — M=1 for Pairformer
        q_input = q       # (B, H, N, dh)
        k_input = k
        v_input = v

        if mask is None:
            mask_input = torch.ones(B, N, device=s.device, dtype=s.dtype)
        else:
            mask_input = mask.float()

        out, _ = cuet.attention_pair_bias(
            s=s_input,
            q=q_input,
            k=k_input,
            v=v_input,
            z=z,
            mask=mask_input,
            num_heads=H,
            w_proj_z=self.z_proj.weight,
            w_proj_g=self.gate.weight,
            w_proj_o=self.out_proj.weight,
            w_ln_z=self.norm_z.weight,
            b_ln_z=self.norm_z.bias,
            b_proj_g=None,
        )

        return out


# ============================================================================
# Pairformer Block (Phase 2c)
# ============================================================================

class PairformerBlock(nn.Module):
    """Single Pairformer block with triangle operations and attention."""

    def __init__(self, config: HelicoConfig, has_single: bool = True):
        super().__init__()
        c = config
        self.has_single = has_single

        # Triangle multiplicative updates
        self.tri_mul_out = TriangleMultiplicativeUpdate(c.d_pair, direction="outgoing")
        self.tri_mul_in = TriangleMultiplicativeUpdate(c.d_pair, direction="incoming")

        # Triangle attention
        self.tri_att_start = TriangleAttention(c.d_pair, c.n_heads_pair, mode="starting")
        self.tri_att_end = TriangleAttention(c.d_pair, c.n_heads_pair, mode="ending")

        # Pair transition
        self.pair_transition = Transition(c.d_pair)

        if has_single:
            # Single attention with pair bias
            self.single_attention = SingleAttentionWithPairBias(c.d_single, c.d_pair, c.n_heads_single)

            # Single transition
            self.single_transition = Transition(c.d_single)

        self.dropout = nn.Dropout(c.dropout) if c.dropout > 0 else nn.Identity()

    def forward(
        self,
        single: torch.Tensor | None,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Args:
            single: (B, N, d_single) or None if has_single=False
            pair: (B, N, N, d_pair)
            mask: (B, N) token mask
            pair_mask: (B, N, N) pair mask
        Returns:
            single: (B, N, d_single) or None
            pair: (B, N, N, d_pair)
        """
        # Triangle multiplicative updates on pair
        pair = pair + self.dropout(self.tri_mul_out(pair, mask=pair_mask))
        pair = pair + self.dropout(self.tri_mul_in(pair, mask=pair_mask))

        # Triangle attention on pair
        pair = pair + self.dropout(self.tri_att_start(pair, mask=pair_mask))
        pair = pair + self.dropout(self.tri_att_end(pair, mask=pair_mask))

        # Pair transition
        pair = pair + self.dropout(self.pair_transition(pair))

        if self.has_single and single is not None:
            # Single attention with pair bias
            single = single + self.dropout(self.single_attention(single, pair, mask=mask))

            # Single transition
            single = single + self.dropout(self.single_transition(single))

        return single, pair


# ============================================================================
# Pairformer Stack (Phase 2d)
# ============================================================================

class Pairformer(nn.Module):
    """Stack of Pairformer blocks."""

    def __init__(self, config: HelicoConfig):
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


# ============================================================================
# MSA Module (Algorithm 8 in AF3)
# ============================================================================

class OuterProductMean(nn.Module):
    """Outer product mean (Algorithm 9): MSA -> pair representation update.

    Matches Protenix: transpose to (*, N_tok, N_msa, C), outer product via einsum,
    normalize by mask overlap count.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int = 32, eps: float = 1e-3):
        super().__init__()
        self.c_hidden = c_hidden
        self.eps = eps
        self.norm = LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden, bias=False)
        self.linear_2 = nn.Linear(c_m, c_hidden, bias=False)
        self.linear_out = nn.Linear(c_hidden * c_hidden, c_z, bias=True)

    def forward(self, m: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            m: (*, N_msa, N_tok, c_m) MSA embedding
            mask: (*, N_msa, N_tok) MSA mask
        Returns:
            (*, N_tok, N_tok, c_z) pair update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        m = self.norm(m)
        mask = mask.unsqueeze(-1)  # (*, N_msa, N_tok, 1)

        a = self.linear_1(m) * mask  # (*, N_msa, N_tok, c_hidden)
        b = self.linear_2(m) * mask

        # Transpose: (*, N_msa, N_tok, C) -> (*, N_tok, N_msa, C)
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # Outer product (matches Protenix einsum "...bac,...dae->...bdce")
        # a: (*, N_tok_i, N_msa, C), b: (*, N_tok_j, N_msa, C)
        # -> (*, N_tok_i, N_tok_j, C, C)
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.flatten(-2)  # (*, N_tok, N_tok, C*C)

        # Protenix applies linear_out BEFORE norm division (order matters due to bias)
        outer = self.linear_out(outer)

        # Normalize by mask overlap count
        # mask shape: (*, N_msa, N_tok, 1) -> einsum sums over N_msa
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)  # (*, N_tok, N_tok, 1)
        outer = outer / (norm + self.eps)

        return outer


class MSAPairWeightedAveraging(nn.Module):
    """Algorithm 10: MSA pair-weighted averaging."""

    def __init__(self, c_m: int, c_z: int, n_heads: int = 8, head_dim: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.layernorm_m = LayerNorm(c_m)
        self.linear_mv = nn.Linear(c_m, n_heads * head_dim, bias=False)
        self.layernorm_z = LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, n_heads, bias=False)
        self.linear_mg = nn.Linear(c_m, n_heads * head_dim, bias=False)
        self.linear_out = nn.Linear(n_heads * head_dim, c_m, bias=False)

        nn.init.zeros_(self.linear_mg.weight)
        nn.init.zeros_(self.linear_out.weight)

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: (*, N_msa, N_tok, c_m)
            z: (*, N_tok, N_tok, c_z)
        Returns:
            (*, N_msa, N_tok, c_m) updated MSA embedding
        """
        H, dh = self.n_heads, self.head_dim

        m_norm = self.layernorm_m(m)
        v = self.linear_mv(m_norm)  # (*, N_msa, N_tok, H*dh)
        v = v.unflatten(-1, (H, dh))  # (*, N_msa, N_tok, H, dh)

        # Pair weights: softmax over j dimension
        w = self.linear_z(self.layernorm_z(z))  # (*, N_tok, N_tok, H)
        w = F.softmax(w, dim=-2)  # softmax over source token dim

        # Gate
        g = torch.sigmoid(self.linear_mg(m_norm))  # (*, N_msa, N_tok, H*dh)
        g = g.unflatten(-1, (H, dh))  # (*, N_msa, N_tok, H, dh)

        # Weighted average: o_mih = sum_j w_ijh * v_mjhc
        o = torch.einsum("...ijh,...mjhc->...mihc", w, v)

        # Gate and project
        o = g * o  # (*, N_msa, N_tok, H, dh)
        o = o.flatten(-2)  # (*, N_msa, N_tok, H*dh)
        return self.linear_out(o)


class MSAStack(nn.Module):
    """MSA pair-weighted averaging + transition with residual connections."""

    def __init__(self, c_m: int, c_z: int, n_heads: int = 8, head_dim: int = 8):
        super().__init__()
        self.pair_avg = MSAPairWeightedAveraging(c_m, c_z, n_heads, head_dim)
        self.transition = Transition(c_m, factor=4)

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: (*, N_msa, N_tok, c_m)
            z: (*, N_tok, N_tok, c_z)
        Returns:
            (*, N_msa, N_tok, c_m)
        """
        m = m + self.pair_avg(m, z)
        m = m + self.transition(m)
        return m


class MSABlock(nn.Module):
    """Single MSA processing block: OPM + pair_stack + optional msa_stack."""

    def __init__(self, config: HelicoConfig, is_last_block: bool = False):
        super().__init__()
        c = config

        self.opm = OuterProductMean(c.d_msa, c.d_pair, c.c_msa_opm_hidden)
        self.pair_stack = PairformerBlock(config, has_single=False)

        self.has_msa_stack = not is_last_block
        if self.has_msa_stack:
            self.msa_stack = MSAStack(c.d_msa, c.d_pair, c.n_msa_pw_heads, c.msa_pw_head_dim)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m: (B, N_msa, N_tok, c_m) MSA embedding
            z: (B, N_tok, N_tok, c_z) pair embedding
            msa_mask: (B, N_msa, N_tok) MSA mask
            pair_mask: (B, N_tok, N_tok) pair mask
        Returns:
            m: updated MSA embedding
            z: updated pair embedding
        """
        # OPM: MSA -> pair update
        z = z + self.opm(m, mask=msa_mask)

        # MSA stack (not in last block) — runs BEFORE pair_stack (matching Protenix)
        if self.has_msa_stack:
            m = self.msa_stack(m, z)

        # Pair stack (PairformerBlock without single path)
        _, z = self.pair_stack(None, z, pair_mask=pair_mask)

        return m, z


class MSAModule(nn.Module):
    """Algorithm 8: MSA module — processes MSA and updates pair representation."""

    def __init__(self, config: HelicoConfig):
        super().__init__()
        c = config
        self.config = config

        # Input projections
        self.linear_m = nn.Linear(34, c.d_msa, bias=False)  # 32 (one-hot) + 1 (has_deletion) + 1 (deletion_value)
        self.linear_s = nn.Linear(c.c_s_inputs, c.d_msa, bias=False)

        # MSA blocks
        self.blocks = nn.ModuleList([
            MSABlock(config, is_last_block=(i + 1 == c.n_msa_blocks))
            for i in range(c.n_msa_blocks)
        ])

    def forward(
        self,
        m_raw: torch.Tensor,
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        msa_mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            m_raw: (B, N_msa, N_tok, 34) raw MSA features
            z: (B, N_tok, N_tok, c_z) pair embedding
            s_inputs: (B, N_tok, c_s_inputs) single input features
            msa_mask: (B, N_msa, N_tok) MSA mask
            pair_mask: (B, N_tok, N_tok) pair mask
        Returns:
            z: (B, N_tok, N_tok, c_z) updated pair embedding
        """
        # Project MSA features and add single input broadcast
        m = self.linear_m(m_raw) + self.linear_s(s_inputs).unsqueeze(-3)

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                m, z = grad_checkpoint(block, m, z, msa_mask, pair_mask, use_reentrant=False)
            else:
                m, z = block(m, z, msa_mask=msa_mask, pair_mask=pair_mask)

        return z


# ============================================================================
# Diffusion Module — Protenix-matching architecture
# ============================================================================

def linear_no_bias(d_in: int, d_out: int, zeros_init: bool = False) -> nn.Linear:
    """Factory for Linear(bias=False) with optional zero weight init."""
    lin = nn.Linear(d_in, d_out, bias=False)
    if zeros_init:
        nn.init.zeros_(lin.weight)
    return lin


class BiasInitLinear(nn.Module):
    """Linear with weight=0, bias=constant. Used for conditioning gates."""

    def __init__(self, d_in: int, d_out: int, bias_init: float = -2.0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class AdaptiveLayerNorm(nn.Module):
    """FiLM-style modulated normalization (Protenix Algorithm 26)."""

    def __init__(self, d_a: int, d_s: int):
        super().__init__()
        self.norm_a = nn.LayerNorm(d_a, elementwise_affine=False)
        self.norm_s = LayerNorm(d_s)
        self.scale_proj = nn.Linear(d_s, d_a, bias=True)
        self.shift_proj = nn.Linear(d_s, d_a, bias=False)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_norm = self.norm_s(s)
        return torch.sigmoid(self.scale_proj(s_norm)) * self.norm_a(a) + self.shift_proj(s_norm)


class FourierEmbedding(nn.Module):
    """Fixed random Fourier features for noise level."""

    def __init__(self, d: int = 256, seed: int = 42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("w", torch.randn(d, generator=gen))
        self.register_buffer("b", torch.randn(d, generator=gen))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) scalar per batch element. Returns (B, d)."""
        return torch.cos(2 * math.pi * (t.unsqueeze(-1) * self.w + self.b))


class ConditionedTransitionBlock(nn.Module):
    """AdaLN + SwiGLU + s-gate."""

    def __init__(self, d_a: int, d_s: int, factor: int = 2):
        super().__init__()
        self.ada_ln = AdaptiveLayerNorm(d_a, d_s)
        self.linear_a = nn.Linear(d_a, d_a * factor, bias=False)
        self.linear_b = nn.Linear(d_a, d_a * factor, bias=False)
        self.linear_out = nn.Linear(d_a * factor, d_a, bias=False)
        self.s_gate = BiasInitLinear(d_s, d_a, bias_init=-2.0)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.ada_ln(a, s)
        return torch.sigmoid(self.s_gate(s)) * self.linear_out(F.silu(self.linear_a(h)) * self.linear_b(h))


class DiffusionAttentionPairBias(nn.Module):
    """Pure PyTorch attention with AdaLN and pair bias."""

    def __init__(self, d_a: int, d_s: int, d_z: int, n_heads: int, head_dim: int,
                 cross_attention_mode: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.cross_attention_mode = cross_attention_mode
        hdim_total = n_heads * head_dim

        self.ada_ln_q = AdaptiveLayerNorm(d_a, d_s)
        self.ada_ln_kv = AdaptiveLayerNorm(d_a, d_s) if cross_attention_mode else None

        self.q_proj = nn.Linear(d_a, hdim_total, bias=True)
        self.k_proj = nn.Linear(d_a, hdim_total, bias=False)
        self.v_proj = nn.Linear(d_a, hdim_total, bias=False)
        self.g_proj = nn.Linear(d_a, hdim_total, bias=False)
        self.out_proj = nn.Linear(hdim_total, d_a, bias=False)

        self.z_norm = nn.LayerNorm(d_z)
        self.z_proj = nn.Linear(d_z, n_heads, bias=False)

        self.s_gate = BiasInitLinear(d_s, d_a, bias_init=-2.0)

    def forward(self, a: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
                kv_a: torch.Tensor | None = None, kv_s: torch.Tensor | None = None,
                n_queries: int | None = None, n_keys: int | None = None,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            a: (B, N, d_a) query input
            s: (B, N, d_s) conditioning
            z: (B, N, N, d_z) global pair rep OR (B, n_blocks, n_q, n_k, d_z) windowed pair
            kv_a: (B, N, d_a) KV input for cross attention
            kv_s: (B, N, d_s) KV conditioning for cross attention
            n_queries: query window size (None = global attention)
            n_keys: key window size
            pad_mask: (n_blocks, n_q, n_k) validity mask for windowed mode
        """
        B, N, _ = a.shape
        H, dh = self.n_heads, self.head_dim

        q_in = self.ada_ln_q(a, s)
        if self.cross_attention_mode and kv_a is not None:
            kv_in = self.ada_ln_kv(kv_a, kv_s if kv_s is not None else s)
        elif self.cross_attention_mode:
            # Match Protenix: apply kv norm to already-normalized q_in
            kv_in = self.ada_ln_kv(q_in, s)
        else:
            kv_in = q_in

        q = self.q_proj(q_in).reshape(B, N, H, dh).permute(0, 2, 1, 3)  # (B, H, N, dh)
        k = self.k_proj(kv_in).reshape(B, N, H, dh).permute(0, 2, 1, 3)
        v = self.v_proj(kv_in).reshape(B, N, H, dh).permute(0, 2, 1, 3)
        g = self.g_proj(q_in).reshape(B, N, H, dh).permute(0, 2, 1, 3)

        if n_queries is None:
            # Global attention path (unchanged)
            bias = self.z_proj(self.z_norm(z)).permute(0, 3, 1, 2)  # (B, H, N, N)

            attn = (q @ k.transpose(-2, -1)) * self.scale + bias
            attn = F.softmax(attn, dim=-1)
            out = attn @ v  # (B, H, N, dh)

            out = torch.sigmoid(g) * out
            out = out.permute(0, 2, 1, 3).reshape(B, N, H * dh)
        else:
            # Windowed attention path
            n_blocks = (N + n_queries - 1) // n_queries
            q_pad = n_blocks * n_queries - N
            pad_left = (n_keys - n_queries) // 2

            # Pair bias: z is already (B, n_blocks, n_q, n_k, d_z)
            bias = self.z_proj(self.z_norm(z)).permute(0, 4, 1, 2, 3)  # (B, H, n_blocks, n_q, n_k)

            # Partition Q, G into query windows: (B, H, N, dh) -> (B, H, n_blocks, n_q, dh)
            def _partition_q(t):
                # t: (B, H, N, dh) -> pad -> reshape
                t_padded = F.pad(t, (0, 0, 0, q_pad))  # (B, H, N_padded, dh)
                return t_padded.reshape(B, H, n_blocks, n_queries, dh)

            # Partition K, V into key windows: (B, H, N, dh) -> (B, H, n_blocks, n_k, dh)
            def _partition_k(t):
                t_padded = F.pad(t, (0, 0, 0, q_pad))  # (B, H, N_padded, dh)
                t_for_keys = F.pad(t_padded, (0, 0, pad_left, n_keys - n_queries - pad_left))
                return t_for_keys.unfold(2, n_keys, n_queries).permute(0, 1, 2, 4, 3)

            q_w = _partition_q(q)  # (B, H, n_blocks, n_q, dh)
            g_w = _partition_q(g)
            k_w = _partition_k(k)  # (B, H, n_blocks, n_k, dh)
            v_w = _partition_k(v)

            attn = (q_w @ k_w.transpose(-2, -1)) * self.scale + bias
            # Apply pad_mask: invalid positions get -inf
            if pad_mask is not None:
                attn = attn.masked_fill(~pad_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = F.softmax(attn, dim=-1)
            # NaN from all-masked rows -> 0
            attn = attn.nan_to_num(0.0)
            out_w = attn @ v_w  # (B, H, n_blocks, n_q, dh)

            out = torch.sigmoid(g_w) * out_w
            # Unpartition: (B, H, n_blocks, n_q, dh) -> (B, H, N, dh)
            out = out.reshape(B, H, n_blocks * n_queries, dh)[:, :, :N]
            out = out.permute(0, 2, 1, 3).reshape(B, N, H * dh)

        # External conditioning gate
        return torch.sigmoid(self.s_gate(s)) * self.out_proj(out)


class DiffusionTransformerBlock(nn.Module):
    """Attention + transition with residual connections."""

    def __init__(self, d_a: int, d_s: int, d_z: int, n_heads: int, head_dim: int,
                 cross_attention_mode: bool = False):
        super().__init__()
        self.attention = DiffusionAttentionPairBias(
            d_a, d_s, d_z, n_heads, head_dim, cross_attention_mode)
        self.transition = ConditionedTransitionBlock(d_a, d_s)

    def forward(self, a: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
                kv_a: torch.Tensor | None = None, kv_s: torch.Tensor | None = None,
                n_queries: int | None = None, n_keys: int | None = None,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        a = a + self.attention(a, s, z, kv_a=kv_a, kv_s=kv_s,
                               n_queries=n_queries, n_keys=n_keys, pad_mask=pad_mask)
        a = a + self.transition(a, s)
        return a


class DiffusionTransformer(nn.Module):
    """Stack of DiffusionTransformerBlocks with gradient checkpointing."""

    def __init__(self, n_blocks: int, d_a: int, d_s: int, d_z: int,
                 n_heads: int, head_dim: int, cross_attention_mode: bool = False,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(d_a, d_s, d_z, n_heads, head_dim, cross_attention_mode)
            for _ in range(n_blocks)
        ])

    def forward(self, a: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
                kv_a: torch.Tensor | None = None, kv_s: torch.Tensor | None = None,
                n_queries: int | None = None, n_keys: int | None = None,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                a = grad_checkpoint(block, a, s, z, kv_a, kv_s, n_queries, n_keys, pad_mask, use_reentrant=False)
            else:
                a = block(a, s, z, kv_a=kv_a, kv_s=kv_s,
                          n_queries=n_queries, n_keys=n_keys, pad_mask=pad_mask)
        return a


def _partition_to_windows(
    x: torch.Tensor, n_queries: int, n_keys: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Partition flat atom tensor into overlapping query/key windows.

    Args:
        x: (B, N, D) flat atom tensor
        n_queries: non-overlapping query block size (e.g. 32)
        n_keys: overlapping key window size (e.g. 128)

    Returns:
        x_q: (B, n_blocks, n_queries, D) query blocks
        x_k: (B, n_blocks, n_keys, D) overlapping key windows
        pad_mask: (n_blocks, n_queries, n_keys) True where both positions are valid
        n_blocks: number of blocks
        q_pad: amount of query padding added
    """
    B, N, D = x.shape
    n_blocks = (N + n_queries - 1) // n_queries
    q_pad = n_blocks * n_queries - N

    # Pad to multiple of n_queries
    x_padded = F.pad(x, (0, 0, 0, q_pad))  # (B, n_blocks * n_queries, D)

    # Queries: non-overlapping blocks
    x_q = x_padded.reshape(B, n_blocks, n_queries, D)

    # Keys: overlapping windows centered on each query block
    pad_left = (n_keys - n_queries) // 2
    pad_right = n_keys - n_queries - pad_left
    x_for_keys = F.pad(x_padded, (0, 0, pad_left, pad_right))  # (B, pad_left + N_padded + pad_right, D)
    x_k = x_for_keys.unfold(1, n_keys, n_queries)  # (B, n_blocks, D, n_keys)
    x_k = x_k.permute(0, 1, 3, 2)  # (B, n_blocks, n_keys, D)

    # Build pad_mask: valid positions are [0, N) in original sequence
    # Query positions per block
    q_pos = torch.arange(n_blocks * n_queries, device=x.device).reshape(n_blocks, n_queries)
    q_valid = q_pos < N  # (n_blocks, n_queries)

    # Key positions per block (offset by -pad_left relative to query block start)
    block_starts = torch.arange(n_blocks, device=x.device) * n_queries
    k_offsets = torch.arange(n_keys, device=x.device) - pad_left
    k_pos = block_starts.unsqueeze(1) + k_offsets.unsqueeze(0)  # (n_blocks, n_keys)
    k_valid = (k_pos >= 0) & (k_pos < N)  # (n_blocks, n_keys)

    pad_mask = q_valid.unsqueeze(2) & k_valid.unsqueeze(1)  # (n_blocks, n_queries, n_keys)

    return x_q, x_k, pad_mask, n_blocks, q_pad


def _unpartition_from_windows(x_q: torch.Tensor, n_orig: int) -> torch.Tensor:
    """Reshape windowed query output back to flat atom tensor.

    Args:
        x_q: (B, n_blocks, n_queries, D)
        n_orig: original sequence length

    Returns:
        (B, n_orig, D)
    """
    B, n_blocks, n_queries, D = x_q.shape
    return x_q.reshape(B, n_blocks * n_queries, D)[:, :n_orig]


class AtomAttentionEncoder(nn.Module):
    """Encode atom-level features into token representation with skip connections.

    Uses windowed within-token atom attention: n_atom_queries × n_atom_keys
    overlapping windows, with geometric pair features masked to within-token
    pairs via atom_to_token (ref_space_uid).

    Args:
        config: Model configuration
        has_coords: If True (default), includes noisy_pos and trunk injection layers.
            Set to False for input embedding (no coordinates/trunk).
        c_token_override: Override the aggregation output dim (default: config.c_token).
    """

    def __init__(self, config: HelicoConfig, has_coords: bool = True,
                 c_token_override: int | None = None):
        super().__init__()
        c = config
        c_atom = c.c_atom
        c_atompair = c.c_atompair
        c_s = c.d_single
        c_z = c.d_pair
        c_token = c_token_override if c_token_override is not None else c.c_token
        self.has_coords = has_coords
        self.n_queries = c.n_atom_queries
        self.n_keys = c.n_atom_keys

        # Reference feature projections
        self.ref_pos_proj = linear_no_bias(3, c_atom)
        self.ref_charge_proj = linear_no_bias(1, c_atom)
        self.n_ref_feat = 1 + 128 + 256  # mask + element_onehot(128) + atom_name_chars(256)
        self.ref_feat_proj = linear_no_bias(self.n_ref_feat, c_atom)

        if has_coords:
            # Noisy coordinate projection
            self.noisy_pos_proj = linear_no_bias(3, c_atom)

            # Trunk injection
            self.trunk_s_norm = LayerNorm(c_s)
            self.trunk_s_proj = linear_no_bias(c_s, c_atom, zeros_init=True)
            self.trunk_z_norm = LayerNorm(c_z)
            self.trunk_z_proj = linear_no_bias(c_z, c_atompair, zeros_init=True)

        # Atom-pair projections
        self.pair_dist_proj = linear_no_bias(3, c_atompair)
        self.pair_inv_dist_proj = linear_no_bias(1, c_atompair)
        self.pair_valid_proj = linear_no_bias(1, c_atompair)

        # Cross-pair features from atoms
        self.cross_pair_q = linear_no_bias(c_atom, c_atompair)
        self.cross_pair_k = linear_no_bias(c_atom, c_atompair)

        # Pair MLP: ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear(zeros)
        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            linear_no_bias(c_atompair, c_atompair),
            nn.ReLU(),
            linear_no_bias(c_atompair, c_atompair),
            nn.ReLU(),
            linear_no_bias(c_atompair, c_atompair, zeros_init=True),
        )

        # Atom transformer
        self.atom_transformer = DiffusionTransformer(
            n_blocks=c.n_atom_encoder_blocks,
            d_a=c_atom, d_s=c_atom, d_z=c_atompair,
            n_heads=c.n_heads_atom, head_dim=c.atom_head_dim,
            cross_attention_mode=True,
            gradient_checkpointing=c.gradient_checkpointing,
        )

        # Aggregation to token level
        self.agg_proj = linear_no_bias(c_atom, c_token)

    def forward(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_features: torch.Tensor,
        atom_to_token: torch.Tensor,
        atom_mask: torch.Tensor,
        n_tokens: int,
        *,
        noisy_pos: torch.Tensor | None = None,
        s_trunk: torch.Tensor | None = None,
        z_trunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            a_token: (B, N_tok, c_token) — token representation
            q_skip: (B, N_atom, c_atom) — skip connection for decoder
            c_skip: (B, N_atom, c_atom) — conditioning skip
            p_skip: (B, n_blocks, n_q, n_k, c_atompair) — windowed pair skip
            pad_mask: (n_blocks, n_q, n_k) — validity mask for windowed attention
        """
        B, N_atom, _ = ref_pos.shape
        n_q, n_k = self.n_queries, self.n_keys

        # 1. Build atom features c_l
        c_l = self.ref_pos_proj(ref_pos) + self.ref_charge_proj(ref_charge) + self.ref_feat_proj(ref_features)
        c_l = c_l * atom_mask.unsqueeze(-1)

        # 2. Inject s_trunk (only when has_coords)
        if self.has_coords and s_trunk is not None:
            s_trunk_atom = self._broadcast_token_to_atom(self.trunk_s_proj(self.trunk_s_norm(s_trunk)), atom_to_token)
            c_l = c_l + s_trunk_atom

        # 3. Build q_l
        if self.has_coords and noisy_pos is not None:
            q_l = c_l + self.noisy_pos_proj(noisy_pos)
        else:
            q_l = c_l

        # 4. Build windowed atom-pair features using REFERENCE positions (AF3 Algo 5 line 3)
        ref_pos_q, ref_pos_k, pad_mask, n_blocks, q_pad = _partition_to_windows(
            ref_pos, n_q, n_k)

        # Pairwise distances within windows
        diff = ref_pos_q.unsqueeze(3) - ref_pos_k.unsqueeze(2)  # (B, n_blocks, n_q, n_k, 3)
        dist_sq = diff.pow(2).sum(-1, keepdim=True)
        inv_dist = 1.0 / (1.0 + dist_sq)

        # Within-token validity mask (atom_to_token serves as ref_space_uid)
        # Use sentinel -1 for padding atoms so they don't match real tokens
        a2t_padded = F.pad(atom_to_token, (0, q_pad), value=-1)
        a2t_q, a2t_k, _, _, _ = _partition_to_windows(
            a2t_padded.unsqueeze(-1).float(), n_q, n_k)
        v_lm = (a2t_q.squeeze(-1).long().unsqueeze(3) == a2t_k.squeeze(-1).long().unsqueeze(2))
        v_lm = v_lm.unsqueeze(-1).to(diff.dtype)  # (B, n_blocks, n_q, n_k, 1)

        # Build pair features (geometric: masked by v_lm; validity: embedded directly)
        p = self.pair_dist_proj(diff) * v_lm
        p = p + self.pair_inv_dist_proj(inv_dist) * v_lm
        p = p + self.pair_valid_proj(v_lm)
        p = p * pad_mask.unsqueeze(0).unsqueeze(-1).to(diff.dtype)

        # 5. Inject z_trunk (windowed gather) — only when has_coords
        if self.has_coords and z_trunk is not None:
            z_trunk_proj = self.trunk_z_proj(self.trunk_z_norm(z_trunk))  # (B, N_tok, N_tok, c_atompair)
            z_windowed = self._gather_trunk_pair_windowed(z_trunk_proj, atom_to_token, n_blocks, q_pad)
            p = p + z_windowed

        # 6. Cross-pair features (windowed additive broadcast, relu before projection)
        c_l_q_w, c_l_k_w, _, _, _ = _partition_to_windows(c_l, n_q, n_k)
        p = p + self.cross_pair_q(F.relu(c_l_q_w)).unsqueeze(3) + self.cross_pair_k(F.relu(c_l_k_w)).unsqueeze(2)

        # 7. Pair MLP residual
        p = p + self.pair_mlp(p)

        # Save skips
        c_skip = c_l
        p_skip = p

        # 8. Atom transformer (cross attention: q_l with c_l conditioning, windowed pair bias)
        q_l = self.atom_transformer(q_l, c_l, p,
                                     n_queries=n_q, n_keys=n_k, pad_mask=pad_mask)

        q_skip = q_l

        # 9. Aggregate to tokens
        a_token = self._aggregate_to_tokens(F.relu(self.agg_proj(q_l)), atom_to_token, atom_mask, n_tokens)

        return a_token, q_skip, c_skip, p_skip, pad_mask

    def _broadcast_token_to_atom(self, token_feat: torch.Tensor, atom_to_token: torch.Tensor) -> torch.Tensor:
        idx = atom_to_token.unsqueeze(-1).expand(-1, -1, token_feat.shape[-1])
        return torch.gather(token_feat, 1, idx)

    def _gather_trunk_pair_windowed(self, z_trunk_proj: torch.Tensor, atom_to_token: torch.Tensor,
                                     n_blocks: int, q_pad: int) -> torch.Tensor:
        """Gather token-pair features into windowed atom-pair format."""
        B = atom_to_token.shape[0]
        n_q, n_k = self.n_queries, self.n_keys
        pad_left = (n_k - n_q) // 2
        C = z_trunk_proj.shape[-1]

        # Pad and partition atom_to_token (use 0 as sentinel — masked in attention)
        a2t_padded = F.pad(atom_to_token, (0, q_pad), value=0)
        tok_q = a2t_padded.reshape(B, n_blocks, n_q)
        a2t_for_keys = F.pad(a2t_padded, (pad_left, n_k - n_q - pad_left), value=0)
        tok_k = a2t_for_keys.unfold(1, n_k, n_q)  # (B, n_blocks, n_k)

        # Advanced indexing: z_trunk_proj[b, tok_q[b,bl,i], tok_k[b,bl,j], :]
        b_idx = torch.arange(B, device=z_trunk_proj.device).view(B, 1, 1, 1)
        result = z_trunk_proj[b_idx, tok_q.unsqueeze(3), tok_k.unsqueeze(2)]
        return result  # (B, n_blocks, n_q, n_k, C)

    def _aggregate_to_tokens(self, atom_feat: torch.Tensor, atom_to_token: torch.Tensor,
                              atom_mask: torch.Tensor, n_tokens: int) -> torch.Tensor:
        B, N_atom, D = atom_feat.shape
        device = atom_feat.device
        dt = atom_feat.dtype
        masked = atom_feat * atom_mask.unsqueeze(-1).to(dt)
        token_sum = torch.zeros(B, n_tokens, D, device=device, dtype=dt)
        token_count = torch.zeros(B, n_tokens, 1, device=device, dtype=dt)
        idx = atom_to_token.unsqueeze(-1).expand(-1, -1, D)
        token_sum.scatter_add_(1, idx, masked)
        token_count.scatter_add_(1, atom_to_token.unsqueeze(-1), atom_mask.unsqueeze(-1).to(dt))
        return token_sum / token_count.clamp(min=1)


class AtomAttentionDecoder(nn.Module):
    """Decode token representation back to atom coordinates."""

    def __init__(self, config: HelicoConfig):
        super().__init__()
        c = config
        self.n_queries = c.n_atom_queries
        self.n_keys = c.n_atom_keys
        self.token_to_atom_proj = linear_no_bias(c.c_token, c.c_atom)
        self.atom_transformer = DiffusionTransformer(
            n_blocks=c.n_atom_decoder_blocks,
            d_a=c.c_atom, d_s=c.c_atom, d_z=c.c_atompair,
            n_heads=c.n_heads_atom, head_dim=c.atom_head_dim,
            cross_attention_mode=True,
            gradient_checkpointing=c.gradient_checkpointing,
        )
        self.out_norm = nn.LayerNorm(c.c_atom)
        self.out_proj = linear_no_bias(c.c_atom, 3)

    def forward(self, a_token: torch.Tensor, atom_to_token: torch.Tensor,
                q_skip: torch.Tensor, c_skip: torch.Tensor, p_skip: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        """Returns: (B, N_atom, 3) coordinate output in float32."""
        # Broadcast token to atom
        projected = self.token_to_atom_proj(a_token)  # (B, N_tok, c_atom)
        idx = atom_to_token.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
        q = torch.gather(projected, 1, idx) + q_skip

        q = self.atom_transformer(q, c_skip, p_skip,
                                   n_queries=self.n_queries, n_keys=self.n_keys,
                                   pad_mask=pad_mask)
        return self.out_proj(self.out_norm(q)).float()


# ============================================================================
# Input Feature Embedder (Protenix input_embedder)
# ============================================================================

class InputFeatureEmbedder(nn.Module):
    """Wraps AtomAttentionEncoder(has_coords=False) for input embedding.

    Produces per-token features by concatenating atom encoder output with
    residue type, MSA profile, and deletion mean.
    """

    def __init__(self, config: HelicoConfig):
        super().__init__()
        self.atom_attention_encoder = AtomAttentionEncoder(
            config, has_coords=False, c_token_override=config.d_single)

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
    ) -> torch.Tensor:
        """
        Returns:
            s_inputs: (B, N_tok, c_s_inputs=449)
                a_token (384) + restype (32) + profile (32) + deletion_mean (1) = 449
        """
        a_token, _, _, _, _ = self.atom_attention_encoder(
            ref_pos, ref_charge, ref_features,
            atom_to_token, atom_mask, n_tokens)
        return torch.cat([a_token, restype, profile, deletion_mean], dim=-1)


# ============================================================================
# Template Embedder (Protenix template_embedder)
# ============================================================================

# The Protenix template embedder uses PairformerBlocks at d_template=64 but with
# internal hidden_dim=128 (n_heads=4, d_head=32, transition_factor=2). This doesn't
# match the cuEq kernel shape assumptions (which require hidden=d), so we use
# standard PyTorch modules with matching parameter names for weight transfer.


class _TemplateTriMul(nn.Module):
    """Triangle multiplicative update matching Protenix template shapes (hidden != d)."""

    def __init__(self, d: int, hidden: int, direction: str):
        super().__init__()
        self.direction = direction
        self.layer_norm_in = LayerNorm(d)
        self.linear_p = nn.Linear(d, 2 * hidden, bias=False)
        self.linear_g = nn.Linear(d, 2 * hidden, bias=False)
        self.layer_norm_out = LayerNorm(hidden)
        self.output_projection = nn.Linear(hidden, d, bias=False)
        self.output_gate = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.output_gate.weight)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.layer_norm_in(z)
        p = self.linear_p(h)
        g_in = torch.sigmoid(self.linear_g(h))
        pg = p * g_in
        a, b = pg.chunk(2, dim=-1)
        if self.direction == "outgoing":
            out = torch.einsum("...ikd,...jkd->...ijd", a, b)
        else:
            out = torch.einsum("...kid,...kjd->...ijd", a, b)
        out = self.layer_norm_out(out)
        g_out = torch.sigmoid(self.output_gate(h))
        return self.output_projection(out) * g_out


class _TemplateTriAtt(nn.Module):
    """Triangle attention matching Protenix template shapes (n_heads * d_head != d)."""

    def __init__(self, d: int, n_heads: int, d_head: int, mode: str):
        super().__init__()
        assert mode in ("starting", "ending")
        self.mode = mode
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.scale = 1.0 / math.sqrt(d_head)

        self.norm = LayerNorm(d)
        self.qkv_proj = nn.Linear(d, 3 * inner, bias=False)
        self.bias_proj = nn.Linear(d, n_heads, bias=False)
        self.out_proj = nn.Linear(inner, d, bias=False)
        self.gate = nn.Linear(d, inner, bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, _, D = z.shape
        H, dh = self.n_heads, self.d_head
        z_in = z if self.mode == "starting" else z.transpose(1, 2).contiguous()
        h = self.norm(z_in)
        q, k, v = self.qkv_proj(h).reshape(B, N, N, 3, H, dh).unbind(3)
        q = q.permute(0, 1, 3, 2, 4)  # (B, N, H, N, dh)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        attn = torch.einsum("bnhid,bnhjd->bnhij", q, k) * self.scale
        bias = self.bias_proj(h).permute(0, 3, 1, 2).unsqueeze(1)  # (B, 1, H, N, N)
        attn = attn + bias
        if mask is not None:
            m = mask if self.mode == "starting" else mask.transpose(1, 2)
            attn = attn.masked_fill(~m.unsqueeze(2).unsqueeze(3).bool(), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bnhij,bnhjd->bnhid", attn, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, N, N, H * dh)
        gate = torch.sigmoid(self.gate(h))
        out = self.out_proj(out * gate)
        if self.mode == "ending":
            out = out.transpose(1, 2).contiguous()
        return out


class _TemplatePairformerBlock(nn.Module):
    """Template pairformer block with Protenix shapes (pair-only, hidden != d)."""

    def __init__(self, d: int, n_heads: int = 4, d_head: int = 32, transition_factor: int = 2):
        super().__init__()
        hidden = n_heads * d_head
        self.tri_mul_out = _TemplateTriMul(d, hidden, "outgoing")
        self.tri_mul_in = _TemplateTriMul(d, hidden, "incoming")
        self.tri_att_start = _TemplateTriAtt(d, n_heads, d_head, "starting")
        self.tri_att_end = _TemplateTriAtt(d, n_heads, d_head, "ending")
        self.pair_transition = Transition(d, factor=transition_factor)

    def forward(
        self,
        single: torch.Tensor | None,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        pair = pair + self.tri_mul_out(pair, mask=pair_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_mask)
        pair = pair + self.tri_att_start(pair, mask=pair_mask)
        pair = pair + self.tri_att_end(pair, mask=pair_mask)
        pair = pair + self.pair_transition(pair)
        return None, pair


class TemplateEmbedder(nn.Module):
    """Template embedding: projection -> pair-only PairformerBlocks at d_template -> projection.

    Currently returns 0 (matches Protenix — no template features in checkpoint config),
    but parameters exist for weight transfer. Uses non-cuEq blocks because the Protenix
    template architecture has hidden_dim=2*d_template (cuEq kernels require hidden=d).
    """

    def __init__(self, config: HelicoConfig):
        super().__init__()
        c = config
        input_dim = 108  # 39 distogram + 1 frame_mask + 3 unit_vec + 1 pseudo_beta + 32+32 restype

        self.z_norm = LayerNorm(c.d_pair)
        self.linear_z = linear_no_bias(c.d_pair, c.d_template)   # 128->64
        self.linear_a = linear_no_bias(input_dim, c.d_template)  # 108->64

        # Protenix template uses n_heads=4, d_head=32 (same as trunk) → hidden=128
        self.pairformer_stack = nn.ModuleList([
            _TemplatePairformerBlock(
                d=c.d_template,
                n_heads=c.n_heads_pair,
                d_head=c.pair_head_dim,
                transition_factor=2,
            )
            for _ in range(c.n_template_blocks)
        ])

        self.out_norm = LayerNorm(c.d_template)
        self.linear_out = linear_no_bias(c.d_template, c.d_pair)  # 64->128

    def forward(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """Returns 0 — templates disabled (matching Protenix checkpoint config)."""
        return 0


# ============================================================================
# Distogram Head (separate from ConfidenceHead)
# ============================================================================

class DistogramHead(nn.Module):
    """Predict symmetrized distance distribution from pair representation."""

    def __init__(self, config: HelicoConfig):
        super().__init__()
        self.linear = nn.Linear(config.d_pair, config.n_distogram_bins)  # WITH bias

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        return logits + logits.transpose(-2, -3)


class RelativePositionEncoding(nn.Module):
    """Algorithm 3: Relative position encoding (matches Protenix).

    Computes 139-dim one-hot feature vector from pairwise token relationships,
    then projects to pair dimension via a single linear layer.
    Features: a_rel_pos (66) + a_rel_token (66) + b_same_entity (1) + a_rel_chain (6) = 139.
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
        """
        Args:
            residue_index: (B, N) per-token residue index
            token_index: (B, N) per-token position within residue
            asym_id: (B, N) chain identifier
            entity_id: (B, N) entity identifier (chains with same sequence share entity)
            sym_id: (B, N) symmetry copy index
        Returns:
            (B, N, N, c_z) pair features
        """
        r_max = self.r_max
        s_max = self.s_max
        dtype = self.linear_no_bias.weight.dtype

        same_chain = (asym_id.unsqueeze(2) == asym_id.unsqueeze(1))  # (B, N, N)

        # a_rel_pos: relative residue position (66 dims)
        d_res = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
        d_res_clipped = (d_res + r_max).clamp(0, 2 * r_max)
        sentinel = 2 * r_max + 1
        d_res_final = torch.where(same_chain, d_res_clipped, torch.full_like(d_res_clipped, sentinel))
        a_rel_pos = F.one_hot(d_res_final.long(), 2 * (r_max + 1)).to(dtype)

        # a_rel_token: relative token position within residue (66 dims)
        d_tok = token_index.unsqueeze(2) - token_index.unsqueeze(1)
        d_tok_clipped = (d_tok + r_max).clamp(0, 2 * r_max)
        same_res = (residue_index.unsqueeze(2) == residue_index.unsqueeze(1))
        d_tok_final = torch.where(same_chain & same_res, d_tok_clipped, torch.full_like(d_tok_clipped, sentinel))
        a_rel_token = F.one_hot(d_tok_final.long(), 2 * (r_max + 1)).to(dtype)

        # b_same_entity: (1 dim)
        same_entity = (entity_id.unsqueeze(2) == entity_id.unsqueeze(1))
        b_same_entity = same_entity.unsqueeze(-1).to(dtype)

        # a_rel_chain: relative symmetry copy (6 dims)
        d_sym = sym_id.unsqueeze(2) - sym_id.unsqueeze(1)
        d_sym_clipped = (d_sym + s_max).clamp(0, 2 * s_max)
        sym_sentinel = 2 * s_max + 1
        d_sym_final = torch.where(same_entity, d_sym_clipped, torch.full_like(d_sym_clipped, sym_sentinel))
        a_rel_chain = F.one_hot(d_sym_final.long(), 2 * (s_max + 1)).to(dtype)

        features = torch.cat([a_rel_pos, a_rel_token, b_same_entity, a_rel_chain], dim=-1)
        return self.linear_no_bias(features)


class DiffusionConditioning(nn.Module):
    """Condition single/pair representations for diffusion."""

    def __init__(self, config: HelicoConfig):
        super().__init__()
        c = config
        c_s = c.d_single
        c_z = c.d_pair

        # Pair path
        self.relpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=c_z)
        self.pair_norm = nn.LayerNorm(2 * c_z)
        self.pair_proj = linear_no_bias(2 * c_z, c_z)
        self.pair_transition_1 = Transition(c_z, factor=2)
        self.pair_transition_2 = Transition(c_z, factor=2)

        # Single path
        self.fourier = FourierEmbedding(c.c_noise_embedding)
        # s_inputs dim: c_s (single_pre_trunk) + 65 (zeros placeholder) = c_s + 65
        self.s_inputs_dim = c_s + 65
        self.single_norm = nn.LayerNorm(c_s + self.s_inputs_dim)
        self.single_proj = linear_no_bias(c_s + self.s_inputs_dim, c_s)
        self.noise_norm = nn.LayerNorm(c.c_noise_embedding)
        self.noise_proj = linear_no_bias(c.c_noise_embedding, c_s)
        self.single_transition_1 = Transition(c_s, factor=2)
        self.single_transition_2 = Transition(c_s, factor=2)

    def forward(self, s_trunk: torch.Tensor, z_trunk: torch.Tensor,
                s_inputs: torch.Tensor, sigma: torch.Tensor,
                relpe_feats: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_trunk: (B, N, c_s)
            z_trunk: (B, N, N, c_z)
            s_inputs: (B, N, c_s+65)
            sigma: (B,) noise level
            relpe_feats: dict with keys residue_index, token_index, asym_id, entity_id, sym_id
                         each (B, N) per-token tensors
        Returns:
            s_cond: (B, N, c_s), z_cond: (B, N, N, c_z)
        """
        sigma_data = 16.0  # EDM constant

        # Pair conditioning
        relpe = self.relpe(**relpe_feats)
        z = self.pair_proj(self.pair_norm(torch.cat([z_trunk, relpe], dim=-1)))
        z = z + self.pair_transition_1(z)
        z = z + self.pair_transition_2(z)

        # Noise embedding
        noise_input = torch.log(sigma / sigma_data) / 4.0  # (B,)
        n = self.noise_proj(self.noise_norm(self.fourier(noise_input)))  # (B, c_s)

        # Single conditioning
        s = self.single_proj(self.single_norm(torch.cat([s_trunk, s_inputs], dim=-1)))
        s = s + n.unsqueeze(1)
        s = s + self.single_transition_1(s)
        s = s + self.single_transition_2(s)

        return s, z


class DiffusionModule(nn.Module):
    """Full diffusion module with EDM preconditioning (Protenix architecture)."""

    def __init__(self, config: HelicoConfig):
        super().__init__()
        self.config = config
        c = config

        self.conditioning = DiffusionConditioning(config)
        self.atom_encoder = AtomAttentionEncoder(config)
        self.atom_decoder = AtomAttentionDecoder(config)

        self.s_to_token_norm = nn.LayerNorm(c.d_single)
        self.s_to_token_proj = linear_no_bias(c.d_single, c.c_token, zeros_init=True)

        self.token_transformer = DiffusionTransformer(
            n_blocks=c.n_diffusion_token_blocks,
            d_a=c.c_token, d_s=c.d_single, d_z=c.d_pair,
            n_heads=c.n_heads_diffusion_token, head_dim=c.diffusion_token_head_dim,
            cross_attention_mode=False,
            gradient_checkpointing=c.gradient_checkpointing,
        )

        self.out_norm = nn.LayerNorm(c.c_token)

        # Inference schedule
        self.n_steps = c.n_diffusion_steps

    def _edm_precondition(self, sigma: torch.Tensor):
        """EDM preconditioning coefficients."""
        sigma_data = self.config.sigma_data
        sigma_sq = sigma ** 2
        sd_sq = sigma_data ** 2
        c_in = 1.0 / (sd_sq + sigma_sq).sqrt()
        c_skip = sd_sq / (sd_sq + sigma_sq)
        c_out = sigma * sigma_data / (sd_sq + sigma_sq).sqrt()
        return c_in, c_skip, c_out

    def _f_forward(self, x_scaled: torch.Tensor, sigma: torch.Tensor,
                   ref_pos: torch.Tensor, ref_charge: torch.Tensor, ref_features: torch.Tensor,
                   atom_to_token: torch.Tensor, atom_mask: torch.Tensor,
                   s_trunk: torch.Tensor, z_trunk: torch.Tensor, s_inputs: torch.Tensor,
                   relpe_feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inner forward: network prediction."""
        n_tokens = s_trunk.shape[1]

        # Conditioning
        s_cond, z_cond = self.conditioning(s_trunk, z_trunk, s_inputs, sigma, relpe_feats)

        # Atom encoder (uses raw s_trunk for single injection, conditioned z for pair injection)
        a_token, q_skip, c_skip, p_skip, pad_mask = self.atom_encoder(
            ref_pos, ref_charge, ref_features,
            atom_to_token, atom_mask, n_tokens,
            noisy_pos=x_scaled, s_trunk=s_trunk, z_trunk=z_cond)

        # Inject conditioned single into token representation
        a_token = a_token + self.s_to_token_proj(self.s_to_token_norm(s_cond))

        # Token transformer
        a_token = self.token_transformer(a_token, s_cond, z_cond)
        a_token = self.out_norm(a_token)

        # Atom decoder
        r_update = self.atom_decoder(a_token, atom_to_token, q_skip, c_skip, p_skip, pad_mask)
        return r_update

    def forward_training(
        self,
        gt_coords: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_features: torch.Tensor,
        atom_to_token: torch.Tensor,
        atom_mask: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        relpe_feats: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward: single denoising step with EDM preconditioning.

        Returns:
            x_denoised: (B, N_atoms, 3) predicted denoised coordinates
            gt_coords: (B, N_atoms, 3) ground truth
            sigma: (B,) noise level used
        """
        B = gt_coords.shape[0]
        device = gt_coords.device

        # Sample noise level (log-normal)
        log_sigma = self.config.noise_log_mean + self.config.noise_log_std * torch.randn(B, device=device)
        sigma = torch.exp(log_sigma) * self.config.sigma_data

        # Add noise
        noise = torch.randn_like(gt_coords)
        sigma_expand = sigma.view(B, 1, 1)
        x_noisy = gt_coords + sigma_expand * noise

        # EDM preconditioning
        c_in, c_skip, c_out = self._edm_precondition(sigma)
        c_in = c_in.view(B, 1, 1)
        c_skip = c_skip.view(B, 1, 1)
        c_out = c_out.view(B, 1, 1)

        # Forward
        r_update = self._f_forward(
            c_in * x_noisy, sigma,
            ref_pos, ref_charge, ref_features,
            atom_to_token, atom_mask,
            s_trunk, z_trunk, s_inputs,
            relpe_feats)

        x_denoised = c_skip * x_noisy + c_out * r_update
        return x_denoised, gt_coords, sigma

    @torch.no_grad()
    def sample(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_features: torch.Tensor,
        atom_to_token: torch.Tensor,
        atom_mask: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        relpe_feats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inference: Euler denoising with EDM scaling."""
        B, N_atoms, _ = ref_pos.shape
        device = ref_pos.device

        # Log-linear sigma schedule
        s_max, s_min = 160.0, 0.001
        sigmas = torch.exp(torch.linspace(math.log(s_max), math.log(s_min), self.n_steps + 1, device=device))

        x = torch.randn(B, N_atoms, 3, device=device) * s_max

        for i in range(self.n_steps):
            sigma_cur = sigmas[i].expand(B)
            c_in, c_skip, c_out = self._edm_precondition(sigma_cur)
            c_in = c_in.view(B, 1, 1)
            c_skip = c_skip.view(B, 1, 1)
            c_out = c_out.view(B, 1, 1)

            r_update = self._f_forward(
                c_in * x, sigma_cur,
                ref_pos, ref_charge, ref_features,
                atom_to_token, atom_mask,
                s_trunk, z_trunk, s_inputs,
                relpe_feats)

            x_denoised = c_skip * x + c_out * r_update

            # Euler step
            sigma_next = sigmas[i + 1]
            d = (x - x_denoised) / sigmas[i]
            x = x + d * (sigma_next - sigmas[i])

        return x


# ============================================================================
# Loss Functions (Phase 4a)
# ============================================================================

def diffusion_loss(
    x_denoised: torch.Tensor,
    gt_coords: torch.Tensor,
    sigma: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """EDM diffusion loss: weighted MSE on denoised coordinates."""
    while sigma.dim() < gt_coords.dim():
        sigma = sigma.unsqueeze(-1)
    weight = 1.0 / sigma.pow(2).clamp(min=1e-6)

    loss = weight * (x_denoised - gt_coords).pow(2).sum(dim=-1)  # (B, N_atoms)

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
    """Differentiable local distance difference test (lDDT) loss."""
    # Compute pairwise distances
    pred_dists = torch.cdist(pred_coords, pred_coords)  # (B, N, N)
    gt_dists = torch.cdist(gt_coords, gt_coords)

    # Only consider pairs within cutoff in ground truth
    close_mask = (gt_dists < cutoff) & (gt_dists > 0.01)

    if atom_mask is not None:
        pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
        close_mask = close_mask & pair_mask

    # Distance differences
    diff = torch.abs(pred_dists - gt_dists)

    # Smooth scoring at thresholds [0.5, 1.0, 2.0, 4.0]
    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=pred_coords.device)
    # Sigmoid approximation instead of step function
    scores = torch.sigmoid(5.0 * (thresholds.view(1, 1, 1, -1) - diff.unsqueeze(-1)))
    score = scores.mean(dim=-1)  # (B, N, N) average over thresholds

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
    """Binned distance prediction loss on token centers (Ca positions).

    Args:
        pred_logits: (B, N, N, n_bins) predicted distance bin logits
        gt_coords: (B, N, 3) token center coordinates (e.g., Ca)
        atom_mask: (B, N) token mask
        n_bins: number of distance bins
    """
    gt_dists = torch.cdist(gt_coords, gt_coords)  # (B, N, N)

    # Bin boundaries
    boundaries = torch.linspace(min_dist, max_dist, n_bins - 1, device=gt_coords.device)
    gt_bins = torch.bucketize(gt_dists, boundaries)  # (B, N, N)

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
    """Penalize steric clashes (atoms too close together)."""
    dists = torch.cdist(pred_coords, pred_coords)  # (B, N, N)

    # Exclude self-distances
    eye = torch.eye(dists.shape[1], device=dists.device).unsqueeze(0)
    dists = dists + eye * 1e6

    # Clash penalty: soft penalty for distances below threshold
    clash = F.relu(clash_threshold - dists)

    if atom_mask is not None:
        pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
        clash = clash * pair_mask
        return clash.sum() / pair_mask.sum().clamp(min=1)
    return clash.mean()


# ============================================================================
# Confidence Head (Protenix architecture)
# ============================================================================

class ConfidenceHead(nn.Module):
    """Protenix confidence head: pairformer + PAE/PDE/pLDDT/resolved heads.

    Uses z_init from s_inputs, distance pair embeddings from predicted coords,
    a 4-block pairformer, then per-atom pLDDT and resolved via einsum weights.
    """

    def __init__(self, config: HelicoConfig):
        super().__init__()
        c = config

        # Input processing
        self.input_s_norm = LayerNorm(c.d_single)
        self.linear_s1 = linear_no_bias(c.c_s_inputs, c.d_pair)   # 449->128
        self.linear_s2 = linear_no_bias(c.c_s_inputs, c.d_pair)   # 449->128

        # Distance pair embeddings
        n_dist_bins = c.n_distance_bins  # 39
        lower = torch.linspace(3.25, 50.75, n_dist_bins)
        upper = torch.cat([torch.linspace(4.50, 52.0, n_dist_bins - 1), torch.tensor([1e6])])
        self.register_buffer("lower_bins", lower)
        self.register_buffer("upper_bins", upper)
        self.linear_d = linear_no_bias(n_dist_bins, c.d_pair)     # 39->128
        self.linear_d_raw = linear_no_bias(1, c.d_pair)           # 1->128

        # 4-block PairformerStack (same dims as trunk)
        conf_config = HelicoConfig(
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
        self.linear_pae = linear_no_bias(c.d_pair, c.n_pae_bins)  # 128->64
        self.pde_norm = LayerNorm(c.d_pair)
        self.linear_pde = linear_no_bias(c.d_pair, c.n_pae_bins)  # 128->64 (PDE uses same bins)
        self.plddt_norm = LayerNorm(c.d_single)
        self.plddt_weight = nn.Parameter(torch.zeros(c.max_atoms_per_token, c.d_single, c.n_plddt_bins))
        self.resolved_norm = LayerNorm(c.d_single)
        self.resolved_weight = nn.Parameter(torch.zeros(c.max_atoms_per_token, c.d_single, 2))

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
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            s_trunk: (B, N, d_single) trunk single (detached)
            z_trunk: (B, N, N, d_pair) trunk pair (detached)
            s_inputs: (B, N, c_s_inputs) single input features (detached)
            pred_coords: (B, N_atoms, 3) predicted coordinates
            atom_to_token: (B, N_atoms) atom-to-token mapping
            atom_mask: (B, N_atoms) atom mask
            mask: (B, N) token mask
            pair_mask: (B, N, N) pair mask
        Returns:
            dict with pae, pde, plddt, resolved logits
        """
        s = self.input_s_norm(s_trunk.detach())
        s_inp = s_inputs.detach()

        # z_init from s_inputs outer product
        z = z_trunk.detach() + self.linear_s1(s_inp).unsqueeze(2) + self.linear_s2(s_inp).unsqueeze(1)

        # Distance pair embeddings from pred_coords
        # Compute token center coords from predicted atom coords
        B, N_tok = s.shape[:2]
        token_centers = self._get_token_centers(pred_coords, atom_to_token, atom_mask, N_tok)

        dists = torch.cdist(token_centers, token_centers)  # (B, N, N)
        # One-hot distance binning
        d_unsq = dists.unsqueeze(-1)  # (B, N, N, 1)
        one_hot = ((d_unsq > self.lower_bins) & (d_unsq < self.upper_bins)).to(z.dtype)
        z = z + self.linear_d(one_hot)
        z = z + self.linear_d_raw(d_unsq.to(z.dtype))

        # Pairformer
        s, z = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

        # Output heads
        pae_logits = self.linear_pae(self.pae_norm(z))
        pde_logits = self.linear_pde(self.pde_norm(z + z.transpose(-2, -3)))
        plddt_logits = torch.einsum("...tc,acb->...tab", self.plddt_norm(s), self.plddt_weight)
        resolved_logits = torch.einsum("...tc,acb->...tab", self.resolved_norm(s), self.resolved_weight)

        return {
            "pae_logits": pae_logits,          # (B, N, N, n_pae_bins)
            "pde_logits": pde_logits,          # (B, N, N, n_pae_bins)
            "plddt_logits": plddt_logits,      # (B, N, max_atoms * n_plddt_bins)
            "resolved_logits": resolved_logits,  # (B, N, max_atoms * 2)
        }

    def _get_token_centers(self, coords: torch.Tensor, atom_to_token: torch.Tensor,
                           atom_mask: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Compute mean atom coordinates per token."""
        B = coords.shape[0]
        device = coords.device
        dt = coords.dtype
        centers = torch.zeros(B, n_tokens, 3, device=device, dtype=dt)
        counts = torch.zeros(B, n_tokens, 1, device=device, dtype=dt)
        masked_coords = coords * atom_mask.unsqueeze(-1).to(dt)
        idx3 = atom_to_token.unsqueeze(-1).expand(-1, -1, 3)
        centers.scatter_add_(1, idx3, masked_coords)
        counts.scatter_add_(1, atom_to_token.unsqueeze(-1), atom_mask.unsqueeze(-1).to(dt))
        return centers / counts.clamp(min=1)


# ============================================================================
# Confidence Score Computation
# ============================================================================

def compute_plddt(plddt_logits: torch.Tensor) -> torch.Tensor:
    """Compute per-atom pLDDT from logits.

    Args:
        plddt_logits: (B, N_tok, max_atoms_per_token, n_plddt_bins) raw logits

    Returns:
        (B, N_tok, max_atoms_per_token) pLDDT scores in [0, 100]
    """
    n_bins = plddt_logits.shape[-1]  # 50
    bin_centers = torch.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins,
                                 device=plddt_logits.device, dtype=plddt_logits.dtype)
    probs = F.softmax(plddt_logits, dim=-1)
    plddt = (probs * bin_centers).sum(dim=-1)  # (B, N_tok, max_atoms)
    return plddt * 100.0


def compute_pae(pae_logits: torch.Tensor) -> torch.Tensor:
    """Compute predicted aligned error matrix from logits.

    Args:
        pae_logits: (B, N, N, n_pae_bins) raw logits

    Returns:
        (B, N, N) PAE in Angstroms, range [0, 32]
    """
    n_bins = pae_logits.shape[-1]  # 64
    # 64 bins covering 0-32A in 0.5A steps, bin centers at 0.25, 0.75, ..., 31.75
    bin_centers = torch.linspace(0.25, 31.75, n_bins,
                                 device=pae_logits.device, dtype=pae_logits.dtype)
    probs = F.softmax(pae_logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


def _compute_tm_term(pae_logits: torch.Tensor, d0: torch.Tensor) -> torch.Tensor:
    """Compute TM-score term from PAE logits.

    Args:
        pae_logits: (B, N, N, n_bins)
        d0: (B, 1, 1) or scalar — TM-score distance scaling factor

    Returns:
        (B, N, N) expected TM-score contribution per pair
    """
    n_bins = pae_logits.shape[-1]
    bin_centers = torch.linspace(0.25, 31.75, n_bins,
                                 device=pae_logits.device, dtype=pae_logits.dtype)
    probs = F.softmax(pae_logits, dim=-1)  # (B, N, N, n_bins)
    # TM term: 1 / (1 + (d/d0)^2) per bin
    tm_per_bin = 1.0 / (1.0 + (bin_centers / d0.unsqueeze(-1)) ** 2)  # (B, 1, 1, n_bins)
    return (probs * tm_per_bin).sum(dim=-1)  # (B, N, N)


def compute_ptm(pae_logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute predicted TM-score from PAE logits.

    Args:
        pae_logits: (B, N, N, n_pae_bins)
        mask: (B, N) token mask, or None for all tokens

    Returns:
        (B,) pTM scores in [0, 1]
    """
    B, N = pae_logits.shape[:2]
    device = pae_logits.device

    if mask is None:
        mask = torch.ones(B, N, device=device, dtype=pae_logits.dtype)
    else:
        mask = mask.to(dtype=pae_logits.dtype)

    # d0 = 1.24 * max(N_res - 15, 19)^(1/3) - 1.8
    n_res = mask.sum(dim=-1).clamp(min=19)  # (B,)
    d0 = 1.24 * (n_res.clamp(min=19 + 15) - 15).pow(1.0 / 3.0) - 1.8  # (B,)
    d0 = d0.reshape(B, 1, 1)

    tm_pair = _compute_tm_term(pae_logits, d0)  # (B, N, N)

    # mask pairs: mask_i * mask_j
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B, N, N)
    tm_pair = tm_pair * pair_mask

    # For each alignment residue i, compute mean TM over scored residues j
    n_scored = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
    tm_per_aligned = tm_pair.sum(dim=-1) / n_scored  # (B, N)

    # pTM = max over alignment dimension, masked
    tm_per_aligned = tm_per_aligned.masked_fill(mask == 0, 0.0)
    ptm = tm_per_aligned.max(dim=-1).values  # (B,)
    return ptm


def compute_iptm(
    pae_logits: torch.Tensor,
    chain_indices: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute interface predicted TM-score (across different chains).

    Args:
        pae_logits: (B, N, N, n_pae_bins)
        chain_indices: (B, N) chain index per token
        mask: (B, N) token mask, or None

    Returns:
        (B,) ipTM scores in [0, 1]
    """
    B, N = pae_logits.shape[:2]
    device = pae_logits.device

    if mask is None:
        mask = torch.ones(B, N, device=device, dtype=pae_logits.dtype)
    else:
        mask = mask.to(dtype=pae_logits.dtype)

    # Inter-chain mask: different chains
    inter_mask = (chain_indices.unsqueeze(-1) != chain_indices.unsqueeze(-2)).float()  # (B, N, N)
    pair_mask = inter_mask * mask.unsqueeze(-1) * mask.unsqueeze(-2)

    # d0 based on total number of residues
    n_res = mask.sum(dim=-1).clamp(min=19)
    d0 = 1.24 * (n_res.clamp(min=19 + 15) - 15).pow(1.0 / 3.0) - 1.8
    d0 = d0.reshape(B, 1, 1)

    tm_pair = _compute_tm_term(pae_logits, d0)  # (B, N, N)
    tm_pair = tm_pair * pair_mask

    # Sum over scored dimension, then max over aligned dimension
    n_inter = pair_mask.sum(dim=-1).clamp(min=1)  # (B, N) — number of inter-chain partners per token
    tm_per_aligned = tm_pair.sum(dim=-1) / n_inter  # (B, N)

    # Mask tokens with no inter-chain partners
    has_inter = (pair_mask.sum(dim=-1) > 0).float()
    tm_per_aligned = tm_per_aligned * has_inter
    # If no inter-chain pairs at all, return 0
    any_inter = has_inter.sum(dim=-1) > 0  # (B,)
    iptm = tm_per_aligned.max(dim=-1).values  # (B,)
    iptm = iptm * any_inter.float()
    return iptm


def compute_ranking_score(
    ptm: torch.Tensor,
    iptm: torch.Tensor,
    has_interface: torch.Tensor,
) -> torch.Tensor:
    """Compute ranking score: 0.8*iptm + 0.2*ptm for multi-chain, ptm for single-chain.

    Args:
        ptm: (B,) pTM scores
        iptm: (B,) ipTM scores
        has_interface: (B,) bool tensor, True when >1 unique chain

    Returns:
        (B,) ranking scores
    """
    multi = has_interface.float()
    return multi * (0.8 * iptm + 0.2 * ptm) + (1.0 - multi) * ptm


def _flatten_plddt(
    plddt: torch.Tensor,
    atom_to_token: torch.Tensor,
    atoms_per_token: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token pLDDT to per-atom pLDDT.

    Args:
        plddt: (B, N_tok, max_atoms_per_token) per-token-atom pLDDT scores
        atom_to_token: (B, N_atoms) token index for each atom
        atoms_per_token: (B, N_tok) number of atoms per token
        atom_mask: (B, N_atoms) atom validity mask

    Returns:
        (B, N_atoms) per-atom pLDDT scores
    """
    B, N_atoms = atom_to_token.shape
    device = plddt.device

    # Compute within-token atom index for each atom
    # For each atom, count how many previous atoms share the same token
    tok_ids = atom_to_token  # (B, N_atoms)

    # Use cumsum approach: for each token, atoms are contiguous
    # atoms_per_token cumsum gives token start offsets
    tok_starts = torch.zeros_like(atoms_per_token)
    tok_starts[:, 1:] = atoms_per_token[:, :-1].cumsum(dim=-1)  # (B, N_tok)

    # For each atom, within_idx = atom_global_idx - tok_starts[token_id]
    atom_indices = torch.arange(N_atoms, device=device).unsqueeze(0).expand(B, -1)  # (B, N_atoms)
    token_start_per_atom = tok_starts.gather(1, tok_ids)  # (B, N_atoms)
    within_idx = atom_indices - token_start_per_atom  # (B, N_atoms)
    within_idx = within_idx.clamp(min=0, max=plddt.shape[-1] - 1)

    # Gather: plddt[b, tok_ids[b, a], within_idx[b, a]]
    flat_plddt = plddt.gather(
        1, tok_ids.unsqueeze(-1).expand(-1, -1, plddt.shape[-1])
    )  # (B, N_atoms, max_atoms_per_token)
    result = flat_plddt.gather(2, within_idx.unsqueeze(-1)).squeeze(-1)  # (B, N_atoms)
    return result * atom_mask.float()


# ============================================================================
# Affinity Module (Phase 4c) — Boltz2 feature
# ============================================================================

class AffinityModule(nn.Module):
    """Binding affinity prediction module (Boltz2 extension).

    Uses a small separate PairFormer operating on pocket region.
    Dual output: binary binder/non-binder + continuous affinity regression.
    """

    def __init__(self, config: HelicoConfig):
        super().__init__()
        c = config

        # Project pocket features to affinity dimension
        self.single_proj = nn.Linear(c.d_single, c.d_affinity)
        self.pair_proj = nn.Linear(c.d_pair, c.d_affinity)

        # Small Pairformer for pocket
        pocket_config = HelicoConfig(
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

        # Output heads
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
        """
        Args:
            single: (B, N, d_single) from Pairformer
            pair: (B, N, N, d_pair) from Pairformer
            pocket_mask: (B, N) binary mask for pocket tokens
        Returns:
            dict with bind_logits (B, 1) and affinity (B, 1)
        """
        # Extract pocket tokens
        B, N, _ = single.shape

        # Project to affinity dimensions
        s = self.single_proj(single)  # (B, N, d_affinity)
        z = self.pair_proj(pair)      # (B, N, N, d_affinity)

        # Mask non-pocket tokens
        s = s * pocket_mask.unsqueeze(-1)
        z = z * pocket_mask.unsqueeze(-1).unsqueeze(-2) * pocket_mask.unsqueeze(-2).unsqueeze(-1)

        # Small Pairformer on full representation (masked)
        pair_mask = pocket_mask.unsqueeze(-1) & pocket_mask.unsqueeze(-2)
        s, z = self.pocket_pairformer(s, z, mask=pocket_mask, pair_mask=pair_mask)

        # Pool pocket tokens for classification/regression
        pocket_single = s * pocket_mask.unsqueeze(-1)
        pooled = pocket_single.sum(dim=1) / pocket_mask.sum(dim=1, keepdim=True).clamp(min=1)

        bind_logits = self.classifier(pooled)  # (B, 1)
        affinity = self.regressor(pooled)       # (B, 1) log10 IC50/Ki/Kd

        return {
            "bind_logits": bind_logits,
            "affinity": affinity,
        }


# ============================================================================
# Full Model (Phase 2-4 combined)
# ============================================================================

class Helico(nn.Module):
    """Complete Helico model with Protenix-matching architecture."""

    def __init__(self, config: HelicoConfig | None = None):
        super().__init__()
        if config is None:
            config = HelicoConfig()
        self.config = config

        # Input embedding (AtomAttentionEncoder without coords)
        self.input_embedder = InputFeatureEmbedder(config)

        # Trunk initialization
        self.linear_sinit = linear_no_bias(config.c_s_inputs, config.d_single)     # 449->384
        self.linear_zinit1 = linear_no_bias(config.d_single, config.d_pair)        # 384->128
        self.linear_zinit2 = linear_no_bias(config.d_single, config.d_pair)        # 384->128
        self.trunk_relpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=config.d_pair)
        self.linear_token_bond = linear_no_bias(1, config.d_pair)                  # 1->128

        # Recycling (zero-initialized)
        self.layernorm_s = LayerNorm(config.d_single)
        self.linear_s = linear_no_bias(config.d_single, config.d_single, zeros_init=True)
        self.layernorm_z_cycle = LayerNorm(config.d_pair)
        self.linear_z_cycle = linear_no_bias(config.d_pair, config.d_pair, zeros_init=True)

        # Template embedder
        self.template_embedder = TemplateEmbedder(config)

        # MSA module
        self.msa_module = MSAModule(config)

        # Pairformer trunk
        self.pairformer = Pairformer(config)

        # Diffusion module
        self.diffusion = DiffusionModule(config)

        # Heads
        self.confidence_head = ConfidenceHead(config)
        self.distogram_head = DistogramHead(config)
        self.affinity = AffinityModule(config)

    def _build_ref_features(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build reference features for atom attention encoder.

        Returns:
            ref_charge: (B, N_atoms, 1) — zeros
            ref_features: (B, N_atoms, 385) — mask + element_onehot(128) + zeros(256)
        """
        B, N_atoms = batch["atom_element_idx"].shape
        device = batch["atom_element_idx"].device
        dtype = batch.get("ref_coords", batch["atom_coords"]).dtype

        ref_charge = torch.zeros(B, N_atoms, 1, device=device, dtype=dtype)

        # Element one-hot (128 dims, padded from n_elements)
        elem_onehot = F.one_hot(batch["atom_element_idx"].clamp(max=127), 128).to(dtype)

        # Atom mask as feature
        atom_mask = batch.get("atom_mask")
        if atom_mask is not None:
            mask_feat = atom_mask.unsqueeze(-1).to(dtype)
        else:
            mask_feat = torch.ones(B, N_atoms, 1, device=device, dtype=dtype)

        # atom_name_chars placeholder (256 zeros)
        name_chars = torch.zeros(B, N_atoms, 256, device=device, dtype=dtype)

        ref_features = torch.cat([mask_feat, elem_onehot, name_chars], dim=-1)  # (B, N_atoms, 385)
        return ref_charge, ref_features

    def _build_relpe_feats(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Build relpe feature dict from batch for RelativePositionEncoding."""
        return {
            "residue_index": batch["rel_pos"],
            "token_index": batch["token_index"],
            "asym_id": batch["chain_indices"],
            "entity_id": batch["entity_id"],
            "sym_id": batch["sym_id"],
        }

    def _build_s_inputs(self, batch: dict[str, torch.Tensor], ref_charge: torch.Tensor,
                        ref_features: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """Build s_inputs (B, N_tok, 449) via InputFeatureEmbedder."""
        B, N_tok = batch["token_types"].shape
        device = batch["token_types"].device
        dtype = ref_charge.dtype

        # restype one-hot (32 dims from token_types)
        restype = F.one_hot(batch["token_types"].clamp(max=31), 32).to(dtype)

        # MSA profile (32 dims: 22 profile + 10 zeros padding)
        profile_22 = batch.get("msa_profile", torch.zeros(B, N_tok, 22, device=device, dtype=dtype))
        profile = F.pad(profile_22, (0, 10))  # (B, N_tok, 32)

        # deletion_mean (1 dim)
        deletion_mean = torch.zeros(B, N_tok, 1, device=device, dtype=dtype)

        return self.input_embedder(
            ref_pos=batch["ref_coords"],
            ref_charge=ref_charge,
            ref_features=ref_features,
            atom_to_token=batch["atom_to_token"],
            atom_mask=atom_mask,
            n_tokens=N_tok,
            restype=restype,
            profile=profile,
            deletion_mean=deletion_mean,
        )

    def _build_msa_raw(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build raw MSA features (B, N_msa, N_tok, 34) and mask."""
        B, N_tok = batch["token_types"].shape
        device = batch["token_types"].device
        dtype = batch.get("ref_coords", batch["atom_coords"]).dtype

        cluster_msa = batch.get("cluster_msa")
        if cluster_msa is None:
            msa_raw = torch.zeros(B, 1, N_tok, 34, device=device, dtype=dtype)
            return msa_raw, None

        N_msa = cluster_msa.shape[1]
        # One-hot encode MSA residues (32 types -> 32 dims, padded to 32)
        msa_onehot = F.one_hot(cluster_msa.clamp(max=31), 32).to(dtype)

        # has_deletion and deletion_value (1 + 1 dims)
        has_del = torch.zeros(B, N_msa, N_tok, 1, device=device, dtype=dtype)
        del_val = torch.zeros(B, N_msa, N_tok, 1, device=device, dtype=dtype)

        msa_raw = torch.cat([msa_onehot, has_del, del_val], dim=-1)  # (B, N_msa, N_tok, 34)
        return msa_raw, None

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        compute_confidence: bool = True,
        compute_affinity: bool = False,
        pocket_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for training."""
        mask = batch.get("token_mask")
        pair_mask = None
        if mask is not None:
            pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()

        B, N_tok = batch["token_types"].shape
        device = batch["token_types"].device

        # Build reference features and atom mask
        ref_charge, ref_features = self._build_ref_features(batch)
        atom_mask = batch.get("atom_mask", torch.ones(B, batch["atom_coords"].shape[1], device=device))
        atom_mask = atom_mask.float()

        # 1. Input embedding -> s_inputs (B, N_tok, 449)
        s_inputs = self._build_s_inputs(batch, ref_charge, ref_features, atom_mask)

        # 2. Trunk initialization
        s_init = self.linear_sinit(s_inputs)
        z_init = self.linear_zinit1(s_init).unsqueeze(2) + self.linear_zinit2(s_init).unsqueeze(1)

        relpe_feats = self._build_relpe_feats(batch)
        z_init = z_init + self.trunk_relpe(**relpe_feats)

        # Token bonds
        token_bonds = batch.get("token_bonds")
        if token_bonds is not None:
            z_init = z_init + self.linear_token_bond(token_bonds.unsqueeze(-1).to(z_init.dtype))

        # 3. Recycling loop
        msa_raw, msa_mask = self._build_msa_raw(batch)
        n_cycles = self.config.n_cycles

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        for cycle in range(n_cycles):
            z = z_init + self.linear_z_cycle(self.layernorm_z_cycle(z))
            z = z + self.template_embedder(batch, z)  # returns 0 for now
            z = self.msa_module(msa_raw, z, s_inputs, msa_mask, pair_mask)
            s = s_init + self.linear_s(self.layernorm_s(s))
            s, z = self.pairformer(s, z, mask=mask, pair_mask=pair_mask)

        results = {"single": s, "pair": z}

        # 4. Diffusion — s_inputs is already (B, N_tok, 449 = d_single + 65)

        x_denoised, gt_coords, sigma = self.diffusion.forward_training(
            gt_coords=batch["atom_coords"],
            ref_pos=batch["ref_coords"],
            ref_charge=ref_charge,
            ref_features=ref_features,
            atom_to_token=batch["atom_to_token"],
            atom_mask=atom_mask,
            s_trunk=s,
            z_trunk=z,
            s_inputs=s_inputs,
            relpe_feats=relpe_feats,
        )

        results["x_denoised"] = x_denoised
        results["sigma"] = sigma
        results["diffusion_loss"] = diffusion_loss(x_denoised, gt_coords, sigma, atom_mask)

        # 5. Distogram (from trunk pair)
        distogram_logits = self.distogram_head(z)
        results["distogram_logits"] = distogram_logits

        # 6. Confidence head (uses pred_coords from diffusion)
        if compute_confidence:
            confidence = self.confidence_head(
                s_trunk=s, z_trunk=z, s_inputs=s_inputs,
                pred_coords=x_denoised.detach(),
                atom_to_token=batch["atom_to_token"],
                atom_mask=atom_mask,
                mask=mask, pair_mask=pair_mask,
            )
            results.update(confidence)

            token_centers = self._get_token_centers(batch)
            results["distogram_loss"] = distogram_loss(
                distogram_logits, token_centers, mask,
            )

        # Affinity module
        if compute_affinity and pocket_mask is not None:
            affinity = self.affinity(s, z, pocket_mask)
            results.update(affinity)

        return results

    def _get_token_centers(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute token center coordinates (mean of atom coords per token)."""
        B = batch["atom_coords"].shape[0]
        N = batch["token_types"].shape[1]
        device = batch["atom_coords"].device

        centers = torch.zeros(B, N, 3, device=device, dtype=batch["atom_coords"].dtype)
        counts = torch.zeros(B, N, 1, device=device, dtype=batch["atom_coords"].dtype)

        for b in range(B):
            n_atoms = batch["n_atoms"][b]
            idx = batch["atom_to_token"][b, :n_atoms]
            centers[b].scatter_add_(0, idx.unsqueeze(1).expand(-1, 3), batch["atom_coords"][b, :n_atoms])
            counts[b].scatter_add_(0, idx.unsqueeze(1), torch.ones(n_atoms, 1, device=device, dtype=batch["atom_coords"].dtype))

        counts = counts.clamp(min=1)
        return centers / counts

    @torch.no_grad()
    def predict(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int = 5,
    ) -> dict[str, torch.Tensor]:
        """Run inference: generate structure predictions."""
        self.eval()
        mask = batch.get("token_mask")
        pair_mask = None
        if mask is not None:
            pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()

        B, N_tok = batch["token_types"].shape
        device = batch["token_types"].device

        ref_charge, ref_features = self._build_ref_features(batch)
        atom_mask = batch.get("atom_mask", torch.ones(B, batch["ref_coords"].shape[1], device=device))
        atom_mask = atom_mask.float()

        # Build s_inputs
        s_inputs = self._build_s_inputs(batch, ref_charge, ref_features, atom_mask)

        # Trunk init
        s_init = self.linear_sinit(s_inputs)
        z_init = self.linear_zinit1(s_init).unsqueeze(2) + self.linear_zinit2(s_init).unsqueeze(1)
        relpe_feats = self._build_relpe_feats(batch)
        z_init = z_init + self.trunk_relpe(**relpe_feats)
        token_bonds = batch.get("token_bonds")
        if token_bonds is not None:
            z_init = z_init + self.linear_token_bond(token_bonds.unsqueeze(-1).to(z_init.dtype))

        # Recycling
        msa_raw, msa_mask = self._build_msa_raw(batch)
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        for cycle in range(self.config.n_cycles):
            z = z_init + self.linear_z_cycle(self.layernorm_z_cycle(z))
            z = z + self.template_embedder(batch, z)
            z = self.msa_module(msa_raw, z, s_inputs, msa_mask, pair_mask)
            s = s_init + self.linear_s(self.layernorm_s(s))
            s, z = self.pairformer(s, z, mask=mask, pair_mask=pair_mask)

        # Generate samples
        all_coords = []
        for _ in range(n_samples):
            coords = self.diffusion.sample(
                ref_pos=batch["ref_coords"],
                ref_charge=ref_charge,
                ref_features=ref_features,
                atom_to_token=batch["atom_to_token"],
                atom_mask=atom_mask,
                s_trunk=s,
                z_trunk=z,
                s_inputs=s_inputs,
                relpe_feats=relpe_feats,
            )
            all_coords.append(coords)

        all_coords = torch.stack(all_coords, dim=1)  # (B, n_samples, N_atoms, 3)

        # Use first sample for confidence
        confidence = self.confidence_head(
            s_trunk=s, z_trunk=z, s_inputs=s_inputs,
            pred_coords=all_coords[:, 0],
            atom_to_token=batch["atom_to_token"],
            atom_mask=atom_mask,
            mask=mask, pair_mask=pair_mask,
        )

        # Compute confidence scores from logits
        plddt = compute_plddt(confidence["plddt_logits"])
        pae = compute_pae(confidence["pae_logits"])
        ptm = compute_ptm(confidence["pae_logits"], mask=mask)

        chain_indices = batch.get("chain_indices")
        if chain_indices is not None:
            iptm = compute_iptm(confidence["pae_logits"], chain_indices, mask=mask)
            unique_counts = []
            for b in range(B):
                ci = chain_indices[b]
                if mask is not None:
                    ci = ci[mask[b]]
                unique_counts.append(ci.unique().numel())
            has_interface = torch.tensor([c > 1 for c in unique_counts], device=device, dtype=torch.bool)
        else:
            iptm = ptm.clone()
            has_interface = torch.zeros(B, device=device, dtype=torch.bool)

        ranking = compute_ranking_score(ptm, iptm, has_interface)

        # Flatten pLDDT to per-atom
        plddt_flat = _flatten_plddt(
            plddt, batch["atom_to_token"], batch["atoms_per_token"], atom_mask,
        )

        return {
            "coords": all_coords[:, 0],          # (B, N_atoms, 3)
            "all_coords": all_coords,             # (B, n_samples, N_atoms, 3)
            "plddt": plddt_flat,                  # (B, N_atoms) 0-100 scale
            "pae": pae,                           # (B, N_tok, N_tok) Angstroms
            "ptm": ptm,                           # (B,)
            "iptm": iptm,                         # (B,)
            "ranking_score": ranking,             # (B,)
            # Raw logits for downstream use
            "pae_logits": confidence["pae_logits"],
            "plddt_logits": confidence["plddt_logits"],
            "pde_logits": confidence["pde_logits"],
        }
