"""Triangle operations on the pair representation (AF3 SI §3.4).

Two fused cuEquivariance kernels: triangle multiplicative update and
triangle attention. Each comes in "starting" / "outgoing" and
"ending" / "incoming" variants, together giving the pair tensor z_ij a
way to mix information about the i→k→j and i→k←j triangles respectively.

These are the most compute-heavy parts of the trunk; we rely on the
fused cuequivariance-torch kernels rather than a PyTorch-native
implementation. Target GPUs are H100/B200.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import cuequivariance_torch as cuet

from .blocks import LayerNorm


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle multiplicative update — AF3 SI Algorithm 12 (outgoing) / 13 (incoming).

    LayerNorm(z) → project to (a, b) with sigmoid gate → einsum:
      outgoing: z_ij = Σ_k a_ik ⊙ b_jk
      incoming: z_ij = Σ_k a_ki ⊙ b_kj
    → LayerNorm → output projection, then element-wise gated by
    sigmoid(linear(h)). The fused cuequivariance kernel does all of this
    in one call.

    Kernel constraint: ``d`` must be divisible by 32.
    """

    def __init__(self, d: int, direction: str = "outgoing"):
        super().__init__()
        assert direction in ("outgoing", "incoming")
        assert d % 32 == 0, f"d={d} must be multiple of 32 for cuEquivariance kernel"
        self.direction = direction
        self.d = d

        self.layer_norm_in = LayerNorm(d)
        self.linear_p = nn.Linear(d, 2 * d, bias=False)  # (a, b) projection
        self.linear_g = nn.Linear(d, 2 * d, bias=False)  # (a_gate, b_gate)

        self.layer_norm_out = LayerNorm(d)
        self.output_projection = nn.Linear(d, d, bias=False)
        self.output_gate = nn.Linear(d, d, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Output-gate zero init → each block is a no-op at initialization
        # (sigmoid(0)=0.5 on the gate, but the gate multiplies linear(h)
        # which starts at zero). Lets the stack train incrementally.
        nn.init.zeros_(self.output_gate.weight)

    @torch.compiler.disable
    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """z: (B, N, N, D). mask: (B, N, N) or None."""
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
    """Triangle attention — AF3 SI Algorithm 14 (starting) / 15 (ending).

    Attention over one edge of the triangle:
      starting: softmax over k of z_ij ↔ z_ik, scored via q_ij·k_ik + bias(z_kj)
      ending:   same but on the transposed pair tensor (edges ending at i)
    fused into a single cuequivariance kernel. Output is then
    element-wise gated and projected back to ``d``.

    Kernel constraint: ``n_heads * head_dim == d``.
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
        """z: (B, N, N, D). mask: (B, N, N) or None."""
        B, N, _, D = z.shape
        H = self.n_heads
        dh = self.head_dim

        # "ending" mode = starting-mode attention on transposed z
        z_input = z if self.mode == "starting" else z.transpose(1, 2).contiguous()
        mask_input = (
            mask if self.mode == "starting"
            else (mask.transpose(1, 2).contiguous() if mask is not None else None)
        )

        z_norm = self.norm(z_input)
        qkv = self.qkv_proj(z_norm)  # (B, N, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # cuequivariance expects (B, N, H, N, dh)
        q = q.reshape(B, N, N, H, dh).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(B, N, N, H, dh).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(B, N, N, H, dh).permute(0, 1, 3, 2, 4).contiguous()

        bias = self.bias_proj(z_norm).permute(0, 3, 1, 2).unsqueeze(1).contiguous()

        tri_mask = None
        if mask_input is not None:
            tri_mask = mask_input.unsqueeze(2).unsqueeze(3)

        out = cuet.triangle_attention(q, k, v, bias, mask=tri_mask, scale=self.scale)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, N, N, D)

        # Gate-then-project (AF3 ordering: out_proj(gate * attn_out))
        gate = torch.sigmoid(self.gate(z_norm))
        out = self.out_proj(gate * out)

        if self.mode == "ending":
            out = out.transpose(1, 2)

        return out
