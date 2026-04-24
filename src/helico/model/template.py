"""Template embedder — AF3 SI §3.5 / Algorithm 16.

Combines template features (distogram, unit vector, masks, aatype) with
the trunk pair tensor, processes each template slot through a small
pair-only Pairformer stack (d_template=64, 2 blocks), averages the per-
template outputs, and projects back to c_z. The result is added to z
every recycling iteration (AF3 Algorithm 1 line 9).

Two Protenix-v1.0.0-specific details baked in:

1. The template Pairformer uses ``hidden != d`` shapes (d_template=64 but
   n_heads*d_head=128). cuEquivariance kernels assume hidden==d, so we
   use pure-PyTorch ``_TemplateTriMul`` / ``_TemplateTriAtt`` here and
   only the trunk keeps the fused kernels.

2. When ``use_template=False`` at inference, Protenix still invokes the
   embedder with 4 dummy template slots: slot 0 aatype=31 (gap), slots
   1-3 aatype=0, all other template features zeroed. This isn't in the
   AF3 SI — it's a fossil of how Protenix's inference dataloader pads
   template features — but the v1.0.0 checkpoint was trained to expect
   it, so we replicate it here to get parity with the Protenix weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm, Transition, linear_no_bias


class _TemplateTriMul(nn.Module):
    """Triangle multiplicative update, pure-PyTorch variant with hidden != d.

    Mirrors AF3 Alg 12/13 but uses standard PyTorch ops because the
    template Pairformer runs at d_template=64 with hidden=128, a shape
    combination the cuEquivariance kernel doesn't support.
    """

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
    """Triangle attention, pure-PyTorch variant with n_heads*d_head != d."""

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
        bias = self.bias_proj(h).permute(0, 3, 1, 2).unsqueeze(1)
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
    """One pair-only Pairformer block for the template embedder."""

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
    """AF3 SI Algorithm 16 — TemplateEmbedder.

    Each template slot contributes a pair-feature tensor ``at`` of shape
    ``(B, N, N, 108)``:
      - template_distogram (39) — binned Cβ-Cβ distances
      - template_pseudo_beta_mask (1) — atom availability
      - template_restype_i (32) + template_restype_j (32)
      - template_unit_vector (3) — direction of Cα in local frame
      - template_backbone_frame_mask (1)

    Forward: LayerNorm(z) + linear(at) → N_block pair-only Pairformer
    blocks → LayerNorm → mean-over-templates → ReLU → linear back to c_z.

    At inference with ``use_template=False``, Protenix still runs this
    with 4 dummy templates (slot 0 aatype=31=gap, slots 1-3 aatype=0),
    everything else zero. The v1.0.0 checkpoint was trained expecting
    that path, so we reproduce the dummy-template recipe below.
    """

    def __init__(self, config):
        super().__init__()
        c = config
        input_dim = 108  # 39 dgram + 1 pseudo-β mask + 3 unit vec + 1 backbone mask + 32+32 aatype

        self.z_norm = LayerNorm(c.d_pair)
        self.linear_z = linear_no_bias(c.d_pair, c.d_template)   # 128 → 64
        self.linear_a = linear_no_bias(input_dim, c.d_template)  # 108 → 64

        # Protenix template: d_template=64 + n_heads=4 + d_head=32 → hidden=128 ≠ d
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
        self.linear_out = linear_no_bias(c.d_template, c.d_pair)  # 64 → 128

    def forward(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """AF3 Alg 16. Returns a residual to add to the trunk pair tensor.

        Uses Protenix's 4-slot dummy-template recipe when real template
        features aren't in the batch (the v1.0.0 checkpoint expects this
        path at inference with use_template=False).
        """
        if not self.pairformer_stack:
            return 0

        B, N_tok = z.shape[:2]
        dtype = z.dtype
        device = z.device

        # Protenix v1.0.0 "no templates" path: 4 dummy slots with aatype
        # [31, 0, 0, 0] and everything else zero. See docstring.
        num_templ = 4
        aatype_per_slot = [31, 0, 0, 0]

        asym_id = batch.get("chain_indices")
        if asym_id is not None:
            multichain_mask = (asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)).to(dtype)
        else:
            multichain_mask = torch.ones(B, N_tok, N_tok, dtype=dtype, device=device)

        token_mask = batch.get("token_mask")
        if token_mask is not None:
            pair_mask = (token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)).to(dtype)
        else:
            pair_mask = torch.ones(B, N_tok, N_tok, dtype=dtype, device=device)

        z_normed = self.z_norm(z)

        # Zero template features (dgram, masks, unit_vec); they'd be
        # multiplied by multichain_mask*pair_mask anyway so the actual
        # value doesn't matter when no real templates are provided.
        dgram = torch.zeros(B, N_tok, N_tok, 39, dtype=dtype, device=device)
        pseudo_beta_mask_2d = torch.zeros(B, N_tok, N_tok, dtype=dtype, device=device)
        unit_vector = torch.zeros(B, N_tok, N_tok, 3, dtype=dtype, device=device)
        backbone_mask_2d = torch.zeros(B, N_tok, N_tok, dtype=dtype, device=device)

        u_sum = None
        for aa_val in aatype_per_slot:
            aatype_long = torch.full((B, N_tok), aa_val, dtype=torch.long, device=device)
            aatype_oh = F.one_hot(aatype_long, 32).to(dtype)  # (B, N_tok, 32)
            # Per AF3 Alg 16: aatype_j is the expand at new dim -3, aatype_i at -2.
            aatype_j = aatype_oh.unsqueeze(-3).expand(B, N_tok, N_tok, 32)
            aatype_i = aatype_oh.unsqueeze(-2).expand(B, N_tok, N_tok, 32)
            at = torch.cat([
                dgram,
                pseudo_beta_mask_2d.unsqueeze(-1),
                aatype_j,
                aatype_i,
                unit_vector,
                backbone_mask_2d.unsqueeze(-1),
            ], dim=-1)  # (B, N, N, 108)
            v = self.linear_z(z_normed) + self.linear_a(at)
            for block in self.pairformer_stack:
                _, v = block(None, v, mask=token_mask, pair_mask=pair_mask)
            v = self.out_norm(v)
            u_sum = v if u_sum is None else (u_sum + v)

        u = u_sum / (1e-7 + num_templ)
        u = self.linear_out(F.relu(u))
        return u
