"""MSA Module — AF3 SI §3.3 / Algorithm 8.

Updates the pair representation ``z`` using the multiple sequence
alignment ``m_raw`` (34 channels = 32 restype one-hot + has_deletion +
deletion_value). The MSA axis flows through the module via
pair-weighted averaging (MSAPairWeightedAveraging, Algorithm 10) and is
projected out to pair space via OuterProductMean (Algorithm 9). After
``n_msa_blocks`` blocks (AF3 default: 4), only ``z`` is returned — the
single representation is not updated from the MSA directly.

Two AF3-spec details:

1. **Per-call random row subsample** (§3.5): each forward pass takes a
   random number of MSA rows in ``[1, N_msa]`` and a random permutation
   of them. Since MSAModule is called once per recycle, each cycle sees
   a different subsample — a regularizer that prevents overfitting to
   specific co-evolution signals.

2. **MSA features are 34-dim** (SI §2.8 Table 5):
   32 restype one-hot + 1 has_deletion + 1 deletion_value. The ``34``
   hard-coded here matches those three features concatenated along the
   last axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .blocks import LayerNorm, Transition
from .pairformer import PairformerBlock


# ---------------------------------------------------------------------------
# Outer product mean (AF3 Algorithm 9)
# ---------------------------------------------------------------------------

class OuterProductMean(nn.Module):
    """AF3 SI Algorithm 9 — MSA → pair update via outer product mean.

    For each MSA row, project m → (a, b) of dim c_hidden, take the outer
    product a_i ⊗ b_j averaged across rows, flatten to c_hidden² and
    project to c_z. Intuitively: if two positions i, j covary in the
    MSA, their outer product signals that correlation and updates z_ij.

    Chunked-MSA implementation preserves math identity: the flattened
    outer product is linear, so summing chunk contributions and applying
    linear_out once equals the non-chunked path. Only linear_out carries
    a bias; that bias is applied to the accumulated sum, matching the
    un-chunked case exactly.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int = 32, eps: float = 1e-3):
        super().__init__()
        self.c_hidden = c_hidden
        self.eps = eps
        self.norm = LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden, bias=False)
        self.linear_2 = nn.Linear(c_m, c_hidden, bias=False)
        self.linear_out = nn.Linear(c_hidden * c_hidden, c_z, bias=True)

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """m: (*, N_msa, N_tok, c_m), mask: (*, N_msa, N_tok) or None.

        Returns: (*, N_tok, N_tok, c_z).
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        m = self.norm(m)
        mask = mask.unsqueeze(-1)  # (*, N_msa, N_tok, 1)

        N_msa = m.shape[-3]
        # Normalize by mask overlap count — same whether chunked or not.
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)  # (*, N_tok, N_tok, 1)

        if chunk_size is None or chunk_size >= N_msa:
            a = self.linear_1(m) * mask            # (*, N_msa, N_tok, C)
            b = self.linear_2(m) * mask
            a = a.transpose(-2, -3)                # (*, N_tok, N_msa, C)
            b = b.transpose(-2, -3)
            outer = torch.einsum("...bac,...dae->...bdce", a, b)  # (*, Ni, Nj, C, C)
            outer = outer.flatten(-2)
            outer = self.linear_out(outer)
            return outer / (norm + self.eps)

        # Chunked: accumulate flattened outer products BEFORE linear_out.
        # linear_out has a bias; summing first then applying linear_out once
        # is mathematically identical to the un-chunked version.
        *batch, _N_msa, N_tok, _c_m = m.shape
        C = self.c_hidden
        outer_acc = m.new_zeros(*batch, N_tok, N_tok, C * C)
        for start in range(0, N_msa, chunk_size):
            end = min(start + chunk_size, N_msa)
            m_chunk = m[..., start:end, :, :]
            mask_chunk = mask[..., start:end, :, :]
            a = self.linear_1(m_chunk) * mask_chunk
            b = self.linear_2(m_chunk) * mask_chunk
            a = a.transpose(-2, -3)
            b = b.transpose(-2, -3)
            outer_chunk = torch.einsum("...bac,...dae->...bdce", a, b)
            outer_chunk = outer_chunk.flatten(-2)
            outer_acc = outer_acc + outer_chunk
        outer = self.linear_out(outer_acc)
        return outer / (norm + self.eps)


# ---------------------------------------------------------------------------
# MSA pair-weighted averaging (AF3 Algorithm 10)
# ---------------------------------------------------------------------------

class MSAPairWeightedAveraging(nn.Module):
    """AF3 SI Algorithm 10 — pair → MSA update via weighted averaging.

    Pair weights ``w_ij`` are derived from z via LayerNorm → linear → softmax
    over j, giving per-head attention weights independent of the MSA axis.
    Each MSA row's value ``v_mj`` is then averaged over j weighted by w,
    producing an update to the MSA row. Row-independent → safely chunkable.
    """

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

        # Zero-init for the gate and the output projection → block is a
        # no-op at init (residual in parent preserves identity).
        nn.init.zeros_(self.linear_mg.weight)
        nn.init.zeros_(self.linear_out.weight)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """m: (*, N_msa, N_tok, c_m), z: (*, N_tok, N_tok, c_z). Returns (*, N_msa, N_tok, c_m)."""
        H, dh = self.n_heads, self.head_dim

        # Pair weights: independent of MSA axis — compute once.
        w = self.linear_z(self.layernorm_z(z))      # (*, N_tok, N_tok, H)
        w = F.softmax(w, dim=-2)

        N_msa = m.shape[-3]
        if chunk_size is None or chunk_size >= N_msa:
            m_norm = self.layernorm_m(m)
            v = self.linear_mv(m_norm).unflatten(-1, (H, dh))
            g = torch.sigmoid(self.linear_mg(m_norm)).unflatten(-1, (H, dh))
            o = torch.einsum("...ijh,...mjhc->...mihc", w, v)
            o = (g * o).flatten(-2)
            return self.linear_out(o)

        chunks = []
        for start in range(0, N_msa, chunk_size):
            end = min(start + chunk_size, N_msa)
            m_c = m[..., start:end, :, :]
            m_norm = self.layernorm_m(m_c)
            v = self.linear_mv(m_norm).unflatten(-1, (H, dh))
            g = torch.sigmoid(self.linear_mg(m_norm)).unflatten(-1, (H, dh))
            o = torch.einsum("...ijh,...mjhc->...mihc", w, v)
            o = (g * o).flatten(-2)
            chunks.append(self.linear_out(o))
        return torch.cat(chunks, dim=-3)


# ---------------------------------------------------------------------------
# Composite blocks
# ---------------------------------------------------------------------------

class MSAStack(nn.Module):
    """MSA pair-weighted averaging + transition, both residual.

    This is the "update m" side of AF3's MSA block. AF3 SI Algorithm 8
    runs this BEFORE the z-side (OPM + pair triangle ops) — we keep that
    ordering in ``MSABlock`` below.
    """

    def __init__(self, c_m: int, c_z: int, n_heads: int = 8, head_dim: int = 8):
        super().__init__()
        self.pair_avg = MSAPairWeightedAveraging(c_m, c_z, n_heads, head_dim)
        self.transition = Transition(c_m, factor=4)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        N_msa = m.shape[-3]
        if chunk_size is None or chunk_size >= N_msa:
            m = m + self.pair_avg(m, z)
            m = m + self.transition(m)
            return m

        # Both pair_avg and transition are per-row-independent (z is shared).
        out_chunks = []
        for start in range(0, N_msa, chunk_size):
            end = min(start + chunk_size, N_msa)
            m_chunk = m[..., start:end, :, :]
            m_chunk = m_chunk + self.pair_avg(m_chunk, z)
            m_chunk = m_chunk + self.transition(m_chunk)
            out_chunks.append(m_chunk)
        return torch.cat(out_chunks, dim=-3)


class MSABlock(nn.Module):
    """One MSA block: OPM(z +=) → MSAStack(m :=) → PairformerBlock-pair-only(z +=).

    AF3 SI Algorithm 8 inner loop, n_msa_blocks=4 by default. The last
    block skips the MSAStack (m is unused after the final block since we
    only return z).
    """

    def __init__(self, config, is_last_block: bool = False):
        super().__init__()
        c = config

        self.opm = OuterProductMean(c.d_msa, c.d_pair, c.c_msa_opm_hidden)
        self.pair_stack = PairformerBlock(config, has_single=False)

        self.has_msa_stack = not is_last_block
        if self.has_msa_stack:
            self.msa_stack = MSAStack(
                c.d_msa, c.d_pair, c.n_msa_pw_heads, c.msa_pw_head_dim,
            )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        msa_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (1) MSA → pair update via OPM
        z = z + self.opm(m, mask=msa_mask, chunk_size=msa_chunk_size)

        # (2) pair → MSA update + MSA transition (skipped in the last block)
        if self.has_msa_stack:
            m = self.msa_stack(m, z, chunk_size=msa_chunk_size)

        # (3) pair-only Pairformer block (triangle ops + pair transition)
        _, z = self.pair_stack(None, z, pair_mask=pair_mask)

        return m, z


# ---------------------------------------------------------------------------
# MSA Module
# ---------------------------------------------------------------------------

class MSAModule(nn.Module):
    """AF3 SI Algorithm 8 — MSA Module.

    Top-level MSA-axis → pair-axis transfer. Called once per recycle
    cycle of the main inference loop.

    Per AF3 SI §3.5 (and empirically required to match Protenix v1.0.0
    weights), the MSA is randomly subsampled to a random row count
    ``∈ [1, N_msa]`` every call — this is a strong regularizer during
    training and remains stochastic at inference (so running the model
    twice gives slightly different predictions unless seed is fixed).
    """

    def __init__(self, config):
        super().__init__()
        c = config
        self.config = config

        # 34 = 32 restype one-hot + 1 has_deletion + 1 deletion_value (SI Table 5)
        self.linear_m = nn.Linear(34, c.d_msa, bias=False)
        self.linear_s = nn.Linear(c.c_s_inputs, c.d_msa, bias=False)

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
        msa_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """m_raw: (B, N_msa, N_tok, 34), z: (B, N_tok, N_tok, c_z),
        s_inputs: (B, N_tok, c_s_inputs). Returns updated z only.

        ``msa_chunk_size`` chunks the MSA axis inside OPM/MSAStack — essential
        for deep MSAs (>4k rows) to avoid OOM. Mathematically identical to
        the non-chunked path.
        """
        # Random row subsample every call (AF3 SI §3.5)
        N_msa = m_raw.shape[-3]
        if N_msa > 1:
            device = m_raw.device
            sample_size = torch.randint(low=1, high=N_msa + 1, size=(1,), device=device).item()
            indices = torch.randperm(n=N_msa, device=device)[:sample_size]
            m_raw = m_raw.index_select(-3, indices)
            if msa_mask is not None:
                msa_mask = msa_mask.index_select(-3, indices)

        # Project MSA features + broadcast single input (AF3 SI Alg 8 line 2)
        m = self.linear_m(m_raw) + self.linear_s(s_inputs).unsqueeze(-3)

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                m, z = grad_checkpoint(
                    block, m, z, msa_mask, pair_mask, msa_chunk_size,
                    use_reentrant=False,
                )
            else:
                m, z = block(
                    m, z, msa_mask=msa_mask, pair_mask=pair_mask,
                    msa_chunk_size=msa_chunk_size,
                )

        return z
