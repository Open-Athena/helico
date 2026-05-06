"""Diffusion module — AF3 SI §3.7 (Algorithms 18-26).

Replaces AF2's Structure Module with a non-equivariant point-cloud
diffusion model over all heavy atoms. During training, Gaussian noise
is added to ground-truth atom positions at a random sigma sampled from
a log-normal distribution; the network learns to denoise conditioned
on the trunk's s and z. At inference, an EDM-style predictor-corrector
sampler walks σ from ~2560 Å (essentially random positions) down to 0.

File contents, in order:

- DiffusionAttentionPairBias (Alg 24)
- DiffusionTransformerBlock (attention + conditioned transition)
- DiffusionTransformer (Alg 23)
- Sequence-local window partition helpers (Alg 7's logic, inlined)
- AtomAttentionEncoder (Alg 5) — atoms → tokens
- AtomAttentionDecoder (Alg 6) — tokens → atom position updates
- DiffusionConditioning (Alg 21) — trunk s,z + σ → single/pair cond
- _centre_random_augmentation (Alg 19) — COM-remove + rot + trans
- DiffusionModule (Alg 20) — the whole stack, with forward_training
  and sample() methods.

Inference uses the EDM preconditioning + power-law σ schedule from SI
Eq. 7:  σ(t) = σ_data · (s_max^(1/ρ) + t·(s_min^(1/ρ) − s_max^(1/ρ)))^ρ
with s_max=160, s_min=4e-4, ρ=7, σ_data=16; gamma-injection of extra
noise above σ_min=1 per AF3's specific Karras-sampler variant.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .blocks import (
    LayerNorm,
    AdaptiveLayerNorm,
    linear_no_bias,
    BiasInitLinear,
    FourierEmbedding,
    ConditionedTransitionBlock,
    Transition,
)
from .pairformer import RelativePositionEncoding


# ---------------------------------------------------------------------------
# Attention + transformer blocks
# ---------------------------------------------------------------------------

class DiffusionAttentionPairBias(nn.Module):
    """AF3 SI Algorithm 24 — AttentionPairBias with adaLN-Zero.

    AdaLN-conditioned self/cross attention with pair bias. Two modes:
      - Global (n_queries=None): full N×N attention, z bias is (B, N, N, d_z).
      - Windowed (n_queries set): sequence-local attention on
        overlapping blocks used by the atom-attention encoder/decoder.
        z is precomputed as (B, n_blocks, n_q, n_k, d_z).

    Output is element-wise sigmoid-gated by ``s_gate`` (bias_init=-2 → ≈0.12)
    which implements adaLN-Zero's weak initial contribution.
    """

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
        """a: (B, N, d_a) query, s: (B, N, d_s) cond, z: pair bias input.

        Windowed path (n_queries set): z is (B, n_blocks, n_q, n_k, d_z);
        queries are partitioned into non-overlapping blocks of size n_q,
        keys into centered overlapping windows of size n_k.
        """
        B, N, _ = a.shape
        H, dh = self.n_heads, self.head_dim

        q_in = self.ada_ln_q(a, s)
        if self.cross_attention_mode and kv_a is not None:
            kv_in = self.ada_ln_kv(kv_a, kv_s if kv_s is not None else s)
        elif self.cross_attention_mode:
            # AdaLN-Zero self-cross: reuse normalized q as kv
            kv_in = self.ada_ln_kv(q_in, s)
        else:
            kv_in = q_in

        q = self.q_proj(q_in).reshape(B, N, H, dh).permute(0, 2, 1, 3)
        k = self.k_proj(kv_in).reshape(B, N, H, dh).permute(0, 2, 1, 3)
        v = self.v_proj(kv_in).reshape(B, N, H, dh).permute(0, 2, 1, 3)
        g = self.g_proj(q_in).reshape(B, N, H, dh).permute(0, 2, 1, 3)

        if n_queries is None:
            # Global attention
            bias = self.z_proj(self.z_norm(z)).permute(0, 3, 1, 2)  # (B, H, N, N)
            attn = (q @ k.transpose(-2, -1)) * self.scale + bias
            attn = F.softmax(attn, dim=-1)
            out = attn @ v
            out = torch.sigmoid(g) * out
            out = out.permute(0, 2, 1, 3).reshape(B, N, H * dh)
        else:
            # Windowed (sequence-local) attention — AF3 SI Alg 7 body
            n_blocks = (N + n_queries - 1) // n_queries
            q_pad = n_blocks * n_queries - N
            pad_left = (n_keys - n_queries) // 2

            bias = self.z_proj(self.z_norm(z)).permute(0, 4, 1, 2, 3)  # (B, H, n_bl, n_q, n_k)

            def _partition_q(t):
                t_padded = F.pad(t, (0, 0, 0, q_pad))
                return t_padded.reshape(B, H, n_blocks, n_queries, dh)

            def _partition_k(t):
                t_padded = F.pad(t, (0, 0, 0, q_pad))
                t_for_keys = F.pad(t_padded, (0, 0, pad_left, n_keys - n_queries - pad_left))
                return t_for_keys.unfold(2, n_keys, n_queries).permute(0, 1, 2, 4, 3)

            q_w = _partition_q(q)
            g_w = _partition_q(g)
            k_w = _partition_k(k)
            v_w = _partition_k(v)

            attn = (q_w @ k_w.transpose(-2, -1)) * self.scale + bias
            if pad_mask is not None:
                attn = attn.masked_fill(~pad_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = F.softmax(attn, dim=-1)
            # Rows fully masked → NaN from softmax; zero them.
            attn = attn.nan_to_num(0.0)
            out_w = attn @ v_w
            out = torch.sigmoid(g_w) * out_w
            out = out.reshape(B, H, n_blocks * n_queries, dh)[:, :, :N]
            out = out.permute(0, 2, 1, 3).reshape(B, N, H * dh)

        # adaLN-Zero output gate (sigmoid(linear(s) with bias=-2) ≈ 0.12 at init)
        return torch.sigmoid(self.s_gate(s)) * self.out_proj(out)


class DiffusionTransformerBlock(nn.Module):
    """One block of AF3 SI Algorithm 23: AttentionPairBias + ConditionedTransition."""

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
    """AF3 SI Algorithm 23 — stack of DiffusionTransformerBlocks.

    Used at token level (Alg 20 line 5, n_blocks=24) and at atom level
    inside the AtomAttentionEncoder/Decoder (Alg 7, n_blocks=3).
    """

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
                a = grad_checkpoint(
                    block, a, s, z, kv_a, kv_s, n_queries, n_keys, pad_mask,
                    use_reentrant=False,
                )
            else:
                a = block(a, s, z, kv_a=kv_a, kv_s=kv_s,
                          n_queries=n_queries, n_keys=n_keys, pad_mask=pad_mask)
        return a


# ---------------------------------------------------------------------------
# Sequence-local windowing helpers (AF3 §3.2 Alg 7)
# ---------------------------------------------------------------------------

def _partition_to_windows(
    x: torch.Tensor, n_queries: int, n_keys: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Partition a flat atom tensor into overlapping query/key windows.

    Query windows are non-overlapping blocks of size ``n_queries``; key
    windows are centered on each query block with overlap ``n_keys > n_queries``.
    This is AF3's ``sequence-local atom attention`` pattern (SI §3.2):
    atoms within a window attend to ~n_keys neighbours (not the full
    N_atom sequence), keeping the attention O(N_atom · n_keys) instead
    of O(N_atom²).

    Returns (x_q, x_k, pad_mask, n_blocks, q_pad).
    """
    B, N, D = x.shape
    n_blocks = (N + n_queries - 1) // n_queries
    q_pad = n_blocks * n_queries - N

    x_padded = F.pad(x, (0, 0, 0, q_pad))

    # Queries: non-overlapping blocks
    x_q = x_padded.reshape(B, n_blocks, n_queries, D)

    # Keys: centered overlapping windows
    pad_left = (n_keys - n_queries) // 2
    pad_right = n_keys - n_queries - pad_left
    x_for_keys = F.pad(x_padded, (0, 0, pad_left, pad_right))
    x_k = x_for_keys.unfold(1, n_keys, n_queries)
    x_k = x_k.permute(0, 1, 3, 2)  # (B, n_blocks, n_keys, D)

    # Validity mask: (n_blocks, n_queries, n_keys) True where both ends are in range [0, N)
    q_pos = torch.arange(n_blocks * n_queries, device=x.device).reshape(n_blocks, n_queries)
    q_valid = q_pos < N
    block_starts = torch.arange(n_blocks, device=x.device) * n_queries
    k_offsets = torch.arange(n_keys, device=x.device) - pad_left
    k_pos = block_starts.unsqueeze(1) + k_offsets.unsqueeze(0)
    k_valid = (k_pos >= 0) & (k_pos < N)

    pad_mask = q_valid.unsqueeze(2) & k_valid.unsqueeze(1)
    return x_q, x_k, pad_mask, n_blocks, q_pad


def _unpartition_from_windows(x_q: torch.Tensor, n_orig: int) -> torch.Tensor:
    """Reshape (B, n_blocks, n_queries, D) → (B, n_orig, D) (inverse of partition)."""
    B, n_blocks, n_queries, D = x_q.shape
    return x_q.reshape(B, n_blocks * n_queries, D)[:, :n_orig]


# ---------------------------------------------------------------------------
# Atom-level encoder / decoder
# ---------------------------------------------------------------------------

class AtomAttentionEncoder(nn.Module):
    """AF3 SI Algorithm 5 — encode atoms into token-level activations.

    Used twice:
      - ``has_coords=False`` (input embedding, Alg 2): encode the reference
        conformer to produce per-token ``s_inputs``. Output agg dim = c_token=384.
      - ``has_coords=True`` (diffusion encoder, Alg 20 line 3): also
        consume the noisy coords ``r_l^noisy`` and inject trunk s/z. Output
        agg dim = c_token=768.

    Uses sequence-local (windowed) atom attention: each query atom
    attends only to its n_keys neighbours, with a pair-feature bias
    masked to within-token (same ref_space_uid) edges only. This keeps
    the atom-axis compute O(N_atom · n_keys) ≈ O(N_atom) even at 20k
    atoms.
    """

    def __init__(self, config, has_coords: bool = True,
                 c_token_override: int | None = None):
        super().__init__()
        self.config = config
        c = config
        c_atom = c.c_atom
        c_atompair = c.c_atompair
        c_s = c.d_single
        c_z = c.d_pair
        c_token = c_token_override if c_token_override is not None else c.c_token
        self.has_coords = has_coords
        self.n_queries = c.n_atom_queries
        self.n_keys = c.n_atom_keys

        # Reference feature projections (Alg 5 line 1)
        self.ref_pos_proj = linear_no_bias(3, c_atom)
        self.ref_charge_proj = linear_no_bias(1, c_atom)
        # 1 (ref_mask) + 128 (ref_element one-hot) + 256 (ref_atom_name_chars: 4×64 one-hot)
        self.n_ref_feat = 1 + 128 + 256
        self.ref_feat_proj = linear_no_bias(self.n_ref_feat, c_atom)

        if has_coords:
            # Noisy coords projection
            self.noisy_pos_proj = linear_no_bias(3, c_atom)

            # Trunk s,z injection (zero-init → no-op at start of training).
            # NOTE: in DiffusionModule, this z_trunk arg is actually z_cond
            # (DiffusionConditioning output, always c_z), not the raw trunk
            # pair tensor. So gh#9's distogram swap only needs to happen
            # inside DiffusionConditioning — by the time z reaches the atom
            # encoder it's already been projected back to c_z.
            self.trunk_s_norm = LayerNorm(c_s)
            self.trunk_s_proj = linear_no_bias(c_s, c_atom, zeros_init=True)
            self.trunk_z_norm = LayerNorm(c_z)
            self.trunk_z_proj = linear_no_bias(c_z, c_atompair, zeros_init=True)

        # Atom-pair projections
        self.pair_dist_proj = linear_no_bias(3, c_atompair)
        self.pair_inv_dist_proj = linear_no_bias(1, c_atompair)
        self.pair_valid_proj = linear_no_bias(1, c_atompair)

        # Cross-pair (atom_i, atom_j) features from atom embeddings
        self.cross_pair_q = linear_no_bias(c_atom, c_atompair)
        self.cross_pair_k = linear_no_bias(c_atom, c_atompair)

        # Pair MLP (3 ReLU-Linear layers, last zero-init)
        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            linear_no_bias(c_atompair, c_atompair),
            nn.ReLU(),
            linear_no_bias(c_atompair, c_atompair),
            nn.ReLU(),
            linear_no_bias(c_atompair, c_atompair, zeros_init=True),
        )

        # Atom-level transformer (AF3 Alg 7), cross-attention mode, windowed
        self.atom_transformer = DiffusionTransformer(
            n_blocks=c.n_atom_encoder_blocks,
            d_a=c_atom, d_s=c_atom, d_z=c_atompair,
            n_heads=c.n_heads_atom, head_dim=c.atom_head_dim,
            cross_attention_mode=True,
            gradient_checkpointing=c.gradient_checkpointing,
        )

        # Aggregation: ReLU → linear → mean over atoms of each token
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
        ref_space_uid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (a_token, q_skip, c_skip, p_skip, pad_mask).

        The skip connections are fed to the AtomAttentionDecoder (Alg 6).
        """
        B, N_atom, _ = ref_pos.shape
        n_q, n_k = self.n_queries, self.n_keys

        # 1. Build c_l from reference conformer features
        c_l = (
            self.ref_pos_proj(ref_pos)
            + self.ref_charge_proj(ref_charge)
            + self.ref_feat_proj(ref_features)
        )
        c_l = c_l * atom_mask.unsqueeze(-1)

        # 2. Trunk single injection (only at diffusion-encoder use)
        if self.has_coords and s_trunk is not None:
            s_trunk_atom = self._broadcast_token_to_atom(
                self.trunk_s_proj(self.trunk_s_norm(s_trunk)), atom_to_token,
            )
            c_l = c_l + s_trunk_atom

        # 3. Build q_l (adds noisy coords at diffusion time)
        if self.has_coords and noisy_pos is not None:
            q_l = c_l + self.noisy_pos_proj(noisy_pos)
        else:
            q_l = c_l

        # 4. Windowed atom-pair features built from REFERENCE positions
        ref_pos_q, ref_pos_k, pad_mask, n_blocks, q_pad = _partition_to_windows(
            ref_pos, n_q, n_k,
        )
        diff = ref_pos_q.unsqueeze(3) - ref_pos_k.unsqueeze(2)  # (B, bl, q, k, 3)
        dist_sq = diff.pow(2).sum(-1, keepdim=True)
        inv_dist = 1.0 / (1.0 + dist_sq)

        # Within-residue mask: atoms of the same ref_space_uid can see each other's
        # pair geometry; across residues, geometric features are masked (just the
        # validity-bit projection survives) so the model doesn't see stale
        # cross-residue reference geometry.
        uid = ref_space_uid if ref_space_uid is not None else atom_to_token
        uid_padded = F.pad(uid, (0, q_pad), value=-1)
        uid_q, uid_k, _, _, _ = _partition_to_windows(
            uid_padded.unsqueeze(-1).float(), n_q, n_k,
        )
        v_lm = (uid_q.squeeze(-1).long().unsqueeze(3) == uid_k.squeeze(-1).long().unsqueeze(2))
        v_lm = v_lm.unsqueeze(-1).to(diff.dtype)

        p = self.pair_dist_proj(diff) * v_lm
        p = p + self.pair_inv_dist_proj(inv_dist) * v_lm
        p = p + self.pair_valid_proj(v_lm)
        p = p * pad_mask.unsqueeze(0).unsqueeze(-1).to(diff.dtype)

        # 5. Trunk pair injection (windowed gather of token-pair z into atom-pair).
        # ``z_trunk`` here is z_cond (post DiffusionConditioning, always c_z
        # channels) — see note in __init__.
        if self.has_coords and z_trunk is not None:
            z_trunk_proj = self.trunk_z_proj(self.trunk_z_norm(z_trunk))
            z_windowed = self._gather_trunk_pair_windowed(
                z_trunk_proj, atom_to_token, n_blocks, q_pad,
            )
            p = p + z_windowed

        # 6. Cross-pair features from atom embeddings
        c_l_q_w, c_l_k_w, _, _, _ = _partition_to_windows(c_l, n_q, n_k)
        p = p + self.cross_pair_q(F.relu(c_l_q_w)).unsqueeze(3) \
              + self.cross_pair_k(F.relu(c_l_k_w)).unsqueeze(2)

        # 7. Pair MLP (residual)
        p = p + self.pair_mlp(p)

        # Save skip connections for the decoder
        c_skip = c_l
        p_skip = p

        # 8. Atom-level transformer (cross-attention, windowed)
        q_l = self.atom_transformer(q_l, c_l, p,
                                    n_queries=n_q, n_keys=n_k, pad_mask=pad_mask)
        q_skip = q_l

        # 9. Aggregate atoms to tokens (AF3 Alg 5 line 14-ish)
        a_token = self._aggregate_to_tokens(
            F.relu(self.agg_proj(q_l)), atom_to_token, atom_mask, n_tokens,
        )
        return a_token, q_skip, c_skip, p_skip, pad_mask

    def _broadcast_token_to_atom(self, token_feat: torch.Tensor,
                                  atom_to_token: torch.Tensor) -> torch.Tensor:
        idx = atom_to_token.unsqueeze(-1).expand(-1, -1, token_feat.shape[-1])
        return torch.gather(token_feat, 1, idx)

    def _gather_trunk_pair_windowed(self, z_trunk_proj: torch.Tensor,
                                     atom_to_token: torch.Tensor,
                                     n_blocks: int, q_pad: int) -> torch.Tensor:
        """Fetch z[b, tok(q_atom), tok(k_atom), :] for each windowed (q, k) atom pair."""
        B = atom_to_token.shape[0]
        n_q, n_k = self.n_queries, self.n_keys
        pad_left = (n_k - n_q) // 2

        a2t_padded = F.pad(atom_to_token, (0, q_pad), value=0)
        tok_q = a2t_padded.reshape(B, n_blocks, n_q)
        a2t_for_keys = F.pad(a2t_padded, (pad_left, n_k - n_q - pad_left), value=0)
        tok_k = a2t_for_keys.unfold(1, n_k, n_q)

        b_idx = torch.arange(B, device=z_trunk_proj.device).view(B, 1, 1, 1)
        result = z_trunk_proj[b_idx, tok_q.unsqueeze(3), tok_k.unsqueeze(2)]
        return result

    def _aggregate_to_tokens(self, atom_feat: torch.Tensor,
                              atom_to_token: torch.Tensor,
                              atom_mask: torch.Tensor, n_tokens: int) -> torch.Tensor:
        B, N_atom, D = atom_feat.shape
        device = atom_feat.device
        dt = atom_feat.dtype
        masked = atom_feat * atom_mask.unsqueeze(-1).to(dt)
        token_sum = torch.zeros(B, n_tokens, D, device=device, dtype=dt)
        token_count = torch.zeros(B, n_tokens, 1, device=device, dtype=dt)
        idx = atom_to_token.unsqueeze(-1).expand(-1, -1, D)
        token_sum.scatter_add_(1, idx, masked)
        token_count.scatter_add_(1, atom_to_token.unsqueeze(-1),
                                  atom_mask.unsqueeze(-1).to(dt))
        return token_sum / token_count.clamp(min=1)


class AtomAttentionDecoder(nn.Module):
    """AF3 SI Algorithm 6 — decode token activations back to atom position updates.

    Broadcasts token rep a_i back to atoms, adds the encoder's q_skip,
    runs another windowed atom-transformer, and projects to 3D updates.
    Output r_update is in float32 regardless of autocast dtype so the
    EDM combination (c_skip·x + c_out·r) is precise.
    """

    def __init__(self, config):
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
        projected = self.token_to_atom_proj(a_token)
        idx = atom_to_token.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
        q = torch.gather(projected, 1, idx) + q_skip

        q = self.atom_transformer(q, c_skip, p_skip,
                                  n_queries=self.n_queries, n_keys=self.n_keys,
                                  pad_mask=pad_mask)
        return self.out_proj(self.out_norm(q)).float()


# ---------------------------------------------------------------------------
# Diffusion conditioning (Alg 21)
# ---------------------------------------------------------------------------

class DiffusionConditioning(nn.Module):
    """AF3 SI Algorithm 21 — DiffusionConditioning.

    Takes the trunk outputs (s_trunk, z_trunk) plus the input embedding
    s_inputs and the current noise level σ, and produces (s_cond, z_cond)
    that will condition the denoiser. Pair conditioning is s_trunk's
    relpe features mixed in; single conditioning adds a Fourier embedding
    of log(σ/σ_data)/4.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        c_s = c.d_single
        c_z = c.d_pair

        # Pair path
        self.relpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=c_z)
        self.pair_norm = nn.LayerNorm(2 * c_z)
        self.pair_proj = linear_no_bias(2 * c_z, c_z)
        self.pair_transition_1 = Transition(c_z, factor=2)
        self.pair_transition_2 = Transition(c_z, factor=2)

        # gh#9: parallel pair input for diffusion_pair_source="distogram_logits".
        # Input is concat(distogram_logits, relpe) — c.n_distogram_bins + c_z.
        # Always present so checkpoints from "z" mode round-trip; only
        # active when config.diffusion_pair_source != "z".
        self.pair_norm_dist = nn.LayerNorm(c.n_distogram_bins + c_z)
        self.pair_proj_dist = linear_no_bias(c.n_distogram_bins + c_z, c_z)

        # Single path
        self.fourier = FourierEmbedding(c.c_noise_embedding)
        self.s_inputs_dim = c_s + 65  # s_inputs dim = c_s (from atom encoder) + 65
        self.single_norm = nn.LayerNorm(c_s + self.s_inputs_dim)
        self.single_proj = linear_no_bias(c_s + self.s_inputs_dim, c_s)
        self.noise_norm = nn.LayerNorm(c.c_noise_embedding)
        self.noise_proj = linear_no_bias(c.c_noise_embedding, c_s)
        self.single_transition_1 = Transition(c_s, factor=2)
        self.single_transition_2 = Transition(c_s, factor=2)

    def forward(self, s_trunk: torch.Tensor, z_trunk: torch.Tensor,
                s_inputs: torch.Tensor, sigma: torch.Tensor,
                relpe_feats: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """s_trunk: (B, N, c_s). z_trunk: (B, N, N, c_z). sigma: (B,).

        Returns (s_cond, z_cond) of the same shapes.
        """
        sigma_data = 16.0  # EDM constant (σ_data)

        # Pair conditioning: concat(z_trunk, relpe) → norm → linear → 2x Transition.
        # gh#9: in "distogram_logits" mode, z_trunk is actually the distogram
        # logits (B, N, N, n_distogram_bins) and we use the parallel
        # pair_proj_dist sized for that channel count.
        relpe = self.relpe(**relpe_feats)
        z_in = torch.cat([z_trunk, relpe], dim=-1)
        if self.config.diffusion_pair_source == "distogram_logits":
            z = self.pair_proj_dist(self.pair_norm_dist(z_in))
        else:
            z = self.pair_proj(self.pair_norm(z_in))
        z = z + self.pair_transition_1(z)
        z = z + self.pair_transition_2(z)

        # Noise embedding: log(σ/σ_data)/4 → FourierEmbedding → norm → proj
        noise_input = torch.log(sigma / sigma_data) / 4.0
        n = self.noise_proj(self.noise_norm(self.fourier(noise_input)))

        # Single conditioning: concat(s_trunk, s_inputs) → norm → proj → + noise → 2x Transition
        s = self.single_proj(self.single_norm(torch.cat([s_trunk, s_inputs], dim=-1)))
        s = s + n.unsqueeze(1)
        s = s + self.single_transition_1(s)
        s = s + self.single_transition_2(s)

        return s, z


# ---------------------------------------------------------------------------
# Augmentation (Alg 19)
# ---------------------------------------------------------------------------

def _uniform_random_rotation(B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Uniform SO(3) random rotation via normalized random quaternion.

    Deterministic given ``torch.manual_seed``. Used by Alg 19 to enforce
    rotation equivariance of the predicted structure during training (and
    as part of the sampler's centring step at inference).
    """
    q = torch.randn(B, 4, device=device, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    rot = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),    2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),    1 - 2*(x*x + y*y),
    ], dim=-1).reshape(B, 3, 3)
    return rot.to(dtype)


def _centre_random_augmentation(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    s_trans: float = 1.0,
) -> torch.Tensor:
    """AF3 SI Algorithm 19 — CentreRandomAugmentation.

    Remove centre of mass, rotate by a uniform-random SO(3) rotation,
    translate by N(0, s_trans). Applied inside the sampler at each step
    so the denoiser sees an arbitrary pose.
    """
    B, N_atoms, _ = x.shape
    device = x.device
    dtype = x.dtype

    # Masked centroid
    if mask is not None and mask.any():
        m = mask.unsqueeze(-1).to(dtype)
        center = (x * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True).clamp(min=1e-12)
    else:
        center = x.mean(dim=-2, keepdim=True)
    x = x - center

    rot = _uniform_random_rotation(B, device, dtype)
    x = torch.einsum("bij,baj->bai", rot, x)

    trans = s_trans * torch.randn(B, 1, 3, device=device, dtype=dtype)
    x = x + trans

    if mask is not None:
        x = x * mask.unsqueeze(-1).to(dtype)
    return x


# ---------------------------------------------------------------------------
# Diffusion module (Alg 20 + sampling loop)
# ---------------------------------------------------------------------------

class DiffusionModule(nn.Module):
    """AF3 SI Algorithm 20 — DiffusionModule.

    One forward of the conditional denoiser: scale x → atom encoder → token
    transformer → atom decoder → combine with x to produce x^out. At
    training we use ``forward_training`` (adds noise, runs a single
    denoising step). At inference, ``sample`` runs the full sampling
    schedule (AF3 SI Eq. 7) using the EDM-style predictor-corrector with
    gamma noise injection.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.conditioning = DiffusionConditioning(config)
        self.atom_encoder = AtomAttentionEncoder(config)
        self.atom_decoder = AtomAttentionDecoder(config)

        # Token-level additive from the conditioned single rep (Alg 20 line 4)
        self.s_to_token_norm = nn.LayerNorm(c.d_single)
        self.s_to_token_proj = linear_no_bias(c.d_single, c.c_token, zeros_init=True)

        # Token-level transformer (Alg 20 line 5)
        self.token_transformer = DiffusionTransformer(
            n_blocks=c.n_diffusion_token_blocks,
            d_a=c.c_token, d_s=c.d_single, d_z=c.d_pair,
            n_heads=c.n_heads_diffusion_token, head_dim=c.diffusion_token_head_dim,
            cross_attention_mode=False,
            gradient_checkpointing=c.gradient_checkpointing,
        )
        self.out_norm = nn.LayerNorm(c.c_token)

        self.n_steps = c.n_diffusion_steps

    def _edm_precondition(self, sigma: torch.Tensor):
        """EDM preconditioning (c_in, c_skip, c_out) given σ.

        Karras et al. 2022, applied per AF3 Alg 20 line 8. c_in scales
        inputs to unit variance; c_skip, c_out blend raw x with network
        update for the final prediction.
        """
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
                   relpe_feats: dict[str, torch.Tensor],
                   ref_space_uid: torch.Tensor | None = None) -> torch.Tensor:
        """One denoiser forward (Alg 20 lines 1-7)."""
        n_tokens = s_trunk.shape[1]

        s_cond, z_cond = self.conditioning(s_trunk, z_trunk, s_inputs, sigma, relpe_feats)

        # Atom encoder: uses raw s_trunk for per-atom single injection but
        # the conditioned z_cond for pair injection (matches AF3 Alg 20 line 3).
        a_token, q_skip, c_skip, p_skip, pad_mask = self.atom_encoder(
            ref_pos, ref_charge, ref_features,
            atom_to_token, atom_mask, n_tokens,
            noisy_pos=x_scaled, s_trunk=s_trunk, z_trunk=z_cond,
            ref_space_uid=ref_space_uid,
        )

        # Token-level additive conditioning (Alg 20 line 4)
        a_token = a_token + self.s_to_token_proj(self.s_to_token_norm(s_cond))

        # Token-level transformer (Alg 20 line 5) + LayerNorm
        a_token = self.token_transformer(a_token, s_cond, z_cond)
        a_token = self.out_norm(a_token)

        # Atom decoder → per-atom 3D update (Alg 20 line 7)
        return self.atom_decoder(a_token, atom_to_token, q_skip, c_skip, p_skip, pad_mask)

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
        ref_space_uid: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward — add noise + one denoising pass.

        Amortizes the expensive trunk forward by running ``n_samples``
        denoising passes (with independent σ and noise) per trunk forward
        (AF3 SI §3.7.1 / "Main Article Fig. 2c"). Per-row inputs are
        repeat_interleaved to B*n_samples; gt_coords, sigma, and x_denoised
        return at that expanded shape.
        """
        B = gt_coords.shape[0]
        N_d = max(1, int(n_samples))
        device = gt_coords.device

        if N_d > 1:
            gt_coords = gt_coords.repeat_interleave(N_d, dim=0)
            ref_pos = ref_pos.repeat_interleave(N_d, dim=0)
            ref_charge = ref_charge.repeat_interleave(N_d, dim=0)
            ref_features = ref_features.repeat_interleave(N_d, dim=0)
            atom_to_token = atom_to_token.repeat_interleave(N_d, dim=0)
            atom_mask = atom_mask.repeat_interleave(N_d, dim=0)
            s_trunk = s_trunk.repeat_interleave(N_d, dim=0)
            z_trunk = z_trunk.repeat_interleave(N_d, dim=0)
            s_inputs = s_inputs.repeat_interleave(N_d, dim=0)
            relpe_feats = {
                k: v.repeat_interleave(N_d, dim=0) if isinstance(v, torch.Tensor) else v
                for k, v in relpe_feats.items()
            }
            if ref_space_uid is not None:
                ref_space_uid = ref_space_uid.repeat_interleave(N_d, dim=0)

        B_eff = B * N_d

        # Log-normal σ sampling — AF3 SI §3.7.1 "During training the noise
        # level is sampled from σ_data · exp(-1.2 + 1.5·N(0,1))"
        log_sigma = self.config.noise_log_mean + self.config.noise_log_std * torch.randn(B_eff, device=device)
        sigma = torch.exp(log_sigma) * self.config.sigma_data

        noise = torch.randn_like(gt_coords)
        sigma_expand = sigma.view(B_eff, 1, 1)
        x_noisy = gt_coords + sigma_expand * noise

        c_in, c_skip, c_out = self._edm_precondition(sigma)
        c_in = c_in.view(B_eff, 1, 1)
        c_skip = c_skip.view(B_eff, 1, 1)
        c_out = c_out.view(B_eff, 1, 1)

        r_update = self._f_forward(
            c_in * x_noisy, sigma,
            ref_pos, ref_charge, ref_features,
            atom_to_token, atom_mask,
            s_trunk, z_trunk, s_inputs,
            relpe_feats,
            ref_space_uid=ref_space_uid,
        )

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
        ref_space_uid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference sampler — EDM predictor + gamma noise injection.

        Implements AF3 SI Algorithm 18 (SampleDiffusion) with the power-law
        σ schedule from Eq. 7. At each step:
          1. CentreRandomAugmentation (Alg 19)
          2. Gamma noise injection: if σ > σ_min, raise σ to σ·(1+γ) and
             add √(σ_new² - σ²)·λ·N(0,I)
          3. Denoiser forward at σ_new
          4. Euler step with step scale η
        """
        B, N_atoms, _ = ref_pos.shape
        device = ref_pos.device
        dtype = ref_pos.dtype

        sigma_data = self.config.sigma_data   # 16.0
        s_max, s_min = 160.0, 4e-4            # AF3 SI Eq. 7
        rho = 7.0
        gamma0 = 0.8
        gamma_min = 1.0                       # gamma only fires when σ > gamma_min
        noise_scale_lambda = 1.003
        step_scale_eta = 1.5

        # Power-law σ schedule (AF3 SI Eq. 7)
        N = self.n_steps
        step_indices = torch.arange(N + 1, device=device, dtype=torch.float64)
        t_steps = (
            sigma_data
            * (s_max ** (1 / rho)
               + step_indices / N * (s_min ** (1 / rho) - s_max ** (1 / rho))
              ) ** rho
        )
        t_steps[-1] = 0.0
        t_steps = t_steps.to(dtype)

        # Initial pure noise at σ_0 = σ_data · s_max ≈ 2560
        x = torch.randn(B, N_atoms, 3, device=device, dtype=dtype) * t_steps[0]

        for i in range(N):
            c_tau_last = t_steps[i]
            c_tau = t_steps[i + 1]

            # (1) Centre + random rotation + random translation
            x = _centre_random_augmentation(x, atom_mask)

            # (2) Gamma noise injection
            gamma = gamma0 if float(c_tau) > gamma_min else 0.0
            t_hat = c_tau_last * (gamma + 1)
            if gamma > 0:
                delta_noise = (t_hat ** 2 - c_tau_last ** 2).sqrt()
                x = x + noise_scale_lambda * delta_noise * torch.randn_like(x)

            # (3) Denoiser forward at σ = t_hat
            sigma_cur = t_hat.expand(B)
            c_in, c_skip, c_out = self._edm_precondition(sigma_cur)
            c_in = c_in.view(B, 1, 1)
            c_skip = c_skip.view(B, 1, 1)
            c_out = c_out.view(B, 1, 1)

            r_update = self._f_forward(
                c_in * x, sigma_cur,
                ref_pos, ref_charge, ref_features,
                atom_to_token, atom_mask,
                s_trunk, z_trunk, s_inputs,
                relpe_feats,
                ref_space_uid=ref_space_uid,
            )
            x_denoised = c_skip * x + c_out * r_update

            # (4) Euler step with scaled step size
            d = (x - x_denoised) / t_hat
            dt = c_tau - t_hat
            x = x + step_scale_eta * dt * d

        return x
