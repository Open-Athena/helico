from __future__ import annotations

import math
import time
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
    # Training-only: number of diffusion noise samples per trunk forward.
    # AF3 / Protenix amortize the expensive trunk over N denoising passes.
    # gh#6. 1 = legacy behavior.
    n_diffusion_samples: int = 8

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

    @classmethod
    def protenix_v2(cls, **overrides) -> "HelicoConfig":
        """Config matching Protenix v2.0.0 (464M params).

        Protenix v2 is a width scale-up of v1: c_z 128→256, c_m 64→128, with
        ``hidden_scale_up=True`` which doubles triangle-mul inner width and bumps
        pair-attention heads (head_dim=32 preserved, so n_heads_pair 4→8).
        The MSA pair-weighted-averaging head count similarly doubles
        (n_msa_pw_heads 8→16, head_dim=8 preserved).

        Note: v2 weights are not yet publicly released (as of April 2026 the
        checkpoint URL returns HTTP 403). This config is provided so the model
        can be instantiated at v2 shapes once weights become available.
        """
        v2 = dict(
            d_pair=256,
            d_msa=128,
            n_heads_pair=8,
            n_msa_pw_heads=16,
        )
        v2.update(overrides)
        return cls(**v2)


# ============================================================================
# Building Blocks + Triangle Ops
# ============================================================================

from helico.model.blocks import (
    LayerNorm,
    Transition,
    AdaptiveLayerNorm,
    linear_no_bias,
    BiasInitLinear,
    FourierEmbedding,
    ConditionedTransitionBlock,
)
from helico.model.triangle import (
    TriangleMultiplicativeUpdate,
    TriangleAttention,
)
from helico.model.pairformer import (
    SingleAttentionWithPairBias,
    PairformerBlock,
    Pairformer,
    RelativePositionEncoding,
)
from helico.model.msa import (
    OuterProductMean,
    MSAPairWeightedAveraging,
    MSAStack,
    MSABlock,
    MSAModule,
)
from helico.model.diffusion import (
    DiffusionAttentionPairBias,
    DiffusionTransformerBlock,
    DiffusionTransformer,
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    DiffusionConditioning,
    DiffusionModule,
    _centre_random_augmentation,
    _partition_to_windows,
    _unpartition_from_windows,
)
from helico.model.input_embedder import InputFeatureEmbedder


# ============================================================================
# Template Embedder
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

    Matches Protenix v1.0.9 Algorithm 16. Protenix v1.0.0's checkpoint has
    n_blocks=2 and the embedder is *always* invoked with dummy template
    features when use_template=False — it still contributes a
    pairformer-transformed version of the trunk pair tensor each cycle.
    Helico previously stubbed this to 0, which caused z to diverge from
    Protenix by rel_L2 ~0.76 after recycling (confirmed by pipeline diff
    on 8t59).
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
        """Algorithm 16. Produces a residual to add to the trunk pair tensor.

        Uses dummy (all-zero) template features when real template features
        are absent in the batch — matching Protenix's inference behavior with
        use_template=False.
        """
        if not self.pairformer_stack:
            return 0

        B, N_tok = z.shape[:2]
        dtype = z.dtype
        device = z.device
        # Protenix's InferenceTemplateFeaturizer pads to max_templates=4
        # even when use_template=False: slot 0 gets aatype=31 (gap), slots
        # 1..3 are zero-padded (aatype=0). All other template features
        # (distogram, unit_vector, masks, atom_positions) are zeros.
        # Verified empirically from Protenix's 00_batch dump on 8t59/8v52.
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

        # Template features that are masked-zeroed (multichain_mask * pair_mask)
        # end up zero regardless of contents, so we just use zeros directly.
        dgram = torch.zeros(B, N_tok, N_tok, 39, dtype=dtype, device=device)
        pseudo_beta_mask_2d = torch.zeros(B, N_tok, N_tok, dtype=dtype, device=device)
        unit_vector = torch.zeros(B, N_tok, N_tok, 3, dtype=dtype, device=device)
        backbone_mask_2d = torch.zeros(B, N_tok, N_tok, dtype=dtype, device=device)

        u_sum = None
        for aa_val in aatype_per_slot:
            aatype_long = torch.full((B, N_tok), aa_val, dtype=torch.long, device=device)
            aatype_oh = F.one_hot(aatype_long, 32).to(dtype)  # (B, N_tok, 32)
            # Protenix: aatype_j = expand_at_dim(aatype, dim=-3); aatype_i = expand_at_dim(aatype, dim=-2)
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


# ============================================================================
# Distogram Head
# ============================================================================

class DistogramHead(nn.Module):
    """Predict symmetrized distance distribution from pair representation."""

    def __init__(self, config: HelicoConfig):
        super().__init__()
        self.linear = nn.Linear(config.d_pair, config.n_distogram_bins)  # WITH bias

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        return logits + logits.transpose(-2, -3)




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
# Confidence Head
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
        rep_atom_idx: torch.Tensor | None = None,
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
            rep_atom_idx: (B, N_tok) index of representative atom per token
        Returns:
            dict with pae, pde, plddt, resolved logits
        """
        s = self.input_s_norm(torch.clamp(s_trunk.detach(), min=-512, max=512))
        s_inp = s_inputs.detach()

        # z_init from s_inputs outer product
        z = z_trunk.detach() + self.linear_s1(s_inp).unsqueeze(2) + self.linear_s2(s_inp).unsqueeze(1)

        # Distance pair embeddings from representative atom coords (CB/CA/C4/C2)
        # Compute distances in float32 for numerical stability (matching Protenix)
        B, N_tok = s.shape[:2]
        if rep_atom_idx is not None:
            # Gather representative atom coordinates
            idx3 = rep_atom_idx.unsqueeze(-1).expand(-1, -1, 3)
            token_centers = torch.gather(pred_coords, 1, idx3)
        else:
            token_centers = self._get_token_centers(pred_coords, atom_to_token, atom_mask, N_tok)

        with torch.amp.autocast("cuda", enabled=False):
            dists = torch.cdist(
                token_centers.float(), token_centers.float()
            )  # (B, N, N) in float32
        # One-hot distance binning
        d_unsq = dists.unsqueeze(-1)  # (B, N, N, 1)
        one_hot = ((d_unsq > self.lower_bins) & (d_unsq < self.upper_bins)).to(z.dtype)
        z = z + self.linear_d(one_hot)
        z = z + self.linear_d_raw(d_unsq.to(z.dtype))

        # Pairformer
        s, z = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

        # Upcast after pairformer for output heads (matching Protenix)
        z = z.float()
        s = s.float()

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


def compute_ptm(
    pae_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    has_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute predicted TM-score from PAE logits.

    Args:
        pae_logits: (B, N, N, n_pae_bins)
        mask: (B, N) token mask, or None for all tokens
        has_frame: (B, N) bool mask — max is taken only over has_frame tokens

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

    # pTM = max over alignment dimension, filtered by has_frame if provided
    frame_mask = mask.clone()
    if has_frame is not None:
        frame_mask = frame_mask * has_frame.to(dtype=frame_mask.dtype)
    tm_per_aligned = tm_per_aligned.masked_fill(frame_mask == 0, 0.0)
    ptm = tm_per_aligned.max(dim=-1).values  # (B,)
    return ptm


def compute_iptm(
    pae_logits: torch.Tensor,
    chain_indices: torch.Tensor,
    mask: torch.Tensor | None = None,
    has_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute interface predicted TM-score (across different chains).

    Args:
        pae_logits: (B, N, N, n_pae_bins)
        chain_indices: (B, N) chain index per token
        mask: (B, N) token mask, or None
        has_frame: (B, N) bool mask — max is taken only over has_frame tokens

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

    # Mask tokens with no inter-chain partners, and filter by has_frame
    has_inter = (pair_mask.sum(dim=-1) > 0).float()
    frame_mask = has_inter.clone()
    if has_frame is not None:
        frame_mask = frame_mask * has_frame.to(dtype=frame_mask.dtype)
    tm_per_aligned = tm_per_aligned * frame_mask
    # If no inter-chain pairs at all, return 0
    any_inter = has_inter.sum(dim=-1) > 0  # (B,)
    iptm = tm_per_aligned.max(dim=-1).values  # (B,)
    iptm = iptm * any_inter.float()
    return iptm


def compute_clash(
    pred_coords: torch.Tensor,
    chain_indices: torch.Tensor,
    atom_to_token: torch.Tensor,
    atom_mask: torch.Tensor,
    threshold: float = 1.1,
) -> torch.Tensor:
    """Detect inter-chain atom clashes (AF3 clash criterion).

    Args:
        pred_coords: (B, N_atoms, 3) predicted coordinates
        chain_indices: (B, N_tok) chain index per token
        atom_to_token: (B, N_atoms) atom-to-token mapping
        atom_mask: (B, N_atoms) atom mask
        threshold: clash distance threshold in Angstroms (AF3 uses 1.1)

    Returns:
        (B,) float tensor: 1.0 if clash detected, 0.0 otherwise
    """
    B = pred_coords.shape[0]
    device = pred_coords.device
    has_clash = torch.zeros(B, device=device)

    for b in range(B):
        mask = atom_mask[b].bool()
        coords = pred_coords[b][mask]  # (N_real, 3)
        tok_ids = atom_to_token[b][mask]  # (N_real,)
        chain_ids = chain_indices[b][tok_ids]  # (N_real,)

        # Check inter-chain pairs
        n = coords.shape[0]
        if n > 5000:
            # Subsample for large structures to avoid OOM
            idx = torch.randperm(n, device=device)[:5000]
            coords = coords[idx]
            chain_ids = chain_ids[idx]

        dists = torch.cdist(coords.float(), coords.float())  # (N, N)
        inter_chain = chain_ids.unsqueeze(0) != chain_ids.unsqueeze(1)
        clash_mask = (dists < threshold) & inter_chain
        has_clash[b] = clash_mask.any().float()

    return has_clash


def _maybe_build_dumper(dump_dir: str | None):
    """Return a callable `dump(stage_name, tensors_dict)` or None.

    Used by Helico.predict to optionally persist intermediate state for
    pipeline-diff analysis (see scripts/pm/diff_pipelines.py). Each call
    writes `<dump_dir>/<stage_name>.npz` with all given tensors as
    float32 numpy arrays. Falls through harmlessly when dump_dir is None.
    """
    if dump_dir is None:
        return None
    from pathlib import Path
    import numpy as np

    out = Path(dump_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _dump(stage: str, tensors: dict):
        arrs = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                arrs[k] = v.detach().to(dtype=torch.float32, device="cpu").numpy()
            elif isinstance(v, (int, float, bool)):
                arrs[k] = np.array(v)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        arrs[f"{k}.{sk}"] = sv.detach().to(dtype=torch.float32, device="cpu").numpy()
        np.savez_compressed(out / f"{stage}.npz", **arrs)

    _dump.dump_dir = out  # type: ignore[attr-defined]
    return _dump


def compute_ranking_score(
    ptm: torch.Tensor,
    iptm: torch.Tensor,
    has_interface: torch.Tensor,
    has_clash: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ranking score: 0.8*iptm + 0.2*ptm - 100*has_clash (Protenix formula).

    Args:
        ptm: (B,) pTM scores
        iptm: (B,) ipTM scores
        has_interface: (B,) bool tensor, True when >1 unique chain
        has_clash: (B,) float tensor, 1.0 if clash detected

    Returns:
        (B,) ranking scores
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
# Affinity Module — Boltz2 feature
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
# Full Model
# ============================================================================

class Helico(nn.Module):
    """Complete Helico model."""

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
            ref_charge: (B, N_atoms, 1) — formal charges from CCD (arcsinh-transformed)
            ref_features: (B, N_atoms, 385) — mask(1) + element_onehot(128) + atom_name_chars(256)
        """
        B, N_atoms = batch["atom_element_idx"].shape
        device = batch["atom_element_idx"].device
        dtype = batch.get("ref_coords", batch["atom_coords"]).dtype

        # Use real ref_charge from CCD if available, else zeros
        if "ref_charge" in batch:
            ref_charge = torch.arcsinh(batch["ref_charge"].to(dtype)).unsqueeze(-1)
        else:
            ref_charge = torch.zeros(B, N_atoms, 1, device=device, dtype=dtype)

        # Element one-hot (128 dims, padded from n_elements)
        elem_onehot = F.one_hot(batch["atom_element_idx"].clamp(max=127), 128).to(dtype)

        # Atom mask as feature
        atom_mask = batch.get("atom_mask")
        if atom_mask is not None:
            mask_feat = atom_mask.unsqueeze(-1).to(dtype)
        else:
            mask_feat = torch.ones(B, N_atoms, 1, device=device, dtype=dtype)

        # Atom name chars: 4 chars × 64-class one-hot = 256 features
        name_chars = batch.get("atom_name_chars")
        if name_chars is None:
            name_chars = torch.zeros(B, N_atoms, 256, device=device, dtype=dtype)
        else:
            name_chars = name_chars.to(device=device, dtype=dtype)

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

        # restype one-hot (32 dims): precomputed in data pipeline (handles RNA/DNA correctly)
        restype = F.one_hot(batch["restype"], 32).to(dtype)

        # MSA profile (32 dims, already Protenix 32-class from data pipeline)
        profile = batch.get("msa_profile", torch.zeros(B, N_tok, 32, device=device, dtype=dtype))

        # deletion_mean (1 dim): use real values from data pipeline
        deletion_mean = batch.get("deletion_mean", torch.zeros(B, N_tok, 1, device=device, dtype=dtype))
        if deletion_mean.dim() == 2:
            deletion_mean = deletion_mean.unsqueeze(-1)

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
            ref_space_uid=batch.get("ref_space_uid"),
        )

    def _build_msa_raw(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build raw MSA features (B, N_msa, N_tok, 34) and mask.

        Protenix feeds the RAW MSA rows (with deletion matrix) to the MSA
        module, not a clustered view: a 13k-row MSA enters as 13k rows
        (optionally subsampled to test_cutoff=16384). We previously fed the
        64-row cluster summary via `cluster_msa` which threw away >99% of the
        alignment diversity and explains most of the reproduction gap on
        multichain targets. Now prefer the raw `msa` + `deletion_matrix` when
        provided, falling back to clustered fields for backward compat.
        """
        B, N_tok = batch["token_types"].shape
        device = batch["token_types"].device
        dtype = batch.get("ref_coords", batch["atom_coords"]).dtype

        msa_int = batch.get("msa")
        if msa_int is not None:
            del_raw = batch.get(
                "deletion_matrix",
                torch.zeros(B, msa_int.shape[1], N_tok, device=device, dtype=dtype),
            )
        else:
            msa_int = batch.get("cluster_msa")
            del_raw = batch.get("cluster_deletion_mean")
        if msa_int is None:
            msa_raw = torch.zeros(B, 1, N_tok, 34, device=device, dtype=dtype)
            return msa_raw, None

        N_msa = msa_int.shape[1]
        if del_raw is None:
            del_raw = torch.zeros(B, N_msa, N_tok, device=device, dtype=dtype)

        # One-hot encode MSA residues (already in Protenix 32-class encoding)
        msa_onehot = F.one_hot(msa_int.clamp(max=31), 32).to(dtype)

        # Features per Protenix (pairformer.py:724-725):
        # has_deletion = clip(deletion_matrix, 0, 1), deletion_value = arctan(d/3)*2/pi
        del_raw = del_raw.to(dtype)
        has_del = del_raw.clamp(0, 1).unsqueeze(-1)  # (B, N_msa, N_tok, 1)
        del_val = (torch.arctan(del_raw / 3.0) * (2.0 / math.pi)).unsqueeze(-1)

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
            z = z + self.template_embedder(batch, z)
            z = self.msa_module(
                msa_raw, z, s_inputs, msa_mask, pair_mask,
                msa_chunk_size=(None if self.training else 2048),
            )
            s = s_init + self.linear_s(self.layernorm_s(s))
            s, z = self.pairformer(s, z, mask=mask, pair_mask=pair_mask)

        results = {"single": s, "pair": z}

        # 4. Diffusion — s_inputs is already (B, N_tok, 449 = d_single + 65)
        # n_diffusion_samples > 1 amortizes the expensive trunk over several
        # denoising passes per batch entry (gh#6). Outputs are (B*N_d, ...).
        n_d = max(1, int(getattr(self.config, "n_diffusion_samples", 1)))
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
            n_samples=n_d,
        )

        results["x_denoised"] = x_denoised
        results["sigma"] = sigma
        # diffusion_loss averages over all B*N_d samples — atom_mask must
        # match the expanded batch.
        atom_mask_d = atom_mask.repeat_interleave(n_d, dim=0) if n_d > 1 else atom_mask
        results["diffusion_loss"] = diffusion_loss(x_denoised, gt_coords, sigma, atom_mask_d)

        # 5. Distogram (from trunk pair)
        distogram_logits = self.distogram_head(z)
        results["distogram_logits"] = distogram_logits

        # 6. Confidence head (uses pred_coords from diffusion). Use only
        # the first denoising sample per batch entry — the head expects
        # (B, N_atoms, 3), not (B*N_d, ...).
        if compute_confidence:
            x_for_conf = x_denoised[::n_d] if n_d > 1 else x_denoised
            confidence = self.confidence_head(
                s_trunk=s, z_trunk=z, s_inputs=s_inputs,
                pred_coords=x_for_conf.detach(),
                atom_to_token=batch["atom_to_token"],
                atom_mask=atom_mask,
                mask=mask, pair_mask=pair_mask,
                rep_atom_idx=batch.get("rep_atom_idx"),
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
        n_cycles: int | None = None,
        verbose_timing: bool = False,
        dump_intermediates_to: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run inference: generate structure predictions.

        Args:
            batch: Input feature dict.
            n_samples: Number of diffusion samples per input.
            n_cycles: Override number of recycling cycles (default: self.config.n_cycles).
            verbose_timing: Print detailed timing breakdown for each phase.
            dump_intermediates_to: If set, write intermediate tensors to
                this directory as .npz files for pipeline-diff analysis.
                Writes batch inputs, pre-recycle embeddings, post-recycle
                (final) s/z, diffusion outputs, and confidence-head
                outputs. Costs ~100-200 MB per target on 2048-token inputs.
        """
        self.eval()

        # Lazy-imported helper for optional intermediate dumping
        _dump = _maybe_build_dumper(dump_intermediates_to)

        def _sync_time():
            if verbose_timing:
                torch.cuda.synchronize()
                return time.perf_counter()
            return 0.0

        t_overall_start = _sync_time()

        mask = batch.get("token_mask")
        pair_mask = None
        if mask is not None:
            pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()

        B, N_tok = batch["token_types"].shape
        device = batch["token_types"].device

        if _dump is not None:
            # Persist input batch first — this is where featurization
            # bugs show up if they exist.
            _dump("00_batch", {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)})

        t0 = _sync_time()
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
        t_embed = _sync_time() - t0

        # Recycling
        msa_raw, msa_mask = self._build_msa_raw(batch)
        if _dump is not None:
            _dump("01_pre_recycle", {
                "s_inputs": s_inputs, "s_init": s_init, "z_init": z_init,
                "msa_raw": msa_raw, "msa_mask": msa_mask,
                "ref_charge": ref_charge, "ref_features": ref_features,
                "atom_mask": atom_mask, "pair_mask": pair_mask,
                "relpe_feats": relpe_feats,
            })

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        actual_cycles = n_cycles if n_cycles is not None else self.config.n_cycles
        t_recycle_start = _sync_time()
        cycle_times = []

        for cycle in range(actual_cycles):
            t_c0 = _sync_time()
            z = z_init + self.linear_z_cycle(self.layernorm_z_cycle(z))
            z = z + self.template_embedder(batch, z)
            z = self.msa_module(
                msa_raw, z, s_inputs, msa_mask, pair_mask,
                msa_chunk_size=(None if self.training else 2048),
            )
            s = s_init + self.linear_s(self.layernorm_s(s))
            s, z = self.pairformer(s, z, mask=mask, pair_mask=pair_mask)
            cycle_times.append(_sync_time() - t_c0)
        t_recycle = _sync_time() - t_recycle_start

        if _dump is not None:
            # Dump final s/z only — intermediate cycles are ~64MB each;
            # final state is the signal the diffusion sees.
            _dump("02_post_recycle", {"s": s, "z": z})

        # Generate all samples in one batched call: expand (B, ...) → (B*n_samples, ...)
        def _expand(t):
            return t.unsqueeze(1).expand(-1, n_samples, *[-1] * (t.dim() - 1)).reshape(B * n_samples, *t.shape[1:])

        ref_space_uid = batch.get("ref_space_uid")
        t_diffusion_start = _sync_time()
        batched_coords = self.diffusion.sample(
            ref_pos=_expand(batch["ref_coords"]),
            ref_charge=_expand(ref_charge),
            ref_features=_expand(ref_features),
            atom_to_token=_expand(batch["atom_to_token"]),
            atom_mask=_expand(atom_mask),
            s_trunk=_expand(s),
            z_trunk=_expand(z),
            s_inputs=_expand(s_inputs),
            relpe_feats={k: _expand(v) for k, v in relpe_feats.items()},
            ref_space_uid=_expand(ref_space_uid) if ref_space_uid is not None else None,
        )  # (B*n_samples, N_atoms, 3)
        all_coords = batched_coords.reshape(B, n_samples, *batched_coords.shape[1:])
        t_diffusion = _sync_time() - t_diffusion_start

        # Score all samples and pick the best by ranking_score
        best_ranking = None
        best_idx = torch.zeros(B, dtype=torch.long, device=device)
        best_confidence = None
        best_plddt = None
        best_pae = None
        best_ptm = None
        best_iptm = None

        chain_indices = batch.get("chain_indices")

        rep_atom_idx = batch.get("rep_atom_idx")
        has_frame = batch.get("has_frame")
        t_confidence_start = _sync_time()
        conf_times = []
        for si in range(n_samples):
            t_ci = _sync_time()
            confidence = self.confidence_head(
                s_trunk=s, z_trunk=z, s_inputs=s_inputs,
                pred_coords=all_coords[:, si],
                atom_to_token=batch["atom_to_token"],
                atom_mask=atom_mask,
                mask=mask, pair_mask=pair_mask,
                rep_atom_idx=rep_atom_idx,
            )

            plddt = compute_plddt(confidence["plddt_logits"])
            pae = compute_pae(confidence["pae_logits"])
            ptm = compute_ptm(confidence["pae_logits"], mask=mask, has_frame=has_frame)

            if chain_indices is not None:
                iptm = compute_iptm(confidence["pae_logits"], chain_indices, mask=mask, has_frame=has_frame)
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

            # Compute clash penalty
            if chain_indices is not None:
                has_clash = compute_clash(
                    all_coords[:, si], chain_indices,
                    batch["atom_to_token"], atom_mask,
                )
            else:
                has_clash = torch.zeros(B, device=device)

            ranking = compute_ranking_score(ptm, iptm, has_interface, has_clash=has_clash)
            conf_times.append(_sync_time() - t_ci)

            if best_ranking is None:
                best_ranking = ranking.clone()
                best_confidence = confidence
                best_plddt = plddt
                best_pae = pae
                best_ptm = ptm
                best_iptm = iptm
            else:
                # Update best per batch element
                better = ranking > best_ranking
                if better.any():
                    best_ranking = torch.where(better, ranking, best_ranking)
                    best_ptm = torch.where(better, ptm, best_ptm)
                    best_iptm = torch.where(better, iptm, best_iptm)
                    for b in range(B):
                        if better[b]:
                            best_idx[b] = si
                            best_plddt[b] = plddt[b]
                            best_pae[b] = pae[b]
                            best_confidence = {k: v.clone() for k, v in confidence.items()}  # save logits
        t_confidence = _sync_time() - t_confidence_start

        # Gather best coords per batch element
        best_coords = torch.stack([all_coords[b, best_idx[b]] for b in range(B)])

        # Flatten pLDDT to per-atom
        plddt_flat = _flatten_plddt(
            best_plddt, batch["atom_to_token"], batch["atoms_per_token"], atom_mask,
        )

        if _dump is not None:
            _dump("03_post_diffusion", {
                "all_coords": all_coords,        # (B, n_samples, N_atoms, 3)
                "best_coords": best_coords,      # (B, N_atoms, 3)
                "best_idx": best_idx,
                "ranking_score_per_sample": best_ranking,
                "ptm": best_ptm, "iptm": best_iptm,
                "plddt_flat": plddt_flat,
                "pae": best_pae,
            })

        t_overall = _sync_time() - t_overall_start

        if verbose_timing:
            n_steps = self.diffusion.n_steps
            print(f"\n{'='*60}")
            print(f"  Helico predict() timing  (N_tok={N_tok}, B={B})")
            print(f"{'='*60}")
            print(f"  Input embedding:      {t_embed:8.2f}s")
            print(f"  Recycling ({actual_cycles} cycles):  {t_recycle:8.2f}s")
            for i, ct in enumerate(cycle_times):
                print(f"    cycle {i:2d}:            {ct:8.2f}s")
            print(f"  Diffusion ({n_samples} samples): {t_diffusion:8.2f}s  (batched, B*S={B*n_samples})")
            print(f"    avg per step:       {t_diffusion/n_steps:8.3f}s  ({n_steps} steps)")
            print(f"  Confidence ({n_samples} samples):{t_confidence:8.2f}s")
            for i, ct in enumerate(conf_times):
                print(f"    sample {i}:           {ct:8.2f}s")
            print(f"  {'─'*40}")
            print(f"  Total wall time:      {t_overall:8.2f}s")
            print(f"{'='*60}\n")

        return {
            "coords": best_coords,               # (B, N_atoms, 3)
            "all_coords": all_coords,             # (B, n_samples, N_atoms, 3)
            "plddt": plddt_flat,                  # (B, N_atoms) 0-100 scale
            "pae": best_pae,                      # (B, N_tok, N_tok) Angstroms
            "ptm": best_ptm,                      # (B,)
            "iptm": best_iptm,                    # (B,)
            "ranking_score": best_ranking,        # (B,)
            # Raw logits for downstream use
            "pae_logits": best_confidence["pae_logits"],
            "plddt_logits": best_confidence["plddt_logits"],
            "pde_logits": best_confidence["pde_logits"],
        }
