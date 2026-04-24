"""Model configuration.

``HelicoConfig`` collects all the hyperparameters the model needs:
representation dimensions, attention shapes, block counts, diffusion
schedule, etc. Defaults match AF3 at the Protenix v1.0.0 checkpoint
scale (≈ 368M parameters).
"""

from __future__ import annotations

from dataclasses import dataclass

from helico.data import NUM_TOKEN_TYPES, UNK_ELEM_IDX


@dataclass
class HelicoConfig:
    """Model configuration with AlphaFold 3 defaults.

    Default values match the Protenix v1.0.0 checkpoint — about 368 M
    parameters. See ``protenix_v2`` classmethod for the (unreleased) v2
    shape configuration.
    """

    # --- Representation dimensions (AF3 SI §3, defaults match Alg 1) ---
    d_single: int = 384              # c_s — per-token single-repr dim
    d_pair: int = 128                # c_z — per-pair dim
    d_msa: int = 64                  # c_m — per-MSA-row dim
    n_msa_blocks: int = 4            # AF3 Alg 8 N_block
    c_msa_opm_hidden: int = 32       # Alg 9 c
    n_msa_pw_heads: int = 8          # Alg 10 N_head
    msa_pw_head_dim: int = 8

    # --- Pairformer (AF3 Alg 17) ---
    n_pairformer_blocks: int = 48    # AF3 default: 48 blocks
    n_heads_pair: int = 4            # d_pair / 32
    n_heads_single: int = 16         # d_single / 24 (note: Alg 24 uses N_head=16)
    pair_head_dim: int = 32
    single_head_dim: int = 24

    # --- Diffusion module (AF3 Alg 20) ---
    c_token: int = 768               # token-axis transformer dim (Alg 20 c_token)
    c_atom: int = 128                # atom-axis feature dim (c_atom in Alg 20)
    c_atompair: int = 16             # atom-pair dim (c_atompair)
    c_noise_embedding: int = 256     # Fourier-embedding output dim
    sigma_data: float = 16.0         # EDM σ_data, see AF3 SI §3.7.1
    n_diffusion_token_blocks: int = 24    # Alg 20 line 5 N_block
    n_heads_diffusion_token: int = 16     # Alg 20 line 5 N_head
    diffusion_token_head_dim: int = 48    # 768 / 16
    n_atom_encoder_blocks: int = 3        # Alg 5 / Alg 7 N_block
    n_atom_decoder_blocks: int = 3        # Alg 6 / Alg 7 N_block
    n_heads_atom: int = 4                 # atom-level transformer heads
    atom_head_dim: int = 32               # 128 / 4

    # Training-only noise sampling
    noise_log_mean: float = -1.2          # Log-normal σ ~ σ_data · exp(μ + σ·N(0,1))
    noise_log_std: float = 1.5

    # Inference sampler
    n_diffusion_steps: int = 200          # AF3 default sampling steps

    # Sequence-local atom attention windows (Alg 7 defaults)
    n_atom_queries: int = 32
    n_atom_keys: int = 128

    # Diffusion training: number of denoising passes per trunk forward
    # (amortize the expensive trunk — AF3 SI §3.7.1 Fig 2c).
    n_diffusion_samples: int = 8

    # --- Atom feature dims (from AF3 SI §2.8 Table 5) ---
    n_elements: int = UNK_ELEM_IDX + 1    # Number of element types + 1 UNK
    n_token_types: int = NUM_TOKEN_TYPES

    # --- Template embedder (AF3 Alg 16) ---
    n_template_blocks: int = 2
    d_template: int = 64                  # Template pair dim (not same as d_pair)

    # --- Confidence head (AF3 SI §4.3) ---
    n_plddt_bins: int = 50                # §4.3.1
    n_pae_bins: int = 64                  # §4.3.2
    n_distogram_bins: int = 64            # §4.4
    n_confidence_blocks: int = 4          # Pairformer blocks inside confidence head
    n_distance_bins: int = 39             # distance embedding for confidence (3.25-52 Å)

    # --- Recycling (AF3 Alg 1 outer loop) ---
    n_cycles: int = 1                     # Inference default: 10; tests use 1

    # --- Training ---
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
        """InputFeatureEmbedder output dim (AF3 Alg 2): d_single + 32 restype + 32 profile + 1 deletion_mean."""
        return self.d_single + 65

    @classmethod
    def protenix_v2(cls, **overrides) -> "HelicoConfig":
        """Config matching Protenix v2.0.0 (≈ 464 M params).

        Width scale-up of v1: c_z 128→256, c_m 64→128, with
        ``hidden_scale_up=True`` in Protenix. Head counts scale so
        head_dim stays 32 for pair attention and 8 for MSA pair-weighted
        averaging.

        Note: v2 weights are not yet publicly released (as of April 2026
        the checkpoint URL returns HTTP 403). This config exists so the
        model can be *instantiated* at v2 shapes once weights become
        available.
        """
        v2 = dict(
            d_pair=256,
            d_msa=128,
            n_heads_pair=8,
            n_msa_pw_heads=16,
        )
        v2.update(overrides)
        return cls(**v2)
