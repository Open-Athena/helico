"""Integration tests for Helico model components."""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add project root so scripts/ can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helico.data import make_synthetic_batch, make_synthetic_structure, tokenize_structure
from helico.model import (
    Helico,
    HelicoConfig,
    InputFeatureEmbedder,
    TemplateEmbedder,
    DistogramHead,
    TriangleMultiplicativeUpdate,
    TriangleAttention,
    SingleAttentionWithPairBias,
    PairformerBlock,
    Pairformer,
    OuterProductMean,
    MSAPairWeightedAveraging,
    MSAModule,
    AdaptiveLayerNorm,
    FourierEmbedding,
    ConditionedTransitionBlock,
    DiffusionAttentionPairBias,
    DiffusionTransformerBlock,
    DiffusionTransformer,
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    DiffusionConditioning,
    DiffusionModule,
    ConfidenceHead,
    AffinityModule,
    diffusion_loss,
    smooth_lddt_loss,
    distogram_loss,
    violation_loss,
    compute_plddt,
    compute_pae,
    compute_ptm,
    compute_iptm,
    compute_ranking_score,
    _flatten_plddt,
)

# Use small config for tests
TEST_CONFIG = HelicoConfig(
    d_single=64,
    d_pair=64,
    d_msa=32,
    n_pairformer_blocks=2,
    n_heads_pair=2,
    n_heads_single=4,
    pair_head_dim=32,
    single_head_dim=16,
    # MSA module
    n_msa_blocks=4,
    c_msa_opm_hidden=16,
    n_msa_pw_heads=4,
    msa_pw_head_dim=8,
    # Diffusion
    c_token=96,
    c_atom=64,
    c_atompair=8,
    c_noise_embedding=32,
    n_diffusion_token_blocks=2,
    n_heads_diffusion_token=3,
    diffusion_token_head_dim=32,  # 96/3
    n_atom_encoder_blocks=1,
    n_atom_decoder_blocks=1,
    n_heads_atom=2,
    atom_head_dim=32,  # 64/2
    n_diffusion_steps=10,
    d_affinity=64,
    n_affinity_pairformer_blocks=1,
    # Template embedder
    n_template_blocks=2,
    d_template=64,
    # Confidence head
    n_confidence_blocks=2,
    n_distance_bins=39,
    max_atoms_per_token=24,
    # Recycling
    n_cycles=1,
    gradient_checkpointing=False,
    dropout=0.0,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
N_TOKENS = 16
N_ATOMS_PER_TOKEN = 4
BATCH_SIZE = 1


def _make_batch(n_tokens=N_TOKENS, batch_size=BATCH_SIZE):
    return make_synthetic_batch(
        n_tokens=n_tokens,
        n_atoms_per_token=N_ATOMS_PER_TOKEN,
        batch_size=batch_size,
        device=DEVICE,
    )


# ============================================================================
# Triangle Operations Tests
# ============================================================================

class TestTriangleOps:
    def test_tri_mul_outgoing_shape(self):
        tri = TriangleMultiplicativeUpdate(TEST_CONFIG.d_pair, "outgoing").to(device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        mask = torch.ones(BATCH_SIZE, N_TOKENS, N_TOKENS, device=DEVICE, dtype=DTYPE)
        out = tri(z, mask=mask)
        assert out.shape == z.shape

    def test_tri_mul_incoming_shape(self):
        tri = TriangleMultiplicativeUpdate(TEST_CONFIG.d_pair, "incoming").to(device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        mask = torch.ones(BATCH_SIZE, N_TOKENS, N_TOKENS, device=DEVICE, dtype=DTYPE)
        out = tri(z, mask=mask)
        assert out.shape == z.shape

    def test_tri_mul_gradient(self):
        tri = TriangleMultiplicativeUpdate(TEST_CONFIG.d_pair, "outgoing").to(DEVICE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(BATCH_SIZE, N_TOKENS, N_TOKENS, device=DEVICE, dtype=torch.float32)
        out = tri(z, mask=mask)
        out.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_tri_att_starting_shape(self):
        att = TriangleAttention(TEST_CONFIG.d_pair, TEST_CONFIG.n_heads_pair, "starting").to(device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        out = att(z)
        assert out.shape == z.shape

    def test_tri_att_ending_shape(self):
        att = TriangleAttention(TEST_CONFIG.d_pair, TEST_CONFIG.n_heads_pair, "ending").to(device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        out = att(z)
        assert out.shape == z.shape

    def test_tri_att_gradient(self):
        att = TriangleAttention(TEST_CONFIG.d_pair, TEST_CONFIG.n_heads_pair, "starting").to(DEVICE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=torch.float32, requires_grad=True)
        out = att(z)
        out.sum().backward()
        assert z.grad is not None


class TestSingleAttention:
    def test_shape(self):
        att = SingleAttentionWithPairBias(TEST_CONFIG.d_single, TEST_CONFIG.d_pair, TEST_CONFIG.n_heads_single).to(device=DEVICE, dtype=DTYPE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        mask = torch.ones(BATCH_SIZE, N_TOKENS, device=DEVICE, dtype=DTYPE)
        out = att(s, z, mask=mask)
        assert out.shape == s.shape


# ============================================================================
# Pairformer Tests
# ============================================================================

class TestPairformer:
    def test_block_shapes(self):
        block = PairformerBlock(TEST_CONFIG).to(device=DEVICE, dtype=DTYPE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        mask = torch.ones(BATCH_SIZE, N_TOKENS, device=DEVICE, dtype=torch.bool)
        pair_mask = torch.ones(BATCH_SIZE, N_TOKENS, N_TOKENS, device=DEVICE, dtype=DTYPE)
        s_out, z_out = block(s, z, mask=mask, pair_mask=pair_mask)
        assert s_out.shape == s.shape
        assert z_out.shape == z.shape

    def test_residual_preserves_scale(self):
        """Residual connections should keep values from exploding."""
        block = PairformerBlock(TEST_CONFIG).to(device=DEVICE, dtype=DTYPE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE, dtype=DTYPE) * 0.1
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE) * 0.1
        pair_mask = torch.ones(BATCH_SIZE, N_TOKENS, N_TOKENS, device=DEVICE, dtype=DTYPE)
        s_out, z_out = block(s, z, pair_mask=pair_mask)
        # Output should be similar magnitude to input (within 100x)
        assert s_out.abs().max() < 100 * max(s.abs().max(), 1.0)
        assert z_out.abs().max() < 100 * max(z.abs().max(), 1.0)

    def test_stack(self):
        config = HelicoConfig(**{**TEST_CONFIG.__dict__, "n_pairformer_blocks": 2})
        stack = Pairformer(config).to(device=DEVICE, dtype=DTYPE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        pair_mask = torch.ones(BATCH_SIZE, N_TOKENS, N_TOKENS, device=DEVICE, dtype=DTYPE)
        s_out, z_out = stack(s, z, pair_mask=pair_mask)
        assert s_out.shape == s.shape
        assert z_out.shape == z.shape


# ============================================================================
# MSA Module Tests
# ============================================================================

class TestOuterProductMean:
    def test_shape(self):
        opm = OuterProductMean(TEST_CONFIG.d_msa, TEST_CONFIG.d_pair, TEST_CONFIG.c_msa_opm_hidden).to(device=DEVICE, dtype=DTYPE)
        N_msa = 8
        m = torch.randn(BATCH_SIZE, N_msa, N_TOKENS, TEST_CONFIG.d_msa, device=DEVICE, dtype=DTYPE)
        mask = torch.ones(BATCH_SIZE, N_msa, N_TOKENS, device=DEVICE, dtype=DTYPE)
        out = opm(m, mask=mask)
        assert out.shape == (BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair)

    def test_gradient(self):
        opm = OuterProductMean(TEST_CONFIG.d_msa, TEST_CONFIG.d_pair, TEST_CONFIG.c_msa_opm_hidden).to(DEVICE)
        N_msa = 4
        m = torch.randn(BATCH_SIZE, N_msa, N_TOKENS, TEST_CONFIG.d_msa, device=DEVICE, requires_grad=True)
        out = opm(m)
        out.sum().backward()
        assert m.grad is not None
        assert torch.isfinite(m.grad).all()


class TestMSAPairWeightedAveraging:
    def test_shape(self):
        avg = MSAPairWeightedAveraging(TEST_CONFIG.d_msa, TEST_CONFIG.d_pair, TEST_CONFIG.n_msa_pw_heads, TEST_CONFIG.msa_pw_head_dim).to(device=DEVICE, dtype=DTYPE)
        N_msa = 8
        m = torch.randn(BATCH_SIZE, N_msa, N_TOKENS, TEST_CONFIG.d_msa, device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        out = avg(m, z)
        assert out.shape == m.shape

    def test_gradient(self):
        avg = MSAPairWeightedAveraging(TEST_CONFIG.d_msa, TEST_CONFIG.d_pair, TEST_CONFIG.n_msa_pw_heads, TEST_CONFIG.msa_pw_head_dim).to(DEVICE)
        N_msa = 4
        m = torch.randn(BATCH_SIZE, N_msa, N_TOKENS, TEST_CONFIG.d_msa, device=DEVICE, requires_grad=True)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, requires_grad=True)
        out = avg(m, z)
        out.sum().backward()
        assert m.grad is not None
        assert z.grad is not None


class TestMSAModule:
    def test_output_shape(self):
        module = MSAModule(TEST_CONFIG).to(device=DEVICE, dtype=DTYPE)
        N_msa = 8
        m_raw = torch.randn(BATCH_SIZE, N_msa, N_TOKENS, 34, device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        s_inputs = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.c_s_inputs, device=DEVICE, dtype=DTYPE)
        pair_mask = torch.ones(BATCH_SIZE, N_TOKENS, N_TOKENS, device=DEVICE, dtype=DTYPE)
        z_out = module(m_raw, z, s_inputs, pair_mask=pair_mask)
        assert z_out.shape == z.shape

    def test_gradient_flow(self):
        module = MSAModule(TEST_CONFIG).to(DEVICE)
        N_msa = 4
        m_raw = torch.randn(BATCH_SIZE, N_msa, N_TOKENS, 34, device=DEVICE, requires_grad=True)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, requires_grad=True)
        s_inputs = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.c_s_inputs, device=DEVICE, requires_grad=True)
        z_out = module(m_raw, z, s_inputs)
        z_out.sum().backward()
        assert m_raw.grad is not None
        assert z.grad is not None
        assert s_inputs.grad is not None

    def test_block_structure(self):
        """Verify MSA module has correct block structure: 4 OPM, 4 pair_stacks, 3 msa_stacks."""
        module = MSAModule(TEST_CONFIG)
        assert len(module.blocks) == 4
        for i, block in enumerate(module.blocks):
            assert hasattr(block, 'opm')
            assert hasattr(block, 'pair_stack')
            assert not block.pair_stack.has_single
            if i < 3:
                assert block.has_msa_stack
                assert hasattr(block, 'msa_stack')
            else:
                assert not block.has_msa_stack


class TestPairformerBlockPairOnly:
    def test_no_single_params(self):
        """PairformerBlock with has_single=False should have no single params."""
        block = PairformerBlock(TEST_CONFIG, has_single=False)
        param_names = [n for n, _ in block.named_parameters()]
        assert not any("single" in n for n in param_names)

    def test_pair_only_forward(self):
        block = PairformerBlock(TEST_CONFIG, has_single=False).to(device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        pair_mask = torch.ones(BATCH_SIZE, N_TOKENS, N_TOKENS, device=DEVICE, dtype=DTYPE)
        s_out, z_out = block(None, z, pair_mask=pair_mask)
        assert s_out is None
        assert z_out.shape == z.shape


# ============================================================================
# New Diffusion Primitive Tests
# ============================================================================

class TestAdaptiveLayerNorm:
    def test_shape(self):
        d_a, d_s = 64, 32
        aln = AdaptiveLayerNorm(d_a, d_s).to(DEVICE)
        a = torch.randn(BATCH_SIZE, N_TOKENS, d_a, device=DEVICE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, d_s, device=DEVICE)
        out = aln(a, s)
        assert out.shape == a.shape

    def test_zeros_init(self):
        """At init, scale_proj outputs 0 -> sigmoid(0) = 0.5."""
        d_a, d_s = 64, 32
        aln = AdaptiveLayerNorm(d_a, d_s).to(DEVICE)
        a = torch.randn(BATCH_SIZE, N_TOKENS, d_a, device=DEVICE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, d_s, device=DEVICE)
        out = aln(a, s)
        # Output should be ~0.5 * LN(a) since shift is 0
        expected = 0.5 * aln.norm_a(a)
        assert torch.allclose(out, expected, atol=1e-5)


class TestFourierEmbedding:
    def test_shape(self):
        d = 32
        fe = FourierEmbedding(d).to(DEVICE)
        t = torch.randn(BATCH_SIZE, device=DEVICE)
        out = fe(t)
        assert out.shape == (BATCH_SIZE, d)

    def test_determinism(self):
        d = 32
        fe1 = FourierEmbedding(d, seed=42).to(DEVICE)
        fe2 = FourierEmbedding(d, seed=42).to(DEVICE)
        t = torch.randn(BATCH_SIZE, device=DEVICE)
        assert torch.allclose(fe1(t), fe2(t))


class TestConditionedTransitionBlock:
    def test_shape(self):
        d_a, d_s = 64, 32
        block = ConditionedTransitionBlock(d_a, d_s).to(DEVICE)
        a = torch.randn(BATCH_SIZE, N_TOKENS, d_a, device=DEVICE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, d_s, device=DEVICE)
        out = block(a, s)
        assert out.shape == a.shape


class TestDiffusionAttentionPairBias:
    def test_shape(self):
        d_a, d_s, d_z = 64, 32, 16
        attn = DiffusionAttentionPairBias(d_a, d_s, d_z, n_heads=2, head_dim=32).to(DEVICE)
        a = torch.randn(BATCH_SIZE, N_TOKENS, d_a, device=DEVICE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, d_s, device=DEVICE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, d_z, device=DEVICE)
        out = attn(a, s, z)
        assert out.shape == a.shape

    def test_cross_attention_mode(self):
        d_a, d_s, d_z = 64, 32, 16
        attn = DiffusionAttentionPairBias(d_a, d_s, d_z, n_heads=2, head_dim=32,
                                           cross_attention_mode=True).to(DEVICE)
        a = torch.randn(BATCH_SIZE, N_TOKENS, d_a, device=DEVICE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, d_s, device=DEVICE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, d_z, device=DEVICE)
        kv_a = torch.randn(BATCH_SIZE, N_TOKENS, d_a, device=DEVICE)
        out = attn(a, s, z, kv_a=kv_a, kv_s=s)
        assert out.shape == a.shape

    def test_windowed_mode(self):
        d_a, d_s, d_z = 64, 32, 16
        N = 64
        n_q, n_k = 32, 64  # small windows for test
        n_blocks = (N + n_q - 1) // n_q  # 2
        attn = DiffusionAttentionPairBias(d_a, d_s, d_z, n_heads=2, head_dim=32).to(DEVICE)
        a = torch.randn(BATCH_SIZE, N, d_a, device=DEVICE)
        s = torch.randn(BATCH_SIZE, N, d_s, device=DEVICE)
        z = torch.randn(BATCH_SIZE, n_blocks, n_q, n_k, d_z, device=DEVICE)
        pad_mask = torch.ones(n_blocks, n_q, n_k, dtype=torch.bool, device=DEVICE)
        out = attn(a, s, z, n_queries=n_q, n_keys=n_k, pad_mask=pad_mask)
        assert out.shape == a.shape


# ============================================================================
# Atom Attention Tests
# ============================================================================

class TestAtomAttentionEncoder:
    def test_output_shapes(self):
        encoder = AtomAttentionEncoder(TEST_CONFIG).to(DEVICE)
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN  # 64
        noisy_pos = torch.randn(BATCH_SIZE, N_atoms, 3, device=DEVICE)
        ref_pos = torch.randn(BATCH_SIZE, N_atoms, 3, device=DEVICE)
        ref_charge = torch.zeros(BATCH_SIZE, N_atoms, 1, device=DEVICE)
        ref_features = torch.randn(BATCH_SIZE, N_atoms, 385, device=DEVICE)
        a2t = torch.arange(N_TOKENS, device=DEVICE).repeat_interleave(N_ATOMS_PER_TOKEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        atom_mask = torch.ones(BATCH_SIZE, N_atoms, device=DEVICE)
        s_trunk = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE)
        z_trunk = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE)

        a_token, q_skip, c_skip, p_skip, pad_mask = encoder(
            ref_pos, ref_charge, ref_features,
            a2t, atom_mask, N_TOKENS,
            noisy_pos=noisy_pos, s_trunk=s_trunk, z_trunk=z_trunk)

        n_q = TEST_CONFIG.n_atom_queries  # 32
        n_k = TEST_CONFIG.n_atom_keys    # 128
        n_blocks = (N_atoms + n_q - 1) // n_q  # ceil(64/32) = 2

        assert a_token.shape == (BATCH_SIZE, N_TOKENS, TEST_CONFIG.c_token)
        assert q_skip.shape == (BATCH_SIZE, N_atoms, TEST_CONFIG.c_atom)
        assert c_skip.shape == (BATCH_SIZE, N_atoms, TEST_CONFIG.c_atom)
        assert p_skip.shape == (BATCH_SIZE, n_blocks, n_q, n_k, TEST_CONFIG.c_atompair)
        assert pad_mask.shape == (n_blocks, n_q, n_k)


class TestAtomAttentionEncoderNoCoords:
    def test_no_trunk_params(self):
        """AtomAttentionEncoder with has_coords=False should have no trunk/noisy params."""
        encoder = AtomAttentionEncoder(TEST_CONFIG, has_coords=False, c_token_override=TEST_CONFIG.d_single)
        param_names = [n for n, _ in encoder.named_parameters()]
        assert not any("noisy_pos_proj" in n for n in param_names)
        assert not any("trunk_s_proj" in n for n in param_names)
        assert not any("trunk_z_proj" in n for n in param_names)

    def test_forward(self):
        encoder = AtomAttentionEncoder(TEST_CONFIG, has_coords=False, c_token_override=TEST_CONFIG.d_single).to(DEVICE)
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN
        ref_pos = torch.randn(BATCH_SIZE, N_atoms, 3, device=DEVICE)
        ref_charge = torch.zeros(BATCH_SIZE, N_atoms, 1, device=DEVICE)
        ref_features = torch.randn(BATCH_SIZE, N_atoms, 385, device=DEVICE)
        a2t = torch.arange(N_TOKENS, device=DEVICE).repeat_interleave(N_ATOMS_PER_TOKEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        atom_mask = torch.ones(BATCH_SIZE, N_atoms, device=DEVICE)

        a_token, _, _, _, _ = encoder(
            ref_pos, ref_charge, ref_features,
            a2t, atom_mask, N_TOKENS)
        assert a_token.shape == (BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single)


class TestAtomAttentionDecoder:
    def test_output_shape(self):
        decoder = AtomAttentionDecoder(TEST_CONFIG).to(DEVICE)
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN  # 64
        n_q = TEST_CONFIG.n_atom_queries  # 32
        n_k = TEST_CONFIG.n_atom_keys    # 128
        n_blocks = (N_atoms + n_q - 1) // n_q  # 2

        a_token = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.c_token, device=DEVICE)
        a2t = torch.arange(N_TOKENS, device=DEVICE).repeat_interleave(N_ATOMS_PER_TOKEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        q_skip = torch.randn(BATCH_SIZE, N_atoms, TEST_CONFIG.c_atom, device=DEVICE)
        c_skip = torch.randn(BATCH_SIZE, N_atoms, TEST_CONFIG.c_atom, device=DEVICE)
        p_skip = torch.randn(BATCH_SIZE, n_blocks, n_q, n_k, TEST_CONFIG.c_atompair, device=DEVICE)
        pad_mask = torch.ones(n_blocks, n_q, n_k, dtype=torch.bool, device=DEVICE)

        out = decoder(a_token, a2t, q_skip, c_skip, p_skip, pad_mask)
        assert out.shape == (BATCH_SIZE, N_atoms, 3)
        assert out.dtype == torch.float32


class TestDiffusionConditioning:
    def test_output_shapes(self):
        cond = DiffusionConditioning(TEST_CONFIG).to(DEVICE)
        s_trunk = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE)
        z_trunk = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE)
        s_inputs = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single + 65, device=DEVICE)
        sigma = torch.ones(BATCH_SIZE, device=DEVICE)
        relpe_feats = {
            "residue_index": torch.arange(N_TOKENS, device=DEVICE).unsqueeze(0).expand(BATCH_SIZE, -1),
            "token_index": torch.zeros(BATCH_SIZE, N_TOKENS, dtype=torch.long, device=DEVICE),
            "asym_id": torch.zeros(BATCH_SIZE, N_TOKENS, dtype=torch.long, device=DEVICE),
            "entity_id": torch.zeros(BATCH_SIZE, N_TOKENS, dtype=torch.long, device=DEVICE),
            "sym_id": torch.zeros(BATCH_SIZE, N_TOKENS, dtype=torch.long, device=DEVICE),
        }

        s_cond, z_cond = cond(s_trunk, z_trunk, s_inputs, sigma, relpe_feats)
        assert s_cond.shape == (BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single)
        assert z_cond.shape == (BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair)


# ============================================================================
# Diffusion Module Tests
# ============================================================================

class TestDiffusionModule:
    def _make_diffusion_inputs(self):
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN
        gt_coords = torch.randn(BATCH_SIZE, N_atoms, 3, device=DEVICE)
        ref_pos = torch.randn(BATCH_SIZE, N_atoms, 3, device=DEVICE)
        ref_charge = torch.zeros(BATCH_SIZE, N_atoms, 1, device=DEVICE)
        ref_features = torch.randn(BATCH_SIZE, N_atoms, 385, device=DEVICE)
        a2t = torch.arange(N_TOKENS, device=DEVICE).repeat_interleave(N_ATOMS_PER_TOKEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        atom_mask = torch.ones(BATCH_SIZE, N_atoms, device=DEVICE)
        s_trunk = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE)
        z_trunk = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE)
        s_inputs = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single + 65, device=DEVICE)
        relpe_feats = {
            "residue_index": torch.arange(N_TOKENS, device=DEVICE).unsqueeze(0).expand(BATCH_SIZE, -1),
            "token_index": torch.zeros(BATCH_SIZE, N_TOKENS, dtype=torch.long, device=DEVICE),
            "asym_id": torch.zeros(BATCH_SIZE, N_TOKENS, dtype=torch.long, device=DEVICE),
            "entity_id": torch.zeros(BATCH_SIZE, N_TOKENS, dtype=torch.long, device=DEVICE),
            "sym_id": torch.zeros(BATCH_SIZE, N_TOKENS, dtype=torch.long, device=DEVICE),
        }
        return dict(
            gt_coords=gt_coords, ref_pos=ref_pos, ref_charge=ref_charge,
            ref_features=ref_features, atom_to_token=a2t, atom_mask=atom_mask,
            s_trunk=s_trunk, z_trunk=z_trunk, s_inputs=s_inputs,
            relpe_feats=relpe_feats,
        )

    def test_training_forward(self):
        module = DiffusionModule(TEST_CONFIG).to(DEVICE)
        inputs = self._make_diffusion_inputs()
        x_denoised, gt_coords, sigma = module.forward_training(**inputs)
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN
        assert x_denoised.shape == (BATCH_SIZE, N_atoms, 3)
        assert gt_coords.shape == (BATCH_SIZE, N_atoms, 3)
        assert sigma.shape == (BATCH_SIZE,)

    def test_training_loss_scalar(self):
        module = DiffusionModule(TEST_CONFIG).to(DEVICE)
        inputs = self._make_diffusion_inputs()
        x_denoised, gt_coords, sigma = module.forward_training(**inputs)
        loss = diffusion_loss(x_denoised, gt_coords, sigma)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_inference_shape(self):
        config = HelicoConfig(**{**TEST_CONFIG.__dict__, "n_diffusion_steps": 3})
        module = DiffusionModule(config).to(DEVICE)
        inputs = self._make_diffusion_inputs()
        del inputs["gt_coords"]
        coords = module.sample(**inputs)
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN
        assert coords.shape == (BATCH_SIZE, N_atoms, 3)


# ============================================================================
# Loss Function Tests
# ============================================================================

class TestLosses:
    def test_diffusion_loss_zero_for_perfect(self):
        gt = torch.randn(1, 10, 3)
        sigma = torch.tensor([1.0])
        loss = diffusion_loss(gt, gt, sigma)
        assert loss.item() < 1e-5

    def test_diffusion_loss_positive(self):
        gt = torch.randn(1, 10, 3)
        pred = torch.randn(1, 10, 3)
        sigma = torch.tensor([1.0])
        loss = diffusion_loss(pred, gt, sigma)
        assert loss.item() > 0

    def test_smooth_lddt_perfect(self):
        coords = torch.randn(1, 20, 3)
        loss = smooth_lddt_loss(coords, coords)
        assert loss.item() < 0.1

    def test_smooth_lddt_random(self):
        pred = torch.randn(1, 20, 3) * 10
        gt = torch.randn(1, 20, 3) * 10
        loss = smooth_lddt_loss(pred, gt, cutoff=50.0)
        assert loss.item() > 0.1

    def test_distogram_loss(self):
        n_bins = 64
        pred = torch.randn(1, 10, 10, n_bins)
        gt = torch.randn(1, 10, 3) * 5
        loss = distogram_loss(pred, gt, n_bins=n_bins)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_violation_loss_no_clash(self):
        coords = torch.zeros(1, 5, 3)
        for i in range(5):
            coords[0, i, 0] = i * 5.0
        loss = violation_loss(coords, clash_threshold=1.2)
        assert loss.item() < 1e-5

    def test_violation_loss_with_clash(self):
        coords = torch.zeros(1, 5, 3)
        loss = violation_loss(coords, clash_threshold=1.2)
        assert loss.item() > 0


# ============================================================================
# New Module Tests (InputFeatureEmbedder, TemplateEmbedder, DistogramHead, ConfidenceHead)
# ============================================================================

class TestInputFeatureEmbedder:
    def test_output_shape(self):
        embedder = InputFeatureEmbedder(TEST_CONFIG).to(DEVICE)
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN
        ref_pos = torch.randn(BATCH_SIZE, N_atoms, 3, device=DEVICE)
        ref_charge = torch.zeros(BATCH_SIZE, N_atoms, 1, device=DEVICE)
        ref_features = torch.randn(BATCH_SIZE, N_atoms, 385, device=DEVICE)
        a2t = torch.arange(N_TOKENS, device=DEVICE).repeat_interleave(N_ATOMS_PER_TOKEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        atom_mask = torch.ones(BATCH_SIZE, N_atoms, device=DEVICE)
        restype = torch.randn(BATCH_SIZE, N_TOKENS, 32, device=DEVICE)
        profile = torch.randn(BATCH_SIZE, N_TOKENS, 32, device=DEVICE)
        deletion_mean = torch.randn(BATCH_SIZE, N_TOKENS, 1, device=DEVICE)

        s_inputs = embedder(ref_pos, ref_charge, ref_features, a2t, atom_mask,
                            N_TOKENS, restype, profile, deletion_mean)
        assert s_inputs.shape == (BATCH_SIZE, N_TOKENS, TEST_CONFIG.c_s_inputs)

    def test_gradient_flow(self):
        embedder = InputFeatureEmbedder(TEST_CONFIG).to(DEVICE)
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN
        ref_pos = torch.randn(BATCH_SIZE, N_atoms, 3, device=DEVICE, requires_grad=True)
        ref_charge = torch.zeros(BATCH_SIZE, N_atoms, 1, device=DEVICE)
        ref_features = torch.randn(BATCH_SIZE, N_atoms, 385, device=DEVICE)
        a2t = torch.arange(N_TOKENS, device=DEVICE).repeat_interleave(N_ATOMS_PER_TOKEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        atom_mask = torch.ones(BATCH_SIZE, N_atoms, device=DEVICE)
        restype = torch.randn(BATCH_SIZE, N_TOKENS, 32, device=DEVICE)
        profile = torch.randn(BATCH_SIZE, N_TOKENS, 32, device=DEVICE)
        deletion_mean = torch.randn(BATCH_SIZE, N_TOKENS, 1, device=DEVICE)

        s_inputs = embedder(ref_pos, ref_charge, ref_features, a2t, atom_mask,
                            N_TOKENS, restype, profile, deletion_mean)
        s_inputs.sum().backward()
        assert ref_pos.grad is not None


class TestTemplateEmbedder:
    def test_returns_zero(self):
        te = TemplateEmbedder(TEST_CONFIG).to(DEVICE)
        batch = _make_batch()
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE)
        out = te(batch, z)
        assert out == 0

    def test_has_params(self):
        """Verify template embedder creates parameters (for weight transfer)."""
        te = TemplateEmbedder(TEST_CONFIG)
        param_names = [n for n, _ in te.named_parameters()]
        assert any("z_norm" in n for n in param_names)
        assert any("linear_z" in n for n in param_names)
        assert any("pairformer_stack" in n for n in param_names)


class TestDistogramHead:
    def test_output_shape(self):
        head = DistogramHead(TEST_CONFIG).to(DEVICE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE)
        out = head(z)
        assert out.shape == (BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.n_distogram_bins)

    def test_symmetry(self):
        """Distogram output should be symmetric."""
        head = DistogramHead(TEST_CONFIG).to(DEVICE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE)
        out = head(z)
        assert torch.allclose(out, out.transpose(-2, -3), atol=1e-5)


class TestConfidenceHead:
    def test_output_shapes(self):
        head = ConfidenceHead(TEST_CONFIG).to(DEVICE)
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN
        s_trunk = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE)
        z_trunk = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE)
        s_inputs = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.c_s_inputs, device=DEVICE)
        pred_coords = torch.randn(BATCH_SIZE, N_atoms, 3, device=DEVICE)
        a2t = torch.arange(N_TOKENS, device=DEVICE).repeat_interleave(N_ATOMS_PER_TOKEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        atom_mask = torch.ones(BATCH_SIZE, N_atoms, device=DEVICE)

        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=DEVICE == "cuda"):
            out = head(s_trunk, z_trunk, s_inputs, pred_coords, a2t, atom_mask)

        assert out["pae_logits"].shape == (BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.n_pae_bins)
        assert out["pde_logits"].shape == (BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.n_pae_bins)
        assert out["plddt_logits"].shape == (BATCH_SIZE, N_TOKENS, TEST_CONFIG.max_atoms_per_token, TEST_CONFIG.n_plddt_bins)
        assert out["resolved_logits"].shape == (BATCH_SIZE, N_TOKENS, TEST_CONFIG.max_atoms_per_token, 2)

    def test_gradient_flow(self):
        head = ConfidenceHead(TEST_CONFIG).to(DEVICE)
        N_atoms = N_TOKENS * N_ATOMS_PER_TOKEN
        s_trunk = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE, requires_grad=True)
        z_trunk = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, requires_grad=True)
        s_inputs = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.c_s_inputs, device=DEVICE)
        pred_coords = torch.randn(BATCH_SIZE, N_atoms, 3, device=DEVICE)
        a2t = torch.arange(N_TOKENS, device=DEVICE).repeat_interleave(N_ATOMS_PER_TOKEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        atom_mask = torch.ones(BATCH_SIZE, N_atoms, device=DEVICE)

        out = head(s_trunk, z_trunk, s_inputs, pred_coords, a2t, atom_mask)
        loss = out["pae_logits"].sum() + out["plddt_logits"].sum()
        loss.backward()
        # Gradients should flow through the pairformer_stack via z_trunk/s_trunk
        # but s_trunk and z_trunk are detached inside forward — so check pairformer params instead
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in head.parameters())
        assert has_grad


# ============================================================================
# Affinity Module Tests
# ============================================================================

class TestAffinityModule:
    def test_output_shapes(self):
        module = AffinityModule(TEST_CONFIG).to(DEVICE)
        s = torch.randn(BATCH_SIZE, N_TOKENS, TEST_CONFIG.d_single, device=DEVICE, dtype=DTYPE)
        z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, TEST_CONFIG.d_pair, device=DEVICE, dtype=DTYPE)
        pocket_mask = torch.zeros(BATCH_SIZE, N_TOKENS, device=DEVICE, dtype=torch.bool)
        pocket_mask[:, :8] = True  # first 8 tokens are pocket

        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=DEVICE == "cuda"):
            out = module(s, z, pocket_mask)

        assert out["bind_logits"].shape == (BATCH_SIZE, 1)
        assert out["affinity"].shape == (BATCH_SIZE, 1)


# ============================================================================
# Confidence Score Tests
# ============================================================================

class TestConfidenceScores:
    def test_compute_plddt_shape(self):
        """pLDDT output shape and range."""
        B, N_tok, max_atoms, n_bins = 2, 16, 24, 50
        logits = torch.randn(B, N_tok, max_atoms, n_bins, device=DEVICE, dtype=DTYPE)
        plddt = compute_plddt(logits)
        assert plddt.shape == (B, N_tok, max_atoms)
        assert plddt.min() >= 0.0
        assert plddt.max() <= 100.0

    def test_compute_plddt_uniform(self):
        """Uniform logits should give pLDDT ~50."""
        B, N_tok, max_atoms, n_bins = 1, 8, 4, 50
        logits = torch.zeros(B, N_tok, max_atoms, n_bins, device=DEVICE, dtype=DTYPE)
        plddt = compute_plddt(logits)
        assert torch.allclose(plddt, torch.full_like(plddt, 50.0), atol=1.0)

    def test_compute_pae_shape(self):
        """PAE output shape and range."""
        B, N, n_bins = 2, 16, 64
        logits = torch.randn(B, N, N, n_bins, device=DEVICE, dtype=DTYPE)
        pae = compute_pae(logits)
        assert pae.shape == (B, N, N)
        assert pae.min() >= 0.0
        assert pae.max() <= 32.0

    def test_compute_ptm_shape(self):
        """pTM output shape and range."""
        B, N, n_bins = 2, 32, 64
        logits = torch.randn(B, N, N, n_bins, device=DEVICE, dtype=DTYPE)
        mask = torch.ones(B, N, device=DEVICE)
        ptm = compute_ptm(logits, mask)
        assert ptm.shape == (B,)
        assert (ptm >= 0.0).all()
        assert (ptm <= 1.0).all()

    def test_compute_ptm_perfect(self):
        """Near-zero PAE should give pTM close to 1.0."""
        B, N, n_bins = 1, 32, 64
        # Put all probability mass in the first bin (0.25 A error)
        logits = torch.full((B, N, N, n_bins), -1e6, device=DEVICE, dtype=torch.float32)
        logits[..., 0] = 0.0  # first bin center = 0.25 A
        mask = torch.ones(B, N, device=DEVICE)
        ptm = compute_ptm(logits, mask)
        assert ptm.item() > 0.95, f"Expected pTM > 0.95 for near-zero PAE, got {ptm.item():.3f}"

    def test_compute_iptm_single_chain(self):
        """Single chain should give ipTM = 0 (no inter-chain pairs)."""
        B, N, n_bins = 1, 16, 64
        logits = torch.randn(B, N, N, n_bins, device=DEVICE, dtype=DTYPE)
        chain_indices = torch.zeros(B, N, dtype=torch.long, device=DEVICE)
        mask = torch.ones(B, N, device=DEVICE)
        iptm = compute_iptm(logits, chain_indices, mask)
        assert iptm.shape == (B,)
        assert iptm.item() == 0.0, f"Expected ipTM=0 for single chain, got {iptm.item()}"

    def test_compute_iptm_multi_chain(self):
        """Multi-chain should give non-zero ipTM."""
        B, N, n_bins = 1, 16, 64
        # Put mass in first bin (low error → high TM)
        logits = torch.full((B, N, N, n_bins), -1e6, device=DEVICE, dtype=torch.float32)
        logits[..., 0] = 0.0
        chain_indices = torch.zeros(B, N, dtype=torch.long, device=DEVICE)
        chain_indices[:, N // 2:] = 1  # second half is chain 1
        mask = torch.ones(B, N, device=DEVICE)
        iptm = compute_iptm(logits, chain_indices, mask)
        assert iptm.item() > 0.5, f"Expected ipTM > 0.5 for low-error multi-chain, got {iptm.item()}"

    def test_compute_ranking_score(self):
        """Multi-chain formula = 0.8*iptm + 0.2*ptm."""
        ptm = torch.tensor([0.5, 0.8], device=DEVICE)
        iptm = torch.tensor([0.3, 0.6], device=DEVICE)
        has_interface = torch.tensor([True, False], device=DEVICE)
        ranking = compute_ranking_score(ptm, iptm, has_interface)
        assert ranking.shape == (2,)
        expected_0 = 0.8 * 0.3 + 0.2 * 0.5  # multi-chain
        expected_1 = 0.8  # single-chain → just ptm
        assert abs(ranking[0].item() - expected_0) < 1e-5
        assert abs(ranking[1].item() - expected_1) < 1e-5

    def test_flatten_plddt(self):
        """_flatten_plddt correctly maps per-token to per-atom."""
        B, N_tok, max_atoms = 1, 4, 3
        N_atoms = N_tok * max_atoms
        plddt = torch.arange(N_tok * max_atoms, device=DEVICE, dtype=DTYPE).reshape(B, N_tok, max_atoms) + 1.0
        atom_to_token = torch.arange(N_tok, device=DEVICE).repeat_interleave(max_atoms).unsqueeze(0)
        atoms_per_token = torch.full((B, N_tok), max_atoms, dtype=torch.long, device=DEVICE)
        atom_mask = torch.ones(B, N_atoms, device=DEVICE)
        flat = _flatten_plddt(plddt, atom_to_token, atoms_per_token, atom_mask)
        assert flat.shape == (B, N_atoms)
        # Should match the flattened plddt
        expected = plddt.reshape(B, -1)
        assert torch.allclose(flat.float(), expected.float(), atol=1e-3), f"Mismatch:\n{flat}\nvs\n{expected}"


# ============================================================================
# Full Model Tests
# ============================================================================

class TestFullModel:
    def test_forward_pass(self):
        """Full model forward pass with synthetic data."""
        model = Helico(TEST_CONFIG).to(DEVICE)
        batch = _make_batch()
        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=DEVICE == "cuda"):
            out = model(batch, compute_confidence=True, compute_affinity=False)

        assert "diffusion_loss" in out
        assert out["diffusion_loss"].dim() == 0
        assert out["diffusion_loss"].item() > 0
        assert "distogram_logits" in out
        assert "distogram_loss" in out

    def test_gradient_flow(self):
        """Verify gradients flow through the full model."""
        model = Helico(TEST_CONFIG).to(DEVICE)
        batch = _make_batch()
        out = model(batch, compute_confidence=False)
        loss = out["diffusion_loss"]
        loss.backward()
        # Check some gradients exist
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
                break
        else:
            pytest.fail("No gradients found in model parameters")


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestEndToEnd:
    def test_train_loss_decreases(self):
        """Train small model for a few steps, verify loss decreases."""
        torch.manual_seed(42)
        config = HelicoConfig(**{**TEST_CONFIG.__dict__, "n_diffusion_steps": 5})
        model = Helico(config).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch = _make_batch(n_tokens=8)

        losses = []
        for step in range(50):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=DTYPE, enabled=DEVICE == "cuda"):
                out = model(batch, compute_confidence=False)
            loss = out["diffusion_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Compare average of first 5 vs last 5
        initial_avg = sum(losses[:5]) / 5
        final_avg = sum(losses[-5:]) / 5
        assert final_avg < initial_avg, f"Loss did not decrease: {initial_avg:.4f} -> {final_avg:.4f}"

    def test_from_synthetic_structure(self):
        """Test model with data from synthetic structure pipeline."""
        structure = make_synthetic_structure(n_residues=12)
        tokenized = tokenize_structure(structure)
        features = tokenized.to_features()

        # Add batch dimension and required fields
        batch = {}
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(DEVICE)
            else:
                batch[k] = torch.tensor([v], device=DEVICE)

        # Add masks
        n_tok = features["n_tokens"]
        n_atoms = features["n_atoms"]
        batch["token_mask"] = torch.ones(1, n_tok, dtype=torch.bool, device=DEVICE)
        batch["atom_mask"] = torch.ones(1, n_atoms, dtype=torch.bool, device=DEVICE)
        # Add MSA features
        batch["msa_profile"] = torch.zeros(1, n_tok, 22, device=DEVICE)
        batch["cluster_msa"] = torch.zeros(1, 1, n_tok, dtype=torch.long, device=DEVICE)
        batch["cluster_profile"] = torch.zeros(1, 1, n_tok, 22, device=DEVICE)
        batch["has_msa"] = torch.zeros(1, device=DEVICE)

        model = Helico(TEST_CONFIG).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=DEVICE == "cuda"):
            out = model(batch, compute_confidence=True)

        assert "diffusion_loss" in out
        assert "pae_logits" in out

    def test_predict_returns_confidence(self):
        """predict() returns all confidence scores with correct shapes."""
        torch.manual_seed(42)
        config = HelicoConfig(**{**TEST_CONFIG.__dict__, "n_diffusion_steps": 3})
        model = Helico(config).to(DEVICE)
        batch = _make_batch(n_tokens=8)

        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=DEVICE == "cuda"):
            results = model.predict(batch, n_samples=1)

        B = BATCH_SIZE
        N_atoms = 8 * N_ATOMS_PER_TOKEN
        N_tok = 8

        # Check all expected keys
        assert "coords" in results
        assert "all_coords" in results
        assert "plddt" in results
        assert "pae" in results
        assert "ptm" in results
        assert "iptm" in results
        assert "ranking_score" in results
        assert "pae_logits" in results
        assert "plddt_logits" in results
        assert "pde_logits" in results

        # Check shapes
        assert results["coords"].shape == (B, N_atoms, 3)
        assert results["plddt"].shape == (B, N_atoms)
        assert results["pae"].shape == (B, N_tok, N_tok)
        assert results["ptm"].shape == (B,)
        assert results["iptm"].shape == (B,)
        assert results["ranking_score"].shape == (B,)

        # Check ranges
        assert results["plddt"].min() >= 0.0
        assert results["plddt"].max() <= 100.0
        assert results["pae"].min() >= 0.0
        assert (results["ptm"] >= 0.0).all()
        assert (results["ptm"] <= 1.0).all()


# ============================================================================
# Load Protenix Tests
# ============================================================================

PROTENIX_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "protenix_base_default_v1.0.0.pt"


@pytest.mark.skipif(not PROTENIX_CHECKPOINT.exists(), reason="Protenix checkpoint not downloaded")
class TestLoadProtenix:
    """Integration tests: load real Protenix checkpoint into Helico and run forward pass."""

    @pytest.fixture(scope="class")
    def protenix_sd(self):
        """Load the real Protenix state dict once for all tests in this class."""
        from collections import OrderedDict
        ckpt = torch.load(PROTENIX_CHECKPOINT, map_location="cpu", weights_only=False)
        sd = ckpt["model"]
        return OrderedDict((k.removeprefix("module."), v) for k, v in sd.items())

    @pytest.fixture(scope="class")
    def loaded_model(self, protenix_sd):
        """Load Protenix weights into a full-size Helico model."""
        from scripts.load_protenix import load_protenix_state_dict
        model = Helico(HelicoConfig())
        stats = load_protenix_state_dict(protenix_sd, model)
        return model, stats

    def test_transfer_stats(self, loaded_model):
        """All Protenix params transfer with zero mismatches."""
        _, stats = loaded_model
        assert stats["n_shape_mismatches"] == 0, f"Shape mismatches: {stats['shape_mismatches']}"
        assert stats["n_unmapped_protenix"] == 0, f"Unmapped PTX: {stats['unmapped_protenix']}"
        assert stats["n_skipped"] == 0
        # Total should be > 3315 (previous) now that we have all components
        assert stats["n_transferred"] > 3315, f"Only {stats['n_transferred']} transferred"

    def test_weights_actually_changed(self, loaded_model, protenix_sd):
        """Verify transferred weights are the actual Protenix values, not random init."""
        from scripts.load_protenix import (
            build_mapping, build_pairformer_mapping, build_msa_mapping,
            build_trunk_init_mapping, build_input_embedder_mapping,
            build_template_mapping, build_confidence_mapping, build_distogram_mapping,
        )
        model, _ = loaded_model
        mapping = build_mapping(protenix_sd)
        pf_direct, pf_concat = build_pairformer_mapping(protenix_sd)
        msa_direct, msa_concat = build_msa_mapping(protenix_sd)
        trunk_direct = build_trunk_init_mapping(protenix_sd)
        input_direct = build_input_embedder_mapping(protenix_sd)
        template_direct, template_concat = build_template_mapping(protenix_sd)
        confidence_direct, confidence_concat = build_confidence_mapping(protenix_sd)
        distogram_direct = build_distogram_mapping(protenix_sd)

        hf_sd = model.state_dict()

        n_checked = 0
        # Check all direct mappings
        all_direct = {
            **mapping, **pf_direct, **msa_direct,
            **trunk_direct, **input_direct, **template_direct,
            **confidence_direct, **distogram_direct,
        }
        for ptx_key, hf_key in all_direct.items():
            if hf_key in hf_sd and ptx_key in protenix_sd:
                assert torch.equal(hf_sd[hf_key].cpu(), protenix_sd[ptx_key].cpu()), (
                    f"Weight mismatch: {ptx_key} -> {hf_key}"
                )
                n_checked += 1

        # Check all concat mappings
        all_concat = {**pf_concat, **msa_concat, **template_concat, **confidence_concat}
        for hf_key, (ptx_keys, cat_dim) in all_concat.items():
            if hf_key in hf_sd:
                expected = torch.cat([protenix_sd[k].cpu() for k in ptx_keys], dim=cat_dim)
                assert torch.equal(hf_sd[hf_key].cpu(), expected), (
                    f"Concat weight mismatch: {ptx_keys} -> {hf_key}"
                )
                n_checked += 1

        assert n_checked > 3315  # Should be much more now

    def test_diffusion_forward_pass(self, loaded_model):
        """Run a real forward pass through the diffusion module with Protenix weights on GPU."""
        model, _ = loaded_model
        diffusion = model.diffusion.to(DEVICE)

        N_tok = 16
        N_atoms = N_tok * 4
        c = HelicoConfig()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE, enabled=DEVICE == "cuda"):
            gt_coords = torch.randn(1, N_atoms, 3, device=DEVICE)
            ref_pos = torch.randn(1, N_atoms, 3, device=DEVICE)
            ref_charge = torch.zeros(1, N_atoms, 1, device=DEVICE)
            ref_features = torch.randn(1, N_atoms, 385, device=DEVICE)
            a2t = torch.arange(N_tok, device=DEVICE).repeat_interleave(4).unsqueeze(0)
            atom_mask = torch.ones(1, N_atoms, device=DEVICE)
            s_trunk = torch.randn(1, N_tok, c.d_single, device=DEVICE)
            z_trunk = torch.randn(1, N_tok, N_tok, c.d_pair, device=DEVICE)
            s_inputs = torch.randn(1, N_tok, c.d_single + 65, device=DEVICE)
            relpe_feats = {
                "residue_index": torch.arange(N_tok, device=DEVICE).unsqueeze(0),
                "token_index": torch.zeros(1, N_tok, dtype=torch.long, device=DEVICE),
                "asym_id": torch.zeros(1, N_tok, dtype=torch.long, device=DEVICE),
                "entity_id": torch.zeros(1, N_tok, dtype=torch.long, device=DEVICE),
                "sym_id": torch.zeros(1, N_tok, dtype=torch.long, device=DEVICE),
            }

            x_denoised, gt, sigma = diffusion.forward_training(
                gt_coords=gt_coords, ref_pos=ref_pos, ref_charge=ref_charge,
                ref_features=ref_features, atom_to_token=a2t, atom_mask=atom_mask,
                s_trunk=s_trunk, z_trunk=z_trunk, s_inputs=s_inputs,
                relpe_feats=relpe_feats,
            )

        assert x_denoised.shape == (1, N_atoms, 3)
        assert torch.isfinite(x_denoised).all(), "Non-finite values in diffusion output"
        assert x_denoised.abs().max() > 0.01, "Output is near-zero — weights may not have loaded"
