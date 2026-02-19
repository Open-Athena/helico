#!/usr/bin/env python3
"""Load Protenix checkpoint weights into Helico's diffusion and pairformer modules.

Usage:
    python scripts/load_protenix.py --protenix-checkpoint path/to/model_v0.5.0.pt --output helico_from_protenix.pt

Programmatic:
    from scripts.load_protenix import load_protenix_checkpoint
    stats = load_protenix_checkpoint("model_v0.5.0.pt", helico_model)
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Protenix → Helico name mapping rules
# ---------------------------------------------------------------------------

# Top-level DiffusionModule mapping: Protenix prefix → Helico prefix
_TOP_LEVEL = {
    "diffusion_module.diffusion_conditioning": "diffusion.conditioning",
    "diffusion_module.atom_attention_encoder": "diffusion.atom_encoder",
    "diffusion_module.atom_attention_decoder": "diffusion.atom_decoder",
    "diffusion_module.diffusion_transformer": "diffusion.token_transformer",
    "diffusion_module.layernorm_s": "diffusion.s_to_token_norm",
    "diffusion_module.linear_no_bias_s": "diffusion.s_to_token_proj",
    "diffusion_module.layernorm_a": "diffusion.out_norm",
}

# Transition block (SwiGLU) sub-key mapping
_TRANSITION = {
    "layernorm1": "norm",
    "linear_no_bias_a": "linear_a",
    "linear_no_bias_b": "linear_b",
    "linear_no_bias": "linear_out",
}

# DiffusionConditioning sub-key mapping
_CONDITIONING = {
    "relpe.linear_no_bias": "relpe.linear_no_bias",
    "layernorm_z": "pair_norm",
    "linear_no_bias_z": "pair_proj",
    "transition_z1": "pair_transition_1",
    "transition_z2": "pair_transition_2",
    "fourier_embedding": "fourier",
    "layernorm_s": "single_norm",
    "linear_no_bias_s": "single_proj",
    "layernorm_n": "noise_norm",
    "linear_no_bias_n": "noise_proj",
    "transition_s1": "single_transition_1",
    "transition_s2": "single_transition_2",
}

# DiffusionTransformerBlock sub-key mapping
_TRANSFORMER_BLOCK = {
    "attention_pair_bias": "attention",
    "conditioned_transition_block": "transition",
}

# AttentionPairBias sub-key mapping
_ATTENTION_PAIR_BIAS = {
    "layernorm_a": "ada_ln_q",
    "layernorm_kv": "ada_ln_kv",
    "attention.linear_q": "q_proj",
    "attention.linear_k": "k_proj",
    "attention.linear_v": "v_proj",
    "attention.linear_g": "g_proj",
    "attention.linear_o": "out_proj",
    "layernorm_z": "z_norm",
    "linear_nobias_z": "z_proj",
    "linear_a_last": "s_gate.linear",
}

# AdaptiveLayerNorm sub-key mapping
_ADAPTIVE_LN = {
    "layernorm_a": "norm_a",
    "layernorm_s": "norm_s",
    "linear_s": "scale_proj",
    "linear_nobias_s": "shift_proj",
}

# ConditionedTransitionBlock sub-key mapping
_CONDITIONED_TRANSITION = {
    "adaln": "ada_ln",
    "linear_nobias_a1": "linear_a",
    "linear_nobias_a2": "linear_b",
    "linear_nobias_b": "linear_out",
    "linear_s": "s_gate.linear",
}

# AtomAttentionEncoder sub-key mapping
_ATOM_ENCODER = {
    "linear_no_bias_ref_pos": "ref_pos_proj",
    "linear_no_bias_ref_charge": "ref_charge_proj",
    "linear_no_bias_f": "ref_feat_proj",
    "linear_no_bias_r": "noisy_pos_proj",
    "linear_no_bias_d": "pair_dist_proj",
    "linear_no_bias_invd": "pair_inv_dist_proj",
    "linear_no_bias_v": "pair_valid_proj",
    "layernorm_s": "trunk_s_norm",
    "linear_no_bias_s": "trunk_s_proj",
    "layernorm_z": "trunk_z_norm",
    "linear_no_bias_z": "trunk_z_proj",
    "linear_no_bias_cl": "cross_pair_q",
    "linear_no_bias_cm": "cross_pair_k",
    "linear_no_bias_q": "agg_proj",
}

# AtomAttentionDecoder sub-key mapping
_ATOM_DECODER = {
    "linear_no_bias_a": "token_to_atom_proj",
    "layernorm_q": "out_norm",
    "linear_no_bias_out": "out_proj",
}

# Keys to skip (structural mismatches)
_SKIP_PREFIXES = (
)


def _map_adaptive_ln(suffix: str) -> str | None:
    """Map AdaptiveLayerNorm internal keys."""
    for ptx, hf in _ADAPTIVE_LN.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            # norm_a has no learnable params in Helico — skip
            if hf == "norm_a":
                return None
            return hf + rest
    return None


def _map_conditioned_transition(suffix: str) -> str | None:
    """Map ConditionedTransitionBlock internal keys."""
    for ptx, hf in _CONDITIONED_TRANSITION.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            if rest.startswith("."):
                # Recurse into adaln
                if hf == "ada_ln":
                    inner = _map_adaptive_ln(rest[1:])
                    if inner is None:
                        return None
                    return hf + "." + inner
                return hf + rest
            return hf + rest
    return None


def _map_attention_pair_bias(suffix: str) -> str | None:
    """Map AttentionPairBias internal keys."""
    for ptx, hf in _ATTENTION_PAIR_BIAS.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            if rest.startswith("."):
                # Recurse into adaptive LN
                if hf in ("ada_ln_q", "ada_ln_kv"):
                    inner = _map_adaptive_ln(rest[1:])
                    if inner is None:
                        return None
                    return hf + "." + inner
                return hf + rest
            return hf + rest
    return None


def _map_transition(suffix: str) -> str | None:
    """Map Transition (SwiGLU) internal keys."""
    for ptx, hf in _TRANSITION.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            return hf + rest
    return None


def _map_transformer_block(suffix: str) -> str | None:
    """Map DiffusionTransformerBlock internal keys."""
    for ptx, hf in _TRANSFORMER_BLOCK.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            if rest.startswith("."):
                inner_suffix = rest[1:]
                if hf == "attention":
                    mapped = _map_attention_pair_bias(inner_suffix)
                elif hf == "transition":
                    mapped = _map_conditioned_transition(inner_suffix)
                else:
                    mapped = None
                if mapped is None:
                    return None
                return hf + "." + mapped
            return hf + rest
    return None


def _map_transformer_blocks(suffix: str) -> str | None:
    """Map blocks.N.* inside a DiffusionTransformer.

    Protenix: diffusion_transformer.blocks.N.*
    Helico: blocks.N.*
    """
    if suffix.startswith("blocks."):
        # blocks.N.rest
        parts = suffix.split(".", 2)  # ['blocks', 'N', 'rest']
        if len(parts) == 3:
            block_idx = parts[1]
            block_rest = parts[2]
            mapped = _map_transformer_block(block_rest)
            if mapped is None:
                return None
            return f"blocks.{block_idx}.{mapped}"
    return None


def _map_small_mlp(suffix: str) -> str | None:
    """Map small_mlp (nn.Sequential) keys. Structure is identical, just rename."""
    # small_mlp.1.weight -> pair_mlp.1.weight, etc.
    return "pair_mlp" + suffix


def _map_atom_encoder(suffix: str) -> str | None:
    """Map AtomAttentionEncoder internal keys."""
    # Direct mappings
    for ptx, hf in _ATOM_ENCODER.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            return hf + rest

    # small_mlp -> pair_mlp
    if suffix.startswith("small_mlp"):
        rest = suffix[len("small_mlp"):]
        return "pair_mlp" + rest

    # atom_transformer.diffusion_transformer.blocks.N.* -> atom_transformer.blocks.N.*
    if suffix.startswith("atom_transformer.diffusion_transformer."):
        inner = suffix[len("atom_transformer.diffusion_transformer."):]
        mapped = _map_transformer_blocks(inner)
        if mapped is None:
            return None
        return "atom_transformer." + mapped

    return None


def _map_atom_decoder(suffix: str) -> str | None:
    """Map AtomAttentionDecoder internal keys."""
    for ptx, hf in _ATOM_DECODER.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            return hf + rest

    # atom_transformer.diffusion_transformer.blocks.N.* -> atom_transformer.blocks.N.*
    if suffix.startswith("atom_transformer.diffusion_transformer."):
        inner = suffix[len("atom_transformer.diffusion_transformer."):]
        mapped = _map_transformer_blocks(inner)
        if mapped is None:
            return None
        return "atom_transformer." + mapped

    return None


def _map_conditioning(suffix: str) -> str | None:
    """Map DiffusionConditioning internal keys."""
    for ptx, hf in _CONDITIONING.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            if rest.startswith("."):
                # Recurse into transitions
                if hf.endswith(("transition_1", "transition_2")):
                    inner = _map_transition(rest[1:])
                    if inner is None:
                        return None
                    return hf + "." + inner
                return hf + rest
            return hf + rest
    return None


def _map_token_transformer(suffix: str) -> str | None:
    """Map diffusion_transformer (token-level) keys."""
    return _map_transformer_blocks(suffix)


# ---------------------------------------------------------------------------
# Pairformer mapping rules
# ---------------------------------------------------------------------------

# TriangleMultiplicativeUpdate direct mappings (Protenix suffix → Helico suffix)
# Note: linear_g here is the OUTPUT gate, distinct from input gates linear_a_g/linear_b_g
_TRI_MUL_DIRECT = {
    "layer_norm_in": "layer_norm_in",
    "layer_norm_out": "layer_norm_out",
    "linear_z": "output_projection",
    "linear_g": "output_gate",
}

# TriangleAttention direct mappings
_TRI_ATT_DIRECT = {
    "layer_norm": "norm",
    "linear": "bias_proj",
    "mha.linear_o": "out_proj",
    "mha.linear_g": "gate",
}

# Pairformer SingleAttentionWithPairBias mappings (has_s=False, simple LayerNorm)
_PAIRFORMER_SINGLE_ATT = {
    "layernorm_a": "norm_s",
    "attention.linear_q": "q_proj",
    "attention.linear_k": "k_proj",
    "attention.linear_v": "v_proj",
    "attention.linear_o": "out_proj",
    "attention.linear_g": "gate",
    "layernorm_z": "norm_z",
    "linear_nobias_z": "z_proj",
}

# Concat rules: hf_suffix → ([ptx_suffixes], cat_dim)
_TRI_MUL_CONCAT = {
    "linear_p.weight": (["linear_a_p.weight", "linear_b_p.weight"], 0),
    "linear_g.weight": (["linear_a_g.weight", "linear_b_g.weight"], 0),
}

_TRI_ATT_CONCAT = {
    "qkv_proj.weight": (["mha.linear_q.weight", "mha.linear_k.weight", "mha.linear_v.weight"], 0),
}

# Pairformer block sub-module dispatch: ptx_sub → (hf_sub, mapper)
_PAIRFORMER_BLOCK_MODULES = {
    "tri_mul_out": ("tri_mul_out", None),       # uses _map_tri_mul
    "tri_mul_in": ("tri_mul_in", None),         # uses _map_tri_mul
    "tri_att_start": ("tri_att_start", None),   # uses _map_tri_att
    "tri_att_end": ("tri_att_end", None),       # uses _map_tri_att
    "pair_transition": ("pair_transition", _map_transition),
    "attention_pair_bias": ("single_attention", None),  # uses _map_pairformer_single_att
    "single_transition": ("single_transition", _map_transition),
}

# Which sub-modules have concat rules: (ptx_sub, hf_sub, rules)
_PAIRFORMER_CONCAT_MODULES = [
    ("tri_mul_out", "tri_mul_out", _TRI_MUL_CONCAT),
    ("tri_mul_in", "tri_mul_in", _TRI_MUL_CONCAT),
    ("tri_att_start", "tri_att_start", _TRI_ATT_CONCAT),
    ("tri_att_end", "tri_att_end", _TRI_ATT_CONCAT),
]


def _map_tri_mul(suffix: str) -> str | None:
    """Map TriangleMultiplicativeUpdate sub-keys. Returns None for concat keys."""
    for ptx, hf in _TRI_MUL_DIRECT.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            return hf + rest
    # Concat keys (linear_a_p, linear_b_p, linear_a_g, linear_b_g) handled separately
    return None


def _map_tri_att(suffix: str) -> str | None:
    """Map TriangleAttention sub-keys. Returns None for concat keys."""
    for ptx, hf in _TRI_ATT_DIRECT.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            return hf + rest
    # Concat keys (mha.linear_q, mha.linear_k, mha.linear_v) handled separately
    return None


def _map_pairformer_single_att(suffix: str) -> str | None:
    """Map pairformer SingleAttentionWithPairBias sub-keys (simple LayerNorm, no AdaLN)."""
    for ptx, hf in _PAIRFORMER_SINGLE_ATT.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            return hf + rest
    return None


def _get_pairformer_mapper(ptx_sub: str):
    """Return the mapper function for a pairformer sub-module."""
    if ptx_sub in ("tri_mul_out", "tri_mul_in"):
        return _map_tri_mul
    elif ptx_sub in ("tri_att_start", "tri_att_end"):
        return _map_tri_att
    elif ptx_sub == "attention_pair_bias":
        return _map_pairformer_single_att
    elif ptx_sub in ("pair_transition", "single_transition"):
        return _map_transition
    return None


def build_pairformer_mapping(
    protenix_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, str], dict[str, tuple[list[str], int]]]:
    """Build pairformer weight mapping with concatenation support.

    Returns:
        direct: {ptx_key: hf_key} for 1:1 mappings
        concat: {hf_key: ([ptx_key1, ...], cat_dim)} for concatenation
    """
    direct = {}
    block_indices = set()

    for ptx_key in protenix_sd:
        if not ptx_key.startswith("pairformer_stack.blocks."):
            continue

        # pairformer_stack.blocks.0.tri_mul_out.layer_norm_in.weight
        parts = ptx_key.split(".", 3)  # ['pairformer_stack', 'blocks', '0', rest]
        if len(parts) < 4:
            continue
        block_idx = parts[2]
        block_rest = parts[3]
        block_indices.add(block_idx)

        hf_block = f"pairformer.blocks.{block_idx}"

        # Try each sub-module
        for ptx_sub, (hf_sub, _) in _PAIRFORMER_BLOCK_MODULES.items():
            if block_rest.startswith(ptx_sub + "."):
                inner = block_rest[len(ptx_sub) + 1:]
                mapper = _get_pairformer_mapper(ptx_sub)
                mapped = mapper(inner)
                if mapped is not None:
                    direct[ptx_key] = f"{hf_block}.{hf_sub}.{mapped}"
                # None means concat key — handled below
                break

    # Build concat groups
    concat = {}
    for block_idx in sorted(block_indices, key=int):
        ptx_block = f"pairformer_stack.blocks.{block_idx}"
        hf_block = f"pairformer.blocks.{block_idx}"

        for ptx_sub, hf_sub, rules in _PAIRFORMER_CONCAT_MODULES:
            for hf_suffix, (ptx_suffixes, cat_dim) in rules.items():
                hf_key = f"{hf_block}.{hf_sub}.{hf_suffix}"
                ptx_keys = [f"{ptx_block}.{ptx_sub}.{s}" for s in ptx_suffixes]
                if all(k in protenix_sd for k in ptx_keys):
                    concat[hf_key] = (ptx_keys, cat_dim)

    return direct, concat


# ---------------------------------------------------------------------------
# MSA module mapping rules
# ---------------------------------------------------------------------------

# OuterProductMean direct mappings
_OPM_DIRECT = {
    "layer_norm": "norm",
    "linear_1": "linear_1",
    "linear_2": "linear_2",
    "linear_out": "linear_out",
}

# MSAPairWeightedAveraging direct mappings
_MSA_PAIR_AVG = {
    "layernorm_m": "layernorm_m",
    "linear_no_bias_mv": "linear_mv",
    "layernorm_z": "layernorm_z",
    "linear_no_bias_z": "linear_z",
    "linear_no_bias_mg": "linear_mg",
    "linear_no_bias_out": "linear_out",
}

# MSA block sub-module dispatch (pair_stack reuses existing pairformer mappers)
_MSA_PAIR_STACK_MODULES = {
    "tri_mul_out": ("tri_mul_out", None),
    "tri_mul_in": ("tri_mul_in", None),
    "tri_att_start": ("tri_att_start", None),
    "tri_att_end": ("tri_att_end", None),
    "pair_transition": ("pair_transition", _map_transition),
}

# MSA pair_stack concat rules (same as pairformer)
_MSA_PAIR_STACK_CONCAT_MODULES = [
    ("tri_mul_out", "tri_mul_out", _TRI_MUL_CONCAT),
    ("tri_mul_in", "tri_mul_in", _TRI_MUL_CONCAT),
    ("tri_att_start", "tri_att_start", _TRI_ATT_CONCAT),
    ("tri_att_end", "tri_att_end", _TRI_ATT_CONCAT),
]


def _map_opm(suffix: str) -> str | None:
    """Map OuterProductMean sub-keys."""
    for ptx, hf in _OPM_DIRECT.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            return hf + rest
    return None


def _map_msa_pair_avg(suffix: str) -> str | None:
    """Map MSAPairWeightedAveraging sub-keys."""
    for ptx, hf in _MSA_PAIR_AVG.items():
        if suffix.startswith(ptx):
            rest = suffix[len(ptx):]
            return hf + rest
    return None


def _get_msa_pair_stack_mapper(ptx_sub: str):
    """Return the mapper function for an MSA pair_stack sub-module."""
    if ptx_sub in ("tri_mul_out", "tri_mul_in"):
        return _map_tri_mul
    elif ptx_sub in ("tri_att_start", "tri_att_end"):
        return _map_tri_att
    elif ptx_sub == "pair_transition":
        return _map_transition
    return None


def build_msa_mapping(
    protenix_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, str], dict[str, tuple[list[str], int]]]:
    """Build MSA module weight mapping with concatenation support.

    Returns:
        direct: {ptx_key: hf_key} for 1:1 mappings
        concat: {hf_key: ([ptx_key1, ...], cat_dim)} for concatenation
    """
    direct = {}
    block_indices = set()

    for ptx_key in protenix_sd:
        if not ptx_key.startswith("msa_module."):
            continue

        suffix = ptx_key[len("msa_module."):]

        # Module-level projections
        if suffix.startswith("linear_no_bias_m."):
            rest = suffix[len("linear_no_bias_m."):]
            direct[ptx_key] = f"msa_module.linear_m.{rest}"
            continue
        if suffix.startswith("linear_no_bias_s."):
            rest = suffix[len("linear_no_bias_s."):]
            direct[ptx_key] = f"msa_module.linear_s.{rest}"
            continue

        # Block-level: msa_module.blocks.N.*
        if not suffix.startswith("blocks."):
            continue

        parts = suffix.split(".", 2)  # ['blocks', 'N', rest]
        if len(parts) < 3:
            continue
        block_idx = parts[1]
        block_rest = parts[2]
        block_indices.add(block_idx)

        hf_block = f"msa_module.blocks.{block_idx}"

        # OuterProductMean: outer_product_mean_msa.*
        if block_rest.startswith("outer_product_mean_msa."):
            inner = block_rest[len("outer_product_mean_msa."):]
            mapped = _map_opm(inner)
            if mapped is not None:
                direct[ptx_key] = f"{hf_block}.opm.{mapped}"
            continue

        # MSA stack: msa_stack.*
        if block_rest.startswith("msa_stack."):
            stack_rest = block_rest[len("msa_stack."):]

            # MSAPairWeightedAveraging: msa_pair_weighted_averaging.*
            if stack_rest.startswith("msa_pair_weighted_averaging."):
                inner = stack_rest[len("msa_pair_weighted_averaging."):]
                mapped = _map_msa_pair_avg(inner)
                if mapped is not None:
                    direct[ptx_key] = f"{hf_block}.msa_stack.pair_avg.{mapped}"
                continue

            # MSA Transition: transition_m.*
            if stack_rest.startswith("transition_m."):
                inner = stack_rest[len("transition_m."):]
                mapped = _map_transition(inner)
                if mapped is not None:
                    direct[ptx_key] = f"{hf_block}.msa_stack.transition.{mapped}"
                continue

            continue

        # Pair stack: pair_stack.*
        if block_rest.startswith("pair_stack."):
            pair_rest = block_rest[len("pair_stack."):]

            for ptx_sub, (hf_sub, _) in _MSA_PAIR_STACK_MODULES.items():
                if pair_rest.startswith(ptx_sub + "."):
                    inner = pair_rest[len(ptx_sub) + 1:]
                    mapper = _get_msa_pair_stack_mapper(ptx_sub)
                    mapped = mapper(inner)
                    if mapped is not None:
                        direct[ptx_key] = f"{hf_block}.pair_stack.{hf_sub}.{mapped}"
                    break

            continue

    # Build concat groups for pair_stack
    concat = {}
    for block_idx in sorted(block_indices, key=int):
        ptx_block = f"msa_module.blocks.{block_idx}"
        hf_block = f"msa_module.blocks.{block_idx}"

        for ptx_sub, hf_sub, rules in _MSA_PAIR_STACK_CONCAT_MODULES:
            for hf_suffix, (ptx_suffixes, cat_dim) in rules.items():
                hf_key = f"{hf_block}.pair_stack.{hf_sub}.{hf_suffix}"
                ptx_keys = [f"{ptx_block}.pair_stack.{ptx_sub}.{s}" for s in ptx_suffixes]
                if all(k in protenix_sd for k in ptx_keys):
                    concat[hf_key] = (ptx_keys, cat_dim)

    return direct, concat


def build_mapping(protenix_sd: dict[str, torch.Tensor]) -> dict[str, str]:
    """Build Protenix key → Helico key mapping for all diffusion_module.* parameters.

    Returns:
        mapping: dict of protenix_key → helico_key
    """
    mapping = {}

    for ptx_key in protenix_sd:
        if not ptx_key.startswith("diffusion_module."):
            continue

        # Check skip list
        if any(ptx_key.startswith(skip) for skip in _SKIP_PREFIXES):
            continue

        # Try top-level exact matches first (e.g. layernorm_s, linear_no_bias_s)
        matched = False
        for ptx_prefix, hf_prefix in _TOP_LEVEL.items():
            if ptx_key.startswith(ptx_prefix):
                rest = ptx_key[len(ptx_prefix):]
                if rest.startswith("."):
                    inner_suffix = rest[1:]

                    # Compound modules: dispatch to sub-mapper
                    if hf_prefix == "diffusion.conditioning":
                        mapped = _map_conditioning(inner_suffix)
                    elif hf_prefix == "diffusion.atom_encoder":
                        mapped = _map_atom_encoder(inner_suffix)
                    elif hf_prefix == "diffusion.atom_decoder":
                        mapped = _map_atom_decoder(inner_suffix)
                    elif hf_prefix == "diffusion.token_transformer":
                        mapped = _map_token_transformer(inner_suffix)
                    else:
                        # Leaf module (LayerNorm, Linear): rest is .weight/.bias
                        mapped = None
                        mapping[ptx_key] = hf_prefix + rest
                        matched = True
                        break

                    if mapped is not None:
                        mapping[ptx_key] = hf_prefix + "." + mapped
                    matched = True
                    break
                else:
                    # Direct param (e.g. layernorm_s.weight -> s_to_token_norm.weight)
                    mapping[ptx_key] = hf_prefix + rest
                    matched = True
                    break

        if not matched:
            pass  # Key will show up as unmapped

    return mapping


# ---------------------------------------------------------------------------
# Trunk init / Recycling / RelPE mapping
# ---------------------------------------------------------------------------

_TRUNK_INIT_DIRECT = {
    "linear_no_bias_sinit.weight": "linear_sinit.weight",
    "linear_no_bias_zinit1.weight": "linear_zinit1.weight",
    "linear_no_bias_zinit2.weight": "linear_zinit2.weight",
    "linear_no_bias_token_bond.weight": "linear_token_bond.weight",
    "layernorm_s.weight": "layernorm_s.weight",
    "layernorm_s.bias": "layernorm_s.bias",
    "linear_no_bias_s.weight": "linear_s.weight",
    "layernorm_z_cycle.weight": "layernorm_z_cycle.weight",
    "layernorm_z_cycle.bias": "layernorm_z_cycle.bias",
    "linear_no_bias_z_cycle.weight": "linear_z_cycle.weight",
    "relative_position_encoding.linear_no_bias.weight": "trunk_relpe.linear_no_bias.weight",
}


def build_trunk_init_mapping(protenix_sd: dict[str, torch.Tensor]) -> dict[str, str]:
    """Map trunk initialization, recycling, and RelPE keys (11 direct)."""
    direct = {}
    for ptx_suffix, hf_key in _TRUNK_INIT_DIRECT.items():
        if ptx_suffix in protenix_sd:
            direct[ptx_suffix] = hf_key
    return direct


# ---------------------------------------------------------------------------
# Input Embedder mapping (AtomAttentionEncoder with has_coords=False)
# ---------------------------------------------------------------------------

# Same as diffusion encoder but WITHOUT: noisy_pos_proj, trunk_s_*, trunk_z_*
_INPUT_ENCODER_DIRECT = {
    "linear_no_bias_ref_pos": "ref_pos_proj",
    "linear_no_bias_ref_charge": "ref_charge_proj",
    "linear_no_bias_f": "ref_feat_proj",
    "linear_no_bias_d": "pair_dist_proj",
    "linear_no_bias_invd": "pair_inv_dist_proj",
    "linear_no_bias_v": "pair_valid_proj",
    "linear_no_bias_cl": "cross_pair_q",
    "linear_no_bias_cm": "cross_pair_k",
    "linear_no_bias_q": "agg_proj",
}


def build_input_embedder_mapping(protenix_sd: dict[str, torch.Tensor]) -> dict[str, str]:
    """Map input_embedder.atom_attention_encoder.* keys."""
    direct = {}
    prefix = "input_embedder.atom_attention_encoder."

    for ptx_key in protenix_sd:
        if not ptx_key.startswith(prefix):
            continue

        suffix = ptx_key[len(prefix):]
        hf_prefix = "input_embedder.atom_attention_encoder."

        # Direct mappings
        for ptx_sub, hf_sub in _INPUT_ENCODER_DIRECT.items():
            if suffix.startswith(ptx_sub):
                rest = suffix[len(ptx_sub):]
                direct[ptx_key] = hf_prefix + hf_sub + rest
                break
        else:
            # small_mlp -> pair_mlp
            if suffix.startswith("small_mlp"):
                rest = suffix[len("small_mlp"):]
                direct[ptx_key] = hf_prefix + "pair_mlp" + rest
            # atom_transformer.diffusion_transformer.blocks.N.*
            elif suffix.startswith("atom_transformer.diffusion_transformer."):
                inner = suffix[len("atom_transformer.diffusion_transformer."):]
                mapped = _map_transformer_blocks(inner)
                if mapped is not None:
                    direct[ptx_key] = hf_prefix + "atom_transformer." + mapped

    return direct


# ---------------------------------------------------------------------------
# Template Embedder mapping
# ---------------------------------------------------------------------------

_TEMPLATE_TOP = {
    "template_embedder.layernorm_z": "template_embedder.z_norm",
    "template_embedder.linear_no_bias_z": "template_embedder.linear_z",
    "template_embedder.linear_no_bias_a": "template_embedder.linear_a",
    "template_embedder.layernorm_v": "template_embedder.out_norm",
    "template_embedder.linear_no_bias_u": "template_embedder.linear_out",
}


def build_template_mapping(
    protenix_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, str], dict[str, tuple[list[str], int]]]:
    """Map template_embedder.* keys."""
    direct = {}
    block_indices = set()

    for ptx_key in protenix_sd:
        if not ptx_key.startswith("template_embedder."):
            continue

        # Top-level direct mappings
        matched = False
        for ptx_prefix, hf_prefix in _TEMPLATE_TOP.items():
            if ptx_key.startswith(ptx_prefix):
                rest = ptx_key[len(ptx_prefix):]
                direct[ptx_key] = hf_prefix + rest
                matched = True
                break
        if matched:
            continue

        # Pairformer blocks: template_embedder.pairformer_stack.blocks.N.*
        pf_prefix = "template_embedder.pairformer_stack.blocks."
        if ptx_key.startswith(pf_prefix):
            rest = ptx_key[len(pf_prefix):]
            parts = rest.split(".", 1)
            if len(parts) == 2:
                block_idx = parts[0]
                block_rest = parts[1]
                block_indices.add(block_idx)
                hf_block = f"template_embedder.pairformer_stack.{block_idx}"

                # pair_stack modules only (no single)
                for ptx_sub, (hf_sub, _) in _MSA_PAIR_STACK_MODULES.items():
                    if block_rest.startswith(ptx_sub + "."):
                        inner = block_rest[len(ptx_sub) + 1:]
                        mapper = _get_msa_pair_stack_mapper(ptx_sub)
                        mapped = mapper(inner)
                        if mapped is not None:
                            direct[ptx_key] = f"{hf_block}.{hf_sub}.{mapped}"
                        break

    # Concat rules for template pairformer blocks
    concat = {}
    for block_idx in sorted(block_indices, key=int):
        ptx_block = f"template_embedder.pairformer_stack.blocks.{block_idx}"
        hf_block = f"template_embedder.pairformer_stack.{block_idx}"

        for ptx_sub, hf_sub, rules in _MSA_PAIR_STACK_CONCAT_MODULES:
            for hf_suffix, (ptx_suffixes, cat_dim) in rules.items():
                hf_key = f"{hf_block}.{hf_sub}.{hf_suffix}"
                ptx_keys = [f"{ptx_block}.{ptx_sub}.{s}" for s in ptx_suffixes]
                if all(k in protenix_sd for k in ptx_keys):
                    concat[hf_key] = (ptx_keys, cat_dim)

    return direct, concat


# ---------------------------------------------------------------------------
# Confidence Head mapping
# ---------------------------------------------------------------------------

_CONFIDENCE_TOP = {
    "confidence_head.input_strunk_ln": "confidence_head.input_s_norm",
    "confidence_head.linear_no_bias_s1": "confidence_head.linear_s1",
    "confidence_head.linear_no_bias_s2": "confidence_head.linear_s2",
    # Longer prefix must come before shorter to avoid partial match
    "confidence_head.linear_no_bias_d_wo_onehot": "confidence_head.linear_d_raw",
    "confidence_head.linear_no_bias_d": "confidence_head.linear_d",
    "confidence_head.pae_ln": "confidence_head.pae_norm",
    "confidence_head.pde_ln": "confidence_head.pde_norm",
    "confidence_head.plddt_ln": "confidence_head.plddt_norm",
    "confidence_head.resolved_ln": "confidence_head.resolved_norm",
    "confidence_head.linear_no_bias_pae": "confidence_head.linear_pae",
    "confidence_head.linear_no_bias_pde": "confidence_head.linear_pde",
}

# Direct params (no prefix stripping needed)
_CONFIDENCE_DIRECT_PARAMS = {
    "confidence_head.plddt_weight": "confidence_head.plddt_weight",
    "confidence_head.resolved_weight": "confidence_head.resolved_weight",
    "confidence_head.lower_bins": "confidence_head.lower_bins",
    "confidence_head.upper_bins": "confidence_head.upper_bins",
}


def build_confidence_mapping(
    protenix_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, str], dict[str, tuple[list[str], int]]]:
    """Map confidence_head.* keys."""
    direct = {}
    block_indices = set()

    for ptx_key in protenix_sd:
        if not ptx_key.startswith("confidence_head."):
            continue

        # Direct param passthrough
        if ptx_key in _CONFIDENCE_DIRECT_PARAMS:
            direct[ptx_key] = _CONFIDENCE_DIRECT_PARAMS[ptx_key]
            continue

        # Top-level prefix mappings
        matched = False
        for ptx_prefix, hf_prefix in _CONFIDENCE_TOP.items():
            if ptx_key.startswith(ptx_prefix):
                rest = ptx_key[len(ptx_prefix):]
                direct[ptx_key] = hf_prefix + rest
                matched = True
                break
        if matched:
            continue

        # Pairformer blocks: confidence_head.pairformer_stack.blocks.N.*
        pf_prefix = "confidence_head.pairformer_stack.blocks."
        if ptx_key.startswith(pf_prefix):
            rest = ptx_key[len(pf_prefix):]
            parts = rest.split(".", 1)
            if len(parts) == 2:
                block_idx = parts[0]
                block_rest = parts[1]
                block_indices.add(block_idx)
                hf_block = f"confidence_head.pairformer_stack.blocks.{block_idx}"

                # Full pairformer (with single) — same as trunk pairformer
                for ptx_sub, (hf_sub, _) in _PAIRFORMER_BLOCK_MODULES.items():
                    if block_rest.startswith(ptx_sub + "."):
                        inner = block_rest[len(ptx_sub) + 1:]
                        mapper = _get_pairformer_mapper(ptx_sub)
                        mapped = mapper(inner)
                        if mapped is not None:
                            direct[ptx_key] = f"{hf_block}.{hf_sub}.{mapped}"
                        break

    # Concat rules for confidence pairformer blocks
    concat = {}
    for block_idx in sorted(block_indices, key=int):
        ptx_block = f"confidence_head.pairformer_stack.blocks.{block_idx}"
        hf_block = f"confidence_head.pairformer_stack.blocks.{block_idx}"

        for ptx_sub, hf_sub, rules in _PAIRFORMER_CONCAT_MODULES:
            for hf_suffix, (ptx_suffixes, cat_dim) in rules.items():
                hf_key = f"{hf_block}.{hf_sub}.{hf_suffix}"
                ptx_keys = [f"{ptx_block}.{ptx_sub}.{s}" for s in ptx_suffixes]
                if all(k in protenix_sd for k in ptx_keys):
                    concat[hf_key] = (ptx_keys, cat_dim)

    return direct, concat


# ---------------------------------------------------------------------------
# Distogram Head mapping
# ---------------------------------------------------------------------------

_DISTOGRAM_DIRECT = {
    "distogram_head.linear.weight": "distogram_head.linear.weight",
    "distogram_head.linear.bias": "distogram_head.linear.bias",
}


def build_distogram_mapping(protenix_sd: dict[str, torch.Tensor]) -> dict[str, str]:
    """Map distogram_head.* keys (2 direct)."""
    direct = {}
    for ptx_key, hf_key in _DISTOGRAM_DIRECT.items():
        if ptx_key in protenix_sd:
            direct[ptx_key] = hf_key
    return direct


def load_protenix_checkpoint(
    protenix_path: str | Path,
    helico_model: torch.nn.Module,
    strict: bool = False,
) -> dict[str, object]:
    """Load Protenix diffusion weights into a Helico model.

    Args:
        protenix_path: Path to Protenix checkpoint (.pt)
        helico_model: Instantiated Helico model
        strict: If True, raise on any unmapped keys

    Returns:
        dict with stats: transferred, skipped, shape_mismatch, unmapped_protenix, unmapped_helico
    """
    checkpoint = torch.load(protenix_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if "model" in checkpoint:
        ptx_sd = checkpoint["model"]
    elif "state_dict" in checkpoint:
        ptx_sd = checkpoint["state_dict"]
    else:
        ptx_sd = checkpoint

    # Strip DDP "module." prefix if present
    ptx_sd = OrderedDict(
        (k.removeprefix("module."), v) for k, v in ptx_sd.items()
    )

    return load_protenix_state_dict(ptx_sd, helico_model, strict=strict)


def load_protenix_state_dict(
    ptx_sd: dict[str, torch.Tensor],
    helico_model: torch.nn.Module,
    strict: bool = False,
) -> dict[str, object]:
    """Transfer weights from a Protenix state dict into a Helico model.

    Handles both diffusion_module.* and pairformer_stack.* keys,
    including concatenation of split projections.

    Args:
        ptx_sd: Protenix state dict (already stripped of "module." prefix)
        helico_model: Instantiated Helico model
        strict: If True, raise on any unmapped keys

    Returns:
        dict with stats
    """
    # Build all mappings
    diffusion_mapping = build_mapping(ptx_sd)
    pf_direct, pf_concat = build_pairformer_mapping(ptx_sd)
    msa_direct, msa_concat = build_msa_mapping(ptx_sd)
    trunk_direct = build_trunk_init_mapping(ptx_sd)
    input_direct = build_input_embedder_mapping(ptx_sd)
    template_direct, template_concat = build_template_mapping(ptx_sd)
    confidence_direct, confidence_concat = build_confidence_mapping(ptx_sd)
    distogram_direct = build_distogram_mapping(ptx_sd)

    # Merge all direct mappings
    all_direct = {
        **diffusion_mapping, **pf_direct, **msa_direct,
        **trunk_direct, **input_direct, **template_direct,
        **confidence_direct, **distogram_direct,
    }
    all_concat = {**pf_concat, **msa_concat, **template_concat, **confidence_concat}

    hf_sd = helico_model.state_dict()

    transferred = []
    skipped = []
    shape_mismatches = []
    unmapped_protenix = []

    # Relevant Protenix key prefixes
    _RELEVANT_PREFIXES = (
        "diffusion_module.", "pairformer_stack.blocks.", "msa_module.",
        "input_embedder.", "template_embedder.", "confidence_head.",
        "distogram_head.", "linear_no_bias_sinit.", "linear_no_bias_zinit",
        "linear_no_bias_token_bond.", "layernorm_s.", "linear_no_bias_s.",
        "layernorm_z_cycle.", "linear_no_bias_z_cycle.",
        "relative_position_encoding.",
    )
    ptx_relevant_keys = [
        k for k in ptx_sd if any(k.startswith(p) for p in _RELEVANT_PREFIXES)
    ]

    consumed_ptx_keys = set()

    # --- Direct mappings ---
    for ptx_key in ptx_relevant_keys:
        if any(ptx_key.startswith(skip) for skip in _SKIP_PREFIXES):
            skipped.append(ptx_key)
            consumed_ptx_keys.add(ptx_key)
            continue

        if ptx_key not in all_direct:
            continue

        consumed_ptx_keys.add(ptx_key)
        hf_key = all_direct[ptx_key]

        if hf_key not in hf_sd:
            unmapped_protenix.append(f"{ptx_key} -> {hf_key} (not in Helico)")
            continue

        ptx_tensor = ptx_sd[ptx_key]
        hf_tensor = hf_sd[hf_key]

        if ptx_tensor.shape != hf_tensor.shape:
            shape_mismatches.append(
                f"{ptx_key} {tuple(ptx_tensor.shape)} -> {hf_key} {tuple(hf_tensor.shape)}"
            )
            continue

        hf_sd[hf_key] = ptx_tensor
        transferred.append(f"{ptx_key} -> {hf_key}")

    # --- Concat mappings (pairformer + MSA split projections) ---
    for hf_key, (ptx_keys, cat_dim) in all_concat.items():
        for k in ptx_keys:
            consumed_ptx_keys.add(k)

        if hf_key not in hf_sd:
            unmapped_protenix.append(f"{ptx_keys} -> {hf_key} (not in Helico)")
            continue

        catted = torch.cat([ptx_sd[k] for k in ptx_keys], dim=cat_dim)
        hf_tensor = hf_sd[hf_key]

        if catted.shape != hf_tensor.shape:
            shape_mismatches.append(
                f"{ptx_keys} {tuple(catted.shape)} -> {hf_key} {tuple(hf_tensor.shape)}"
            )
            continue

        hf_sd[hf_key] = catted
        transferred.append(f"{ptx_keys} -> {hf_key} (concat)")

    # Load the modified state dict
    helico_model.load_state_dict(hf_sd, strict=False)

    # Unmapped Protenix keys (relevant keys not consumed by any mapping)
    for k in ptx_relevant_keys:
        if k not in consumed_ptx_keys:
            unmapped_protenix.append(k)

    # Unmapped Helico params
    _HF_RELEVANT_PREFIXES = (
        "diffusion.", "pairformer.", "msa_module.",
        "input_embedder.", "template_embedder.", "confidence_head.",
        "distogram_head.", "linear_sinit.", "linear_zinit",
        "linear_token_bond.", "layernorm_s.", "linear_s.",
        "layernorm_z_cycle.", "linear_z_cycle.", "trunk_relpe.",
    )
    mapped_hf_keys = set(all_direct.values()) | set(all_concat.keys())
    unmapped_helico = [
        k for k in hf_sd
        if any(k.startswith(p) for p in _HF_RELEVANT_PREFIXES)
        and k not in mapped_hf_keys
    ]

    stats = {
        "transferred": transferred,
        "skipped": skipped,
        "shape_mismatches": shape_mismatches,
        "unmapped_protenix": unmapped_protenix,
        "unmapped_helico": unmapped_helico,
        "n_transferred": len(transferred),
        "n_skipped": len(skipped),
        "n_shape_mismatches": len(shape_mismatches),
        "n_unmapped_protenix": len(unmapped_protenix),
        "n_unmapped_helico": len(unmapped_helico),
    }

    if strict and (unmapped_protenix or shape_mismatches):
        raise ValueError(
            f"Strict mode: {len(unmapped_protenix)} unmapped, "
            f"{len(shape_mismatches)} shape mismatches"
        )

    return stats


def print_stats(stats: dict[str, object]) -> None:
    """Print a human-readable summary of transfer stats."""
    print(f"\n{'='*60}")
    print(f"Protenix → Helico Weight Transfer Summary")
    print(f"{'='*60}")
    print(f"  Transferred:       {stats['n_transferred']}")
    print(f"  Skipped (known):   {stats['n_skipped']}")
    print(f"  Shape mismatches:  {stats['n_shape_mismatches']}")
    print(f"  Unmapped Protenix: {stats['n_unmapped_protenix']}")
    print(f"  Unmapped Helico: {stats['n_unmapped_helico']}")
    print(f"{'='*60}")

    if stats["shape_mismatches"]:
        print("\nShape mismatches:")
        for s in stats["shape_mismatches"]:
            print(f"  {s}")

    if stats["unmapped_protenix"]:
        print("\nUnmapped Protenix keys:")
        for s in stats["unmapped_protenix"]:
            print(f"  {s}")

    if stats["unmapped_helico"]:
        print("\nUnmapped Helico keys:")
        for s in stats["unmapped_helico"]:
            print(f"  {s}")


def main():
    parser = argparse.ArgumentParser(
        description="Load Protenix checkpoint weights into Helico"
    )
    parser.add_argument(
        "--protenix-checkpoint", required=True, type=str,
        help="Path to Protenix .pt checkpoint"
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Output path for Helico checkpoint with transferred weights"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Raise on unmapped keys or shape mismatches"
    )
    args = parser.parse_args()

    # Import here to avoid circular imports when used as library
    from helico.model import Helico, HelicoConfig

    print("Instantiating Helico model with default config...")
    config = HelicoConfig()
    model = Helico(config)

    print(f"Loading Protenix checkpoint from {args.protenix_checkpoint}...")
    stats = load_protenix_checkpoint(args.protenix_checkpoint, model, strict=args.strict)

    print_stats(stats)

    print(f"\nSaving Helico checkpoint to {args.output}...")
    torch.save({"model": model.state_dict(), "config": config}, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
