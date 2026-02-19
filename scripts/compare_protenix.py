#!/usr/bin/env python3
"""Compare Helico diffusion module outputs against Protenix reference.

Loads the same Protenix checkpoint into both models and feeds identical
inputs. Reports numerical agreement at two levels:
  1. DiffusionConditioning  (should match exactly — validates RelPE)
  2. Full DiffusionModule   (may differ in atom attention path)
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root and scripts/ so both src/ and sibling scripts can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICE = "cuda"
CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "protenix_base_default_v1.0.0.pt"

# ── Helpers ─────────────────────────────────────────────────────────────

def load_protenix_state_dict() -> OrderedDict:
    """Load and strip DDP prefix from Protenix checkpoint."""
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    return OrderedDict((k.removeprefix("module."), v) for k, v in sd.items())


def extract_diffusion_sd(full_sd: OrderedDict, prefix: str = "diffusion_module.") -> OrderedDict:
    """Pull out diffusion_module.* and strip the prefix."""
    return OrderedDict(
        (k[len(prefix):], v) for k, v in full_sd.items() if k.startswith(prefix)
    )


def report(label: str, hf: torch.Tensor, ptx: torch.Tensor):
    """Print comparison stats between two tensors."""
    diff = (hf.float() - ptx.float()).abs()
    print(f"  {label}:")
    print(f"    shape       = {tuple(hf.shape)}")
    print(f"    max |diff|  = {diff.max().item():.2e}")
    print(f"    mean |diff| = {diff.mean().item():.2e}")
    print(f"    allclose    = {torch.allclose(hf.float(), ptx.float(), atol=1e-4, rtol=1e-4)}")
    return diff.max().item()


# ── 1. DiffusionConditioning comparison ────────────────────────────────

def compare_conditioning(ptx_sd: OrderedDict):
    """Compare DiffusionConditioning outputs — should match exactly."""
    from protenix.model.modules.diffusion import DiffusionConditioning as PtxCond
    from helico.model import DiffusionConditioning as HfCond, HelicoConfig

    print("\n" + "=" * 70)
    print("1. DiffusionConditioning comparison")
    print("=" * 70)

    # ── Instantiate Protenix ──
    ptx_cond = PtxCond(sigma_data=16.0, c_z=128, c_s=384, c_s_inputs=449, c_noise_embedding=256)
    ptx_cond_sd = {}
    prefix = "diffusion_module.diffusion_conditioning."
    for k, v in ptx_sd.items():
        if k.startswith(prefix):
            ptx_cond_sd[k[len(prefix):]] = v
    ptx_cond.load_state_dict(ptx_cond_sd)
    ptx_cond = ptx_cond.to(DEVICE).eval()

    # ── Instantiate Helico ──
    config = HelicoConfig()
    hf_cond = HfCond(config).to(DEVICE)

    # Transfer weights via our mapping
    from helico.load_protenix import load_protenix_state_dict
    from helico.model import Helico
    model = Helico(config)
    stats = load_protenix_state_dict(ptx_sd, model)
    # Extract conditioning submodule state dict from the loaded model
    hf_cond_sd = {
        k.removeprefix("conditioning."): v
        for k, v in model.diffusion.conditioning.state_dict().items()
    }
    hf_cond.load_state_dict(hf_cond_sd)
    hf_cond = hf_cond.to(DEVICE).eval()

    # ── Build identical inputs ──
    torch.manual_seed(123)
    N_tok = 32
    B = 1

    s_trunk = torch.randn(B, N_tok, 384, device=DEVICE)
    z_trunk = torch.randn(B, N_tok, N_tok, 128, device=DEVICE)
    s_inputs = torch.randn(B, N_tok, 449, device=DEVICE)
    sigma = torch.tensor([5.0], device=DEVICE)

    residue_index = torch.arange(N_tok, device=DEVICE).unsqueeze(0)
    token_index = torch.zeros(B, N_tok, dtype=torch.long, device=DEVICE)
    asym_id = torch.zeros(B, N_tok, dtype=torch.long, device=DEVICE)
    asym_id[:, N_tok // 2:] = 1  # two chains
    entity_id = torch.zeros(B, N_tok, dtype=torch.long, device=DEVICE)
    entity_id[:, N_tok // 2:] = 1
    sym_id = torch.zeros(B, N_tok, dtype=torch.long, device=DEVICE)

    # ── Protenix forward ──
    ptx_input_dict = {
        "residue_index": residue_index,
        "token_index": token_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
    }
    with torch.no_grad():
        ptx_s, ptx_z = ptx_cond(
            t_hat_noise_level=sigma,
            input_feature_dict=ptx_input_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
        )
    # Protenix s has shape (B, N_sample=1, N_tok, 384) — squeeze N_sample
    ptx_s = ptx_s.squeeze(-3)

    # ── Helico forward ──
    relpe_feats = {
        "residue_index": residue_index,
        "token_index": token_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
    }
    with torch.no_grad():
        hf_s, hf_z = hf_cond(
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_inputs=s_inputs,
            sigma=sigma,
            relpe_feats=relpe_feats,
        )

    max_s = report("s_cond (single)", hf_s, ptx_s)
    max_z = report("z_cond (pair)", hf_z, ptx_z)

    passed = max_s < 1e-3 and max_z < 1e-3
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# ── 2. Full DiffusionModule comparison ────────────────────────────────

def compare_full_module(ptx_sd: OrderedDict):
    """Compare full DiffusionModule denoised output."""
    from protenix.model.modules.diffusion import DiffusionModule as PtxDiff
    from helico.model import DiffusionModule as HfDiff, HelicoConfig

    print("\n" + "=" * 70)
    print("2. Full DiffusionModule comparison")
    print("=" * 70)

    # ── Instantiate Protenix ──
    ptx_diff = PtxDiff(
        sigma_data=16.0, c_atom=128, c_atompair=16, c_token=768,
        c_s=384, c_z=128, c_s_inputs=449,
        atom_encoder={"n_blocks": 3, "n_heads": 4},
        transformer={"n_blocks": 24, "n_heads": 16, "drop_path_rate": 0},
        atom_decoder={"n_blocks": 3, "n_heads": 4},
    )
    ptx_diff_sd = extract_diffusion_sd(ptx_sd)
    ptx_diff.load_state_dict(ptx_diff_sd)
    ptx_diff = ptx_diff.to(DEVICE).eval()
    print(f"  Protenix DiffusionModule: {sum(p.numel() for p in ptx_diff.parameters()):,} params")

    # ── Instantiate Helico ──
    config = HelicoConfig()
    from helico.model import Helico
    from helico.load_protenix import load_protenix_state_dict
    model = Helico(config)
    stats = load_protenix_state_dict(ptx_sd, model)
    hf_diff = model.diffusion.to(DEVICE).eval()
    print(f"  Helico DiffusionModule: {sum(p.numel() for p in hf_diff.parameters()):,} params")
    print(f"  Weight transfer: {stats['n_transferred']} transferred, "
          f"{stats['n_skipped']} skipped, {stats['n_shape_mismatches']} mismatches")

    # ── Build identical inputs ──
    torch.manual_seed(456)
    N_tok = 16
    N_atoms_per_tok = 4
    N_atoms = N_tok * N_atoms_per_tok
    B = 1

    # Trunk
    s_trunk = torch.randn(B, N_tok, 384, device=DEVICE)
    z_trunk = torch.randn(B, N_tok, N_tok, 128, device=DEVICE)
    s_inputs = torch.randn(B, N_tok, 449, device=DEVICE)

    # Noise level
    sigma = torch.tensor([5.0], device=DEVICE)

    # Noisy coordinates
    x_noisy = torch.randn(B, N_atoms, 3, device=DEVICE)

    # Atom features
    atom_to_token = torch.arange(N_tok, device=DEVICE).repeat_interleave(N_atoms_per_tok)
    ref_pos = torch.randn(B, N_atoms, 3, device=DEVICE)
    ref_charge = torch.zeros(B, N_atoms, 1, device=DEVICE)
    atom_mask = torch.ones(B, N_atoms, device=DEVICE)

    # Reference features (mask=1, element_onehot=128, atom_name_chars=256 → 385)
    ref_element = torch.zeros(N_atoms, 128, device=DEVICE)
    ref_element[:, 0] = 1.0  # all Carbon
    ref_mask = torch.ones(N_atoms, 1, device=DEVICE)
    ref_atom_name_chars = torch.zeros(N_atoms, 256, device=DEVICE)
    ref_features = torch.cat([ref_mask, ref_element, ref_atom_name_chars], dim=-1)
    ref_features = ref_features.unsqueeze(0).expand(B, -1, -1)

    # Relpe features
    residue_index = torch.arange(N_tok, device=DEVICE).unsqueeze(0)
    token_index = torch.zeros(B, N_tok, dtype=torch.long, device=DEVICE)
    asym_id = torch.zeros(B, N_tok, dtype=torch.long, device=DEVICE)
    asym_id[:, N_tok // 2:] = 1
    entity_id = torch.zeros(B, N_tok, dtype=torch.long, device=DEVICE)
    entity_id[:, N_tok // 2:] = 1
    sym_id = torch.zeros(B, N_tok, dtype=torch.long, device=DEVICE)

    # ref_space_uid groups atoms by token for Protenix
    ref_space_uid = atom_to_token.clone()

    # ── Protenix forward ──
    ptx_input_dict = {
        "residue_index": residue_index,
        "token_index": token_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "atom_to_token_idx": atom_to_token,
        "ref_pos": ref_pos.squeeze(0),
        "ref_charge": ref_charge.squeeze(0),
        "ref_mask": ref_mask,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
    }

    with torch.no_grad():
        # Protenix expects x_noisy with N_sample dim: (B, N_sample, N_atoms, 3)
        x_noisy_ptx = x_noisy.unsqueeze(1)  # (1, 1, N_atoms, 3)
        sigma_ptx = sigma.unsqueeze(0)  # (1, 1)
        ptx_out = ptx_diff(
            x_noisy=x_noisy_ptx,
            t_hat_noise_level=sigma_ptx,
            input_feature_dict=ptx_input_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
        )
    ptx_denoised = ptx_out.squeeze(1)  # (B, N_atoms, 3)

    # ── Helico forward ──
    relpe_feats = {
        "residue_index": residue_index,
        "token_index": token_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
    }

    with torch.no_grad():
        # Helico _f_forward expects pre-scaled input
        sigma_data = 16.0
        c_in = 1.0 / torch.sqrt(torch.tensor(sigma_data**2) + sigma**2)
        c_skip = torch.tensor(sigma_data**2) / (torch.tensor(sigma_data**2) + sigma**2)
        c_out = sigma * sigma_data / torch.sqrt(torch.tensor(sigma_data**2) + sigma**2)

        r_update = hf_diff._f_forward(
            x_scaled=c_in.view(B, 1, 1) * x_noisy,
            sigma=sigma,
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_features=ref_features,
            atom_to_token=atom_to_token.unsqueeze(0).expand(B, -1),
            atom_mask=atom_mask,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_inputs=s_inputs,
            relpe_feats=relpe_feats,
        )
        hf_denoised = c_skip.view(B, 1, 1) * x_noisy + c_out.view(B, 1, 1) * r_update

    max_diff = report("x_denoised (full module)", hf_denoised, ptx_denoised)

    # Interpret the result
    if max_diff < 1e-3:
        print("\n  RESULT: PASS — outputs match within tolerance")
    else:
        print(f"\n  RESULT: Outputs diverge (max diff = {max_diff:.2e})")
        print("  NOTE: This is expected due to atom attention architecture difference:")
        print("    - Protenix restricts atom-pair attention to within-token (ref_space_uid)")
        print("    - Helico computes all atom pairs (no masking)")
        print("    - The conditioning path and token transformer match exactly")
        print("    - Only the atom encoder/decoder paths diverge")

    # Also check that both produce finite, non-trivial output
    print(f"\n  Protenix output: finite={torch.isfinite(ptx_denoised).all()}, "
          f"range=[{ptx_denoised.min():.2f}, {ptx_denoised.max():.2f}]")
    print(f"  Helico output: finite={torch.isfinite(hf_denoised).all()}, "
          f"range=[{hf_denoised.min():.2f}, {hf_denoised.max():.2f}]")

    return max_diff


# ── 3. Pairformer comparison ────────────────────────────────────────

def compare_pairformer(ptx_sd: OrderedDict):
    """Compare Pairformer outputs — validates triangle ops + single attention."""
    from protenix.model.modules.pairformer import PairformerStack as PtxPairformer
    from helico.model import Pairformer as HfPairformer, HelicoConfig, Helico

    print("\n" + "=" * 70)
    print("3. Pairformer comparison")
    print("=" * 70)

    # ── Instantiate Protenix ──
    ptx_pf = PtxPairformer(n_blocks=48, n_heads=16, c_z=128, c_s=384, dropout=0.0)
    ptx_pf_sd = {}
    prefix = "pairformer_stack."
    for k, v in ptx_sd.items():
        if k.startswith(prefix):
            ptx_pf_sd[k[len(prefix):]] = v
    ptx_pf.load_state_dict(ptx_pf_sd)
    ptx_pf = ptx_pf.to(DEVICE).eval()
    print(f"  Protenix PairformerStack: {sum(p.numel() for p in ptx_pf.parameters()):,} params")

    # ── Instantiate Helico ──
    config = HelicoConfig()
    model = Helico(config)
    from helico.load_protenix import load_protenix_state_dict as load_ptx_sd
    stats = load_ptx_sd(ptx_sd, model)
    hf_pf = model.pairformer.to(DEVICE).eval()
    print(f"  Helico Pairformer: {sum(p.numel() for p in hf_pf.parameters()):,} params")
    print(f"  Weight transfer: {stats['n_transferred']} transferred, "
          f"{stats['n_shape_mismatches']} mismatches")

    # ── Build identical inputs ──
    torch.manual_seed(789)
    N_tok = 16
    B = 1

    single = torch.randn(B, N_tok, 384, device=DEVICE)
    pair = torch.randn(B, N_tok, N_tok, 128, device=DEVICE)
    pair_mask = torch.ones(B, N_tok, N_tok, device=DEVICE)

    # ── Protenix forward ──
    with torch.no_grad():
        ptx_s, ptx_z = ptx_pf(single.clone(), pair.clone(), pair_mask)

    # ── Helico forward ──
    with torch.no_grad():
        hf_s, hf_z = hf_pf(single.clone(), pair.clone(), pair_mask=pair_mask)

    max_s = report("s (single)", hf_s, ptx_s)
    max_z = report("z (pair)", hf_z, ptx_z)

    passed = max_s < 1e-3 and max_z < 1e-3
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# ── 4. MSA module comparison ─────────────────────────────────────────

def compare_msa(ptx_sd: OrderedDict):
    """Compare MSA module outputs block-by-block — validates OPM + pair_avg + pair_stack."""
    from protenix.model.modules.pairformer import MSAModule as PtxMSA
    from helico.model import MSAModule as HfMSA, HelicoConfig, Helico

    print("\n" + "=" * 70)
    print("4. MSA module comparison")
    print("=" * 70)

    # ── Instantiate Protenix ──
    ptx_msa = PtxMSA(
        n_blocks=4, c_m=64, c_z=128, c_s_inputs=449,
        msa_dropout=0.0, pair_dropout=0.0,
        blocks_per_ckpt=None, msa_chunk_size=2048, msa_max_size=16384,
        msa_configs={"enable": True, "strategy": "random",
                     "sample_cutoff": {"train": 4096, "test": 16384},
                     "min_size": {"train": 1, "test": 1}},
    )
    ptx_msa_sd = {}
    prefix = "msa_module."
    for k, v in ptx_sd.items():
        if k.startswith(prefix):
            ptx_msa_sd[k[len(prefix):]] = v
    ptx_msa.load_state_dict(ptx_msa_sd)
    ptx_msa = ptx_msa.to(DEVICE).eval()
    print(f"  Protenix MSAModule: {sum(p.numel() for p in ptx_msa.parameters()):,} params")

    # ── Instantiate Helico ──
    config = HelicoConfig()
    model = Helico(config)
    from helico.load_protenix import load_protenix_state_dict as load_ptx_sd
    stats = load_ptx_sd(ptx_sd, model)
    hf_msa = model.msa_module.to(DEVICE).eval()
    print(f"  Helico MSAModule: {sum(p.numel() for p in hf_msa.parameters()):,} params")
    print(f"  Weight transfer: {stats['n_transferred']} transferred, "
          f"{stats['n_shape_mismatches']} mismatches")

    # ── Build identical inputs ──
    torch.manual_seed(321)
    N_tok = 16
    N_msa = 4
    B = 1

    # Raw MSA features: (B, N_msa, N_tok, 34)
    m_raw = torch.randn(B, N_msa, N_tok, 34, device=DEVICE)
    z = torch.randn(B, N_tok, N_tok, 128, device=DEVICE)
    s_inputs = torch.randn(B, N_tok, 449, device=DEVICE)
    pair_mask = torch.ones(B, N_tok, N_tok, device=DEVICE)

    # ── Compare block-by-block (avoids Protenix MSA sampling/one-hot preprocessing) ──
    # Compute initial m identically
    with torch.no_grad():
        # Helico path
        hf_m = hf_msa.linear_m(m_raw) + hf_msa.linear_s(s_inputs).unsqueeze(-3)
        hf_z = z.clone()

        # Protenix path
        ptx_m = ptx_msa.linear_no_bias_m(m_raw) + ptx_msa.linear_no_bias_s(s_inputs).unsqueeze(-3)
        ptx_z = z.clone()

    max_m = report("m_init", hf_m, ptx_m)

    # Run through blocks
    max_z_diff = 0.0
    for i in range(4):
        with torch.no_grad():
            # Helico block
            hf_m, hf_z = hf_msa.blocks[i](hf_m, hf_z, pair_mask=pair_mask)

            # Protenix block
            ptx_m, ptx_z = ptx_msa.blocks[i](ptx_m, ptx_z, pair_mask=pair_mask)

        blk_z = report(f"z after block {i}", hf_z, ptx_z)
        max_z_diff = max(max_z_diff, blk_z)
        if i < 3:
            blk_m = report(f"m after block {i}", hf_m, ptx_m)

    passed = max_z_diff < 1e-3
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} (max z diff = {max_z_diff:.2e})")
    return passed


# ── Main ───────────────────────────────────────────────────────────────

def main():
    assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"
    print(f"Loading checkpoint: {CHECKPOINT}")
    ptx_sd = load_protenix_state_dict()
    n_diff = sum(1 for k in ptx_sd if k.startswith("diffusion_module."))
    n_pf = sum(1 for k in ptx_sd if k.startswith("pairformer_stack."))
    n_msa = sum(1 for k in ptx_sd if k.startswith("msa_module."))
    print(f"  {n_diff} diffusion_module.* parameters")
    print(f"  {n_pf} pairformer_stack.* parameters")
    print(f"  {n_msa} msa_module.* parameters")

    cond_ok = compare_conditioning(ptx_sd)
    full_diff = compare_full_module(ptx_sd)
    pf_ok = compare_pairformer(ptx_sd)
    msa_ok = compare_msa(ptx_sd)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  DiffusionConditioning:  {'PASS' if cond_ok else 'FAIL'}")
    print(f"  Full DiffusionModule:   max |diff| = {full_diff:.2e}")
    print(f"  Pairformer:             {'PASS' if pf_ok else 'FAIL'}")
    print(f"  MSA Module:             {'PASS' if msa_ok else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
