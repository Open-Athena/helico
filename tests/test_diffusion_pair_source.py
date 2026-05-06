"""gh#9: tests for the diffusion_pair_source config flag + freeze_trunk helper."""

from __future__ import annotations

import pytest
import torch

from helico.data import make_synthetic_batch
from helico.model import Helico, HelicoConfig
from helico.train import _freeze_trunk


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _small_cfg(**overrides):
    base = dict(
        n_pairformer_blocks=2,
        n_diffusion_token_blocks=2,
        n_diffusion_samples=2,
        n_cycles=1,
    )
    base.update(overrides)
    return HelicoConfig(**base)


@cuda_only
def test_z_mode_default_is_legacy_path():
    """``diffusion_pair_source='z'`` (default) should use pair_proj, not pair_proj_dist.
    Sanity check that the new code path doesn't accidentally fire when the
    flag is off — a zero gradient on pair_proj_dist after a backward
    confirms it's untouched.
    """
    cfg = _small_cfg()
    model = Helico(cfg).cuda()
    batch = make_synthetic_batch(n_tokens=24, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch, compute_confidence=False)
    out["diffusion_loss"].backward()
    p_dist = model.diffusion.conditioning.pair_proj_dist.weight
    p_z = model.diffusion.conditioning.pair_proj.weight
    assert p_dist.grad is None or p_dist.grad.abs().sum().item() == 0.0
    assert p_z.grad is not None and p_z.grad.abs().sum().item() > 0.0


@cuda_only
def test_distogram_mode_uses_dist_projection():
    """In distogram mode the dist projection gets gradient, the legacy
    pair_proj does not."""
    cfg = _small_cfg(diffusion_pair_source="distogram_logits")
    model = Helico(cfg).cuda()
    batch = make_synthetic_batch(n_tokens=24, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch, compute_confidence=False)
    out["diffusion_loss"].backward()
    p_dist = model.diffusion.conditioning.pair_proj_dist.weight
    p_z = model.diffusion.conditioning.pair_proj.weight
    assert p_dist.grad is not None and p_dist.grad.abs().sum().item() > 0.0
    assert p_z.grad is None or p_z.grad.abs().sum().item() == 0.0


@cuda_only
def test_freeze_trunk_only_diffusion_trains():
    """``_freeze_trunk`` leaves only ``model.diffusion.*`` trainable.
    After a forward+backward, no trunk param should have a nonzero grad.
    """
    cfg = _small_cfg(diffusion_pair_source="distogram_logits")
    model = Helico(cfg).cuda()
    n_frozen, n_trainable = _freeze_trunk(model)
    assert n_frozen > 0 and n_trainable > 0
    # Every non-diffusion param has requires_grad False; every diffusion
    # param has requires_grad True.
    for name, p in model.named_parameters():
        if name.startswith("diffusion."):
            assert p.requires_grad, name
        else:
            assert not p.requires_grad, name

    batch = make_synthetic_batch(n_tokens=24, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch, compute_confidence=False)
    out["diffusion_loss"].backward()
    for name, p in model.named_parameters():
        if name.startswith("diffusion."):
            continue
        # Frozen params: PyTorch leaves grad None when requires_grad=False.
        assert p.grad is None or p.grad.abs().sum().item() == 0.0, name


@cuda_only
def test_distogram_logits_independent_of_z_mode():
    """The distogram-head output (from the trunk) must be identical
    regardless of which pair source the diffusion module reads from —
    the head is part of the trunk, the swap is downstream of it."""
    torch.manual_seed(0)
    cfg_z = _small_cfg()
    model_z = Helico(cfg_z).cuda()
    cfg_d = _small_cfg(diffusion_pair_source="distogram_logits")
    model_d = Helico(cfg_d).cuda()
    # Copy weights so any difference is purely from the runtime branch.
    model_d.load_state_dict(model_z.state_dict())

    batch = make_synthetic_batch(n_tokens=24, device="cuda")
    torch.manual_seed(0)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_z = model_z(batch, compute_confidence=False)
    torch.manual_seed(0)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_d = model_d(batch, compute_confidence=False)
    # Distogram head reads from `z` (trunk pair); should be ~identical
    # (bf16 noise + cuDNN nondeterminism allow small drift). Compare via
    # relative L2 like the snapshot tests.
    a = out_z["distogram_logits"].float()
    b = out_d["distogram_logits"].float()
    rel_l2 = float((a - b).norm() / (a.norm() + 1e-8))
    assert rel_l2 < 0.05, f"distogram drifted between modes: rel_L2={rel_l2:.4g}"
