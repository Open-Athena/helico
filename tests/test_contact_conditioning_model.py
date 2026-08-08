"""Model-side tests for contact conditioning (use_contacts / use_msa).

The CPU-checkable structural properties run anywhere; the forward/backward
tests need a GPU because the trunk uses cuEquivariance kernels.
"""

from __future__ import annotations

import pytest
import torch

from helico.data import CONTACT_UNKNOWN, NUM_CONTACT_STATES, make_synthetic_batch
from helico.model import Helico, HelicoConfig
from helico.model.features import build_contact_onehot


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _small_cfg(**overrides):
    base = dict(
        n_pairformer_blocks=2,
        n_diffusion_token_blocks=2,
        n_diffusion_samples=2,
        n_msa_blocks=1,
        n_cycles=1,
    )
    base.update(overrides)
    return HelicoConfig(**base)


def _bf16(batch):
    return {
        k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.dtype.is_floating_point else v)
        for k, v in batch.items()
    }


class TestStructure:
    """CPU-checkable wiring."""

    def test_projection_shape_and_zero_init(self):
        cfg = _small_cfg()
        model = Helico(cfg)
        w = model.linear_contact.weight
        assert w.shape == (cfg.d_pair, NUM_CONTACT_STATES)
        assert bool((w == 0).all()), "must be zero-init so it is a no-op on warm start"

    def test_one_hot_builder(self):
        batch = make_synthetic_batch(n_tokens=8, device="cpu")
        onehot = build_contact_onehot(batch, torch.float32)
        assert onehot.shape == (1, 8, 8, NUM_CONTACT_STATES)
        assert bool((onehot.sum(-1) == 1).all())

    def test_one_hot_builder_returns_none_when_absent(self):
        batch = make_synthetic_batch(n_tokens=8, device="cpu")
        del batch["contact_state"]
        assert build_contact_onehot(batch, torch.float32) is None

    def test_config_fields_reach_checkpoint_dict(self):
        """asdict(TrainConfig) is what loaders rebuild HelicoConfig from."""
        from dataclasses import asdict

        from helico.train import TrainConfig

        d = asdict(TrainConfig())
        assert "use_contacts" in d and "use_msa" in d


def _capture_contact_contribution(model, batch):
    """Run a forward, returning what linear_contact contributed to z_init.

    Hooking the projection tests the invariant exactly. Comparing two whole
    forward passes instead would be a proxy measurement at the mercy of
    cuEquivariance kernel nondeterminism.
    """
    captured = []
    handle = model.linear_contact.register_forward_hook(
        lambda _m, _i, out: captured.append(out.detach().float().clone())
    )
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(batch, compute_confidence=False)
    finally:
        handle.remove()
    return captured


@cuda_only
class TestForward:
    def test_zero_init_makes_contacts_a_noop(self):
        """A fresh model must contribute exactly zero, so warm starts are exact."""
        model = Helico(_small_cfg()).cuda().eval()
        batch = _bf16(make_synthetic_batch(n_tokens=24, device="cuda"))
        captured = _capture_contact_contribution(model, batch)
        assert len(captured) == 1, "linear_contact should run exactly once per forward"
        assert captured[0].abs().max().item() == 0.0

    def test_trained_projection_contributes_signal(self):
        """Once the projection is nonzero, the matrix must reach z_init."""
        model = Helico(_small_cfg()).cuda().eval()
        with torch.no_grad():
            model.linear_contact.weight.normal_(0, 0.02)
        batch = _bf16(make_synthetic_batch(n_tokens=24, device="cuda"))
        captured = _capture_contact_contribution(model, batch)
        assert captured[0].abs().max().item() > 0.0
        # and the contribution must differ between contact states
        blank = dict(batch)
        blank["contact_state"] = torch.full_like(batch["contact_state"], CONTACT_UNKNOWN)
        blank_out = _capture_contact_contribution(model, blank)
        assert not torch.equal(captured[0], blank_out[0])

    def test_gradient_reaches_the_contact_projection(self):
        model = Helico(_small_cfg()).cuda()
        batch = _bf16(make_synthetic_batch(n_tokens=24, device="cuda"))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, compute_confidence=False)
        out["diffusion_loss"].backward()
        grad = model.linear_contact.weight.grad
        assert grad is not None and grad.abs().sum().item() > 0.0

    def test_use_contacts_false_skips_the_projection(self):
        """With the flag off the projection must not run at all."""
        model = Helico(_small_cfg(use_contacts=False)).cuda().eval()
        with torch.no_grad():
            model.linear_contact.weight.normal_(0, 0.02)
        batch = _bf16(make_synthetic_batch(n_tokens=24, device="cuda"))
        assert _capture_contact_contribution(model, batch) == []

    def test_batch_without_contact_state_still_runs(self):
        """Legacy batches (and tokenize_sequences inference) carry no contacts."""
        model = Helico(_small_cfg()).cuda().eval()
        batch = _bf16(make_synthetic_batch(n_tokens=24, device="cuda"))
        del batch["contact_state"]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, compute_confidence=False)
        assert torch.isfinite(out["pair"].float()).all()


@cuda_only
class TestMSAFree:
    def test_msa_free_forward_and_backward(self):
        model = Helico(_small_cfg(use_msa=False)).cuda()
        batch = _bf16(make_synthetic_batch(n_tokens=24, device="cuda"))
        for key in ["msa_profile", "cluster_msa", "cluster_profile",
                    "deletion_mean", "cluster_deletion_mean", "has_msa"]:
            batch.pop(key, None)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, compute_confidence=False)
        out["diffusion_loss"].backward()
        assert torch.isfinite(out["diffusion_loss"]).all()
        assert model.linear_contact.weight.grad.abs().sum().item() > 0.0

    def test_msa_module_gets_no_gradient_when_disabled(self):
        model = Helico(_small_cfg(use_msa=False)).cuda()
        batch = _bf16(make_synthetic_batch(n_tokens=24, device="cuda"))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, compute_confidence=False)
        out["diffusion_loss"].backward()
        for p in model.msa_module.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0

    def test_s_inputs_width_is_unchanged(self):
        """c_s_inputs must stay 449 so Protenix warm start still fits."""
        assert HelicoConfig(use_msa=False).c_s_inputs == HelicoConfig().c_s_inputs
