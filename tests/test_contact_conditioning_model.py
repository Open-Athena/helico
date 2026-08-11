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


def _leak_stats(run, clean_batch, poisoned_batch, k=3):
    """Separate a real dependency from cuEquivariance kernel noise.

    The trunk is nondeterministic: two identical forward passes differ by ~0.03
    max-abs. Comparing one clean pass against one poisoned pass therefore fails
    about half the time when the feature is correctly gated, because both
    numbers are draws from the same distribution.

    Averaging k passes per group suppresses that noise while leaving any
    systematic dependency intact. The returned pair is
    ``(noise, signal)``: ``noise`` is the difference between two independent
    groups of *clean* runs, ``signal`` the difference between clean and
    poisoned. Measured separation is ~160x when a feature really is connected,
    so a small multiple of ``noise`` is a safe threshold.
    """
    a = sum(run(clean_batch) for _ in range(k)) / k
    b = sum(run(clean_batch) for _ in range(k)) / k
    c = sum(run(poisoned_batch) for _ in range(k)) / k
    return float((a - b).abs().max()), float((a - c).abs().max())


@cuda_only
class TestMSAGating:
    """`use_msa=False` must remove every alignment-derived signal, not just the module.

    `msa_profile` and `deletion_mean` are per-column conservation features that
    live in `s_inputs`, outside MSAModule. Gating only the module left them
    flowing, so runs labelled "MSA-free" were training and benching with real
    conservation profiles. These tests pin the full gate.

    Exact comparison is not available here: the atom encoder runs cuEquivariance
    kernels, which are nondeterministic, so the same input twice does not give
    bitwise-identical output. Every assertion is therefore made against a
    measured noise floor rather than against zero.
    """

    def _batch(self, n_tok=24, device="cuda"):
        from helico.data import make_synthetic_batch

        batch = make_synthetic_batch(batch_size=1, n_tokens=n_tok, device=device)
        # A conservation profile that is unmistakably not zeros.
        g = torch.Generator(device="cpu").manual_seed(11)
        prof = torch.rand(1, n_tok, 32, generator=g).to(device)
        batch["msa_profile"] = prof / prof.sum(-1, keepdim=True)
        batch["deletion_mean"] = torch.rand(1, n_tok, generator=g).to(device)
        return _bf16(batch)

    @staticmethod
    def _zeroed(batch):
        out = dict(batch)
        out["msa_profile"] = torch.zeros_like(batch["msa_profile"])
        out["deletion_mean"] = torch.zeros_like(batch["deletion_mean"])
        return out

    def _s_inputs(self, model, batch, use_msa):
        from helico.model.features import build_ref_features, build_s_inputs

        ref_charge, ref_features = build_ref_features(batch)
        return build_s_inputs(model.input_embedder, batch, ref_charge, ref_features,
                              batch["atom_mask"], use_msa=use_msa)

    def _noise_and_signal(self, use_msa):
        """Return (kernel noise, effect of zeroing the profile) on s_inputs."""
        from helico.model import Helico

        model = Helico(_small_cfg(use_msa=use_msa)).cuda().to(torch.bfloat16).eval()
        batch = self._batch()

        def run(b):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return self._s_inputs(model, b, use_msa).float()

        return _leak_stats(run, batch, self._zeroed(batch))

    def test_profile_reaches_s_inputs_when_enabled(self):
        """Sanity check on the test itself: with use_msa=True the profile matters."""
        noise, signal = self._noise_and_signal(use_msa=True)
        assert signal > max(10 * noise, 1e-2), (
            f"zeroing the profile changed s_inputs by only {signal:.3g} "
            f"(noise {noise:.3g}) — the test cannot detect the gate"
        )

    def test_profile_blocked_when_disabled(self):
        """The real assertion: with use_msa=False the profile has no effect."""
        noise, signal = self._noise_and_signal(use_msa=False)
        assert signal <= max(3 * noise, 1e-2), (
            f"profile still reached s_inputs with use_msa=False: moved it "
            f"{signal:.3g} against a {noise:.3g} noise floor"
        )

    def test_gate_follows_model_config(self):
        """Helico.forward must pass its own config through, not the default."""
        import inspect

        from helico.model import helico as helico_mod

        src = inspect.getsource(helico_mod)
        assert src.count("use_msa=self.config.use_msa") == 2, (
            "both build_s_inputs call sites must forward config.use_msa"
        )


@cuda_only
class TestNoMSALeak:
    """Audit: with use_msa=False, NO alignment-derived key may affect the model.

    Rather than reasoning key-by-key about which pathway reads what, this
    poisons every MSA-derived entry in the batch at once and checks the full
    trunk output is unchanged. Any future feature that smuggles alignment
    information in through a new key will fail this test.

    The keys below are every MSA-derived tensor the model reads anywhere
    (see `grep 'batch\\.get\\|batch\\[' src/helico/model/`).
    """

    MSA_KEYS = ("msa", "msa_profile", "deletion_matrix", "deletion_mean",
                "cluster_msa", "cluster_profile", "cluster_deletion_mean",
                "has_msa")

    def _batch(self, n_tok=24, device="cuda"):
        from helico.data import AF3_NUM_MSA_CLASSES, make_synthetic_batch

        batch = make_synthetic_batch(batch_size=1, n_tokens=n_tok, device=device)
        g = torch.Generator(device="cpu").manual_seed(7)
        depth = 8
        prof = torch.rand(1, n_tok, AF3_NUM_MSA_CLASSES, generator=g).to(device)
        batch["msa_profile"] = prof / prof.sum(-1, keepdim=True)
        batch["deletion_mean"] = torch.rand(1, n_tok, generator=g).to(device)
        batch["msa"] = torch.randint(0, AF3_NUM_MSA_CLASSES, (1, depth, n_tok),
                                     generator=g).to(device)
        batch["deletion_matrix"] = torch.rand(1, depth, n_tok, generator=g).to(device)
        batch["cluster_msa"] = batch["msa"].clone()
        batch["cluster_deletion_mean"] = batch["deletion_matrix"].clone()
        batch["cluster_profile"] = prof.unsqueeze(1).to(device)
        batch["has_msa"] = torch.ones(1, device=device)
        return _bf16(batch)

    def _poison(self, batch):
        """Replace every MSA-derived tensor with different values."""
        g = torch.Generator(device="cpu").manual_seed(99)
        out = dict(batch)
        for k in self.MSA_KEYS:
            v = batch[k]
            if v.dtype.is_floating_point:
                out[k] = torch.rand(v.shape, generator=g).to(v.device, v.dtype)
            else:
                out[k] = torch.randint(0, 32, v.shape, generator=g).to(v.device, v.dtype)
        return out

    def _noise_and_signal(self, use_msa):
        from helico.model import Helico

        model = Helico(_small_cfg(use_msa=use_msa)).cuda().to(torch.bfloat16).eval()
        batch = self._batch()

        def run(b):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(b, compute_confidence=False)
            return out["pair"].float()

        return _leak_stats(run, batch, self._poison(batch))

    def test_trunk_invariant_to_all_msa_keys(self):
        """use_msa=False: poisoning every MSA key must not move the trunk."""
        noise, signal = self._noise_and_signal(use_msa=False)
        assert signal <= max(3 * noise, 1e-2), (
            f"pair repr moved {signal:.3g} when MSA keys changed against a "
            f"{noise:.3g} noise floor — an MSA feature is still leaking"
        )

    def test_trunk_does_move_when_msa_enabled(self):
        """Control: the same poisoning must matter when use_msa=True."""
        noise, signal = self._noise_and_signal(use_msa=True)
        assert signal > max(10 * noise, 1e-2), (
            f"MSA keys had no effect even with use_msa=True (signal {signal:.3g}, "
            f"noise {noise:.3g}) — the audit cannot detect a leak")
