import random

import pytest
import torch

from helico.contact_compliance import compliance_loss, constraint_distances, prepare_constraints
from helico.contact_diffusion import absorbing_mask, focused_time
from helico.train_contacts import check_continuation


def test_sparse_support_is_label_blind_and_scales_with_pair_count():
    torch.set_num_threads(2)
    for n in (32, 128):
        valid = ~torch.eye(n, dtype=torch.bool)
        target = torch.zeros_like(valid)
        for count in (1, 4, 16, 64):
            mixture = dict(unknown=0, full=0, sparse=1, uniform=0, expected_counts=[count])
            t = focused_time(valid, random.Random(3), mixture)
            assert (1 - t / 1000) * n * (n - 1) / 2 == pytest.approx(count)
            samples = []
            for seed in range(256):
                state = absorbing_mask(target, valid, t, generator=torch.Generator().manual_seed(seed))
                flipped = absorbing_mask(~target, valid, t, generator=torch.Generator().manual_seed(seed))
                assert torch.equal(state != 0, flipped != 0)
                samples.append(int((state != 0).sum()) / 2)
            assert abs(sum(samples) / len(samples) - count) < 5 * (count / len(samples)) ** .5
            if count == 1:
                assert 0 in samples and 1 in samples and 2 in samples


def constraints():
    f = dict(n_tokens=3, atom_to_token=torch.tensor([0, 0, 0, 1, 1, 2]),
             atom_element_idx=torch.tensor([5, 5, 0, 5, 5, 5]),
             coordinate_mask=torch.tensor([True, False, True, True, True, True]),
             contact_valid=~torch.eye(3, dtype=torch.bool),
             contact_target=torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool))
    state = torch.tensor([[0, 2, 1], [2, 0, 0], [1, 0, 0]], dtype=torch.uint8)
    xyz = torch.tensor([[[0., 0, 0], [7., 0, 0], [7., 0, 0], [8., 0, 0], [10., 0, 0], [2., 0, 0]]])
    return f, state, xyz


def test_compliance_gradients_pull_positive_push_negative_ignore_missing_and_hydrogen():
    f, state, xyz = constraints()
    xyz.requires_grad_(True)
    batch = prepare_constraints(f, state, "cpu")
    loss, metrics = compliance_loss(xyz, batch)
    assert metrics["positive_count"] == metrics["negative_count"] == 1
    assert metrics["positive_satisfaction"] == metrics["negative_satisfaction"] == 0
    loss.backward()
    assert xyz.grad[0, 3, 0] > 0  # descent moves positive partner towards atom 0
    assert xyz.grad[0, 5, 0] < 0  # descent moves negative partner away
    assert not xyz.grad[0, [1, 2, 4]].any()
    after, _ = compliance_loss(xyz.detach() - .1 * xyz.grad, batch)
    assert after < loss


def test_exact_minimum_gradient_and_chunking_match_dense_reference():
    torch.manual_seed(7)
    atoms = torch.tensor([[0, 1], [2, -1], [3, 4]])
    pairs = torch.tensor([[0, 1], [1, 2], [0, 2]])
    x = torch.randn(2, 5, 3, requires_grad=True)
    reference = torch.stack([torch.cdist(x[:, atoms[i][atoms[i] >= 0]], x[:, atoms[j][atoms[j] >= 0]],
        compute_mode="donot_use_mm_for_euclid_dist").flatten(-2).min(-1).values for i, j in pairs], -1)
    actual = constraint_distances(x, atoms, pairs, chunk_size=1)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(reference.sum(), x)[0])
    torch.testing.assert_close(constraint_distances(x + 7, atoms, pairs, 2), actual)


def test_empty_invalid_and_satisfied_constraints_have_zero_loss():
    f, state, xyz = constraints()
    xyz.requires_grad_(True)
    f["contact_valid"].zero_()
    loss, _ = compliance_loss(xyz, prepare_constraints(f, state, "cpu"))
    loss.backward()
    assert loss == 0 and not xyz.grad.any()
    f, state, xyz = constraints()
    xyz[0, 3, 0] = 4.; xyz[0, 5, 0] = 12.
    loss, metrics = compliance_loss(xyz.requires_grad_(True), prepare_constraints(f, state, "cpu"))
    assert loss == 0 and metrics["positive_satisfaction"] == metrics["negative_satisfaction"] == 1
    state[0, 1] = state[1, 0] = 1
    with pytest.raises(ValueError, match="clean native"):
        prepare_constraints(f, state, "cpu")


def test_continuation_checks_parent_identity_data_and_architecture():
    ckpt = dict(step=20000, git_sha="parent", data_lock_sha256="lock", model_config={"width": 4},
                training_config={"contact_lr": 2e-4, "lr": 2e-5})
    cfg = dict(continuation={"step": 20000, "git_sha": "parent"}, contact_lr=1e-4, lr=1e-5)
    check_continuation(ckpt, cfg, {"width": 4}, "lock")
    for bad in ({"step": 19999}, {"git_sha": "other"}, {"data_lock_sha256": "other"},
                {"model_config": {"width": 5}}):
        with pytest.raises(ValueError):
            check_continuation({**ckpt, **bad}, cfg, {"width": 4}, "lock")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_geometric_loss_reaches_contact_projection_and_diffusion():
    from helico.model import Helico, HelicoConfig
    from helico.data import make_synthetic_batch
    torch.manual_seed(24)
    model = Helico(HelicoConfig(n_pairformer_blocks=2, n_diffusion_token_blocks=2,
        n_msa_blocks=1, n_diffusion_samples=2, predict_contacts=True)).cuda().train()
    batch = make_synthetic_batch(n_tokens=32, device="cuda")
    batch['contact_state'].zero_()
    batch['contact_state'][0, 0, 16] = batch['contact_state'][0, 16, 0] = 1
    constraints = dict(constraint_atoms=torch.arange(160, device="cuda").reshape(32, 5),
        constraint_pairs=torch.tensor([[0, 16]], device="cuda"),
        constraint_values=torch.tensor([1], device="cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch, compute_confidence=False)
    loss, _ = compliance_loss(out['x_denoised'], constraints, threshold=100.)
    loss.backward()
    grad = model.linear_contact.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    grads = [p.grad for n, p in model.named_parameters() if n.startswith('diffusion') and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads) > 0
