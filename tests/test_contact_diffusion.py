import numpy as np
import torch

from helico.contact_diffusion import absorbing_mask, contact_bce, heavy_atom_contacts


def test_residue_ligand_minimum_heavy_distance_and_boundary():
    # Residue representative atoms are far apart, but sidechains contact.
    coords = [[0, 0, 0], [10, 0, 0], [14, 0, 0], [30, 0, 0], [19, 0, 0]]
    labels = heavy_atom_contacts(coords, [0, 0, 1, 1, 2], 3)
    assert labels[0, 1]
    assert not labels[1, 2]  # exactly 5 A, outside the strict threshold
    assert not labels.diagonal().any()
    assert torch.equal(labels, labels.T)
    # Excluding a non-heavy or unresolved atom removes its claimed contact.
    masked = heavy_atom_contacts(coords, [0, 0, 1, 1, 2], 3,
                                 atom_mask=[1, 0, 1, 1, 1])
    assert not masked[0, 1]


def test_absorbing_endpoints_validity_and_nested_clean_masks():
    target = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    valid = torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=torch.bool)
    full = absorbing_mask(target, valid, 0)
    assert full.tolist() == [[0, 2, 1], [2, 0, 0], [1, 0, 0]]
    assert not absorbing_mask(target, valid, 1000).any()
    earlier = absorbing_mask(target, valid, 250, generator=torch.Generator().manual_seed(9))
    later = absorbing_mask(target, valid, 750, generator=torch.Generator().manual_seed(9))
    assert torch.equal(later, later.T)
    assert not ((later != 0) & (earlier == 0)).any()
    assert torch.equal(later[later != 0], full[later != 0])


def test_revelation_is_independent_of_label():
    n = 400
    target = torch.rand(n, n) > .5
    target = torch.triu(target, 1); target |= target.T.clone()
    valid = ~torch.eye(n, dtype=torch.bool)
    first = absorbing_mask(target, valid, 997, generator=torch.Generator().manual_seed(5))
    flipped = absorbing_mask(~target, valid, 997, generator=torch.Generator().manual_seed(5))
    assert torch.equal(first != 0, flipped != 0)
    reveal_fraction = (first != 0)[valid].float().mean()
    assert .001 < reveal_fraction < .005


def test_masked_loss_ignores_visible_and_invalid_pairs_with_finite_gradients():
    logits = torch.zeros(1, 3, 3, requires_grad=True)
    target = torch.zeros_like(logits, dtype=torch.bool)
    valid = torch.ones_like(target)
    valid[:, 0, 2] = valid[:, 2, 0] = False
    state = torch.zeros_like(logits, dtype=torch.uint8)
    state[:, 0, 1] = state[:, 1, 0] = 1
    losses = contact_bce(logits, target, valid, state)
    assert np.isclose(float(losses["contact_loss"].detach()), np.log(2))
    losses["contact_loss"].backward()
    assert logits.grad[0, 0, 1] == 0
    assert logits.grad[0, 0, 2] == 0
    assert logits.grad[0, 1, 2] != 0
    full = torch.ones_like(state)
    empty = contact_bce(logits, target, valid, full)["contact_loss"]
    assert empty == 0 and empty.requires_grad


def test_average_precision_groups_ties_including_untrained_constant_head():
    from helico.contact_pilot import average_precision
    y = np.array([1, 0, 0, 1, 0], dtype=bool)
    assert average_precision(y, np.ones(5)) == .4
    assert average_precision(y[::-1], np.ones(5)) == .4
    assert average_precision(y, y.astype(float)) == 1.
