import copy
import numpy as np
import torch

from helico.model.losses import diffusion_loss
from helico.protenix_data import adapt_example, batch_example, chain_contiguous_token_order
from helico.train_contacts import weighted_draws


def example():
    n, a = 3, 6
    token = torch.tensor([0, 0, 1, 1, 2, 2])
    f = dict(token_index=torch.arange(n), residue_index=torch.tensor([1, 20, 1]),
        asym_id=torch.tensor([0, 0, 1]), entity_id=torch.tensor([0, 0, 1]), sym_id=torch.zeros(n).long(),
        restype=torch.nn.functional.one_hot(torch.tensor([0, 0, 20]), 32),
        atom_to_token_idx=token, is_protein=torch.tensor([1, 1, 1, 1, 0, 0]),
        ref_element=torch.nn.functional.one_hot(torch.tensor([5, 0, 5, 5, 5, 5]), 128),
        ref_pos=torch.randn(a, 3), ref_mask=torch.ones(a), ref_charge=torch.zeros(a),
        ref_space_uid=token, ref_atom_name_chars=torch.zeros(a, 4, 64),
        distogram_rep_atom_mask=torch.tensor([1, 0, 1, 0, 1, 0]), token_bonds=torch.zeros(n, n),
        msa=torch.zeros(2, n).long(), profile=torch.zeros(n, 32), has_deletion=torch.zeros(2, n),
        deletion_value=torch.zeros(2, n), deletion_mean=torch.zeros(n),
        is_ligand=torch.tensor([0, 0, 0, 0, 1, 1]), is_dna=torch.zeros(a), is_rna=torch.zeros(a))
    xyz = torch.tensor([[0., 0, 0], [8., 0, 0], [10., 0, 0], [11., 0, 0], [13., 0, 0], [14., 0, 0]])
    return dict(input_feature_dict=f, label_dict=dict(coordinate=xyz,
                coordinate_mask=torch.tensor([1, 1, 1, 0, 1, 1])))


def test_missing_atoms_do_not_establish_absence_and_hydrogen_is_excluded():
    f = adapt_example(example())
    assert not f['contact_target'][0, 1]  # hydrogen at x=8 does not establish contact
    assert not f['contact_valid'][0, 1]   # token 1 has missing heavy atom
    assert f['contact_target'][1, 2] and f['contact_valid'][1, 2]  # observed witness suffices
    assert f['contact_valid'][0, 2] and not f['contact_target'][0, 2]
    assert not f['contact_valid'].diagonal().any()


def test_padding_preserves_reference_and_observation_masks_and_msa_features():
    e = example(); e['input_feature_dict']['ref_mask'][4] = 0
    f = adapt_example(e)
    b = batch_example(f, torch.zeros(3, 3).to(torch.uint8))
    assert b['token_mask'].shape == (1, 32)
    assert not b['token_mask'][0, 3:].any()
    assert b['atom_mask'][0, 3] and not b['coordinate_mask'][0, 3]
    assert b['coordinate_mask'][0, 4] and not b['ref_mask'][0, 4]
    assert not b['distogram_mask'][0, 3:].any()
    assert (b['msa'][0, :, 3:] == 31).all()
    assert not b['contact_valid'][0, 3:].any()


def test_diffusion_sample_weights_do_not_mix_between_samples():
    truth = torch.zeros(2, 1, 3)
    pred = truth.clone(); pred[0, 0, 0] = 2; pred[1, 0, 0] = 3
    loss = diffusion_loss(pred, truth, torch.tensor([1., 3.]), torch.ones(2, 1))
    assert torch.allclose(loss, torch.tensor(2.5))  # (4/1 + 9/9) / 2


def test_draw_stream_resume_and_rank_partition():
    args = ([1., 2., 3.], 7, 2, 2)
    ranks = [weighted_draws(*args, rank, 0, 42) for rank in range(2)]
    assert set(d for _, d in ranks[0]).isdisjoint(d for _, d in ranks[1])
    assert sorted(d for r in ranks for _, d in r) == list(range(28))
    for rank in range(2):
        assert weighted_draws(*args, rank, 3, 42) == ranks[rank][6:]


def test_fragmented_chains_become_contiguous_without_losing_or_duplicating_tokens():
    chains = np.array([7, 7, 3, 7, 3, 3])
    residues = np.array([8, 9, 2, 1, 1, 1])
    order = chain_contiguous_token_order(chains, residues)
    np.testing.assert_array_equal(order, [3, 0, 1, 4, 5, 2])
    np.testing.assert_array_equal(np.sort(order), np.arange(len(chains)))
    np.testing.assert_array_equal(chains[order], [7, 7, 7, 3, 3, 3])
    # Atom tokens within the same residue keep their relative order.
    assert list(order).index(4) < list(order).index(5)


def test_well_ordered_chains_keep_their_original_token_order():
    chains = [9, 9, 2, 2, 2]
    residues = [3, 5, 1, 1, 8]
    np.testing.assert_array_equal(chain_contiguous_token_order(chains, residues), np.arange(5))
