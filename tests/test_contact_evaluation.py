import math

import numpy as np
import torch

from helico.contact_eval_metrics import (
    average_precision, contact_metrics, discrepancy_pairs, input_pair_groups,
    oracle_state, reverse_mask_step, sparse_lddt,
)
from helico.contact_evaluate import inference_batch, select_panel, LABEL_KEYS


def test_constant_predictor_ap_is_prevalence_and_missing_labels_excluded():
    assert average_precision([0, 1, 0, 0], [.2] * 4) == .25
    target = torch.zeros(3, 3, dtype=torch.bool)
    target[0, 1] = target[1, 0] = True
    valid = target.clone()
    groups = {'all': torch.triu(torch.ones(3, 3, dtype=torch.bool), 1)}
    rows, bins = contact_metrics(torch.zeros(3, 3), target, valid, groups)
    assert rows[0]['pairs'] == 1 and rows[0]['positives'] == 1
    assert rows[0]['ap'] == 1
    assert sum(r['count'] for r in bins) == 1


def test_reverse_process_is_symmetric_absorbing_and_samples_both_values():
    state = torch.zeros(20, 20, dtype=torch.uint8)
    state[0, 1] = state[1, 0] = 2
    eligible = ~torch.eye(20, dtype=torch.bool)
    probabilities = torch.full((20, 20), .5)
    result = reverse_mask_step(state, probabilities, eligible, 1000, 0,
                              torch.Generator().manual_seed(7))
    assert torch.equal(result, result.T)
    assert result[0, 1] == 2
    assert (result[eligible] == 1).any() and (result[eligible] == 2).any()
    assert not (result[eligible] == 0).any()
    assert not result.diagonal().any()


def test_proposals_do_not_use_native_truth_or_observation_validity():
    p = torch.tensor([[0., .9, .8], [.9, 0., .7], [.8, .7, 0.]])
    eligible = torch.triu(torch.ones(3, 3, dtype=torch.bool), 1)
    generated = torch.zeros_like(eligible)
    generated[0, 1] = generated[1, 0] = True
    state = torch.zeros(3, 3, dtype=torch.uint8)
    assert discrepancy_pairs(p, generated, eligible, state, count=1).tolist() == [[0, 2]]
    state[0, 2] = state[2, 0] = 1
    assert discrepancy_pairs(p, generated, eligible, state, count=1).tolist() == [[1, 2]]


def test_oracle_constraints_are_valid_and_exclude_unknown_pairs():
    target = torch.tensor([[False, True, False], [True, False, False], [False, False, False]])
    valid = target.clone()
    state = oracle_state(target, valid, torch.ones_like(valid), 16, True,
                         torch.Generator().manual_seed(1))
    assert int((state == 2).sum()) == 2
    assert torch.equal(state, state.T)
    negative = oracle_state(target, valid, torch.ones_like(valid), 16, False,
                            torch.Generator().manual_seed(1))
    assert not negative.any()


def test_inference_batch_removes_labels_and_pads_confidence_features():
    features = dict(n_tokens=3, n_atoms=3, token_types=torch.ones(3, dtype=torch.long),
        atoms_per_token=torch.ones(3, dtype=torch.long), has_frame=torch.ones(3, dtype=torch.bool),
        atom_coords=torch.randn(3, 3), coordinate_mask=torch.ones(3, dtype=torch.bool),
        contact_target=torch.ones(3, 3), contact_valid=torch.ones(3, 3),
        distogram_mask=torch.ones(3, dtype=torch.bool))
    batch = inference_batch(features, torch.zeros(3, 3, dtype=torch.uint8), 'cpu')
    assert not LABEL_KEYS & batch.keys()
    assert batch['atoms_per_token'].shape == (1, 32)
    assert batch['has_frame'].shape == (1, 32)
    assert batch['contact_state'].shape == (1, 32, 32)


def test_geometry_ignores_unresolved_atoms():
    native = np.array([[0., 0., 0.], [2., 0., 0.], [1., 0., 0.]])
    prediction = native.copy(); prediction[2] = 1000.
    scores = sparse_lddt(prediction, native, [True, True, False], [0, 1, 1])
    assert scores['observed_lddt'] == 1
    assert scores['interface_lddt'] == 1
    assert scores['lddt_pairs'] == 1


def test_feature_builders_accept_inference_inputs_without_native_coordinates():
    from helico.model.features import build_ref_features, build_msa_raw
    batch = dict(ref_coords=torch.zeros(1, 3, 3),
                 atom_element_idx=torch.tensor([[6, 7, 8]]),
                 token_types=torch.ones(1, 3, dtype=torch.long),
                 msa=torch.zeros(1, 1, 3, dtype=torch.long))
    charge, reference = build_ref_features(batch)
    msa, _ = build_msa_raw(batch)
    assert charge.shape == (1, 3, 1)
    assert reference.shape == (1, 3, 385)
    assert msa.shape == (1, 1, 3, 34)


def test_panel_selection_is_disjoint_and_order_independent():
    records = [dict(pdb_id=str(i), status='ready', eval_types=[kind])
               for i, kind in enumerate(['ligand_prot'] * 3 + ['prot_prot'] * 3 + ['intra_prot'] * 3)]
    a = select_panel(records, per_group=2)
    b = select_panel(list(reversed(records)), per_group=2)
    assert a == b
    assert len({r['pdb_id'] for r in a}) == 6


def test_time_budget_counts_complete_batches_and_selection_ignores_quality():
    from helico.contact_eval_analysis import candidates_at_budget
    candidates = [dict(available_seconds=10., ranking=.9, observed_lddt=.4),
                  dict(available_seconds=10., ranking=.2, observed_lddt=.8),
                  dict(available_seconds=20., ranking=1., observed_lddt=.95)]
    assert candidates_at_budget(candidates, 9.) is None
    result = candidates_at_budget(candidates, 15.)
    assert result['candidates'] == 2
    assert result['selected']['observed_lddt'] == .4
    assert result['oracle_observed_lddt'] == .8


def test_official_summary_selects_confidence_and_keeps_failed_denominator(tmp_path):
    import pandas as pd
    from helico.contact_foldbench_score import summarize
    category='interface_protein_ligand'
    predictions=pd.DataFrame([dict(model='ema',pdb_id='a',seed=0,sample=0,ranking_score=.9),
                              dict(model='ema',pdb_id='a',seed=0,sample=1,ranking_score=.2)])
    scores=pd.DataFrame([{**r,'category':category,'target_index':0,'status':'complete','success':r['sample']==1,
                          'rmsd':4. if r['sample']==0 else 1.,'lddt_pli':.9}
                         for r in predictions.to_dict('records')])
    targets=pd.DataFrame([dict(pdb_id=p,category=category,target_index=i) for i,p in enumerate(['a','b'])])
    result=summarize(predictions,scores,targets,tmp_path).iloc[0]
    assert result.denominator==2 and result.scored==1 and result.failed_or_missing==1
    assert result.successes==0 and result.success_rate_all==0
    oracle=pd.read_csv(tmp_path/'oracle_summary.csv').iloc[0]
    assert oracle.successes==1 and oracle.success_rate_all==.5


def test_rmsd_removes_rigid_motion_but_not_reflections():
    from helico.contact_state_analysis import aligned_rmsd
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
    rotation=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    assert aligned_rmsd(points,points@rotation+7.)<1e-6
    assert aligned_rmsd(points,points*np.array([-1.,1.,1.]))>.1


def test_official_target_chain_schemas_for_polymer_ligand_and_monomer():
    from helico.contact_foldbench_score import native_chains
    assert native_chains(dict(native_chain_id_1='A-2',native_chain_id_2='F-2'))==('A-2','F-2')
    assert native_chains(dict(interface_chain_id_1='A',interface_chain_id_2='C'))==('A','C')
    assert native_chains(dict(chain_id='B'))==('B','B')


def test_official_paired_effect_weights_targets_and_counts_missing_as_failure(tmp_path):
    import pandas as pd
    from helico.contact_foldbench_score import summarize_paired
    category='interface_protein_protein'
    targets=pd.DataFrame([dict(category=category,pdb_id=p,target_index=i)
                          for i,p in enumerate(['a','a','b'])])
    selected=pd.DataFrame([dict(category=category,model=model,target_index=i,status='complete',success=hit)
                          for model,entries in [('ema',[(0,True),(1,True)]),
                                                ('protenix',[(0,False),(1,False),(2,True)])]
                          for i,hit in entries])
    summarize_paired(selected,targets,tmp_path)
    row=pd.read_csv(tmp_path/'paired_differences.csv').iloc[0]
    assert row.denominator==3 and row.assemblies==2 and row.common_scored==2
    assert abs(row['mean']-1/3)<1e-10 and row.mean_common_scored==1
