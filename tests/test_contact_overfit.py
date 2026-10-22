import copy

import pytest

from helico.contact_overfit import summarize, validate_manifest
from helico.train_contacts import check_continuation


def manifest():
    return dict(source_data_lock_sha256='lock',
                split_audit=dict(all_train=True, validation_pool_overlap=[], foldbench_overlap=[]),
                records=[dict(pdb_id='1abc', path='1abc.pt.gz', status='ready')])


@pytest.mark.parametrize('field', ['validation_pool_overlap', 'foldbench_overlap'])
def test_overfit_refuses_validation_or_foldbench_examples(field):
    data = manifest()
    data['split_audit'][field] = ['1abc']
    with pytest.raises(ValueError, match='training split'):
        validate_manifest(data, 'lock')


def test_overfit_checks_data_identity_and_duplicate_targets():
    validate_manifest(manifest(), 'lock')
    with pytest.raises(ValueError, match='data lock'):
        validate_manifest(manifest(), 'other')
    data = manifest()
    data['records'] *= 2
    with pytest.raises(ValueError, match='distinct'):
        validate_manifest(data, 'lock')


def row(pdb, supplied, masked, count=4):
    return dict(pdb_id=pdb, mode='rescue4', metrics=[dict(
        requested_contact_count=count, requested_contact_satisfaction=supplied,
        repeat_requested_contact_satisfaction=masked,
        change_requested_contact_satisfaction=supplied-masked, contact_head_response=.1)])


def test_memorized_unconditioned_structure_is_not_contact_steering():
    result = summarize([row('1abc', 1., 1.), row('2abc', 1., 1.)])
    assert result['mean']['supplied_satisfaction'] == 1
    assert result['mean']['masked_satisfaction'] == 1
    assert result['mean']['steering'] == 0
    assert result['measured_steering_targets'] == 2


def test_summary_weights_targets_equally_and_excludes_empty_constraints():
    result = summarize([row('1abc', 1., 0.)]*6 + [row('2abc', 0., 0.), row('3abc', float('nan'), float('nan'), 0)])
    assert result['mean']['steering'] == .5
    assert result['measured_steering_targets'] == 2
    with pytest.raises(ValueError, match='Nonfinite'):
        summarize([row('1abc', float('nan'), 0.)])


def test_chained_continuation_checks_cumulative_step():
    ckpt = dict(step=1500, total_step=21500, git_sha='parent', data_lock_sha256='lock',
                model_config={'width': 4}, training_config={'lr': 1e-5, 'contact_lr': 1e-4})
    cfg = dict(continuation=dict(step=1500, total_step=21500, git_sha='parent'), lr=1e-5, contact_lr=1e-4)
    check_continuation(ckpt, cfg, {'width': 4}, 'lock')
    wrong = copy.deepcopy(ckpt)
    wrong['total_step'] = 1500
    with pytest.raises(ValueError, match='cumulative'):
        check_continuation(wrong, cfg, {'width': 4}, 'lock')
