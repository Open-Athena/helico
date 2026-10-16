import copy

import pytest

from helico.contact_continuation_gate import assess, expected_keys


def panel(steering):
    config = dict(weights=['raw', 'ema'], seeds=[0], target_groups={
        f'{group}{i}': group for group in ('intra', 'pp', 'pl') for i in range(4)})
    rows = {}
    for key in expected_keys(config):
        weight, pdb, offset, mode = key
        rows[key] = dict(metrics=[dict(observed_lddt=.8, interface_lddt=.5,
            interchain_close_atom_pairs=0, requested_contact_count=4,
            change_requested_contact_satisfaction=steering)])
    parent = {k: copy.deepcopy(v) for k, v in rows.items() if k[0] == 'ema'}
    for row in parent.values():
        row['metrics'][0]['change_requested_contact_satisfaction'] = 0.
    return parent, rows, config


THRESHOLDS = dict(min_lddt_delta=-.02, min_interface_delta=-.05,
                  max_clash_delta=.05, min_steering=.05, min_steering_improvement=.03)


def test_long_continuation_requires_steering_and_retention():
    parent, rows, config = panel(0.)
    assert assess(parent, rows, config, THRESHOLDS, 500)['decision'] == 'continue'
    assert assess(parent, rows, config, THRESHOLDS, 1500)['decision'] == 'stop'
    parent, rows, config = panel(.2)
    assert assess(parent, rows, config, THRESHOLDS, 1500)['decision'] == 'continue'
    for key, row in rows.items():
        if key[-1] in {'unknown', 'repeat'}:
            row['metrics'][0]['observed_lddt'] = .7
    assert assess(parent, rows, config, THRESHOLDS, 500)['decision'] == 'stop'


def test_missing_outputs_nonfinite_and_changed_constraint_counts_fail_closed():
    parent, rows, config = panel(.2)
    key = ('raw', 'pp0', 0, 'unknown')
    del rows[key]
    with pytest.raises(ValueError, match='incomplete'):
        assess(parent, rows, config, THRESHOLDS, 500)
    parent, rows, config = panel(.2)
    rows[key]['metrics'][0]['observed_lddt'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        assess(parent, rows, config, THRESHOLDS, 500)
    parent, rows, config = panel(.2)
    rows['raw', 'pp0', 0, 'rescue1']['metrics'][0]['requested_contact_count'] = 1
    with pytest.raises(ValueError, match='constraints changed'):
        assess(parent, rows, config, THRESHOLDS, 1500)
