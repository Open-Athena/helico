"""Account for every observed Iris execution attempt, including superseded runs."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = Path(__file__).resolve().parent / 'data'
    receipts = {}
    for path in data.glob('*.json'):
        value = json.loads(path.read_text())
        if isinstance(value, dict) and 'job_id' in value and 'spec' in value:
            receipts[value['job_id']] = value['spec']
    source = json.loads(args.attempts.read_text())
    frame = pd.DataFrame(source['attempts'])
    if set(frame.job_id) - receipts.keys():
        raise ValueError('Missing submission receipts')
    if frame.duplicated(['task_id', 'attempt']).any():
        raise ValueError('Duplicate execution attempt')
    frame['gpus'] = frame.job_id.map(lambda key: receipts[key]['gpus'])
    frame['observed_gpu_hours'] = frame.observed_seconds * frame.gpus / 3600
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'attempts.csv', index=False)
    summary = frame.groupby(['job_id', 'job_state'], as_index=False).agg(
        attempts=('attempt', 'size'), observed_gpu_hours=('observed_gpu_hours', 'sum'))
    summary.to_csv(args.output / 'jobs.csv', index=False)
    hours = float(frame.observed_gpu_hours.sum())
    usage = dict(observed_at_ms=source['observed_at_ms'], jobs=int(frame.job_id.nunique()),
                 attempts=len(frame), observed_execution_h100_hours=hours,
                 whole_study_h100_hour_cap=512, incremental_storage_transfer_allowance_usd=96,
                 active_jobs=sorted(set(frame.loc[frame.job_state.isin(['running', 'pending']), 'job_id'])),
                 accounting_scope='All Iris attempts, including failures, preemptions and superseded inference. '
                     'Execution timestamps do not include queue time or exact container-start GPU reservation. '
                     'Local A5000 diagnostic smoke checks are separate; this is not an invoice.')
    (args.output / 'usage.json').write_text(json.dumps(usage, indent=2) + '\n')
    if hours > 512:
        raise ValueError('Observed execution already exceeds the whole-study cap')
    print(json.dumps(usage, indent=2))


if __name__ == '__main__':
    main()
