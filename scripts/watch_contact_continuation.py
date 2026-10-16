"""Bounded experiment-24 controller: train, evaluate, gate, then continue.

Every stage has a fixed deadline, including retries. Only owned jobs can be
cancelled. No FoldBench inputs or fleet mutations occur here. Run in Iris env.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from helico.contact_continuation_gate import assess, fetch_rows
from helico.experiment import ensure_bench_run, ensure_training_run, set_experiment
from scripts.watch_contact_training import storage, read_json, retryable_failure, notify, save


class ExperimentStop(Exception):
    """A positive stop decision, distinct from an unavailable control plane."""


def issue_update(state, message):
    if not state.get('comment_id'):
        return
    body = ('🤖 Focused continuation: ' + message + '\n\n'
            'Eight H100s per job; original Protenix-v1 split and Hugging Face data lock retained. '
            'Whole study: at most 192 prepaid H100-hours and $24 additional first-month storage. '
            '500/1,500-update held-out gates control continuation to 5,000 updates.\n\n'
            'Notebook: https://github.com/Open-Athena/helico/blob/codex/clean-contact-pilot/'
            'experiments/exp24_focused-continuation/README.md\n'
            'W&B: https://wandb.ai/timodonnell/helico\n\n'
            'Controller state: ' + state['state_path'] + '\n')
    payload = json.dumps({'body': body})
    subprocess.run(['gh', 'api', '--method', 'PATCH',
        f"repos/Open-Athena/helico/issues/comments/{state['comment_id']}", '--input', '-'],
        input=payload, text=True, check=True, stdout=subprocess.DEVNULL, timeout=30)


def submit(state, plan, name, retry=False):
    root = Path(state['workspace'])
    source_check = subprocess.run(['git', 'diff', '--exit-code', state['source_sha'], '--',
                    'src', 'configs', 'scripts', 'uv.lock'], cwd=root, check=False,
                   stdout=subprocess.DEVNULL)
    if source_check.returncode:
        raise ExperimentStop('Scientific source changed since controller launch')
    spec = dict(plan['stages'][name])
    job = state['jobs'].get(name)
    if job is None:
        job = dict(deadline=time.time() + spec['timeout_seconds'], retries=0, attempts=[])
        state['jobs'][name] = job
    if retry:
        job['retries'] += 1
    remaining = int(job['deadline'] - time.time())
    if remaining < 300:
        raise ExperimentStop('Insufficient stage budget for submission: ' + name)
    suffix = f"-recovery-{job['retries']}" if job['retries'] else ''
    spec['job_name'] += suffix
    spec['timeout_seconds'] = remaining if retry else spec['timeout_seconds']
    # Persist the exact submission before dispatch. Recovery after a controller
    # crash adopts the same receipt/spec rather than changing its timeout.
    if job.get('status') == 'submitting' and not retry:
        spec = job['pending_spec']
    else:
        job.update(status='submitting', pending_spec=spec)
        save(Path(state['state_path']), state)
    set_experiment(plan['experiment'])
    os.environ.update(HELICO_IRIS_PYTHON=sys.executable, HELICO_IRIS_CONFIG=state['cluster_config'])
    if spec['mode'] == 'train':
        receipt = ensure_training_run(name + suffix, gpu='H100:8', max_steps=5000,
            crop_size=384, lr=1e-5, est_wall_hours=remaining / 3600, coreweave=spec).meta
    else:
        receipt = ensure_bench_run(name + suffix, gpu='H100', workers=8, est_wall_hours=remaining / 3600,
                                  coreweave=spec).meta
    job.update(receipt=receipt, status='submitted')
    job['attempts'].append(dict(job_id=receipt['job_id'], submitted_at=time.time(), spec=receipt['spec']))
    save(Path(state['state_path']), state)
    notify('Submitted ' + receipt['job_id'])


def inspect_jobs(state, plan, fs):
    from iris.cli.connect import open_iris_client
    from iris.cluster.types import JobName
    with open_iris_client(config_file=Path(state['cluster_config']), workspace=Path(state['workspace'])) as client:
        for name, job in state['jobs'].items():
            if job.get('status') in {'complete', 'submitting'}:
                continue
            handle = client.job(JobName.from_wire(job['receipt']['job_id']))
            status = handle.status()
            condition = str(status.state)
            job['last_check'] = dict(time=time.time(), state=condition, pending_reason=status.pending_reason,
                                     error=status.error_message, preemptions=status.preemption_count)
            spec = plan['stages'][name]
            base = spec['output_uri']
            if condition == 'succeeded':
                result = read_json(fs, base + '/result.json')
                if spec['mode'] == 'train':
                    config = json.loads((Path(state['workspace']) / spec['config']).read_text())
                    checkpoint = read_json(fs, base + '/latest.json')
                    if not (result and result.get('stage_complete') and result['step'] == config['stop_after_step']
                            and checkpoint and checkpoint['step'] == result['step'] and checkpoint.get('sha256')
                            and fs.size(checkpoint['checkpoint']) == checkpoint['bytes']):
                        raise ExperimentStop('Incomplete training stage: ' + name)
                    job['checkpoint'] = checkpoint
                elif not result or result.get('status') != 'complete' or result.get('resource_failures'):
                    raise ExperimentStop('Incomplete evaluation stage: ' + name)
                job.update(status='complete', result=result, finished_at=time.time())
                job['attempts'][-1]['finished_at'] = time.time()
                notify('Completed ' + name)
                continue
            if time.time() >= job['deadline']:
                handle.cancel()
                raise ExperimentStop('Stage resource deadline reached: ' + name)
            if condition in {'failed', 'worker_failed', 'killed', 'preempted', 'unschedulable'}:
                failure = (status.error_message or '') + condition
                diagnostics = base + '/diagnostics'
                if fs.exists(diagnostics):
                    folders = sorted(fs.ls(diagnostics, detail=False))
                    if folders and fs.exists(folders[-1] + '/trainer.log'):
                        path = folders[-1] + '/trainer.log'
                        failure += fs.cat_file(path, start=max(0, fs.size(path) - 32768)).decode(errors='replace')
                checkpoint = read_json(fs, base + '/latest.json')
                step = checkpoint['step'] if checkpoint else 0
                if (condition in {'failed', 'worker_failed'} and job['retries'] < 2 and
                        step > job.get('last_recovery_step', -1) and retryable_failure(failure)):
                    job['last_recovery_step'] = step
                    job['attempts'][-1]['finished_at'] = time.time()
                    submit(state, plan, name, retry=True)
                else:
                    job['failure_tail'] = failure[-16000:]
                    raise ExperimentStop('Stopped on job failure: ' + name)


def advance(state, plan, fs):
    jobs = state['jobs']
    complete = lambda name: jobs.get(name, {}).get('status') == 'complete'
    for name, job in list(jobs.items()):
        if job.get('status') == 'submitting':
            submit(state, plan, name)
    for name in ('train-500', 'parent'):
        if name not in jobs:
            submit(state, plan, name)
    for step, next_step in ((500, 1500), (1500, 5000), (5000, None)):
        train, evaluation = f'train-{step}', f'eval-{step}'
        if complete(train) and complete('parent') and evaluation not in jobs:
            submit(state, plan, evaluation)
        if not complete(evaluation):
            return
        key = str(step)
        if key not in state['gates']:
            root = Path(state['workspace'])
            parent_cfg = json.loads((root / plan['stages']['parent']['config']).read_text())
            cfg = json.loads((root / plan['stages'][evaluation]['config']).read_text())
            try:
                result = assess(fetch_rows(fs, parent_cfg), fetch_rows(fs, cfg), cfg, plan['gates'], step)
            except ValueError as error:
                raise ExperimentStop(str(error)) from error
            state['gates'][key] = result
            save(root / plan['experiment_path'] / 'data' / f'gate-{step}.json', result)
            notify(f"Step {step} development gate: {result['decision']}")
            issue_update(state, f"Step {step} development gate: **{result['decision']}**. "
                         + json.dumps(result, allow_nan=False))
        if next_step is None:
            state['status'] = 'completed'
            return
        if state['gates'][key]['decision'] != 'continue':
            state['status'] = 'stopped_by_gate'
            return
        if f'train-{next_step}' not in jobs:
            submit(state, plan, f'train-{next_step}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    lock = args.state.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(args.state.read_text())
    os.chdir(state['workspace'])
    plan = json.loads(Path(state['plan']).read_text())
    if (sum(s['gpus'] * s['timeout_seconds'] / 3600 for s in plan['stages'].values()) > plan['max_gpu_hours']
            or sum(s['estimated_incremental_cost_usd'] for s in plan['stages'].values()) != plan['estimated_incremental_cost_usd']
            or plan['estimated_incremental_cost_usd'] >= 100):
        raise ValueError('Whole-study resource/cost gate failed')
    while state.get('status') not in {'completed', 'stopped_by_gate', 'needs_attention'}:
        try:
            fs = storage(state)
            inspect_jobs(state, plan, fs)
            advance(state, plan, fs)
            state['inspection_errors'] = 0
        except ExperimentStop as error:
            state.update(status='needs_attention', error=str(error))
            notify(str(error))
            # Stop further spending after a controller/scientific failure.
            from iris.cli.connect import open_iris_client
            from iris.cluster.types import JobName
            with open_iris_client(config_file=Path(state['cluster_config']), workspace=Path(state['workspace'])) as client:
                for job in state['jobs'].values():
                    if job.get('status') != 'complete' and job.get('receipt'):
                        client.job(JobName.from_wire(job['receipt']['job_id'])).cancel()
            try:
                issue_update(state, 'Stopped for inspection: ' + str(error))
            except Exception:
                pass
        except Exception as error:
            # A failed control-plane read does not imply a failed worker.
            state['inspection_errors'] = state.get('inspection_errors', 0) + 1
            state['inspection_error'] = type(error).__name__ + ': ' + str(error)
            if state['inspection_errors'] == 3:
                notify('Continuation monitor cannot inspect jobs: ' + type(error).__name__)
        state['updated_at'] = time.time()
        save(args.state, state)
        if args.once:
            return
        if state.get('status') in {'completed', 'stopped_by_gate', 'needs_attention'}:
            return
        time.sleep(120)


if __name__ == '__main__':
    main()
