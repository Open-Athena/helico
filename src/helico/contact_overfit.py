"""Fixed training crops and paired inference for a bounded memorization diagnostic."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

from helico.datasets import file_sha256


def validate_manifest(manifest, lock_sha):
    if manifest['source_data_lock_sha256'] != lock_sha:
        raise ValueError('Fixed crops use a different source data lock')
    audit = manifest['split_audit']
    if not audit['all_train'] or audit['validation_pool_overlap'] or audit['foldbench_overlap']:
        raise ValueError('Overfit crops must belong only to the original training split')
    records = manifest['records']
    if not records or len({r['pdb_id'] for r in records}) != len(records):
        raise ValueError('Fixed crops must have distinct PDB identifiers')
    for record in records:
        if record['path'] != record['pdb_id'] + '.pt.gz' or record['status'] != 'ready':
            raise ValueError('Invalid fixed-crop record')


def stage_features(settings, lock_sha, root):
    """Resolve one immutable HF revision, then verify every file before training."""
    from huggingface_hub import hf_hub_download
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    revision = settings['revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Diagnostic dataset must pin a full immutable revision')
    def fetch(name, sha):
        path = Path(hf_hub_download(settings['repo_id'], name, repo_type='dataset',
                                   revision=revision, local_dir=root))
        if file_sha256(path) != sha:
            raise ValueError('Diagnostic feature checksum mismatch: ' + name)
        return path
    manifest_path = fetch('manifest.json', settings['manifest_sha256'])
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, lock_sha)
    for record in manifest['records']:
        fetch(record['path'], record['sha256'])
    return str(root)


class FixedCrops(torch.utils.data.Dataset):
    """Draw seeds still change masks/noise; the structural crop stays fixed."""
    def __init__(self, root, settings, lock_sha):
        from helico.contact_evaluate import load_payload
        root = Path(root)
        if file_sha256(root / 'manifest.json') != settings['manifest_sha256']:
            raise ValueError('Diagnostic manifest checksum mismatch')
        self.manifest = json.loads((root / 'manifest.json').read_text())
        validate_manifest(self.manifest, lock_sha)
        self.records = self.manifest['records']
        self.payloads = []
        for record in self.records:
            path = root / record['path']
            if file_sha256(path) != record['sha256']:
                raise ValueError('Diagnostic feature checksum mismatch: ' + record['pdb_id'])
            payload = load_payload(path)
            if payload['pdb_id'] != record['pdb_id'] or payload['features']['n_tokens'] != record['tokens']:
                raise ValueError('Diagnostic feature identity mismatch')
            # Training caches predate the inference-only pLDDT flattening field.
            # Derive it from input tokenization; no native geometry is involved.
            features = payload['features']
            payload['features'] = {**features, 'atoms_per_token': torch.bincount(
                features['atom_to_token'], minlength=features['n_tokens'])}
            self.payloads.append(payload)
        self.merged_datapoint_weights = [1.] * len(self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        index, draw = key
        return self.payloads[index]['features'], self.records[index]['pdb_id'], draw


def summarize(rows):
    """Average seeds/modes within each target before averaging targets.

    Report absolute satisfaction separately from conditioned-minus-repeat.
    Memorizing an unconditioned structure must not count as contact steering.
    """
    per_target = defaultdict(lambda: defaultdict(list))
    for row in rows:
        metrics = row['metrics'][0]
        values = per_target[row['pdb_id']]
        if row['mode'] == 'unknown':
            values['unconditioned_lddt'].append(metrics['observed_lddt'])
        if row['mode'] in {'rescue1', 'rescue4', 'rescue16'} and metrics['requested_contact_count']:
            for name, key in [('supplied_satisfaction', 'requested_contact_satisfaction'),
                              ('masked_satisfaction', 'repeat_requested_contact_satisfaction'),
                              ('steering', 'change_requested_contact_satisfaction'),
                              ('head_response', 'contact_head_response')]:
                value = metrics[key]
                if not math.isfinite(value):
                    raise ValueError('Nonfinite measured overfit metric: ' + name)
                values[name].append(value)
    targets = {pdb: {k: statistics.mean(v) for k, v in values.items()} for pdb, values in per_target.items()}
    keys = set().union(*(v.keys() for v in targets.values()))
    means = {key: statistics.mean(v[key] for v in targets.values() if key in v) for key in keys}
    return dict(targets=targets, mean=means,
                measured_steering_targets=sum('steering' in v for v in targets.values()),
                prediction_records=len(rows))


def evaluate_fixed(model, dataset, config, output, step, rank, world, *, weights='raw'):
    """Evaluate full coordinate sampling with labels stripped and frozen constraints."""
    from helico.contact_evaluate import ArtifactStore, run_focused
    from helico.contact_continuation_gate import fetch_rows
    from helico.coreweave_worker import filesystem
    from helico.checkpoint_storage import publish_json
    from helico.datasets import write_json
    settings = config['overfit']
    base = config['output_uri'] + '/overfit-eval'
    cfg = dict(stage='focused', weights=[weights], seeds=settings['evaluation_seeds'],
               target_groups={r['pdb_id']: r['group'] for r in dataset.records},
               output_uri=base + f'/step-{step:06d}', checkpoint_step=step,
               features_dataset=settings['dataset'], parent=config['continuation'],
               cycles=1, diffusion_steps=200, reference_weights='raw')
    if step or weights != 'raw':
        cfg['reference_uri'] = base + '/step-000000'
    # Weight-specific hashes prevent one weight's metadata from changing on resume.
    encoded = json.dumps(cfg, sort_keys=True).encode()
    fs = filesystem(durable=True)
    store = ArtifactStore(Path(output) / 'overfit-eval' / f'step-{step:06d}',
                          cfg['output_uri'], fs, hashlib.sha256(encoded).hexdigest())
    previous_msa, previous_steps = model.config.use_msa, model.config.n_diffusion_steps
    # The sampler caches its step count at construction.
    previous_sampler_steps = model.diffusion.n_steps
    model.config.use_msa = True
    model.config.n_diffusion_steps = model.diffusion.n_steps = 200
    model.eval()
    local_records = dataset.records[rank::world]
    try:
        for index in range(rank, len(dataset), world):
            run_focused(model, dataset.payloads[index], dataset.records[index], cfg, store, weights)
        local_cfg = {**cfg, 'target_groups': {r['pdb_id']: r['group'] for r in local_records}}
        rows = list(fetch_rows(fs, local_cfg).values())
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        result = None
        if rank == 0:
            result = summarize([row for partition in gathered for row in partition])
            result.update(step=step, weights=weights, status='complete', evaluation_config=cfg)
            publish_json(fs, cfg['output_uri'] + f'/summary-{weights}.json', result)
            write_json(Path(output) / f'overfit-summary-{step}-{weights}.json', result)
        return result
    finally:
        model.config.use_msa = previous_msa
        model.config.n_diffusion_steps = previous_steps
        model.diffusion.n_steps = previous_sampler_steps
        model.train()
