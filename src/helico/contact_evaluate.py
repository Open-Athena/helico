"""Restartable distributed inference-only evaluation of masked contacts.

One independent model replica per GPU. Durable target/arm markers make restarts
skip verified completed work; labels are stripped before every model call.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from helico.contact_eval_metrics import (
    contact_distances, contact_metrics, discrepancy_pairs, geometry_contacts,
    input_pair_groups, oracle_state, reveal_pairs, reverse_mask_step, structure_metrics,
)
from helico.datasets import file_sha256
from helico.model import Helico, HelicoConfig
from helico.protenix_data import batch_example


LABEL_KEYS = {'atom_coords', 'coordinate_mask', 'distogram_mask', 'contact_target', 'contact_valid'}


def inference_batch(features, state, device):
    clean = {k: v for k, v in features.items() if k not in LABEL_KEYS}
    batch = batch_example(clean, state, device)
    padding = (-features['n_tokens']) % 32
    for key in ('atoms_per_token', 'has_frame'):
        if key in batch:
            batch[key] = F.pad(batch[key], (0, padding))
    if LABEL_KEYS & batch.keys():
        raise ValueError('Native labels leaked into inference batch')
    return batch


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_panel(manifest, per_group=8, seed=2301):
    """Balanced nonoverlapping panel chosen from metadata before predictions."""
    candidates = [r for r in manifest if r['status'] == 'ready']
    candidates.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['pdb_id']}".encode()).hexdigest())
    selected, used = [], set()
    for name, types in [('pl', {'ligand_prot'}), ('pp', {'prot_prot', 'antibody_antigen'}), ('intra', {'intra_prot'})]:
        count = 0
        for record in candidates:
            if record['pdb_id'] in used or not (set(record['eval_types']) & types):
                continue
            if name == 'intra' and set(record['eval_types']) & {
                'prot_prot', 'antibody_antigen', 'ligand_prot', 'dna_prot', 'rna_prot', 'peptide_prot'
            }:
                continue
            selected.append({**record, 'panel_group': name})
            used.add(record['pdb_id']); count += 1
            if count == per_group:
                break
    return selected


def assign_targets(records, world, balanced=False):
    ready=[r for r in records if r['status']=='ready']
    if not balanced:return [ready[rank::world] for rank in range(world)]
    partitions=[[] for _ in range(world)];loads=[0]*world
    for record in sorted(ready,key=lambda r:(-r['tokens'],r['pdb_id'])):
        rank=min(range(world),key=lambda i:loads[i])
        partitions[rank].append(record)
        loads[rank]+=record['tokens']**2+100*record['tokens']
    return partitions


class ArtifactStore:
    def __init__(self, root, uri, fs, config_hash):
        self.root, self.uri, self.fs, self.config_hash = Path(root), uri.rstrip('/'), fs, config_hash
        self.root.mkdir(parents=True, exist_ok=True)

    def done(self, name):
        uri = self.uri + '/' + name + '.json'
        if not self.fs.exists(uri):
            return False
        row = json.loads(self.fs.cat_file(uri))
        if row.get('config_sha256') != self.config_hash:
            raise ValueError(f'Existing result has a different configuration: {name}')
        if row.get('status') != 'complete':
            return False
        artifact = row.get('artifact')
        return not artifact or self.fs.size(self.uri + '/' + artifact) == row['artifact_bytes']

    def save(self, name, record, tensors=None):
        from helico.checkpoint_storage import transient_storage_failure
        row = {**record, 'config_sha256': self.config_hash, 'code_sha': os.environ.get('HELICO_CODE_SHA')}
        if tensors is not None:
            artifact = name + '.pt.gz'
            path = self.root / artifact; path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, 'wb', compresslevel=1) as handle:
                torch.save(tensors, handle)
            row.update(artifact=artifact, artifact_bytes=path.stat().st_size, artifact_sha256=file_sha256(path))
            for attempt in range(3):
                try:
                    self.fs.put_file(str(path), self.uri + '/' + artifact)
                    if self.fs.size(self.uri + '/' + artifact) != path.stat().st_size:
                        raise ValueError('Evaluation artifact size mismatch')
                    break
                except Exception as error:
                    if attempt == 2 or not transient_storage_failure(str(error)):
                        raise
                    time.sleep(2 ** attempt)
        self.fs.pipe_file(self.uri + '/' + name + '.json', json.dumps(row, allow_nan=True).encode())
        print(json.dumps(dict(event='result', name=name, status=row['status'])), flush=True)
        return row

    def tensors(self, name):
        row = json.loads(self.fs.cat_file(self.uri + '/' + name + '.json'))
        data = self.fs.cat_file(self.uri + '/' + row['artifact'])
        if hashlib.sha256(data).hexdigest() != row['artifact_sha256']:
            raise ValueError('Evaluation artifact checksum mismatch: ' + name)
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
            return torch.load(handle, map_location='cpu', weights_only=False)


def load_model(checkpoint, weights, cfg, initial=None):
    config = HelicoConfig(**checkpoint['model_config'])
    config.n_diffusion_steps = cfg.get('diffusion_steps', 200)
    config.n_cycles = cfg.get('cycles', 1)
    config.msa_sample_cutoff = config.msa_sample_min_eval = 512
    if weights == 'protenix':
        config.predict_contacts = False; config.use_contacts = False
    model = Helico(config)
    if weights == 'protenix':
        from helico.load_protenix import load_protenix_checkpoint
        stats = load_protenix_checkpoint(initial, model)
        if stats['shape_mismatches']:
            raise ValueError(str(stats))
    else:
        model.load_state_dict(checkpoint['ema_state_dict' if weights == 'ema' else 'model_state_dict'])
    return model.cuda().eval()


def forward_contacts(model, features, state, msa, seed):
    seed_all(seed); model.config.use_msa = msa
    batch = inference_batch(features, state, 'cuda')
    torch.cuda.synchronize(); started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(batch, compute_structure=False, compute_confidence=False)
    torch.cuda.synchronize()
    return out['contact_logits'][0, :features['n_tokens'], :features['n_tokens']].float().cpu(), time.monotonic() - started


def predict(model, features, state, msa, seed, samples=1):
    seed_all(seed); model.config.use_msa = msa
    batch = inference_batch(features, state, 'cuda')
    torch.cuda.synchronize(); started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.predict(batch, n_samples=samples, n_cycles=model.config.n_cycles)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    result = {k: v.detach().float().cpu() for k, v in out.items()
              if k in {'all_coords', 'contact_logits', 'contact_probs', 'all_ptm', 'all_iptm',
                       'all_ranking_score', 'all_has_clash'}}
    if not torch.isfinite(result['all_coords']).all():
        raise ValueError('Nonfinite generated coordinates')
    return result, seconds


def run_audit(model, payload, record, cfg, store, weights):
    f = payload['features']; n = f['n_tokens']
    groups = input_pair_groups(f, payload['protein_token'], payload['ligand_token'])
    state = torch.zeros(n, n, dtype=torch.uint8)
    for msa in cfg.get('msa_modes', [True, False]):
        name = f"audit/{weights}/{record['pdb_id']}/msa{int(msa)}"
        if store.done(name): continue
        logits, seconds = forward_contacts(model, f, state, msa, payload['seed'])
        rows, bins = contact_metrics(logits, f['contact_target'], f['contact_valid'], groups, state)
        store.save(name, dict(status='complete', pdb_id=record['pdb_id'], weights=weights,
            msa=msa, tokens=n, seconds=seconds, metrics=rows, calibration=bins),
            dict(logits=logits.to(torch.float16)))


def run_focused(model, payload, record, cfg, store, weights):
    """Fixed parent-derived missed contacts and matched repeated null predictions.

    The parent run saves every constraint matrix. Continuations reuse those
    exact constraints, so changes in which contacts the model misses cannot
    change the evaluation task. Native labels remain outside inference_batch.
    """
    f = payload['features']; n = f['n_tokens']
    groups = input_pair_groups(f, payload['protein_token'], payload['ligand_token'])
    group = cfg['target_groups'][record['pdb_id']]
    eligible = groups[group] if group in {'intra', 'pp', 'pl'} else groups['all']
    reference = (ArtifactStore(store.root, cfg['reference_uri'], store.fs, '')
                 if cfg.get('reference_uri') else store)
    for offset in cfg.get('seeds', [0, 1]):
        seed = payload['seed'] + offset * 100000
        prefix = f"focused/{weights}/{record['pdb_id']}/s{offset}-"
        reference_prefix = f"focused/ema/{record['pdb_id']}/s{offset}-"
        baseline = repeat = None
        modes = ['unknown', 'repeat']
        if group in {'intra', 'pp', 'pl'}:
            modes += ['rescue1', 'rescue4', 'rescue16', 'negative4']
        for mode in modes:
            name = prefix + mode
            if store.done(name):
                saved = store.tensors(name)
                if mode == 'unknown': baseline = saved
                if mode == 'repeat': repeat = saved
                continue
            state = torch.zeros(n, n, dtype=torch.uint8)
            if mode not in {'unknown', 'repeat'}:
                if cfg.get('reference_uri'):
                    state = reference.tensors(reference_prefix + mode)['contact_state']
                else:
                    contacts = geometry_contacts(baseline['all_coords'][0, 0], f)
                    positive = mode.startswith('rescue')
                    count = int(mode[6:]) if positive else 4
                    state = oracle_state(f['contact_target'], f['contact_valid'],
                        eligible & (~contacts if positive else contacts), count, positive,
                        torch.Generator().manual_seed(seed))
            prediction, seconds = predict(model, f, state, True, seed, 1)
            if mode == 'unknown': baseline = prediction
            if mode == 'repeat': repeat = prediction
            metrics, _ = structure_metrics(prediction['all_coords'][0, 0], f, groups, state)
            if mode not in {'unknown', 'repeat'}:
                null, _ = structure_metrics(repeat['all_coords'][0, 0], f, groups, state)
                for key in ('requested_contact_satisfaction', 'requested_no_contact_satisfaction',
                            'observed_lddt', 'interface_lddt'):
                    metrics['repeat_' + key] = null[key]
                    metrics['change_' + key] = metrics[key] - null[key]
                mask = state != 0
                metrics['contact_head_response'] = (float((prediction['contact_probs'][0, :n, :n][mask]
                    - repeat['contact_probs'][0, :n, :n][mask]).mean()) if mask.any() else float('nan'))
            store.save(name, dict(status='complete', pdb_id=record['pdb_id'], weights=weights,
                group=group, seed=seed, mode=mode, oracle=mode not in {'unknown', 'repeat'},
                tokens=n, seconds=seconds, metrics=[metrics]), {**prediction, 'contact_state': state})


def run_generation(model, payload, record, cfg, store, weights, conditioning=False):
    f = payload['features']; n = f['n_tokens']
    groups = input_pair_groups(f, payload['protein_token'], payload['ligand_token'])
    eligible = groups[record['panel_group']] if conditioning else groups['all']
    for seed_offset in cfg.get('seeds', [0, 1]):
        seed = payload['seed'] + seed_offset * 100000
        modes = ['unknown']
        if conditioning:
            modes += ['positive1', 'positive4', 'positive16', 'negative16',
                      'rescue_positive4', 'corrective_negative4', 'full']
        for msa in cfg.get('msa_modes', [True]):
            baseline = None
            for mode in modes:
                name = f"generation/{weights}/{record['pdb_id']}/s{seed_offset}-msa{int(msa)}-{mode}"
                if store.done(name):
                    if conditioning and mode == 'unknown':
                        baseline = store.tensors(name)
                    continue
                state = torch.zeros(n, n, dtype=torch.uint8)
                generator = torch.Generator().manual_seed(seed)
                if mode.startswith('positive'):
                    state = oracle_state(f['contact_target'], f['contact_valid'], eligible,
                        int(mode[8:]), True, generator)
                elif mode.startswith('negative'):
                    state = oracle_state(f['contact_target'], f['contact_valid'], eligible,
                        int(mode[8:]), False, generator)
                elif mode in {'rescue_positive4', 'corrective_negative4'}:
                    baseline_contacts = geometry_contacts(baseline['all_coords'][0, 0], f)
                    positive = mode == 'rescue_positive4'
                    violated = ~baseline_contacts if positive else baseline_contacts
                    state = oracle_state(f['contact_target'], f['contact_valid'],
                        eligible & violated, 4, positive, generator)
                elif mode == 'full':
                    state = torch.where(f['contact_valid'], f['contact_target'].to(torch.uint8) + 1, 0)
                prediction, seconds = predict(model, f, state, msa, seed, cfg.get('samples', 1))
                if mode == 'unknown':
                    baseline = prediction
                rows = []
                for i, coords in enumerate(prediction['all_coords'][0]):
                    metrics, contacts = structure_metrics(coords, f, groups, state)
                    metrics.update(sample=i, ranking=float(prediction['all_ranking_score'][0, i]),
                        ptm=float(prediction['all_ptm'][0, i]), iptm=float(prediction['all_iptm'][0, i]),
                        model_has_clash=float(prediction['all_has_clash'][0, i]))
                    if conditioning and mode != 'unknown':
                        control_metrics, _ = structure_metrics(baseline['all_coords'][0, i], f, groups, state)
                        for key in ('requested_contact_satisfaction', 'requested_no_contact_satisfaction',
                                    'observed_lddt', 'interface_lddt', 'pp_unrequested_recall',
                                    'pl_unrequested_recall', 'intra_unrequested_recall'):
                            metrics['control_' + key] = control_metrics[key]
                            metrics['change_' + key] = metrics[key] - control_metrics[key]
                    if 'contact_probs' in prediction and mode == 'unknown':
                        probabilities = prediction['contact_probs'][0, :n, :n]
                        proposals = []
                        for group in ('pp', 'pl', 'intra'):
                            pairs = discrepancy_pairs(probabilities, contacts, groups[group], state, count=16)
                            distances = contact_distances(coords, f['atom_to_token'], pairs, f['atom_element_idx'] != 0)
                            for j, ((a, b), distance) in enumerate(zip(pairs.tolist(), distances)):
                                proposals.append(dict(group=group, rank=j + 1, token_i=a, token_j=b,
                                    probability=float(probabilities[a, b]), distance=distance,
                                    native_observed=bool(f['contact_valid'][a, b]),
                                    native_contact=bool(f['contact_target'][a, b])))
                        metrics['missing_contact_proposals'] = proposals
                    rows.append(metrics)
                store.save(name, dict(status='complete', pdb_id=record['pdb_id'], weights=weights,
                    msa=msa, seed=seed, mode=mode, oracle=mode != 'unknown', seconds=seconds,
                    tokens=n, metrics=rows), {**prediction, 'contact_state': state})


def run_search(model, payload, record, cfg, store, weights):
    """Compare adaptive candidates against batched ordinary sampling by time.

    Reference geometry is used only inside score(), after a prediction. Neither
    expansion nor ranking can read it. Complete batches become available at
    their measured completion time; partial-batch compute is never discounted.
    """
    import heapq
    f = payload['features']; n = f['n_tokens']
    groups = input_pair_groups(f, payload['protein_token'], payload['ligand_token'])
    eligible = groups[record['panel_group']]
    # The diffusion baseline models all supported pair types, not only the
    # selected target interface; tree proposals prioritize the panel category.
    empty = torch.zeros(n, n, dtype=torch.uint8)
    for seed_offset in cfg.get('seeds', [0, 1]):
        name = f"search/{weights}/{record['pdb_id']}/s{seed_offset}"
        if store.done(name): continue
        seed = payload['seed'] + seed_offset * 100000
        rows, tensors, costs = [], {}, {}

        def score(prediction, state, method, call, parent, cumulative):
            for sample, coords in enumerate(prediction['all_coords'][0]):
                metrics = (structure_metrics(coords, f, groups, state)[0]
                           if 'contact_target' in f else {})
                metrics.update(method=method, call=call, sample=sample, parent=parent,
                    available_seconds=cumulative,
                    ranking=float(prediction['all_ranking_score'][0, sample]),
                    ptm=float(prediction['all_ptm'][0, sample]),
                    iptm=float(prediction['all_iptm'][0, sample]),
                    model_has_clash=float(prediction['all_has_clash'][0, sample]))
                rows.append(metrics)
            tensors[f'{method}-{call}'] = {**prediction, 'contact_state': state, 'parent': parent}

        # Beam expansion uses only predicted confidence and generated geometry.
        prediction, elapsed = predict(model, f, empty, True, seed)
        score(prediction, empty, 'tree', 0, -1, elapsed)
        root_rank = float(prediction['all_ranking_score'][0, 0])
        queue = [(-root_rank, 0, empty, prediction, 0)]
        call = 1
        max_calls = cfg.get('search_calls', 8)
        while queue and call < max_calls:
            _, parent, parent_state, parent_prediction, depth = heapq.heappop(queue)
            if depth >= cfg.get('search_depth', 3): continue
            coords = parent_prediction['all_coords'][0, 0]
            contacts = geometry_contacts(coords, f)
            probs = parent_prediction['contact_probs'][0, :n, :n]
            pairs = discrepancy_pairs(probs, contacts, eligible, parent_state, count=64)
            distances = contact_distances(coords, f['atom_to_token'], pairs, f['atom_element_idx'] != 0)
            pairs = pairs[torch.tensor([d >= cfg.get('min_proposal_distance', 7.) for d in distances], dtype=torch.bool)]
            for pair in pairs[:cfg.get('branching', 2)]:
                if call >= max_calls: break
                state = reveal_pairs(parent_state, pair[None], 2)
                child, seconds = predict(model, f, state, True, seed + call)
                elapsed += seconds
                score(child, state, 'tree', call, parent, elapsed)
                rank_score = float(child['all_ranking_score'][0, 0])
                heapq.heappush(queue, (-rank_score, call, state, child, depth + 1))
                call += 1
            # Keep the two most promising unexpanded nodes at each iteration.
            queue = heapq.nsmallest(cfg.get('beam_width', 2), queue)
        costs['tree'] = elapsed

        state = empty.clone(); elapsed = 0.
        schedule = cfg.get('mask_schedule', [1000, 995, 980, 950, 850, 600, 300, 0])
        generator = torch.Generator().manual_seed(seed)
        for call, current_t in enumerate(schedule):
            prediction, seconds = predict(model, f, state, True, seed + call)
            elapsed += seconds
            score(prediction, state, 'unmask', call, call - 1, elapsed)
            if call + 1 < len(schedule):
                state = reverse_mask_step(state, prediction['contact_probs'][0, :n, :n],
                    groups['all'], current_t, schedule[call + 1], generator)
        costs['unmask'] = elapsed

        # Ordinary diffusion uses batched samples to amortize trunk computation.
        # Continue until it covers both adaptive methods' inference time.
        elapsed, call = 0., 0
        while elapsed < max(costs.values()):
            prediction, seconds = predict(model, f, empty, True, seed + call,
                                           samples=cfg.get('baseline_batch_samples', 4))
            elapsed += seconds
            score(prediction, empty, 'ordinary', call, -1, elapsed)
            call += 1
            if call >= cfg.get('baseline_max_batches', 32):
                break
        costs['ordinary'] = elapsed
        store.save(name, dict(status='complete', pdb_id=record['pdb_id'], weights=weights,
            seed=seed, oracle=False, tokens=n, costs=costs, metrics=rows,
            baseline_covers_search=costs['ordinary'] >= max(costs['tree'], costs['unmask'])), tensors)


def run_blind_generation(model, payload, record, cfg, store, weights):
    """Test inference has no reference labels available, including for scoring."""
    f = payload['features']; n = f['n_tokens']
    if LABEL_KEYS & f.keys():
        raise ValueError('Blind test payload contains reference labels')
    state = torch.zeros(n, n, dtype=torch.uint8)
    for offset in cfg.get('seeds', [0]):
        name = f"blind/{weights}/{record['pdb_id']}/s{offset}"
        if store.done(name): continue
        prediction, seconds = predict(model, f, state, True,
                                      payload['seed'] + offset * 100000, cfg.get('samples', 2))
        metrics = [dict(sample=i, ranking=float(prediction['all_ranking_score'][0, i]),
                       ptm=float(prediction['all_ptm'][0, i]),
                       iptm=float(prediction['all_iptm'][0, i]),
                       model_has_clash=float(prediction['all_has_clash'][0, i]))
                   for i in range(cfg.get('samples', 2))]
        store.save(name, dict(status='complete', pdb_id=record['pdb_id'], weights=weights,
            seed=offset, msa=True, tokens=n, seconds=seconds, metrics=metrics), prediction)


def run_contact_hypotheses(model, payload, record, cfg, store, weights):
    """Trunk-only absorbing transitions followed by one structure per hypothesis."""
    f=payload['features'];n=f['n_tokens']
    groups=input_pair_groups(f,payload['protein_token'],payload['ligand_token'])
    empty=torch.zeros(n,n,dtype=torch.uint8)
    for offset in cfg.get('seeds',[0,1]):
        name=f"search/{weights}/{record['pdb_id']}/s{offset}"
        if store.done(name):continue
        seed=payload['seed']+offset*100000
        rows,tensors,costs=[],{},{}
        def score(prediction,state,method,call,elapsed,head_calls):
            for sample,coords in enumerate(prediction['all_coords'][0]):
                metrics=structure_metrics(coords,f,groups,state)[0]
                metrics.update(method=method,call=call,sample=sample,parent=-1,
                    available_seconds=elapsed,head_calls=head_calls,
                    ranking=float(prediction['all_ranking_score'][0,sample]),
                    ptm=float(prediction['all_ptm'][0,sample]),
                    iptm=float(prediction['all_iptm'][0,sample]),
                    model_has_clash=float(prediction['all_has_clash'][0,sample]))
                rows.append(metrics)
            tensors[f'{method}-{call}']={**prediction,'contact_state':state}
        for method,schedule in cfg['contact_schedules'].items():
            elapsed=0.
            for call in range(cfg.get('hypotheses',4)):
                state=empty.clone();hypothesis_seed=seed+call*1000
                generator=torch.Generator().manual_seed(hypothesis_seed)
                for step,(t,s) in enumerate(zip(schedule,schedule[1:])):
                    logits,seconds=forward_contacts(model,f,state,True,hypothesis_seed+step)
                    elapsed+=seconds
                    state=reverse_mask_step(state,logits.sigmoid(),groups['all'],t,s,generator)
                prediction,seconds=predict(model,f,state,True,hypothesis_seed,1)
                elapsed+=seconds
                score(prediction,state,method,call,elapsed,len(schedule)-1)
            costs[method]=elapsed
        elapsed,call=0.,0
        while elapsed<max(costs.values()):
            prediction,seconds=predict(model,f,empty,True,seed+call,
                                       cfg.get('baseline_batch_samples',4))
            elapsed+=seconds;score(prediction,empty,'ordinary',call,elapsed,0);call+=1
            if call>=cfg.get('baseline_max_batches',32):break
        costs['ordinary']=elapsed
        store.save(name,dict(status='complete',pdb_id=record['pdb_id'],weights=weights,seed=seed,
            oracle=False,tokens=n,costs=costs,metrics=rows,
            baseline_covers_search=elapsed>=max(costs.values())),tensors)


def load_payload(path):
    with gzip.open(path, 'rb') as handle:
        return torch.load(handle, map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    rank, world = int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.set_num_threads(2); torch.cuda.set_device(rank)
    from helico.coreweave_worker import filesystem
    fs = filesystem(durable=True)
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    store = ArtifactStore(args.output, cfg['output_uri'], fs, config_hash)
    manifest = json.loads((args.features / 'manifest.json').read_text())
    if cfg.get('panel_per_group'):
        manifest = select_panel(manifest, cfg['panel_per_group'], cfg.get('seed', 2301))
    if cfg.get('target_ids'):
        manifest = [r for r in manifest if r['pdb_id'] in cfg['target_ids']]
    targets = assign_targets(manifest, world, cfg.get('balance_by_tokens',False))[rank]
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint['step'] != cfg.get('checkpoint_step', 20000) or checkpoint['data_lock_sha256'] != cfg['data_lock_sha256']:
        raise ValueError('Wrong model checkpoint or data lock')
    failed, resource_failed = [], []
    for weights in cfg['weights']:
        model = load_model(checkpoint, weights, cfg, args.initial)
        for record in targets:
            if cfg.get('default_panel_group'):
                record = {**record, 'panel_group': cfg['default_panel_group']}
            print(json.dumps(dict(event='target', rank=rank, weights=weights,
                pdb_id=record['pdb_id'], stage=cfg['stage'])), flush=True)
            try:
                payload = load_payload(args.features / (record['pdb_id'] + '.pt.gz'))
                if cfg['stage'] == 'smoke':
                    if weights != 'protenix':
                        run_audit(model, payload, record, cfg, store, weights)
                    run_generation(model, payload, record, cfg, store, weights)
                elif cfg['stage'] == 'audit':
                    run_audit(model, payload, record, cfg, store, weights)
                elif cfg['stage'] == 'focused':
                    run_focused(model, payload, record, cfg, store, weights)
                elif cfg['stage'] in {'generation', 'conditioning'}:
                    run_generation(model, payload, record, cfg, store, weights, cfg['stage'] == 'conditioning')
                elif cfg['stage'] == 'search':
                    run_search(model, payload, record, cfg, store, weights)
                elif cfg['stage'] == 'blind':
                    run_blind_generation(model, payload, record, cfg, store, weights)
                elif cfg['stage'] == 'contact-hypotheses':
                    run_contact_hypotheses(model,payload,record,cfg,store,weights)
                else:
                    raise ValueError('Unsupported evaluation stage: ' + cfg['stage'])
            except Exception as error:
                row = dict(status='failed', pdb_id=record['pdb_id'], weights=weights,
                    error=repr(error), traceback=traceback.format_exc())
                store.save(f"errors/{weights}/{record['pdb_id']}", row)
                if cfg.get('allow_resource_failures') and isinstance(error, torch.cuda.OutOfMemoryError):
                    resource_failed.append(row)
                else:
                    failed.append(row)
                # Data/OOM failures are explicit; CUDA context failures abort the worker.
                if 'device-side assert' in str(error) or 'illegal memory access' in str(error):
                    raise
                torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()
    store.save(f'ranks/{rank}', dict(status='complete', rank=rank, targets=len(targets),
                                   failures=len(failed), resource_failures=len(resource_failed)))
    if failed:
        raise RuntimeError(f'{len(failed)} target/weight evaluations failed; durable errors recorded')


if __name__ == '__main__':
    main()
