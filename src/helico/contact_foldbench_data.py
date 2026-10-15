"""Full FoldBench chemical inputs, rebuilt without reference complex geometry.

Reference CIFs supply identities and sequences only. The upstream inference
pipeline reconstructs atoms/conformers from CCD before assembling any features.
Reference contacts, coordinates, pockets and inter-entity bonds are not inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import traceback

import numpy as np
import pandas as pd
import torch

from helico.datasets import file_sha256


def chemical_input(task):
    root, output, pid, categories = task
    root, output = Path(root), Path(output)
    dest = output / 'inputs' / (pid + '.json')
    marker = output / 'identity_records' / (pid + '.json')
    if marker.exists():
        return json.loads(marker.read_text())
    source = root / 'examples/ground_truths' / (pid + '.cif.gz')
    reference = output / 'references' / (pid + '.cif')
    for path in (dest, marker, reference): path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from protenix.data.inference.json_maker import cif_to_input_json
        reference.write_bytes(gzip.decompress(source.read_bytes()))
        sample = cif_to_input_json(str(reference), sample_name=pid, save_entity_and_asym_id=True)
        sample.pop('covalent_bonds', None)
        if set(sample) != {'name', 'sequences'}:
            raise ValueError('Unexpected upstream input fields: ' + str(set(sample)))
        dest.write_text(json.dumps(sample, indent=2) + '\n')
        record = dict(pdb_id=pid, status='ready', categories=categories,
            reference_sha256=file_sha256(reference), input_sha256=file_sha256(dest),
            polymer_tokens=sum(len(c.get('sequence', '')) * c['count']
                              for entry in sample['sequences'] for c in entry.values()),
            chains=sum(c['count'] for entry in sample['sequences'] for c in entry.values()))
    except Exception as error:
        record = dict(pdb_id=pid, status='failed', categories=categories,
                      error=repr(error), traceback=traceback.format_exc())
    marker.write_text(json.dumps(record, indent=2) + '\n')
    return record


def extract_chemical_inputs(root, output, workers=4):
    root, output = Path(root), Path(output)
    categories = {}
    for path in sorted((root / 'targets').glob('*.csv')):
        for pid in pd.read_csv(path).pdb_id.unique():
            categories.setdefault(pid, []).append(path.stem)
    tasks = [(str(root), str(output), p, c) for p, c in sorted(categories.items())]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(chemical_input, tasks):
            records.append(record)
            print(json.dumps(dict(stage='identity', done=len(records), total=len(tasks), **record)), flush=True)
    (output / 'identity_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


def adapt_inference_features(f):
    """Public upstream inference features only; this API accepts no labels."""
    index = f['atom_to_token_idx'].long(); n, a = len(f['token_index']), len(index)
    representative = f['distogram_rep_atom_mask'].bool().nonzero().flatten()
    if len(representative) != n or not torch.equal(index[representative], torch.arange(n)):
        raise ValueError('Representative atom/token mismatch')
    out = dict(token_types=f['restype'].argmax(-1), restype=f['restype'].argmax(-1),
        rel_pos=f['residue_index'], token_index=f['token_index'], chain_indices=f['asym_id'],
        entity_id=f['entity_id'], sym_id=f['sym_id'],
        chain_same=f['asym_id'][:, None] == f['asym_id'][None, :],
        token_mask=torch.ones(n, dtype=torch.bool), token_bonds=f['token_bonds'],
        rep_atom_idx=representative, ref_coords=f['ref_pos'].float(), atom_to_token=index,
        atom_element_idx=f['ref_element'].argmax(-1),
        atom_name_chars=f['ref_atom_name_chars'].flatten(-2).float(),
        ref_charge=f['ref_charge'].float(), ref_mask=f['ref_mask'].bool(),
        ref_space_uid=f['ref_space_uid'], atom_mask=torch.ones(a, dtype=torch.bool),
        msa=f['msa'].long(), msa_profile=f['profile'].float(),
        has_deletion=f['has_deletion'].float(), deletion_value=f['deletion_value'].float(),
        deletion_mean=f['deletion_mean'].float(), is_ligand=f['is_ligand'].bool(),
        is_dna=f['is_dna'].bool(), is_rna=f['is_rna'].bool(), n_tokens=n, n_atoms=a,
        atoms_per_token=torch.bincount(index, minlength=n), has_frame=f['has_frame'].bool())
    return out


def attach_alignments(root, output, assets):
    """Stage pinned upstream alignments; keep all remaining coverage explicit."""
    root, output = Path(root), Path(output)
    source_root = assets.root
    maps = dict(proteinChain=json.loads((source_root / 'common/seq_to_pdb_index.json').read_text()),
                rnaSequence=json.loads((source_root / 'rna_msa/rna_sequence_to_pdb_chains.json').read_text()))
    samples = [json.loads(p.read_text()) for p in sorted((output / 'inputs').glob('*.json'))]
    selected = {}
    for sample in samples:
        for entry in sample['sequences']:
            for kind, chain in entry.items():
                seq = chain.get('sequence', '')
                if kind not in maps or (kind, seq) in selected: continue
                match = maps[kind].get(seq)
                if isinstance(match, list): match = match[0] if match else None
                selected[kind, seq] = str(match) if match is not None else None
    requested = []
    for (kind, seq), match in selected.items():
        if match is None: continue
        if kind == 'proteinChain':
            requested.extend(f'mmcif_msa_template/{match}/{name}.a3m'
                             for name in ('pairing', 'non_pairing'))
        else:
            requested.extend(p for p in assets.entries if p.startswith(f'rna_msa/msas/{match}/'))
    assets.fetch_many(p for p in requested if p in assets.entries)
    alignment_dir = output / 'alignments'; alignment_dir.mkdir(exist_ok=True)
    records, lookup = [], {}
    for (kind, seq), match in selected.items():
        key = hashlib.sha256((seq + '\n').encode()).hexdigest()
        pair, unpair, source = '', '', 'query_only'
        if match is not None:
            if kind == 'proteinChain':
                directory = source_root / 'mmcif_msa_template' / match
                p, u = directory / 'pairing.a3m', directory / 'non_pairing.a3m'
                pair = p.read_text() if p.exists() else ''
                unpair = u.read_text() if u.exists() else ''
            else:
                files = sorted((source_root / 'rna_msa/msas' / match).glob('*.a3m'))
                unpair = files[0].read_text() if files else ''
            if pair or unpair: source = 'pinned_protenix'
        bundled = root / 'foldbench-msas' / (key + '.a3m.gz')
        if not pair and not unpair and bundled.exists():
            unpair = gzip.open(bundled, 'rt').read(); source = 'foldbench_bundled'
        # Empty input is represented explicitly as a query-only alignment.
        if not pair and not unpair: unpair = '>query\n' + seq + '\n'
        for value in (pair, unpair):
            if value:
                query = ''.join(value.splitlines()[1:next((i for i,l in enumerate(value.splitlines()[1:],1) if l.startswith('>')),len(value.splitlines()))])
                if re.sub('[a-z.-]', '', query) != seq:
                    raise ValueError('Alignment query differs from full input sequence: ' + key)
        paths = {}
        for name, value in [('pairedMsaPath', pair), ('unpairedMsaPath', unpair)]:
            if not value: continue
            path = alignment_dir / (key + '-' + name + '.a3m'); path.write_text(value)
            paths[name] = str(path)
        lookup[kind, seq] = paths
        records.append(dict(kind=kind, sequence_sha256=key, length=len(seq), source=source,
            paired_rows=pair.count('>'), unpaired_rows=unpair.count('>'),
            files={k:dict(path=Path(v).name,sha256=file_sha256(v)) for k,v in paths.items()}))
    aligned = output / 'aligned_inputs'; aligned.mkdir(exist_ok=True)
    for sample in samples:
        for entry in sample['sequences']:
            for kind, chain in entry.items():
                chain.update(lookup.get((kind, chain.get('sequence','')), {}))
        (aligned / (sample['name'] + '.json')).write_text(json.dumps(sample, indent=2) + '\n')
    (output / 'alignment_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


def prepare_one(task):
    output, pid, max_tokens = task
    output = Path(output)
    dest = output / 'features' / (pid + '.pt.gz')
    marker = dest.with_suffix('').with_suffix('.json')
    if marker.exists(): return json.loads(marker.read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads((output / 'identity_records' / (pid + '.json')).read_text())
    if record['status'] != 'ready': return record
    try:
        torch.set_num_threads(1)
        seed = int(hashlib.sha256(('exp23:' + pid).encode()).hexdigest()[:8], 16)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        sample = json.loads((output / 'aligned_inputs' / (pid + '.json')).read_text())
        if record['polymer_tokens'] > max_tokens:
            raise ValueError(f'Full-input polymer token count exceeds resource ceiling {max_tokens}')
        from ml_collections import ConfigDict
        from protenix.data.inference.infer_dataloader import InferenceDataset
        # The constructor expects a list. Its public process_one method takes
        # the chemical sample; use a shared empty list for constructor metadata.
        config = ConfigDict(dict(input_json_path=str(output / 'empty.json'),
            dump_dir=str(output / 'inference_metadata'), use_msa=True, use_template=False,
            use_rna_msa=True, msa_pair_as_unpair=True, esm=dict(enable=False, model_name='disabled'),
            sample_diffusion=dict(guidance=dict(enable=False))))
        data, atoms, _ = InferenceDataset(config).process_one(sample)
        f = data['input_feature_dict']; features = adapt_inference_features(f)
        n = features['n_tokens']
        if n > max_tokens:
            raise ValueError(f'Full-input {n} tokens exceeds resource ceiling {max_tokens}')
        raw_rows = len(features['msa'])
        if raw_rows > 512:
            rows = torch.cat([torch.zeros(1, dtype=torch.long),
                torch.randperm(raw_rows - 1, generator=torch.Generator().manual_seed(seed))[:511] + 1])
            for key in ('msa', 'has_deletion', 'deletion_value'): features[key] = features[key][rows]
        def token_mask(name):
            return torch.zeros(n, dtype=torch.long).scatter_add_(0, features['atom_to_token'], f[name].long()) > 0
        payload = dict(features=features, protein_token=token_mask('is_protein'),
            ligand_token=token_mask('is_ligand'), pdb_id=pid, seed=seed, atom_array=atoms,
            input_json=sample, categories=record['categories'])
        with gzip.open(dest, 'wb', compresslevel=1) as handle: torch.save(payload, handle)
        record.update(status='ready', tokens=n, atoms=features['n_atoms'],
            msa_rows=len(features['msa']), raw_msa_rows=raw_rows, sha256=file_sha256(dest))
    except Exception as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    marker.write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['identity', 'features'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=2048)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'identity':
        extract_chemical_inputs(args.root, args.output, args.workers)
    else:
        (args.output / 'empty.json').write_text('[]\n')
        records = json.loads((args.output / 'identity_manifest.json').read_text())
        tasks = [(str(args.output), r['pdb_id'], args.max_tokens) for r in records]
        prepared = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(prepare_one, tasks):
                prepared.append(row)
                print(json.dumps(dict(done=len(prepared), **row)), flush=True)
        (args.output / 'features/manifest.json').write_text(json.dumps(prepared, indent=2) + '\n')


if __name__ == '__main__':
    main()
