"""Prepare immutable, full-complex validation inputs for contact evaluation.

Only validation assets are staged. Range reads use the upstream tar inventory,
verify each tar member's identity/size and record its own SHA256. They read the
existing verified whole-shard mirror without claiming a partial data directory
is a fully staged training bundle.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import tarfile
import time

import numpy as np
import torch

from helico.datasets import file_sha256, validate_lock


class RangeAssets:
    def __init__(self, inventory, lock, root, fs, mirror_uri):
        self.root, self.fs, self.mirror_uri = Path(root), fs, mirror_uri
        self.entries, offsets = {}, defaultdict(int)
        self.shards = {e['path']: e for d in lock['datasets']
                       for s in d['manifest']['sources'] for e in s['files']}
        with open(inventory) as handle:
            for row in csv.DictReader(handle):
                size = int(row['size_bytes'])
                row['offset'] = offsets[row['shard_path']]
                offsets[row['shard_path']] += 1536 + ((size + 511) // 512) * 512
                self.entries[row['path']] = row

    def fetch(self, name):
        if name not in self.entries:
            raise FileNotFoundError(f'Asset absent from pinned inventory: {name}')
        row = self.entries[name]
        shard = self.shards[row['shard_path']]
        target = self.root / name
        marker = target.with_name(target.name + '.sha256.json')
        if marker.exists() and target.exists():
            recorded = json.loads(marker.read_text())
            if (recorded['shard_sha256'] == shard['sha256']
                    and target.stat().st_size == int(row['size_bytes'])
                    and file_sha256(target) == recorded['sha256']):
                return target
        uri = self.mirror_uri.rstrip('/') + '/' + shard['sha256']
        for attempt in range(4):
            try:
                remote_marker = json.loads(self.fs.cat_file(uri + '.json'))
                if remote_marker != shard:
                    raise ValueError('Source shard marker differs from dataset lock')
                start, size = row['offset'], int(row['size_bytes'])
                data = self.fs.cat_file(uri, start=start, end=start + 1536 + size)
                with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                    member = archive.next()
                    if member.name != name or member.size != size or not member.isfile():
                        raise ValueError(f'Tar range identity mismatch: {name}')
                    content = archive.extractfile(member).read()
                if len(content) != size:
                    raise ValueError(f'Truncated tar member: {name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_name(target.name + '.pending')
                pending.write_bytes(content)
                pending.replace(target)
                marker.write_text(json.dumps(dict(path=name, bytes=size,
                    shard_sha256=shard['sha256'], offset=start,
                    sha256=hashlib.sha256(content).hexdigest())) + '\n')
                return target
            except (OSError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def fetch_many(self, names, workers=12):
        names = sorted(set(names))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, _ in enumerate(pool.map(self.fetch, names)):
                if i % 50 == 0:
                    print(json.dumps(dict(stage='assets', done=i + 1, total=len(names))), flush=True)


def stage_validation_assets(lock, inventory, metadata_root, root, fs, mirror_uri):
    validate_lock(lock)
    if len(lock['datasets']) != 1:
        raise ValueError('Evaluation currently requires one pinned PDB source')
    root, metadata_root = Path(root), Path(metadata_root)
    dataset = lock['datasets'][0]
    manifest = dataset['manifest']
    split = manifest['splits'][dataset['validation_split']]
    # Verify released split files, including the train and FoldBench exclusions.
    for entry in manifest['files']:
        path = metadata_root / entry['path']
        if not path.is_file() or file_sha256(path) != entry['sha256']:
            raise ValueError(f'Pinned metadata checksum mismatch: {path}')
    ids = (metadata_root / split['pdb_ids']).read_text().split()
    train = set((metadata_root / manifest['splits'][dataset['train_split']]['pdb_ids']).read_text().split())
    heldout = set((metadata_root / 'audit/foldbench_excluded_pdb_ids.txt').read_text().split())
    if set(ids) & (train | heldout):
        raise ValueError('Validation overlaps training or FoldBench')
    assets = RangeAssets(inventory, lock, root, fs, mirror_uri)
    assets.fetch_many(name for name in assets.entries if name.startswith('common/')
                      or name == 'rna_msa/rna_sequence_to_pdb_chains.json')
    assembly_names = [split['bioassembly_dir'] + '/' + pid + '.pkl.gz' for pid in ids]
    assets.fetch_many(assembly_names)
    mappings = [json.loads((root / path).read_text()) for path in
                ['common/seq_to_pdb_index.json', 'rna_msa/rna_sequence_to_pdb_chains.json']]
    dirs = [set(), set()]
    for name in assembly_names:
        with gzip.open(root / name, 'rb') as handle:
            assembly = pickle.load(handle)
        for sequence in assembly.get('sequences', {}).values():
            for mapping, selected in zip(mappings, dirs):
                matches = mapping.get(sequence, [])
                if not isinstance(matches, list):
                    matches = [matches]
                # The pinned MSA featurizer selects the first sequence mapping.
                selected.update(str(v) for v in matches[:1])
    alignments = [name for name in assets.entries
        if (name.startswith('mmcif_msa_template/') and name.split('/')[1] in dirs[0]
            and name.endswith('.a3m'))
        or (name.startswith('rna_msa/msas/') and name.split('/')[2] in dirs[1])]
    assets.fetch_many(alignments)
    report = dict(scope='validation-assets-only', entries=len(ids),
        alignment_files=len(alignments), lock_sha256=lock['lock_sha256'],
        inventory_sha256=file_sha256(inventory), source=manifest['sources'][0]['revision'])
    (root / 'validation-staging.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def validation_dataset(root, index_path, pdb_ids, error_dir):
    """Use the same pinned public featurizer API, without training construction."""
    root = Path(root)
    os.environ['PROTENIX_ROOT_DIR'] = str(root)
    from protenix.data.msa.msa_featurizer import MSAFeaturizer
    from helico.protenix_dataset import ChainContiguousDataset
    # Arguments match the pinned upstream data configuration. Its raw config
    # uses ListValue wrappers, so pass ordinary values through the public API.
    msa = dict(dataset_name='contact-evaluation', enable_prot_msa=True, enable_rna_msa=True,
        prot_seq_or_filename_to_msadir_jsons=[str(root / 'common/seq_to_pdb_index.json')],
        prot_msadir_raw_paths=[str(root / 'mmcif_msa_template')],
        rna_seq_or_filename_to_msadir_jsons=[str(root / 'rna_msa/rna_sequence_to_pdb_chains.json')],
        rna_msadir_raw_paths=[str(root / 'rna_msa/msas')],
        prot_pairing_dbs=['pairing'], prot_non_pairing_dbs=['pairing-non_pairing'],
        prot_indexing_methods=['sequence'], rna_indexing_methods=['sequence'])
    return ChainContiguousDataset(name='contact-evaluation', mmcif_dir=str(root / 'mmcif'),
        bioassembly_dict_dir=str(root / 'recentPDB_bioassembly'), indices_fpath=str(index_path),
        pdb_list=str(pdb_ids), max_n_token=-1, group_by_pdb_id=True,
        sort_by_n_token=False, find_eval_chain_interface=True, random_sample_if_failed=False,
        use_reference_chains_only=False, exclusion={}, error_dir=str(error_dir),
        cropping_configs=dict(crop_size=-1, method_weights=[0., 0., 1.]),
        msa_featurizer=MSAFeaturizer(**msa), template_featurizer=None,
        lig_atom_rename=False, shuffle_mols=False, shuffle_sym_ids=False,
        ref_pos_augment=True, constraint=dict(enable=False))


def prepare_features(root, metadata_root, output, seed=2301):
    from helico.protenix_data import adapt_example
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    metadata_root = Path(metadata_root)
    dataset = validation_dataset(root, metadata_root / 'splits/validation.csv',
        metadata_root / 'splits/validation_pdb_ids.txt', output / 'errors')
    records = []
    for i, frame in enumerate(dataset.indices_list):
        pid = str(frame.iloc[0].pdb_id)
        record_path = output / (pid + '.json')
        if record_path.exists():
            records.append(json.loads(record_path.read_text())); continue
        random.seed(seed + i); np.random.seed(seed + i); torch.manual_seed(seed + i)
        try:
            example = dataset[i]
            features = adapt_example(example)
            raw_msa_rows = len(features['msa'])
            if raw_msa_rows > 512:
                generator = torch.Generator().manual_seed(seed + i)
                rows = torch.cat([torch.zeros(1, dtype=torch.long),
                    torch.randperm(raw_msa_rows - 1, generator=generator)[:511] + 1])
                for key in ('msa', 'has_deletion', 'deletion_value'):
                    features[key] = features[key][rows]
            f = example['input_feature_dict']
            features['atoms_per_token'] = torch.bincount(features['atom_to_token'], minlength=features['n_tokens'])
            if 'has_frame' in f:
                features['has_frame'] = f['has_frame'].bool()
            protein = torch.zeros(features['n_tokens'], dtype=torch.long).scatter_add_(
                0, features['atom_to_token'], f['is_protein'].long()) > 0
            ligand = torch.zeros(features['n_tokens'], dtype=torch.long).scatter_add_(
                0, features['atom_to_token'], f['is_ligand'].long()) > 0
            # Scoring metadata stays outside model inputs.
            payload = dict(features=features, protein_token=protein, ligand_token=ligand,
                eval_types=example['label_dict'].get('eval_type', []),
                chain_1_mask=example['label_dict'].get('chain_1_mask'),
                chain_2_mask=example['label_dict'].get('chain_2_mask'),
                pdb_id=pid, seed=seed + i)
            path = output / (pid + '.pt.gz')
            with gzip.open(path, 'wb', compresslevel=1) as handle:
                torch.save(payload, handle)
            record = dict(pdb_id=pid, status='ready', tokens=features['n_tokens'],
                atoms=features['n_atoms'], msa_rows=len(features['msa']), raw_msa_rows=raw_msa_rows,
                eval_types=sorted(set(frame.eval_type.tolist())), sha256=file_sha256(path))
        except Exception as error:
            import traceback
            record = dict(pdb_id=pid, status='failed', error=repr(error), traceback=traceback.format_exc())
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print(json.dumps(record), flush=True)
    (output / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--metadata-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    prepare_features(args.root, args.metadata_root, args.output)


if __name__ == '__main__':
    main()
