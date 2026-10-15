"""Validate and package the full FoldBench inputs for an immutable HF release."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_feature(task):
    root, record = task
    import torch
    from helico.contact_evaluate import LABEL_KEYS, load_payload
    torch.set_num_threads(1)
    path = Path(root)/'features'/(record['pdb_id']+'.pt.gz')
    if sha(path) != record['sha256']:
        raise ValueError('Feature checksum mismatch: '+record['pdb_id'])
    payload = load_payload(path)
    f = payload['features']
    if LABEL_KEYS & f.keys():
        raise ValueError('Native labels in inference features')
    if f['n_tokens'] != record['tokens'] or f['n_atoms'] != record['atoms']:
        raise ValueError('Feature shape differs from manifest')
    if not 1 <= len(f['msa']) <= 512 or not torch.isfinite(f['ref_coords']).all():
        raise ValueError('Invalid input alignment/reference conformer')
    return record['pdb_id']


def archive(path, entries):
    # Stable archive metadata; entries are (relative name, bytes).
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=1) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for name, data in entries:
                info = tarfile.TarInfo(name)
                info.size = len(data);info.mode = 0o644;info.mtime = 0
                tar.addfile(info, io.BytesIO(data))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-ids',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'features/manifest.json').read_text())
    identities=json.loads((root/'identity_manifest.json').read_text())
    ids={r['pdb_id'] for r in identities}
    if len(ids)!=1522 or len(manifest)!=1522 or {r['pdb_id'] for r in manifest}!=ids:
        raise ValueError('Incomplete or duplicated official target inputs')
    overlap=sorted({p.split('-')[0] for p in ids}&set(args.train_ids.read_text().split()))
    if overlap:raise ValueError('Fine-tuning train/benchmark overlap: '+str(overlap))
    ready=[r for r in manifest if r['status']=='ready']
    failures=[r for r in manifest if r['status']!='ready']
    if any('resource ceiling' not in r.get('error','') for r in failures):
        raise ValueError('Unresolved preparation error')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,_ in enumerate(pool.map(verify_feature,[(str(root),r) for r in ready])):
            if i%100==0:print('VERIFIED',i+1,'of',len(ready),flush=True)
    alignments=json.loads((root/'alignment_manifest.json').read_text())
    names=set()
    for row in alignments:
        if row['source']=='query_only' and row['length']>=15:
            raise ValueError('Unfinished sequence alignment acquisition')
        for item in row['files'].values():
            path=root/'alignments'/item['path']
            if sha(path)!=item['sha256']:raise ValueError('Alignment checksum mismatch')
            names.add(item['path'])
    archive(out/'alignments.tar.gz',((f'alignments/{name}',(root/'alignments'/name).read_bytes()) for name in sorted(names)))
    def inputs():
        for path in sorted((root/'inputs').glob('*.json')):
            yield 'inputs/'+path.name,path.read_bytes()
            sample=json.loads((root/'aligned_inputs'/path.name).read_text())
            for entity in sample['sequences']:
                for chain in entity.values():
                    for key in ['pairedMsaPath','unpairedMsaPath']:
                        if key in chain:chain[key]='alignments/'+Path(chain[key]).name
            yield 'aligned_inputs/'+path.name,(json.dumps(sample,indent=2)+'\n').encode()
    archive(out/'chemical_inputs.tar.gz',inputs())
    for name in ['identity_manifest.json','alignment_manifest.json','source_manifest.json','target_chain_audit.json']:
        shutil.copyfile(root/name,out/name)
    shutil.copyfile(root/'features/manifest.json',out/'manifest.json')
    summary=dict(target_assemblies=len(ids),ready=len(ready),resource_failures=len(failures),
        failure_categories=dict(Counter(c for r in failures for c in r['categories'])),
        alignment_sources=dict(Counter(r['source'] for r in alignments)),
        fine_tuning_train_overlap=overlap,train_ids_sha256=sha(args.train_ids),
        feature_manifest_sha256=sha(out/'manifest.json'),
        archives={name:dict(sha256=sha(out/name),bytes=(out/name).stat().st_size)
                  for name in ['alignments.tar.gz','chemical_inputs.tar.gz']})
    (out/'preparation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'README.md').write_text(f'''---
tags:
- protein-structure
- foldbench
- helico
- evaluation
---

# Full-input FoldBench inference features for Helico exp23

This evaluation-only dataset contains all 1,522 official FoldBench assemblies: {len(ready)} prepared full inputs and {len(failures)} explicit resource-ceiling failures. No structural cropping or native pocket is used. The failures remain in official category denominators.

Features are rebuilt with the public Protenix v1 inference pipeline from full sequences and CCD chemical identities. Native complex coordinates, contacts, observation masks, pocket geometry and inter-entity covalent constraints are absent from model inputs. `ref_coords` are independently reconstructed CCD reference conformers. Reference structures are accessed separately for scoring from the pinned upstream source.

`features/<assembly>.pt.gz` contains trusted PyTorch payloads (including a Biotite AtomArray for mmCIF export); `features/manifest.json` and the identical root `manifest.json` record content hashes and explicit failures. Features are complete tensors; embedded absolute alignment paths are unused preparation metadata. To rebuild features, extract `chemical_inputs.tar.gz` and `alignments.tar.gz`, resolve the relative alignment paths against the extraction directory, and use the pinned upstream inference implementation.

MSAs prefer paired/unpaired alignments from pinned Protenix data, followed by supplied FoldBench alignments. Missing protein alignments were generated with the ColabFold MMseqs2 server on 2026-10-15; individual sequences, sources and file hashes are recorded in `alignment_manifest.json`. Remaining query-only inputs are peptides shorter than 15 residues. All alignment files are included in the archive. Profiles use complete alignments; feature tensors retain the query and at most 511 other rows using per-assembly deterministic seeds. Current MSAs mean results should be compared to the identically configured initialization baseline, not published Protenix benchmark scores with other inputs/settings.

Maximum full-input size is 2,048 tokens. Ligands are atom tokens, so the token cap can exceed the polymer residue count. No fine-tuning training PDB ID overlaps these assemblies; `preparation_summary.json` records the checked training-ID file and the full input audit. This is a PDB-ID split audit, not a claim of sequence dissimilarity.

## Provenance

- FoldBench source: [timodonnell/helico-data@382ad3219097b764500c75475aaae3f82547f4fd](https://huggingface.co/datasets/timodonnell/helico-data/tree/382ad3219097b764500c75475aaae3f82547f4fd/benchmarks/FoldBench). All 1,534 used source files were content-verified; see `source_manifest.json`.
- Protenix implementation: `bytedance/Protenix@85767b811c40ed46e73a9b39519cf6bfca8701ba`.
- Protenix data: `LiteFold/protenix-data@47150f5244c15967e69315f3937d0ebcf863f317`.
- Fine-tuning split: [timodonnell/helico-protenix-v1-pdb@164e41cf98d7fdf4d437192a9947138ca2464011](https://huggingface.co/datasets/timodonnell/helico-protenix-v1-pdb/tree/164e41cf98d7fdf4d437192a9947138ca2464011); training release-date cutoff before 2021-09-30 and FoldBench exclusions retained.
- Preparation: `helico.contact_foldbench_data` and `experiments/exp23_contact-evaluation/prepare_foldbench_release.py` in [the experiment source](https://github.com/Open-Athena/helico/tree/codex/clean-contact-pilot).
- [Experiment 23](https://github.com/Open-Athena/helico/issues/23). Official benchmark inference is label-free; reference scoring is performed after protocol selection using validation only.

Feature-manifest SHA256: `{summary['feature_manifest_sha256']}`.
''')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
