"""Export every blind prediction to full mmCIF without selecting by reference."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from helico.contact_evaluate import load_payload


def export_predictions(records_root, features_root, output, fs=None, uri=None):
    from protenix.data.utils import save_structure_cif
    records_root, features_root, output = map(Path, (records_root, features_root, output))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for record_path in sorted((records_root / 'blind').rglob('*.json')):
        record = json.loads(record_path.read_text())
        if record.get('status') != 'complete': continue
        artifact = records_root / record['artifact']
        if not artifact.exists():
            if fs is None: raise FileNotFoundError(artifact)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(uri.rstrip('/') + '/' + record['artifact'], str(artifact))
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != record['artifact_sha256']:
            raise ValueError('Prediction checksum mismatch: ' + str(artifact))
        with gzip.open(artifact, 'rb') as handle:
            prediction = torch.load(handle, map_location='cpu', weights_only=False)
        payload = load_payload(features_root / (record['pdb_id'] + '.pt.gz'))
        entity_types = {str(i + 1): {'proteinChain':'polypeptide(L)',
            'rnaSequence':'polyribonucleotide','dnaSequence':'polydeoxyribonucleotide'}[kind]
            for i, entry in enumerate(payload['input_json']['sequences'])
            for kind in entry if kind in {'proteinChain','rnaSequence','dnaSequence'}}
        for m in record['metrics']:
            path = output / record['weights'] / record['pdb_id'] / f"s{record['seed']}-{m['sample']}.cif"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                save_structure_cif(payload['atom_array'], prediction['all_coords'][0,m['sample']],
                    str(path), entity_types, record['pdb_id'], save_wo_unresolved=False)
            rows.append(dict(pdb_id=record['pdb_id'], model=record['weights'],
                seed=record['seed'], sample=m['sample'], ranking_score=m['ranking'],
                prediction_path=str(path.resolve()), input_sha256=hashlib.sha256(
                    (features_root / (record['pdb_id'] + '.pt.gz')).read_bytes()).hexdigest(),
                prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = pd.DataFrame(rows)
    result.to_csv(output / 'prediction_reference.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(export_predictions(args.records, args.features, args.output))


if __name__ == '__main__': main()
