"""Freeze matched full-benchmark shards after publishing verified HF inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
from helico.contact_evaluate import assign_targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--dataset-revision',required=True)
    parser.add_argument('--features-uri',required=True)
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.dataset_revision):
        raise ValueError('An immutable HF dataset revision is required')
    root=Path(__file__).resolve().parents[2]
    records=json.loads(args.manifest.read_text())
    if len(records)!=1522 or len({r['pdb_id'] for r in records})!=1522:
        raise ValueError('Expected all 1522 unique benchmark assemblies')
    template=json.loads((root/'configs/eval/contact-blind-preflight-v2.json').read_text())
    partitions=assign_targets(records,16,balanced=True)
    shards=[[r for part in partitions[4*i:4*i+4] for r in part] for i in range(4)]
    rows=[]
    for i,shard in enumerate(shards):
        rows.extend(dict(shard=i,pdb_id=r['pdb_id'],status=r['status'],tokens=r['tokens']) for r in shard)
        for weights in ['ema','protenix']:
            name=f'foldbench-{weights}-{i}-v1'
            config={**template,'target_ids':sorted(r['pdb_id'] for r in shard),
                'run_name':'exp23-contact-eval-'+name,
                'output_uri':'s3://marin-us-east-02a/helico/evaluations/exp23/'+name,
                'features_uri':args.features_uri,
                'features_manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'features_dataset':dict(repo_id='timodonnell/helico-foldbench-eval-v1',revision=args.dataset_revision),
                'weights':[weights], 'allow_resource_failures':True,'balance_by_tokens':True,
                'benchmark_shard':i,'benchmark_shards':4}
            if weights=='protenix':
                config['model_source']=dict(url='https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt',
                    sha256='2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04')
            path=root/'configs/eval'/('contact-'+name+'.json')
            encoded=json.dumps(config,indent=2)+'\n'
            if path.exists() and path.read_text()!=encoded:
                raise ValueError('Refusing to change an existing immutable shard configuration')
            path.write_text(encoded)
    rows.extend(dict(shard=-1,pdb_id=r['pdb_id'],status=r['status'],tokens=r.get('tokens'))
                for r in records if r['status']!='ready')
    output=root/'experiments/exp23_contact-evaluation/data'
    pd.DataFrame(rows).to_csv(output/'foldbench_partitions.csv',index=False)
    protocol=json.loads((output/'foldbench_protocol.json').read_text())
    protocol.update(features_dataset=dict(repo_id='timodonnell/helico-foldbench-eval-v1',revision=args.dataset_revision),
                    feature_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    initialization_and_ema_comparison_frozen_before_scoring=True)
    (output/'foldbench_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('Shard sizes:',[len(s) for s in shards])
    print('Ready:',sum(map(len,shards)),'explicit input resource failures:',sum(r['status']!='ready' for r in records))


if __name__=='__main__':main()
