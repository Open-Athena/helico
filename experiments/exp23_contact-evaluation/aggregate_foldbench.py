"""Combine disjoint completed shards, retaining every official denominator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from helico.contact_foldbench_score import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--foldbench',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--completion',action='store_true')
    parser.add_argument('--unavailable-alignments-reason',type=Path,
                        help='Explicit bounded-recovery failure evidence, if alignments remain unavailable')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    inputs=json.loads(args.manifest.read_text())
    if len(inputs)!=1522 or len({r['pdb_id'] for r in inputs})!=1522:
        raise ValueError('Expected all 1522 official assemblies')
    pending=[r for r in inputs if r['status']=='alignment_pending']
    if pending and not args.unavailable_alignments_reason:
        raise ValueError('Deferred alignments must complete or have explicit failure evidence')
    names=[f'foldbench-{w}-{s}-v1' for s in range(4) for w in ['ema','protenix']]
    if args.completion:names += [f'foldbench-{w}-completion-v1' for w in ['ema','protenix']]
    predictions=[];scores=[];failures=[];stages=[]
    for name in names:
        stage=json.loads((args.results/name/'done.json').read_text());stages.append(stage)
        # Paths resolve after extracting the published per-stage score archives.
        for destination, path in [(predictions, stage['predictions']),
                                  (scores, stage['scores'])]:
            frame=pd.read_csv(path)
            frame['prediction_path']=frame.prediction_path.map(
                lambda p: str(Path(p).resolve().relative_to(args.results.resolve())))
            destination.append(frame)
        for path in (args.raw/'.cache'/name/'errors').rglob('*.json'):
            r=json.loads(path.read_text())
            failures.append(dict(model=r['weights'],pdb_id=r['pdb_id'],kind='inference',error=r['error']))
    predictions=pd.concat(predictions,ignore_index=True);scores=pd.concat(scores,ignore_index=True)
    keys=['model','pdb_id','seed','sample']
    if predictions.duplicated(keys).any() or scores.duplicated(keys+['target_index']).any():
        raise ValueError('Overlapping benchmark shards or duplicated score rows')
    if not np.isfinite(predictions.ranking_score).all():
        raise ValueError('Nonfinite model confidence must be resolved explicitly')
    if set(predictions.seed)!={0} or set(predictions['sample'])!={0,1,2,3}:
        raise ValueError('Unexpected inference seed/sample settings')
    if not predictions.groupby(['model','pdb_id']).size().eq(4).all():
        raise ValueError('Incomplete coordinate sample export')
    ready={r['pdb_id'] for r in inputs if r['status']=='ready'}
    for model in ['ema','protenix']:
        generated=set(predictions[predictions.model==model].pdb_id)
        failed={r['pdb_id'] for r in failures if r['model']==model}
        if generated & failed or generated | failed != ready:
            raise ValueError('Unaccounted ready targets for '+model)
        for record in inputs:
            if record['status']=='ready':continue
            failures.append(dict(model=model,pdb_id=record['pdb_id'],kind=record['status'],
                                 error=record.get('error',''),tokens=record.get('tokens')))
    targets=pd.concat([pd.read_csv(p).assign(category=p.stem) for p in sorted((args.foldbench/'targets').glob('*.csv'))],ignore_index=True)
    targets['target_index']=np.arange(len(targets))
    expected=predictions[keys].merge(targets[['pdb_id','target_index']],on='pdb_id')
    check=expected.merge(scores[keys+['target_index']],on=keys+['target_index'],how='outer',indicator=True)
    if not check['_merge'].eq('both').all():raise ValueError('Missing or extra official score rows')
    result=summarize(predictions,scores,targets,args.output)
    scores.to_csv(args.output/'per_prediction.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    (args.output/'per_prediction.csv').unlink()
    predictions.to_csv(args.output/'prediction_reference.csv',index=False)
    pd.DataFrame(failures).to_csv(args.output/'input_inference_failures.csv',index=False)
    scores[scores.status.ne('complete')].to_csv(args.output/'scoring_failures.csv',index=False)
    targets.groupby('category').agg(targets=('target_index','size'),assemblies=('pdb_id','nunique')).to_csv(args.output/'denominators.csv')
    metadata=dict(assemblies=1522,ready_inputs=len(ready),resource_failed_inputs=sum('resource ceiling' in r.get('error','') for r in inputs),
        unavailable_alignments=len(pending),structures=len(predictions),official_score_rows=len(scores),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),stages=stages)
    if args.unavailable_alignments_reason:
        metadata['alignment_failure_evidence']=args.unavailable_alignments_reason.read_text()
    (args.output/'completion.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(result.to_string(index=False))


if __name__=='__main__':main()
