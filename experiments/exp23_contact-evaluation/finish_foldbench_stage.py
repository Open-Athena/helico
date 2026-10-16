"""Verify, export and officially score one completed immutable benchmark shard."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess

import pandas as pd

from helico.contact_foldbench_outputs import export_predictions


def finish_stage(fs, stage, features, raw, output, foldbench, references, ost_python, workers=8):
    experiment=Path(__file__).resolve().parent;work=experiment.parents[1]
    config_path=work/'configs/eval'/('contact-'+stage+'.json')
    config=json.loads(config_path.read_text());uri=config['output_uri']
    package=runpy.run_path(str(experiment/'package_results.py'))['package_stage']
    manifest=package(fs,uri,raw)
    if manifest['config_sha256']!=hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError('Benchmark results do not match the frozen configuration')
    records_root=Path(raw)/'.cache'/stage
    records=[json.loads(p.read_text()) for p in (records_root/'blind').rglob('*.json')]
    failures=[json.loads(p.read_text()) for p in (records_root/'errors').rglob('*.json')]
    for weight in config['weights']:
        complete={r['pdb_id'] for r in records if r['weights']==weight and r['status']=='complete'}
        failed={r['pdb_id'] for r in failures if r['weights']==weight}
        if complete & failed or complete | failed != set(config['target_ids']):
            raise ValueError('Missing, duplicated or conflicting target outcomes')
    output=Path(output)/stage;output.mkdir(parents=True,exist_ok=True)
    predictions=export_predictions(records_root,features,output/'predictions')
    if len(predictions)!=len(records)*config['samples']:
        raise ValueError('Incomplete coordinate sample export')
    scores=output/'scores'
    env={**os.environ,'PYTHONPATH':str(work/'src'),'OMP_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with (output/'scoring.log').open('w') as handle:
        subprocess.run([str(ost_python),'-m','helico.contact_foldbench_score',
            '--foldbench-root',str(foldbench),'--predictions',str(output/'predictions/prediction_reference.csv'),
            '--references',str(references),'--output',str(scores),'--workers',str(workers)],
            cwd=work,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    per_prediction=pd.read_csv(scores/'per_prediction.csv')
    result=dict(stage=stage,config_sha256=manifest['config_sha256'],
        input_targets=len(config['target_ids']),inference_failures=len(failures),
        structures=len(predictions),target_scores=len(per_prediction),
        failed_target_scores=int(per_prediction.status.ne('complete').sum()),
        predictions=str(output/'predictions/prediction_reference.csv'),
        scores=str(scores/'per_prediction.csv'))
    (output/'done.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
