"""Restartable official FoldBench scoring with explicit failure denominators.

Run this module in the pinned OpenStructure environment. Predictions are
immutable full-input mmCIFs; confidence, never native quality, selects a sample.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def score_prediction(task):
    row, targets, refs, output, ost, timeout, foldbench = task
    folder = Path(output) / row['model'] / row['pdb_id'] / f"s{row['seed']}-{row['sample']}"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'scores.json'
    identity = hashlib.sha256(json.dumps(dict(row=row, targets=targets), sort_keys=True).encode()).hexdigest()
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['identity'] != identity: raise ValueError('Changed prediction or target metadata')
        return saved['scores']
    from evaluation.eval_by_ost import get_structure_value, get_ligand_value
    modes = {'ligand' if t['category']=='interface_protein_ligand' else 'structure' for t in targets}
    failures = {}
    for mode in modes:
        dest = folder / (mode + '.json')
        command = [ost, 'compare-ligand-structures' if mode=='ligand' else 'compare-structures',
                   '-m', row['prediction_path'], '-r', str(Path(refs)/(row['pdb_id']+'.cif')),
                   '--fault-tolerant', '-o', str(dest)]
        command += (['--lddt-pli','--rmsd'] if mode=='ligand' else
                    ['--min-pep-length','4','--min-nuc-length','4','--lddt','--rigid-scores','--tm-score','--dockq'])
        try:
            if not dest.exists():
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
                (folder/(mode+'.log')).write_text(result.stdout+'\n'+result.stderr)
                if result.returncode: raise RuntimeError(f'OST exit {result.returncode}: {result.stderr[-500:]}')
            json.loads(dest.read_text())
        except Exception as error:
            failures[mode] = repr(error)
    scores = []
    for target in targets:
        result = {**row, **target, 'status':'complete'}
        category = target['category']; mode = 'ligand' if category=='interface_protein_ligand' else 'structure'
        a, b = target.get('native_chain_id_1',target.get('chain_id')), target.get('native_chain_id_2',target.get('chain_id'))
        try:
            if mode in failures: raise RuntimeError(failures[mode])
            if mode == 'ligand':
                rmsd, lddt_lp, lddt_pli = get_ligand_value(str(folder/'ligand.json'),a,b)
                if rmsd is None or lddt_pli is None: raise ValueError('Unassigned native ligand')
                result.update(rmsd=rmsd, lddt_lp=lddt_lp, lddt_pli=lddt_pli,
                              success=bool(rmsd<2 and lddt_pli>.8))
            else:
                dockq, irmsd, lrmsd, _, lddt, tm, gdt, rmsd = get_structure_value(str(folder/'structure.json'),a,b)
                if category in {'interface_protein_dna','interface_protein_rna'}:
                    # Official modified DockQv2 handles nucleic-acid interfaces.
                    # A separate bounded subprocess prevents pathological chain
                    # permutations from hanging the scoring pool.
                    na_path = folder / (a+'-'+b+'-dockq.json')
                    if not na_path.exists():
                        command = [sys.executable, '-m', 'helico.contact_foldbench_score',
                            '--dockq-single', row['prediction_path'], str(Path(refs)/(row['pdb_id']+'.cif')),
                            a, b, str(na_path), '--foldbench-root', str(foldbench)]
                        proc=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
                        if proc.returncode: raise RuntimeError(proc.stderr[-500:])
                    na = json.loads(na_path.read_text());dockq,irmsd,lrmsd=na['dockq'],na['irmsd'],na['lrmsd']
                if lddt is None: raise ValueError('OST returned no lDDT')
                if category.startswith('interface') and dockq is None: raise ValueError('Unassigned native interface')
                result.update(dockq=dockq,irmsd=irmsd,lrmsd=lrmsd,lddt=lddt,tm_score=tm,gdt_ts=gdt,rmsd=rmsd)
                if category.startswith('interface'): result['success']=bool(dockq>=.23)
        except Exception as error:
            result.update(status='failed',error=repr(error),success=False)
        scores.append(result)
    marker.write_text(json.dumps(dict(identity=identity,scores=scores),indent=2)+'\n')
    return scores


def summarize(predictions, scores, targets, output):
    output=Path(output)
    scores.to_csv(output/'per_prediction.csv',index=False)
    chosen=predictions.loc[predictions.groupby(['model','pdb_id']).ranking_score.idxmax()]
    selected=scores.merge(chosen[['model','pdb_id','seed','sample']],on=['model','pdb_id','seed','sample'])
    selected.to_csv(output/'model_selected.csv',index=False)
    rows=[]
    for model in predictions.model.unique():
        for category,reference in targets.groupby('category'):
            frame=selected[(selected.model==model)&(selected.category==category)]
            good=frame[frame.status=='complete']
            row=dict(model=model,category=category,denominator=len(reference),predicted=len(frame),
                     scored=len(good),failed_or_missing=len(reference)-len(good))
            if category.startswith('interface'):
                successes=int(good.success.fillna(False).astype(bool).sum())
                row.update(successes=successes,success_rate_all=successes/len(reference),
                           success_rate_scored=successes/len(good) if len(good) else np.nan)
            for metric in ['lddt','dockq','rmsd','lddt_pli','tm_score']:
                if metric in good:row['mean_'+metric]=good[metric].mean()
            rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(output/'summary.csv',index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foldbench-root',type=Path,required=True)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--references',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--ost',default=str(Path(sys.executable).parent/'ost'))
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--dockq-single',nargs=5)
    args=parser.parse_args();sys.path.insert(0,str(args.foldbench_root))
    if args.dockq_single:
        from evaluation.eval_by_dockqv2 import dockq
        model,native,a,b,path=args.dockq_single
        info=dockq(model,native,native_chains=[a,b],small_molecule=False,allowed_mismatches=4)
        result=next(iter(info['best_result'].values()))
        Path(path).write_text(json.dumps(dict(dockq=result['DockQ'],irmsd=result['iRMSD'],lrmsd=result['LRMSD'])))
        return
    predictions=pd.read_csv(args.predictions)
    targets=pd.concat([pd.read_csv(p).assign(category=p.stem) for p in sorted((args.foldbench_root/'targets').glob('*.csv'))],ignore_index=True)
    by_id={pid:[{k:v for k,v in row.items() if pd.notna(v)} for row in frame.to_dict('records')]
           for pid,frame in targets.groupby('pdb_id')}
    tasks=[(row,by_id[row['pdb_id']],str(args.references),str(args.output),args.ost,args.timeout,str(args.foldbench_root))
           for row in predictions.to_dict('records')]
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(score_prediction,tasks)):
            rows.extend(result)
            if i%25==0:print(json.dumps(dict(scored=i+1,total=len(tasks))),flush=True)
    print(summarize(predictions,pd.DataFrame(rows),targets,args.output).to_string(index=False))


if __name__=='__main__':main()
