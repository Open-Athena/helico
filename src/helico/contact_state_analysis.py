"""Geometry-only candidate diversity and held-out alternative-reference audits."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from helico.contact_evaluate import load_payload


def aligned_rmsd(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    a,b=a-a.mean(0),b-b.mean(0)
    u,_,v=np.linalg.svd(a.T@b)
    rotation=u@np.diag([1.,1.,np.linalg.det(u@v)])@v
    return float(np.sqrt(((a@rotation-b)**2).sum(1).mean()))


def protein_ca_indices(payload, observed=False):
    f=payload['features']
    names=[''.join(map(chr,row)).strip()
           for row in (f['atom_name_chars'].reshape(-1,4,64).argmax(-1)+32).tolist()]
    result={}
    for atom,name in enumerate(names):
        token=int(f['atom_to_token'][atom])
        if name!='CA' or not payload['protein_token'][token]:continue
        if observed and not f['coordinate_mask'][atom]:continue
        key=(int(f['chain_indices'][token]),int(f['rel_pos'][token]))
        result[key]=atom
    return result


def read_artifact(record, root, fs=None, uri=None):
    path=Path(root)/record['artifact']
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        fs.get_file(uri.rstrip('/')+'/'+record['artifact'],str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['artifact_sha256']:
        raise ValueError('Search artifact checksum mismatch')
    with gzip.open(path,'rb') as handle:
        return torch.load(handle,map_location='cpu',weights_only=False)


def score_alternative_references(records, root, features_root, panel, output, fs=None, uri=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    by_id={pid:pair for pair in panel['pairs'] for pid in [pair['apo'],pair['bound']]}
    rows=[]
    for record in records:
        if record.get('status')!='complete' or 'costs' not in record:continue
        pair=by_id[record['pdb_id']]
        payloads={p:load_payload(Path(features_root)/(p+'.pt.gz')) for p in [pair['apo'],pair['bound']]}
        # Both diagnostic pairs are single-protein chains; residue indices
        # were verified against exactly identical full sequences.
        maps={p:{key[1]:atom for key,atom in protein_ca_indices(x,True).items()} for p,x in payloads.items()}
        common=sorted(set.intersection(*(set(m) for m in maps.values())))
        references={p:payloads[p]['features']['atom_coords'][[maps[p][i] for i in common]].numpy() for p in maps}
        native_separation=aligned_rmsd(references[pair['apo']],references[pair['bound']])
        if abs(native_separation-pair['rmsd'])>.01:raise ValueError('Alternative-reference atom mapping changed')
        sample_map={key[1]:atom for key,atom in protein_ca_indices(payloads[record['pdb_id']]).items()}
        index=[sample_map[i] for i in common]
        artifact=read_artifact(record,root,fs,uri)
        for metric in record['metrics']:
            coords=artifact[f"{metric['method']}-{metric['call']}"]['all_coords'][0,metric['sample']][index].numpy()
            ra=aligned_rmsd(coords,references[pair['apo']]);rb=aligned_rmsd(coords,references[pair['bound']])
            threshold=min(2.,native_separation/2.)
            rows.append(dict(pdb_id=record['pdb_id'],protein=pair['protein'],seed=record['seed'],
                input_state='apo' if record['pdb_id']==pair['apo'] else 'bound',
                method=metric['method'],call=metric['call'],sample=metric['sample'],ranking=metric['ranking'],
                available_seconds=metric['available_seconds'],tree_budget=record['costs']['tree'],
                unmask_budget=record['costs']['unmask'],rmsd_apo=ra,rmsd_bound=rb,
                native_separation=native_separation,recovery_threshold=threshold,
                recovered_apo=ra<threshold,recovered_bound=rb<threshold))
    df=pd.DataFrame(rows);df.to_csv(output/'alternative_state_candidates.csv',index=False)
    summary=[]
    if len(df):
        for (pid,seed,method),frame in df.groupby(['pdb_id','seed','method']):
            for horizon in ['tree','unmask']:
                frame_at_budget=frame[frame.available_seconds<=frame[horizon+'_budget'].iloc[0]]
                if not len(frame_at_budget):continue
                selected=frame_at_budget.loc[frame_at_budget.ranking.idxmax()]
                summary.append(dict(pdb_id=pid,seed=seed,method=method,horizon=horizon,
                    protein=selected.protein,input_state=selected.input_state,candidates=len(frame_at_budget),
                    recovered_apo=bool(frame_at_budget.recovered_apo.any()),
                    recovered_bound=bool(frame_at_budget.recovered_bound.any()),
                    min_rmsd_apo=float(frame_at_budget.rmsd_apo.min()),min_rmsd_bound=float(frame_at_budget.rmsd_bound.min()),
                    selected_rmsd_apo=selected.rmsd_apo,selected_rmsd_bound=selected.rmsd_bound))
    pd.DataFrame(summary).to_csv(output/'alternative_state_matched_time.csv',index=False)
    return df


def candidate_diversity(records, root, features_root, output, fs=None, uri=None):
    """Greedy 2-Angstrom CA clusters, ordered by predicted confidence."""
    rows=[]
    for record in records:
        if record.get('status')!='complete' or 'costs' not in record:continue
        payload=load_payload(Path(features_root)/(record['pdb_id']+'.pt.gz'))
        indices=list(protein_ca_indices(payload).values())
        if len(indices)<3:continue
        artifact=read_artifact(record,root,fs,uri)
        for horizon in ['tree','unmask']:
            for method in ['ordinary','tree','unmask']:
                metrics=[m for m in record['metrics'] if m['method']==method and m['available_seconds']<=record['costs'][horizon]]
                metrics.sort(key=lambda m:-m['ranking'])
                xyz=[artifact[f"{method}-{m['call']}"]['all_coords'][0,m['sample']][indices].numpy() for m in metrics]
                distances=np.zeros((len(xyz),len(xyz)))
                for i in range(len(xyz)):
                    for j in range(i):distances[i,j]=distances[j,i]=aligned_rmsd(xyz[i],xyz[j])
                centers=[]
                for i in range(len(xyz)):
                    if not centers or min(distances[i,j] for j in centers)>=2.:centers.append(i)
                pairwise=distances[np.triu_indices(len(xyz),1)]
                rows.append(dict(pdb_id=record['pdb_id'],seed=record['seed'],horizon=horizon,method=method,
                    candidates=len(xyz),ca_clusters_2A=len(centers),mean_ca_rmsd=float(pairwise.mean()) if len(pairwise) else 0.,
                    max_ca_rmsd=float(pairwise.max()) if len(pairwise) else 0.,
                    cluster_center_mean_lddt=float(np.mean([metrics[i]['observed_lddt'] for i in centers])) if centers else np.nan))
    df=pd.DataFrame(rows);Path(output).mkdir(parents=True,exist_ok=True)
    df.to_csv(Path(output)/'search_diversity.csv',index=False)
    return df
