"""Compare native-close intratoken distances; this is not chemical bond assignment."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from helico.contact_evaluate import load_payload
from helico.contact_state_analysis import read_artifact
from helico.contact_eval_analysis import bootstrap_mean
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--features',type=Path,required=True)
parser.add_argument('--raw',type=Path,required=True)
args=parser.parse_args()
data=Path(__file__).resolve().parent/'data';raw=args.raw;torch.set_num_threads(2)
rows=[]
for pid in pd.read_csv(data/'development_panel.csv').pdb_id:
 f=load_payload(args.features/(pid+'.pt.gz'))['features'];xyz=f['atom_coords'];dist=torch.cdist(xyz,xyz)
 observed=f['coordinate_mask'].bool()&(f['atom_element_idx']!=0)
 mask=(f['atom_to_token'][:,None]==f['atom_to_token'][None,:])&(dist>1.1)&(dist<1.8)&observed[:,None]&observed[None,:]
 i,j=torch.where(mask.triu(1));reference=torch.linalg.vector_norm(xyz[i]-xyz[j],dim=-1)
 for version in ['v1','v2']:
  root=raw/('development-'+version)
  for path in root.glob('generation/*/'+pid+'/*.json'):
   record=json.loads(path.read_text());artifact=read_artifact(record,root)
   for sample,coords in enumerate(artifact['all_coords'][0]):
    difference=(torch.linalg.vector_norm(coords[i]-coords[j],dim=-1)-reference).abs()
    rows.append(dict(version=version,pdb_id=pid,weights=record['weights'],seed=record['seed'],sample=sample,pairs=len(i),mean_distance_error=float(difference.mean()),fraction_over_0p2=float((difference>.2).float().mean())))
 print('DONE',pid,flush=True)
frame=pd.DataFrame(rows)
assert len(frame)==576 and frame.pdb_id.nunique()==24
frame.to_csv(data/'rotation_geometry_per_sample.csv',index=False)
target=frame.groupby(['version','pdb_id','weights'],as_index=False)[['mean_distance_error','fraction_over_0p2']].mean()
summary=[]
for (version,weights),d in target.groupby(['version','weights']):
 for metric in ['mean_distance_error','fraction_over_0p2']:
  summary.append(dict(version=version,weights=weights,metric=metric,**bootstrap_mean(d[metric])))
pd.DataFrame(summary).to_csv(data/'rotation_geometry_summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False),flush=True)
