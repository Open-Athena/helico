"""Analyze completed contact-only trajectories on the prespecified panels."""
import argparse
import json
from pathlib import Path

import pandas as pd

from helico.contact_eval_analysis import bootstrap_mean, summarize_search
from helico.contact_state_analysis import candidate_diversity, score_alternative_references


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    records=[json.loads(p.read_text()) for p in (args.records/'search').rglob('*.json')]
    if len(records)!=54 or any(r['status']!='complete' for r in records):
        raise ValueError('All 27 targets and both seeds must complete before final analysis')
    panel=set(pd.read_csv(root/'data/development_panel.csv').pdb_id)
    primary=[r for r in records if r['pdb_id'] in panel]
    if len(primary)!=48:raise ValueError('Primary 24-target panel is incomplete')
    summarize_search(primary,args.output)
    diversity=candidate_diversity(primary,args.records,args.features,args.output)
    metrics=['candidates','ca_clusters_2A','mean_ca_rmsd','max_ca_rmsd','cluster_center_mean_lddt']
    means=diversity.groupby(['pdb_id','horizon','method'],as_index=False)[metrics].mean()
    paired=[]
    for horizon,frame in means.groupby('horizon'):
        for metric in metrics:
            wide=frame.pivot(index='pdb_id',columns='method',values=metric)
            for method in ['contact7','contact32']:
                paired.append(dict(horizon=horizon,contrast=method+'-minus-ordinary',metric=metric,
                                   **bootstrap_mean(wide[method]-wide.ordinary)))
    pd.DataFrame(paired).to_csv(args.output/'search_diversity_paired.csv',index=False)
    alternative=json.loads((root/'data/alternative_state_panel.json').read_text())
    ids={p[k] for p in alternative['pairs'] for k in ['apo','bound']}
    score_alternative_references([r for r in records if r['pdb_id'] in ids],args.records,args.features,
                                 alternative,args.output/'alternative-states')
    print((args.output/'search_paired_differences.csv').read_text())


if __name__=='__main__':main()
