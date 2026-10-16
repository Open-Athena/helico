"""Log the completed study and immutable HF release to the shared W&B project."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd
import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-revision', required=True)
    parser.add_argument('--foldbench-data', type=Path, required=True)
    parser.add_argument('--protocol-name', default='foldbench_protocol_v2.json')
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    completion = json.loads((args.foldbench_data / 'completion.json').read_text())
    if completion['assemblies'] != 1522 or not completion['stages']:
        raise ValueError('Full benchmark accounting is required before publication')
    receipt = json.loads((experiment / 'data/results_release.json').read_text())
    if receipt['revision'] != args.results_revision or receipt.get('full_foldbench_pending', True):
        raise ValueError('Final HF publication must be verified first')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=experiment, text=True).strip()
    run = wandb.init(project='helico', entity='timodonnell',
        id='exp23-contact-evaluation-results-v1', name='exp23-contact-evaluation-results-v1',
        job_type='evaluation-summary', resume='allow',
        config=dict(results_dataset=receipt['repo_id'], results_revision=args.results_revision,
                    source_commit=source,
                    protocol=json.loads((experiment / 'data' / args.protocol_name).read_text())),
        allow_val_change=True)
    try:
        tables = {}
        for filename in ['summary', 'paired_differences', 'oracle_summary', 'denominators']:
            tables['foldbench/' + filename] = wandb.Table(
                dataframe=pd.read_csv(args.foldbench_data / (filename + '.csv')))
        for filename in ['audit-v1/contact_summary', 'conditioning-v2/conditioning_summary',
                         'conditioning-v2/conditioning_head_summary',
                         'conditioning-repeat-v1/repeat_control_summary', 'validation-structures-v3/structure_paired_differences',
                         'search-v2/search_paired_differences',
                         'contact-hypotheses-v2/search_paired_differences']:
            path = experiment / 'data' / (filename + '.csv')
            tables['validation/' + filename] = wandb.Table(dataframe=pd.read_csv(path))
        for name in ['full-validation', 'contact-conditioning', 'contact-search',
                     'contact-hypotheses', 'alternative-states', 'rotation-geometry', 'contact-conditioning-repeat', 'foldbench']:
            path = experiment / 'plots/corrected' / (name + '.png')
            if path.exists():
                tables['figures/' + name] = wandb.Image(str(path))
        run.log(tables)
        run.summary.update({k: v for k, v in completion.items() if k != 'stages'})
        run.summary['results_url'] = ('https://huggingface.co/datasets/' + receipt['repo_id'] +
                                      '/tree/' + args.results_revision)
        artifact = wandb.Artifact('exp23-evaluation-results', type='evaluation',
            metadata=dict(results_dataset=receipt['repo_id'], revision=args.results_revision))
        artifact.add_dir(str(args.foldbench_data), name='foldbench')
        artifact.add_file(str(experiment / 'data/results_release.json'))
        run.log_artifact(artifact)
        url = run.url
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    (experiment / 'data/wandb_results_release.json').write_text(json.dumps(
        dict(url=url, results_revision=args.results_revision, source_commit=source), indent=2) + '\n')
    print(url)


if __name__ == '__main__':
    main()
